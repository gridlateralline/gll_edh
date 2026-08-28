# Copyright 2026 ewz - Zurich Municipal Electric Utility.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The loop. Infrastructure -- you should not need to edit this file.

Two things it guarantees, and both matter:

**The controller never sees a price.** It is handed a
:class:`~sandbox.observation.LocalObservation` and its own carry, and nothing
else. ``timestep.reward`` is the settlement for the interval that just
ended -- routing it to the controller would collapse a settlement lag of days
into one interval and delete the premise of the challenge.

**The controller never sees a neighbour.** It is ``jax.vmap``'d per household,
so there is no agent axis inside it to index.

Two speeds, same semantics:

* ``fast=True`` -- ``lax.scan`` over a jitted step, ``vmap`` over households.
* ``fast=False`` -- a Python loop, and a list comprehension instead of vmap.
  Roughly a hundred times slower, and worth every bit of it while you are
  writing a controller: values are concrete, so ``if`` works, ``print`` works,
  and a traceback points at your line instead of at a tracer.

Write it eager, score it fast. The results are identical.
"""

from typing import Any, Optional

import chex
import jax
import jax.numpy as jnp
from gll_env.components.environment import EnvironmentDynamics
from gll_env.components.grid import GridDynamics
from gll_env.env import ProsumerGrid
from gll_env.factories import (
    daytime_dynamics,
    grid_code,
    newton_raphson,
    prosumer_dynamics,
    reward_fn,
)
from gll_env.generator import DynamicsGenerator
from gll_env.observer import RawObserver

from sandbox.controller import Controller
from sandbox.observation import LocalObservation, to_action, to_local
from sandbox.scenarios import (
    EPISODE_STEPS,
    FEEDER_IMPEDANCE_SCALE,
    Population,
    grid_arrays,
)


@chex.dataclass(frozen=True)
class Trajectory:
    """One episode, time-major. Every array has a leading axis of ``n_steps``.

    Attributes:
        p_set_kw: (T, num_agents) What each controller asked for.
        p_realized_kw: (T, num_agents) What it actually got, after the
            environment's feasibility projection. The gap between this and
            ``p_set_kw`` is how much a controller is asking for and not
            receiving.
        meter_kwh: (T, num_pq) Net energy at every connection point, tenants
            included.
        reward_chf: (T, num_agents) Per-agent settlement, aligned with the
            interval it describes.
        settlement_chf: (T, num_pq) The same settlement over *all* connection
            points. Tenants have no agent and are absent from ``reward_chf``
            entirely, so this is what any fairness question must read.
        pv_available_kw: (T, num_agents) What each roof could have produced.
        pv_realized_kw: (T, num_agents) What it was allowed to. The difference
            is curtailment -- generation thrown away because the inverter had
            nowhere to put it. Under a Q(U) grid code this, rather than
            over-voltage, is usually what binds: the code holds voltage by
            absorbing reactive power and, past that, by curtailing.
        q_meter_kvarh: (T, num_pq) Reactive energy at each connection point.
            Mostly the grid code's doing, and a real cost to the network.
        transformer_kw: (T,) Active power through the substation transformer,
            positive when the feeder draws from the grid and negative when it
            exports. The quantity that decides whether a transformer must be
            replaced -- and the one constraint a meshed urban network cannot
            mesh its way out of, since meshing buys voltage stiffness and no
            thermal capacity at all.
        transformer_kvar: (T,) Reactive power through it.
        losses_kw: (T,) Network losses. Quadratic in flow, so synchronised
            behaviour costs more than its average suggests.
        voltage_pu: (T, num_bus) Voltage magnitude everywhere.
        day_step: (T,) Interval within the day, for time-of-day breakdowns.
        valid: (T,) Whether the power flow converged.
    """

    p_set_kw: chex.Array
    p_realized_kw: chex.Array
    meter_kwh: chex.Array
    reward_chf: chex.Array
    settlement_chf: chex.Array
    pv_available_kw: chex.Array
    pv_realized_kw: chex.Array
    q_meter_kvarh: chex.Array
    transformer_kw: chex.Array
    transformer_kvar: chex.Array
    losses_kw: chex.Array
    voltage_pu: chex.Array
    day_step: chex.Array
    valid: chex.Array


def build_model(
    population: Population, impedance_scale: float = FEEDER_IMPEDANCE_SCALE
) -> EnvironmentDynamics:
    """Assemble the environment model for a population.

    Built component by component rather than through
    :func:`gll_env.factories.environment_model`, for one reason: the grid is
    constructed from *modified* asset arrays. The bundled CIGRE feeder is a
    short urban one and the congestion this challenge is about happens on
    suburban feeders, so the LV network is weakened to the IEC 60725 reference
    impedance. See :data:`sandbox.scenarios.FEEDER_IMPEDANCE_SCALE`.

    Everything else comes from the factories unchanged.
    """
    config = population.config
    time = daytime_dynamics(config.n_steps_per_day)
    grid = GridDynamics(
        **grid_arrays(impedance_scale),
        nr=newton_raphson(config.grid.get("newton_raphson", {})),
        time=time,
    )
    prosumer = prosumer_dynamics(config.prosumer, num_pq=grid.num_pq, time=time)
    model = EnvironmentDynamics(
        prosumer=prosumer,
        grid=grid,
        time=time,
        grid_code=grid_code(config.get("grid_code", {}), prosumer),
    )
    return model.replace(reward=reward_fn(config.reward, model))


def build_env(
    population: Population,
    time_limit: int = EPISODE_STEPS,
    impedance_scale: float = FEEDER_IMPEDANCE_SCALE,
) -> ProsumerGrid:
    """The environment for a population.

    Uses :class:`~gll_env.observer.RawObserver` so the timestep carries the
    full typed observation; the harness slices it per household itself rather
    than consuming a pre-flattened MARL vector.
    """
    model = build_model(population, impedance_scale)
    return ProsumerGrid(
        generator=DynamicsGenerator(model),
        observer=RawObserver(model),
        time_limit=time_limit,
    )


def _tile_carry(carry: Any, num_agents: int) -> Any:
    """Broadcast one household's starting memory across the population."""
    return jax.tree_util.tree_map(
        lambda leaf: jnp.broadcast_to(leaf, (num_agents, *jnp.shape(leaf))).copy(), carry
    )


def _record(
    model: Any,
    state: Any,
    p_set_kw: chex.Array,
    timestep: Any,
    pv_available_kw: chex.Array,
) -> dict[str, chex.Array]:
    step_h = float(model.time.step_duration_h)
    prosumer_state = state.prosumer_state
    solar = timestep.observation.prosumer_observation.inverter_observation.solar_observation
    return {
        "p_set_kw": p_set_kw,
        "p_realized_kw": jnp.real(prosumer_state.inverter_state.s_inv_realized_kvah) / step_h,
        "meter_kwh": jnp.real(prosumer_state.s_pq_realized_kvah),
        # Recorded against the availability the controller was shown, so the
        # two describe the same interval and their difference is curtailment.
        "pv_available_kw": pv_available_kw,
        "pv_realized_kw": jnp.asarray(solar.sol_realized) / step_h,
        "q_meter_kvarh": jnp.imag(prosumer_state.s_pq_realized_kvah),
        # The slack IS the medium-voltage side of the transformer, so its
        # injection is the whole feeder's throughput. Load convention:
        # positive means the feeder is DRAWING from the grid, negative means
        # it is exporting into it.
        "transformer_kw": model.grid.pu_to_kw(
            jnp.real(state.grid_state.bus_power_injection_pu)[model.grid.slack_id[0]]
        ),
        "transformer_kvar": model.grid.pu_to_kw(
            jnp.imag(state.grid_state.bus_power_injection_pu)[model.grid.slack_id[0]]
        ),
        # Sum every bus injection and what is left is what the network ate.
        "losses_kw": model.grid.pu_to_kw(
            jnp.sum(jnp.real(state.grid_state.bus_power_injection_pu))
        ),
        "reward_chf": timestep.reward,
        "settlement_chf": timestep.extras["reward"].settlement_chf,
        "voltage_pu": jnp.abs(state.grid_state.bus_voltage_pu),
        "day_step": state.time_state.day_step,
        "valid": state.valid,
    }


def rollout(
    controller: Controller,
    population: Population,
    key: chex.PRNGKey,
    n_steps: int = EPISODE_STEPS,
    params: Optional[Any] = None,
    env: Optional[ProsumerGrid] = None,
    fast: bool = True,
) -> Trajectory:
    """Run one episode and record it.

    Args:
        controller: The household controller, applied to every agent.
        population: Which households live on the feeder.
        key: Seeds the episode. The same key gives the same episode, always.
        n_steps: Intervals to simulate.
        params: Overrides ``controller.params``. This is the handle a tuner
            turns; the controller itself is untouched.
        env: Reuse a prebuilt environment. Building one parses config and
            loads the grid asset, so a sweep should build once.
        fast: ``False`` for the eager debugging path.

    Returns:
        A :class:`Trajectory`, time-major.
    """
    env = env or build_env(population, time_limit=n_steps)
    model = env.environment
    num_agents = model.num_agents
    params = controller.params if params is None else params

    # The controller is written for ONE household. This is where that becomes
    # a guarantee rather than a convention.
    per_agent = jax.vmap(controller.fn, in_axes=(0, 0, None, 0))

    def decide(
        observation: Any, state: Any, carry: Any, key: chex.PRNGKey
    ) -> tuple[chex.Array, Any, LocalObservation]:
        local = to_local(model, observation, state)
        keys = jax.random.split(key, num_agents)
        p_set_kw, carry = per_agent(local, carry, params, keys)
        return p_set_kw, carry, local

    def decide_eager(
        observation: Any, state: Any, carry: Any, key: chex.PRNGKey
    ) -> tuple[chex.Array, Any, LocalObservation]:
        """vmap replaced by an ordinary loop. Same answer, readable failures."""
        local = to_local(model, observation, state)
        keys = jax.random.split(key, num_agents)
        results = [
            controller.fn(
                jax.tree_util.tree_map(lambda leaf, i=i: leaf[i], local),
                jax.tree_util.tree_map(lambda leaf, i=i: leaf[i], carry),
                params,
                keys[i],
            )
            for i in range(num_agents)
        ]
        actions = jnp.stack([jnp.asarray(r[0], dtype=jnp.float32) for r in results])
        carries = jax.tree_util.tree_map(
            lambda *leaves: jnp.stack(leaves), *[r[1] for r in results]
        )
        return actions, carries, local

    reset_key, loop_key = jax.random.split(key)
    state, timestep = env.reset(reset_key)
    carry = _tile_carry(controller.init_carry(), num_agents)

    def body(
        loop_state: tuple[Any, Any, Any, chex.PRNGKey], _: None
    ) -> tuple[tuple[Any, Any, Any, chex.PRNGKey], dict[str, chex.Array]]:
        state, observation, carry, key = loop_state
        key, decide_key = jax.random.split(key)
        p_set_kw, carry, local = decide(observation, state, carry, decide_key)
        # NOTE: timestep.reward is deliberately NOT passed on. See module docs.
        new_state, new_timestep = env.step(state, to_action(model, p_set_kw))
        record = _record(model, new_state, p_set_kw, new_timestep, local.pv_available_kw)
        return (new_state, new_timestep.observation, carry, key), record

    if fast:
        _, records = jax.lax.scan(
            body, (state, timestep.observation, carry, loop_key), None, length=n_steps
        )
        return Trajectory(**records)

    observation = timestep.observation
    collected: list[dict[str, chex.Array]] = []
    for _ in range(n_steps):
        loop_key, decide_key = jax.random.split(loop_key)
        p_set_kw, carry, local = decide_eager(observation, state, carry, decide_key)
        state, timestep = env.step(state, to_action(model, p_set_kw))
        observation = timestep.observation
        collected.append(_record(model, state, p_set_kw, timestep, local.pv_available_kw))

    return Trajectory(
        **{key: jnp.stack([record[key] for record in collected]) for key in collected[0]}
    )


def rollout_seeds(
    controller: Controller,
    population: Population,
    keys: chex.Array,
    n_steps: int = EPISODE_STEPS,
    params: Optional[Any] = None,
) -> Trajectory:
    """Run one episode per key, batched. Adds a leading seed axis to everything.

    Scoring on a single episode rewards luck. Because the whole rollout is a
    pure function of its key, an ensemble is one ``vmap`` and costs no more
    engineering than a single run.
    """
    env = build_env(population, time_limit=n_steps)

    def one(key: chex.PRNGKey) -> Trajectory:
        return rollout(
            controller, population, key, n_steps=n_steps, params=params, env=env, fast=True
        )

    return jax.vmap(one)(keys)


def local_observations(
    env: ProsumerGrid, state: Any, observation: Any
) -> LocalObservation:
    """What every household sees right now. Exposed for notebooks and tests."""
    return to_local(env.environment, observation, state)
