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

"""One of the two seams you edit: the household controller.

A controller decides one number per interval -- **net active power at the
inverter, in kW, positive when injecting** -- for **one** household. The
harness ``jax.vmap``s it over the population.

That is not only a speed trick. Because your function is written for a single
household, there is no agent axis inside it and you *cannot* reference a
neighbour, even by accident. vmap is the fairness contract.

The signature::

    def controller(obs, carry, params, key) -> (p_set_kw, carry)

* ``obs``    -- :class:`~sandbox.observation.LocalObservation`, all scalars.
* ``carry``  -- your household's memory, carried between intervals. Must be a
  fixed-shape, fixed-dtype pytree: no growing lists, no changing dtypes.
* ``params`` -- your tunable numbers, shared across the population. Tune them
  across episodes; there is no price to read within one.
* ``key``    -- a fresh PRNG key per household per interval. Use it if you
  want to break symmetry -- see the note on desynchronization below.

Return anything you like; the harness clips to ``[p_min_kw, p_max_kw]`` and
the environment projects whatever survives onto the physically feasible set.
A controller cannot crash the simulation.

On the carry, and desynchronization
-----------------------------------
The carry is more useful than it looks precisely *because* there is no price.
It is where a load forecast, a voltage trend, or "how long since I last
curtailed" lives. It is also where **deliberate staggering** lives: a
household that remembers what it just did can offset itself against its
neighbours. Hysteresis and randomized start times are legitimate answers to
herding, and they are pure carry mechanisms.
"""

from typing import Any, Callable, Protocol

import chex
import jax.numpy as jnp

from sandbox.observation import LocalObservation


@chex.dataclass(frozen=True)
class Memory:
    """The default carry. Replace it with your own -- any fixed pytree works.

    Attributes:
        p_prev_kw: What this household set last interval.
        voltage_ewma_pu: Exponentially weighted mean of its own bus voltage,
            a cheap read on whether local congestion is building rather than
            merely present.
        intervals: Count of intervals elapsed, for anything phase-dependent.
    """

    p_prev_kw: chex.Array
    voltage_ewma_pu: chex.Array
    intervals: chex.Array


def init_memory() -> Memory:
    """A single household's starting memory -- no agent axis."""
    return Memory(
        p_prev_kw=jnp.float32(0.0),
        voltage_ewma_pu=jnp.float32(1.0),
        intervals=jnp.int32(0),
    )


class ControllerFn(Protocol):
    """``(obs, carry, params, key) -> (p_set_kw, carry)`` for one household."""

    def __call__(
        self,
        obs: LocalObservation,
        carry: Any,
        params: Any,
        key: chex.PRNGKey,
    ) -> tuple[chex.Array, Any]: ...


@chex.dataclass(frozen=True)
class Controller:
    """A controller bundled with the two things the harness needs to run it.

    Attributes:
        name: Label used in results and plots.
        fn: The per-household decision function.
        params: Tunable parameters, shared across the population.
        init_carry: Builds one household's starting memory.

    Parameters are shared deliberately. Heterogeneity lives in the *state* --
    a tenant has ``p_min_kw == p_max_kw == 0`` and the same parameters produce
    no action from them -- so one tuning run covers a mixed population and
    there is no per-agent best-response game to chase. Supply parameters with
    a leading agent axis if you want per-household values anyway.
    """

    name: str
    fn: ControllerFn
    params: Any
    init_carry: Callable[[], Any]


def clip_to_feasible(p_set_kw: chex.Array, obs: LocalObservation) -> chex.Array:
    """Clamp a request into this household's own feasible interval.

    The harness does this for you. It is exported because doing it *inside* a
    controller is often what you want: a rule that saturates should know it
    saturated, and can then record that in its carry.
    """
    return jnp.clip(p_set_kw, obs.p_min_kw, obs.p_max_kw)


def update_memory(carry: Memory, obs: LocalObservation, p_set_kw: chex.Array) -> Memory:
    """Roll the default carry forward. ``voltage_ewma_pu`` uses a 24-interval
    (six hour) time constant -- long enough to describe the neighbourhood
    rather than this instant, short enough to move within a day."""
    alpha = 1.0 / 24.0
    return Memory(
        p_prev_kw=p_set_kw,
        voltage_ewma_pu=(1.0 - alpha) * carry.voltage_ewma_pu + alpha * obs.voltage_pu,
        intervals=carry.intervals + 1,
    )


# ---------------------------------------------------------------------------
# The base controller
# ---------------------------------------------------------------------------


def greedy_self_consumption(
    obs: LocalObservation,
    carry: Memory,
    params: Any,
    key: chex.PRNGKey,
) -> tuple[chex.Array, Memory]:
    """Cover your own load, bank the rest. What every home battery does by default.

    Setting the inverter to exactly the household's own consumption drives the
    meter to zero, and the inverter's internal dispatch does the rest: solar
    first, battery absorbing whatever is left over, charging on surplus and
    discharging on shortfall. Export happens only once the battery is full,
    import only once it is empty.

    It is realistic -- this is the out-of-the-box behaviour of essentially
    every residential storage product -- and it is also **the thing that
    causes the problem**. Every roof peaks at noon, so every battery charges
    at noon and every battery is full by early afternoon, at which point the
    whole feeder exports at once. Every household cooks at seven, so every
    battery discharges together and empties together. No price is involved;
    the correlation is in the weather and the working day.

    Takes no parameters, deliberately: it is the floor, not a design.
    """
    del params, key
    # Cover your own load first. The inverter dispatches solar before battery,
    # so asking for exactly the load drives the meter to zero and the battery
    # takes up whatever is left over -- charging on surplus, discharging on
    # shortfall.
    #
    # The second term is what stops that from silently curtailing. Once the
    # battery cannot absorb any more, surplus generation has nowhere to go and
    # the inverter throws it away rather than exporting it. A real household
    # exports. So ask for the part of the surplus the battery cannot take.
    surplus_kw = jnp.maximum(obs.pv_available_kw - obs.load_kw, 0.0)
    export_kw = jnp.maximum(surplus_kw - obs.bat_charge_max_kw, 0.0)
    p_set_kw = clip_to_feasible(obs.load_kw + export_kw, obs)
    return p_set_kw, update_memory(carry, obs, p_set_kw)


def base_controller() -> Controller:
    """The reference household every submitted tariff is scored against."""
    return Controller(
        name="greedy_self_consumption",
        fn=greedy_self_consumption,
        params={},
        init_carry=init_memory,
    )


def passive(
    obs: LocalObservation,
    carry: Memory,
    params: Any,
    key: chex.PRNGKey,
) -> tuple[chex.Array, Memory]:
    """Never use the battery: whatever the roof makes goes straight to the grid.

    The do-nothing anchor for the economics metrics. A tariff so punitive that
    nobody moves lands here, which is how the scorer tells "solved the
    problem" apart from "suppressed all activity".
    """
    del params, key
    p_set_kw = clip_to_feasible(obs.pv_available_kw, obs)
    return p_set_kw, update_memory(carry, obs, p_set_kw)


def passive_controller() -> Controller:
    return Controller(name="passive", fn=passive, params={}, init_carry=init_memory)


# ---------------------------------------------------------------------------
# Your controller
# ---------------------------------------------------------------------------


def my_controller(
    obs: LocalObservation,
    carry: Memory,
    params: dict[str, chex.Array],
    key: chex.PRNGKey,
) -> tuple[chex.Array, Memory]:
    """**EDIT ME.** Starts as greedy self-consumption with a voltage nudge.

    The nudge is the smallest possible gesture at the real question: back off
    injecting when your own terminal voltage is already high, since that is
    the local, measurable signature of a congested feeder. It is deliberately
    crude -- it reacts to *this* interval's voltage, so every household backs
    off in the same interval, which is exactly the herding this challenge is
    about. Beating it means anticipating instead.

    Ideas that need no price: use ``load_forecast_kw`` rather than ``load_kw``;
    hold charge back for the evening peak using the clock; watch
    ``carry.voltage_ewma_pu`` for a trend rather than a level; stagger against
    your neighbours using ``key``.
    """
    del key
    headroom = jnp.maximum(params["voltage_setpoint_pu"] - obs.voltage_pu, 0.0)
    droop_kw = params["droop_kw_per_pu"] * headroom
    target_kw = jnp.minimum(obs.load_kw + droop_kw, obs.load_kw + params["export_cap_kw"])
    p_set_kw = clip_to_feasible(target_kw, obs)
    return p_set_kw, update_memory(carry, obs, p_set_kw)


def my_controller_params() -> dict[str, chex.Array]:
    """Starting parameters. These are what you tune across episodes."""
    return {
        "voltage_setpoint_pu": jnp.float32(1.06),
        "droop_kw_per_pu": jnp.float32(0.0),
        "export_cap_kw": jnp.float32(0.0),
    }


def my_controller_bundle() -> Controller:
    return Controller(
        name="my_controller",
        fn=my_controller,
        params=my_controller_params(),
        init_carry=init_memory,
    )
