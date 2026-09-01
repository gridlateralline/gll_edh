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

"""The harness's own promises.

Chiefly the two the whole challenge rests on -- that a controller sees neither
a price nor a neighbour -- plus the invariants that make the scoring mean
anything.
"""

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sandbox.controller import (
    LocalObservation,
    Memory,
    base_controller,
    hour_of_day,
    passive_controller,
)
from sandbox.evaluate import Submission, evaluate
from sandbox.export import feeder_dataframe, to_dataframe
from sandbox.metrics import coincidence_factor, revenue_adequate, score
from sandbox.numpy_bridge import numpy_controller
from sandbox.observation import to_local
from sandbox.rollout import Trajectory, build_env, rollout, rollout_seeds
from sandbox.scenarios import reference_scenario
from sandbox.tariff import MyTariff, my_tariff
from sandbox.tuning import parameter_grid, tune

DAY = 96


@pytest.fixture(scope="module")
def population():
    return reference_scenario()


@pytest.fixture(scope="module")
def env(population):
    return build_env(population, time_limit=DAY)


# ---------------------------------------------------------------------------
# The two guarantees
# ---------------------------------------------------------------------------


def test_the_controller_is_handed_a_price_free_observation(population, env) -> None:
    """Everything a controller receives, enumerated. If a price ever appears in
    this list the challenge is over: households would react rather than
    anticipate, and real settlement lags past the end of an episode anyway."""
    seen: list[LocalObservation] = []

    def spy(obs, carry, params, key):
        seen.append(obs)
        return obs.load_kw, carry

    rollout(
        base_controller().replace(fn=spy),
        population,
        jax.random.PRNGKey(0),
        n_steps=4,
        env=env,
        fast=False,
    )

    fields = set(seen[0].as_dict())
    assert fields == {
        "time_sin",
        "time_cos",
        "voltage_pu",
        "meter_kw",
        "load_kw",
        "load_forecast_kw",
        "pv_available_kw",
        "soc_kwh",
        "soc_headroom_kwh",
        "bat_charge_max_kw",
        "bat_discharge_max_kw",
        "p_min_kw",
        "p_max_kw",
    }
    assert not any("price" in name or "chf" in name or "bill" in name for name in fields)


def test_a_controller_cannot_reach_a_neighbour(population, env) -> None:
    """Under vmap every field is a scalar, so there is no agent axis to index.
    That is what makes the isolation structural rather than a rule in a README."""
    shapes: list[tuple[int, ...]] = []

    def spy(obs, carry, params, key):
        shapes.append(jnp.shape(obs.voltage_pu))
        return obs.load_kw, carry

    rollout(
        base_controller().replace(fn=spy),
        population,
        jax.random.PRNGKey(0),
        n_steps=2,
        env=env,
    )
    assert shapes and all(shape == () for shape in shapes)


def test_any_action_at_all_is_survivable(population, env) -> None:
    """A controller may return nonsense; the harness clips and the environment
    projects. There is no such thing as a crashing controller."""
    for value in (1e6, -1e6, 0.0):
        trajectory = rollout(
            base_controller().replace(fn=lambda o, c, p, k, v=value: (jnp.float32(v), c)),
            population,
            jax.random.PRNGKey(0),
            n_steps=8,
            env=env,
        )
        assert bool(jnp.all(trajectory.valid))
        assert bool(jnp.all(jnp.isfinite(trajectory.p_realized_kw)))


def test_the_clock_helper_recovers_the_actual_hour(population, env) -> None:
    """The clock is a sine/cosine pair so it stays continuous across midnight,
    which means no single component of it is an hour.

    `12 * (1 - time_cos)` is the obvious-looking mistake: it reads 24 at noon
    and is symmetric about it, so "after 13:00" silently also matches 11:00.
    It shipped in the cookbook once and made a charge-delay parameter do
    nothing."""
    state, timestep = env.reset(jax.random.PRNGKey(0))
    observation = to_local(env.environment, timestep.observation, state)

    expected = float(state.time_state.day_step) * 24.0 / 96.0
    recovered = float(hour_of_day(observation)[0])
    assert abs(recovered - expected) <= 0.25, (recovered, expected)

    naive = 12.0 * (1.0 - float(observation.time_cos[0]))
    assert abs(naive - expected) > 0.5, "the mistake this test exists to catch"


def test_my_idea_is_wired_end_to_end(population) -> None:
    """The one file a participant edits must actually reach the scorer, for
    both seams, without them assembling anything."""
    from sandbox.check import my_controller_as_bundle, my_tariff_factory

    controller = my_controller_as_bundle()
    env = build_env(population, time_limit=DAY, tariff=my_tariff_factory())
    trajectory = rollout(controller, population, jax.random.PRNGKey(0), DAY, env=env)

    assert bool(jnp.all(trajectory.valid))
    chex.assert_shape(trajectory.settlement_chf, (DAY, population.num_pq))


# ---------------------------------------------------------------------------
# Tariff
# ---------------------------------------------------------------------------


def test_the_congestion_charge_only_redistributes(population) -> None:
    """It has to sum to zero across connection points, or the tariff is a tax
    or a subsidy rather than a price. This is what the revenue gate checks at
    scale; here it is checked exactly."""
    model = build_env(population, time_limit=DAY).environment
    tariff = MyTariff(model.prosumer, headroom_kwh=1.0, price_chf_per_kwh=2.0)

    for flows in (
        jnp.linspace(-4.0, 6.0, population.num_pq),
        jnp.full((population.num_pq,), 3.0),
        jnp.zeros((population.num_pq,)),
    ):
        assert abs(float(tariff.congestion_charge(flows).sum())) < 1e-4


def test_the_revenue_gate_holds_behaviour_fixed(population) -> None:
    """The submitted tariff, scored against unchanged behaviour, must collect
    what fair LEG collects. Comparing re-tuned cells instead would fail every
    tariff that actually worked -- households that export less earn less, and
    that is the tariff succeeding."""
    key = jax.random.PRNGKey(0)
    base = base_controller()
    reference = score(
        rollout(base, population, key, DAY, env=build_env(population, time_limit=DAY)),
        population,
    )
    with_tariff = score(
        rollout(
            base,
            population,
            key,
            DAY,
            env=build_env(population, time_limit=DAY, tariff=my_tariff),
        ),
        population,
    )
    assert revenue_adequate(with_tariff, reference)


# ---------------------------------------------------------------------------
# Tuning
# ---------------------------------------------------------------------------


def test_tuning_sweeps_a_subset_without_deleting_the_rest(population) -> None:
    """A grid over two of three parameters must leave the third alone. Merging
    rather than substituting is the difference between a working sweep and a
    KeyError halfway through a scoring run."""
    controller = base_controller()
    params, table = tune(
        controller,
        population,
        {"export_cap_kw": [1.0e3, 3.0]},
        n_steps=DAY,
        seeds=1,
    )
    assert set(params) == {"export_cap_kw"}
    assert table.shape == (2,)
    merged = {**controller.params, **params}
    assert set(merged) == set(controller.params)


def test_the_tuner_maximises_the_household_bill(population) -> None:
    """Never the grid score. A tuner optimising network welfare would measure
    what a central planner could achieve rather than what a price can induce."""
    candidates = {"export_cap_kw": [1.0e3, 2.0]}
    best, table = tune(base_controller(), population, candidates, n_steps=DAY, seeds=2)
    entries = parameter_grid(candidates)
    assert entries[int(np.argmax(table))] == best
    # Capping exports throws energy away, so the free hand must win on the bill.
    assert float(best["export_cap_kw"]) > 100.0


# ---------------------------------------------------------------------------
# Tiers
# ---------------------------------------------------------------------------


def test_the_numpy_tier_agrees_with_the_jax_one(population, env) -> None:
    """Same rule, written twice. If these diverged the NumPy path would be a
    trap rather than a convenience."""

    @numpy_controller
    def in_numpy(obs, carry, params):
        surplus = max(float(obs["pv_available_kw"]) - float(obs["load_kw"]), 0.0)
        export = max(surplus - float(obs["bat_charge_max_kw"]), 0.0)
        target = float(obs["load_kw"]) + export
        if target > float(obs["p_max_kw"]):  # a real Python branch
            target = float(obs["p_max_kw"])
        return target, Memory(
            p_prev_kw=np.float32(target),
            voltage_ewma_pu=carry.voltage_ewma_pu,
            intervals=carry.intervals + 1,
        )

    key = jax.random.PRNGKey(3)
    in_jax = rollout(base_controller(), population, key, DAY, env=env)
    numpy_run = rollout(in_numpy, population, key, DAY, env=env)
    chex.assert_trees_all_close(numpy_run.p_set_kw, in_jax.p_set_kw, atol=1e-4)


def test_a_seed_ensemble_is_just_a_vmap(population) -> None:
    """Scoring on one week rewards luck. Because a rollout is a pure function
    of its key, an ensemble costs no extra engineering."""
    keys = jax.random.split(jax.random.PRNGKey(0), 3)
    ensemble = rollout_seeds(base_controller(), population, keys, n_steps=DAY)
    chex.assert_shape(ensemble.settlement_chf, (3, DAY, population.num_pq))
    # Different weather really is different.
    assert float(jnp.abs(ensemble.settlement_chf[0] - ensemble.settlement_chf[1]).sum()) > 0.0


# ---------------------------------------------------------------------------
# Metrics and export
# ---------------------------------------------------------------------------


def test_perfect_synchrony_reads_as_a_coincidence_factor_of_one() -> None:
    """The metric's meaning, pinned. Identical households peaking together need
    a connection as large as the sum of their peaks; that is the diversity a
    network is planned on, and losing it is the harm."""
    steps, households = 40, 5
    shape = (steps, households)
    block = steps // households

    # Everyone drawing the same profile at the same moment.
    together = jnp.tile(jnp.sin(jnp.linspace(0.0, jnp.pi, steps))[:, None], (1, households))

    # The same total energy, arranged so no two households ever overlap. Peak
    # of the sum then equals one household's peak, and the factor is exactly
    # 1/households -- the diversity a feeder is planned on.
    slots = jnp.arange(steps)[:, None] // block == jnp.arange(households)[None, :]
    staggered = slots.astype(jnp.float32)

    def as_trajectory(meter: chex.Array) -> Trajectory:
        return Trajectory(
            p_set_kw=meter,
            p_realized_kw=meter,
            meter_kwh=meter * 0.25,
            reward_chf=jnp.zeros(shape),
            settlement_chf=jnp.zeros(shape),
            pv_available_kw=jnp.zeros(shape),
            pv_realized_kw=jnp.zeros(shape),
            q_meter_kvarh=jnp.zeros(shape),
            transformer_kw=meter.sum(1),
            transformer_kvar=jnp.zeros((steps,)),
            losses_kw=jnp.zeros((steps,)),
            voltage_pu=jnp.ones(shape),
            day_step=jnp.arange(steps),
            valid=jnp.ones((steps,), dtype=bool),
        )

    assert coincidence_factor(as_trajectory(together)) == pytest.approx(1.0, abs=1e-5)
    assert coincidence_factor(as_trajectory(staggered)) == pytest.approx(1.0 / households, abs=1e-5)


def test_the_exported_frame_carries_the_households_with_no_agent(population, env) -> None:
    """Six connection points have no inverter and are absent from every
    agent-indexed array. A fairness audit that cannot see them is asking the
    wrong question."""
    trajectory = rollout(base_controller(), population, jax.random.PRNGKey(0), DAY, env=env)
    frame = to_dataframe(trajectory, population, label="base")

    assert len(frame) == DAY * population.num_pq
    assert set(frame.household) == {"tenant", "pv_only", "pv_battery", "large_flex"}
    tenants = frame[frame.household == "tenant"]
    assert len(tenants) == DAY * 6
    assert tenants.p_set_kw.isna().all(), "a tenant sets nothing"
    assert tenants.settlement_chf.sum() < 0.0, "a household that only consumes pays"

    feeder = feeder_dataframe(trajectory)
    assert len(feeder) == DAY
    assert feeder.transformer_kw.min() < 0.0, "the feeder should export at some point"


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


def test_a_submission_scores_four_distinguishable_cells(population) -> None:
    """The whole pipeline, and the reason it is four rollouts. Without the
    re-tuning step a submitted tariff changes nothing physical at all."""
    evaluation = evaluate(
        Submission(
            controller=passive_controller(),
            tariff=my_tariff,
            candidates={"export_cap_kw": [1.0e3, 3.0]},
        ),
        population,
        n_steps=DAY,
        seeds=1,
    )
    assert set(evaluation.cells) == {
        "fair_leg/base",
        "fair_leg/submitted",
        "submitted/base",
        "submitted/submitted",
    }
    assert evaluation.revenue_check is not None
    assert evaluation.revenue_adequate
    assert "fair_leg/base" in str(evaluation)
