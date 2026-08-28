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

"""Acceptance tests for the reference scenario.

These are not unit tests. They assert that the challenge has *signal*: that
the naive population actually stresses the feeder, and that a controller can
do something about it. Both are properties of a calibration, and a calibration
drifts -- change the population, the feeder, or the grid code, and one of
these will catch it before a leaderboard built on it does.
"""

import chex
import jax
import jax.numpy as jnp
import pytest

from sandbox.controller import (
    Controller,
    base_controller,
    clip_to_feasible,
    init_memory,
    passive_controller,
    update_memory,
)
from sandbox.rollout import build_env, rollout
from sandbox.scenarios import (
    REFERENCE_POPULATION,
    end_of_line_impedance_ohm,
    reference_scenario,
)

WEEK = 672
OVER_VOLTAGE_PU = 1.05


@pytest.fixture(scope="module")
def population():
    return reference_scenario()


@pytest.fixture(scope="module")
def env(population):
    return build_env(population, time_limit=WEEK)


def _over_voltage_fraction(trajectory) -> float:
    return float(jnp.mean(trajectory.voltage_pu > OVER_VOLTAGE_PU))


def _peak_kw(trajectory) -> float:
    return float((jnp.abs(trajectory.meter_kwh.sum(-1)) / 0.25).max())


def test_the_feeder_is_realistically_weak() -> None:
    """Around twice IEC 60725's reference LV impedance of 0.4 + j0.25 ohm.

    A long rural or outer-suburban feeder -- the kind where distributed PV
    actually causes trouble, rather than the short urban one the CIGRE asset
    ships as.
    """
    assert 0.8 < end_of_line_impedance_ohm() < 1.05


def test_the_population_is_mixed_and_six_households_cannot_respond(population) -> None:
    """Tenants are the point: no inverter, no agent, no way to react to any
    price at all. If they ever disappear, the fairness pathway loses the
    households it exists to ask about."""
    assert population.num_pq == 18
    assert population.num_agents == 12

    types = population.type_of_pq
    assert types.count("tenant") == 6
    tenant_pq = {i for i, name in enumerate(types) if name == "tenant"}
    assert tenant_pq.isdisjoint(set(population.inverter_id))


def test_the_flexible_households_sit_at_the_far_end(population) -> None:
    """Voltage rise is distance times injection. With the flexible households
    spread evenly the nodal signal is nearly flat and the tariff pathway has
    no gradient to exploit."""
    far_types = {"pv_battery", "large_flex"}
    ranks = [
        population.distance_rank[i]
        for i, name in enumerate(population.type_of_pq)
        if name in far_types
    ]
    near = [
        population.distance_rank[i]
        for i, name in enumerate(population.type_of_pq)
        if name not in far_types
    ]
    assert min(ranks) > max(near)


def test_naive_control_stresses_the_feeder(env, population) -> None:
    """THE acceptance test. If greedy self-consumption keeps this feeder inside
    its limits, every tariff scores identically and there is no challenge.

    Verify by running it, never by assuming it: the last time this scenario
    was sized by argument alone it produced a peak voltage of 1.018 and zero
    violations.
    """
    trajectory = rollout(base_controller(), population, jax.random.PRNGKey(0), WEEK, env=env)

    assert bool(jnp.all(trajectory.valid)), "power flow must converge everywhere"
    assert float(trajectory.voltage_pu.max()) > OVER_VOLTAGE_PU
    assert _over_voltage_fraction(trajectory) > 0.005


def test_control_has_authority_over_the_outcome(env, population) -> None:
    """The second way to mis-size, and the less obvious one.

    Oversize PV and the batteries saturate before noon, at which point the
    peak is just "generation minus load" whatever anyone does and the naive
    and do-nothing baselines converge. A challenge needs the gap.
    """
    naive = rollout(base_controller(), population, jax.random.PRNGKey(0), WEEK, env=env)
    nothing = rollout(passive_controller(), population, jax.random.PRNGKey(0), WEEK, env=env)

    assert _over_voltage_fraction(naive) < _over_voltage_fraction(nothing)
    assert _peak_kw(naive) < _peak_kw(nothing) * 0.98


def test_violations_can_be_removed_but_not_for_free(env, population) -> None:
    """There has to be a frontier, or the challenge is a one-liner.

    A blunt export cap eliminates over-voltage entirely -- and throws away a
    tenth of the generation to do it. Somewhere between "ignore the grid" and
    "throw energy away" is the design space this hackathon is about.
    """

    def export_capped(obs, carry, params, key):
        del key
        surplus = jnp.maximum(obs.pv_available_kw - obs.load_kw, 0.0)
        export = jnp.maximum(surplus - obs.bat_charge_max_kw, 0.0)
        target = jnp.minimum(obs.load_kw + export, obs.load_kw + params["cap_kw"])
        p_set_kw = clip_to_feasible(target, obs)
        return p_set_kw, update_memory(carry, obs, p_set_kw)

    blunt = Controller(
        name="export_cap",
        fn=export_capped,
        params={"cap_kw": jnp.float32(2.5)},
        init_carry=init_memory,
    )
    naive = rollout(base_controller(), population, jax.random.PRNGKey(0), WEEK, env=env)
    capped = rollout(blunt, population, jax.random.PRNGKey(0), WEEK, env=env)

    assert _over_voltage_fraction(capped) < _over_voltage_fraction(naive) * 0.2

    def curtailed(trajectory) -> float:
        lost = jnp.maximum(trajectory.pv_available_kw - trajectory.pv_realized_kw, 0.0).sum()
        return float(lost / trajectory.pv_available_kw.sum())

    assert curtailed(naive) < 0.01
    assert curtailed(capped) > 0.05


def test_tenants_are_settled_even_though_they_have_no_agent(env, population) -> None:
    """The reward array is (num_agents,), so six households are simply absent
    from it. Any fairness question has to read the (num_pq,) settlement."""
    trajectory = rollout(base_controller(), population, jax.random.PRNGKey(0), 96, env=env)

    chex.assert_shape(trajectory.reward_chf, (96, population.num_agents))
    chex.assert_shape(trajectory.settlement_chf, (96, population.num_pq))

    tenants = [i for i, name in enumerate(population.type_of_pq) if name == "tenant"]
    tenant_bill = trajectory.settlement_chf[:, jnp.asarray(tenants)].sum()
    assert float(tenant_bill) < 0.0, "a household that only consumes must pay"


def test_the_eager_path_agrees_with_the_scanned_one(env, population) -> None:
    """Write a controller with `if` and `print`, score it under vmap and scan,
    and get the same answer. If these ever diverge the debugging path is a
    trap rather than a convenience."""
    key = jax.random.PRNGKey(11)
    fast = rollout(base_controller(), population, key, 24, env=env, fast=True)
    eager = rollout(base_controller(), population, key, 24, env=env, fast=False)

    chex.assert_trees_all_close(fast.p_set_kw, eager.p_set_kw, atol=1e-5)
    chex.assert_trees_all_close(fast.settlement_chf, eager.settlement_chf, atol=1e-5)


def test_household_sizing_stays_recognisable() -> None:
    """Guards the calibration against drift into implausibility. A Swiss
    single-family roof carries 6-10 kWp and a home battery holds 10-20 kWh;
    once a scenario needs 30 kWp to be interesting, it is no longer evidence
    about anything."""
    for household in REFERENCE_POPULATION:
        if household.pv_kwp:
            assert 5.0 <= household.pv_kwp <= 12.0, household.name
        if household.battery_kwh:
            assert 10.0 <= household.battery_kwh <= 25.0, household.name
        assert 8.0 <= household.daily_consumption_kwh <= 35.0, household.name
