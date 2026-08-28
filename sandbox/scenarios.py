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

"""The reference scenario: who lives on the feeder, and where.

`gll_env`'s own default gives all eighteen connection points an identical
prosumer. That is the right default for a zero-config environment, and the
wrong population for this challenge, on three counts:

* **Fairness has nothing to measure.** With identical households the Gini is
  zero by construction and any spread is noise from the load process.
* **Herding becomes dismissible.** Clones running one controller synchronize
  perfectly -- "well, obviously" -- while hiding the finding actually worth
  having, that a *heterogeneous* population still synchronizes.
* **Nodal pricing has nothing to price.** With identical injections everywhere,
  nodal prices separate only by line impedance.

So: four household types, placed by electrical distance from the transformer.

Sizing target
-------------
Naive control must **stress the network**, and a better controller must be
able to do something about it. Both are calibration targets to verify, not
assumptions -- see ``tests/test_scenarios.py``.

"Stress" here does not mean over-voltage. On the default urban feeder voltage
never leaves the band, and that is correct rather than a failure to size: a
meshed city network is stiff, and ewz's own operational experience is that
over-voltage is not what constrains it. What binds is reverse flow through a
transformer specified for one direction, the loss of diversity that network
planning depends on, and the ramp. See :mod:`sandbox.metrics`.

Two ways to mis-size, and the second is less obvious:

* Too small, and nothing moves. Every tariff scores identically.
* PV too *large*, and the batteries saturate before noon. Control then has no
  authority at all: the naive and do-nothing baselines converge, because the
  peak becomes "generation minus load" whatever anyone does. This is a real
  property of a fully-solarized quarter, and it is why the reference PV is
  6-10 kWp rather than the 8-14 kWp a large roof would carry.
"""

from dataclasses import dataclass
from typing import Sequence

import jax.numpy as jnp
import numpy as np
from omegaconf import DictConfig, OmegaConf

# 15-minute intervals, seven days. A single day is not enough: battery
# arbitrage and herding both need more than one diurnal cycle before they are
# distinguishable from noise.
STEPS_PER_DAY = 96
EPISODE_DAYS = 7
EPISODE_STEPS = STEPS_PER_DAY * EPISODE_DAYS

GRID_MODEL = "cigre_lv_consumer"

#: Multiplier on the low-voltage network impedance -- see :func:`weaken_feeder`.
#:
#: The bundled CIGRE feeder has an end-of-line Thevenin impedance of about
#: 0.135 ohm, which is a short, generously dimensioned *urban* feeder -- a fair
#: model of a dense city network. But PV congestion is not an urban
#: phenomenon. It bites on suburban and rural feeders: longer runs, thinner
#: conductor, detached houses with large roofs and low coincident load.
#:
#: Named feeder strengths, as a multiplier on the LV branch impedance.
#:
#: ``urban`` is the bundled CIGRE asset untouched -- a short, generously
#: dimensioned, meshed city feeder with an end-of-line Thevenin impedance of
#: 0.135 ohm. It is ewz's own situation, and on it **over-voltage never
#: happens**: voltage stays inside 1.02 pu whatever anyone does, because
#: meshing buys voltage stiffness. What it does not buy is thermal capacity,
#: and the reverse-flow peak here runs at three times the forward peak for
#: some 44 % of the week. That is the real constraint, and it is the default.
#:
#: ``suburban`` scales to 0.46 ohm, the magnitude of IEC 60725's reference LV
#: network impedance (0.4 + j0.25 ohm). ``rural`` reaches 0.91 ohm, roughly
#: twice it -- a long feeder where a single 5 kW injection moves local voltage
#: by around 3 %, and where over-voltage becomes a binding constraint rather
#: than a curiosity.
#:
#: Same population, same jury, different binding constraint. Which one binds
#: where is itself worth a submission.
FEEDER_STRENGTHS: dict[str, float] = {
    "urban": 1.0,
    "suburban": 3.5,
    "rural": 7.0,
}

#: ewz's actual network is the honest default. Over-voltage is a research
#: question on this feeder, not the challenge.
FEEDER_IMPEDANCE_SCALE = FEEDER_STRENGTHS["urban"]


@dataclass(frozen=True)
class HouseholdType:
    """One household archetype, in the units a datasheet uses.

    Attributes:
        name: Short label, used in metrics breakdowns and plots.
        count: How many connection points of this type the feeder carries.
        daily_consumption_kwh: Annual demand spread over a day. A Swiss
            single-family home runs ~4500 kWh/yr; a heat pump adds roughly
            5500 and an EV another 3000.
        s_load_max_kva: Peak apparent load the household can draw.
        s_pq_max_kva: Grid connection rating. 17 kVA is a 25 A three-phase
            connection, 22 kVA a 32 A one.
        pv_kwp: Roof capacity. Zero means no inverter at all -- the household
            is a pure consumer and gets no agent.
        battery_kwh: Usable storage. Zero means PV without storage: the
            household can curtail but cannot shift.
        battery_kw: Charge and discharge rating.
        s_inv_max_kva: Inverter rating.
        far_end: Place this type at the far end of the feeder. Voltage rise
            is a function of distance times injection, so clustering the
            flexible households there localizes it -- without that, the nodal
            signal is nearly flat and the tariff pathway has no gradient to
            exploit.
    """

    name: str
    count: int
    daily_consumption_kwh: float
    s_load_max_kva: float
    s_pq_max_kva: float
    pv_kwp: float
    battery_kwh: float
    battery_kw: float
    s_inv_max_kva: float
    far_end: bool

    @property
    def has_inverter(self) -> bool:
        return self.pv_kwp > 0.0


#: The reference population. Eighteen connection points, twelve of them agents.
#:
#: The six tenants are the point of the exercise: they cannot respond to any
#: price, and they are who a badly designed tariff harms. They are also
#: invisible in the ``(num_agents,)`` reward array, which is why the scorer
#: reads ``extras["reward"].settlement_chf`` instead.
REFERENCE_POPULATION: tuple[HouseholdType, ...] = (
    HouseholdType(
        name="tenant",
        count=6,
        daily_consumption_kwh=9.0,
        s_load_max_kva=8.0,
        s_pq_max_kva=17.0,
        pv_kwp=0.0,
        battery_kwh=0.0,
        battery_kw=0.0,
        s_inv_max_kva=0.0,
        far_end=False,
    ),
    HouseholdType(
        name="pv_only",
        count=5,
        daily_consumption_kwh=12.0,
        s_load_max_kva=15.0,
        s_pq_max_kva=22.0,
        pv_kwp=6.0,
        battery_kwh=0.0,
        battery_kw=0.0,
        s_inv_max_kva=7.0,
        far_end=False,
    ),
    HouseholdType(
        name="pv_battery",
        count=5,
        daily_consumption_kwh=12.0,
        s_load_max_kva=15.0,
        s_pq_max_kva=22.0,
        pv_kwp=8.0,
        battery_kwh=13.0,
        battery_kw=5.0,
        s_inv_max_kva=10.0,
        far_end=True,
    ),
    HouseholdType(
        name="large_flex",
        count=2,
        daily_consumption_kwh=30.0,
        s_load_max_kva=22.0,
        s_pq_max_kva=22.0,
        pv_kwp=10.0,
        battery_kwh=20.0,
        battery_kw=10.0,
        s_inv_max_kva=13.0,
        far_end=True,
    ),
)


@dataclass(frozen=True)
class Population:
    """A population assigned to connection points, with the config that builds it.

    Attributes:
        config: OmegaConf tree for :func:`gll_env.factories.environment_model`.
        type_of_pq: Type name at each of the ``num_pq`` connection points.
        inverter_id: Connection-point index of each agent, ascending. Agents
            are indexed by position in this array, which is the same order
            ``gll_env`` reports rewards in.
        distance_rank: Rank of each connection point by electrical distance
            from the transformer, 0 nearest. Kept for plots and for the
            fairness breakdown.
    """

    config: DictConfig
    type_of_pq: tuple[str, ...]
    inverter_id: tuple[int, ...]
    distance_rank: tuple[int, ...]

    @property
    def num_pq(self) -> int:
        return len(self.type_of_pq)

    @property
    def num_agents(self) -> int:
        return len(self.inverter_id)

    def type_of_agent(self) -> tuple[str, ...]:
        """Type name per agent, in agent order."""
        return tuple(self.type_of_pq[pq] for pq in self.inverter_id)

    def pq_bus_id(self) -> np.ndarray:
        """Global bus index of each connection point.

        Grid quantities are indexed by bus and household quantities by
        connection point; anything joining the two needs this hop.
        """
        return np.asarray(grid_arrays()["pq_id"]).astype(int)

    def mask_for(self, type_name: str) -> np.ndarray:
        """Boolean mask over connection points selecting one household type."""
        return np.array([name == type_name for name in self.type_of_pq], dtype=bool)


def weaken_feeder(admittance: np.ndarray, base_v_kv: np.ndarray, scale: float) -> np.ndarray:
    """Multiply every low-voltage branch impedance by `scale`.

    Longer runs and thinner conductor, which is what separates a suburban
    feeder from an urban one. The transformer branch is deliberately left
    alone: this changes the *network*, not the substation, so the two effects
    stay separable when reading a result.

    A bus admittance matrix holds ``Y_ij = -y_ij`` off the diagonal and
    ``Y_ii = sum_j y_ij + shunt``. Scaling a branch admittance by ``1/scale``
    therefore divides its off-diagonal entries and moves the difference back
    onto both diagonals, which is what keeps the row sums -- and so the shunt
    content, which is negligible at LV but must not be invented -- unchanged.
    """
    if scale == 1.0:
        return admittance
    if scale <= 0.0:
        raise ValueError(f"impedance scale must be positive, got {scale}")

    y = np.array(admittance, dtype=np.complex128, copy=True)
    low_voltage = np.asarray(base_v_kv) < 1.0
    num_bus = y.shape[0]

    for i in range(num_bus):
        for j in range(num_bus):
            if i == j or not (low_voltage[i] and low_voltage[j]):
                continue
            if abs(y[i, j]) < 1e-12:
                continue
            scaled = y[i, j] / scale
            y[i, i] += y[i, j] - scaled
            y[i, j] = scaled

    return y.astype(admittance.dtype)


def grid_arrays(scale: float = FEEDER_IMPEDANCE_SCALE) -> dict:
    """The grid asset's arrays, with the LV network weakened by `scale`."""
    from gll_env.assets.serialization import load_asset_arrays
    from gll_env.factories import GRID_ASSETS_DIR

    arrays = dict(load_asset_arrays(GRID_MODEL, asset_dir=GRID_ASSETS_DIR))
    arrays["admittance"] = jnp.asarray(
        weaken_feeder(
            np.asarray(arrays["admittance"]),
            np.asarray(arrays["base_v_kv"]),
            scale,
        )
    )
    return arrays


def end_of_line_impedance_ohm(scale: float = FEEDER_IMPEDANCE_SCALE) -> float:
    """Thevenin impedance magnitude at the worst connection point.

    The number to compare against IEC 60725's 0.4 + j0.25 ohm reference when
    deciding whether a feeder is realistically weak.
    """
    arrays = grid_arrays(scale)
    admittance = np.asarray(arrays["admittance"]).astype(np.complex128)
    base_v_kv = np.asarray(arrays["base_v_kv"])
    slack = int(np.asarray(arrays["slack_id"])[0])
    pq_id = np.asarray(arrays["pq_id"]).astype(int)

    keep = [i for i in range(admittance.shape[0]) if i != slack]
    reduced = np.linalg.inv(admittance[np.ix_(keep, keep)])
    position = {bus: i for i, bus in enumerate(keep)}
    z_base = float(np.min(base_v_kv[base_v_kv < 1.0])) ** 2 / float(arrays["base_s_mva"])
    return float(max(abs(reduced[position[b], position[b]]) * z_base for b in pq_id))


def _feeder_order() -> np.ndarray:
    """Connection points ordered by distance from the transformer, nearest first.

    Read off the grid asset's own bus coordinates rather than assumed from
    bus numbering, so the placement survives a change of feeder. Distance is
    the physical one; on a radial LV feeder it ranks the same way electrical
    distance does, and it is the quantity the asset actually carries.
    """
    from gll_env.assets.serialization import load_asset_arrays
    from gll_env.factories import GRID_ASSETS_DIR

    arrays = dict(load_asset_arrays(GRID_MODEL, asset_dir=GRID_ASSETS_DIR))
    position = np.asarray(arrays["position"])
    slack = np.asarray(arrays["slack_id"]).reshape(-1)[0]
    pq_id = np.asarray(arrays["pq_id"]).reshape(-1)

    distance = np.linalg.norm(position[pq_id] - position[slack], axis=-1)
    return np.argsort(distance, kind="stable")


def assign_population(
    types: Sequence[HouseholdType] = REFERENCE_POPULATION,
) -> Population:
    """Place `types` on the feeder and build the environment config.

    Types marked ``far_end`` take the connection points furthest from the
    transformer; the rest fill in from the near end. Placement is
    deterministic, so every submission is scored on the same feeder.
    """
    order = _feeder_order()
    num_pq = int(order.shape[0])
    if sum(t.count for t in types) != num_pq:
        raise ValueError(
            f"population covers {sum(t.count for t in types)} connection points, "
            f"but {GRID_MODEL} has {num_pq}."
        )

    near = [t for t in types if not t.far_end]
    far = [t for t in types if t.far_end]

    type_of_pq: list[str | None] = [None] * num_pq
    cursor = 0
    for household in near:
        for _ in range(household.count):
            type_of_pq[int(order[cursor])] = household.name
            cursor += 1
    cursor = num_pq - 1
    for household in far:
        for _ in range(household.count):
            type_of_pq[int(order[cursor])] = household.name
            cursor -= 1

    by_name = {t.name: t for t in types}
    resolved = tuple(name for name in type_of_pq if name is not None)
    assert len(resolved) == num_pq, "every connection point must be assigned"

    at = [by_name[name] for name in resolved]
    inverter_id = tuple(i for i, household in enumerate(at) if household.has_inverter)
    agents = [at[i] for i in inverter_id]

    distance_rank = [0] * num_pq
    for rank, pq in enumerate(order):
        distance_rank[int(pq)] = rank

    config = OmegaConf.create(
        {
            "n_steps_per_day": STEPS_PER_DAY,
            "grid": {"grid_model": GRID_MODEL},
            # Q(U) is the law in force on a Swiss LV feeder, and it reduces the
            # action space to active power alone -- so a controller returns one
            # number, and ActionConstraints.bounds() (exact only in one
            # dimension) becomes available to report it against.
            "grid_code": {"name": "swiss_lv"},
            "prosumer": {
                "s_pq_max_kVA": [h.s_pq_max_kva for h in at],
                "inverter_id": list(inverter_id),
                "load": {
                    "daily_consumption_kWh": [h.daily_consumption_kwh for h in at],
                    "s_load_max_kVA": [h.s_load_max_kva for h in at],
                },
                "inverter": {
                    "s_inv_max_kVA": [h.s_inv_max_kva for h in agents],
                    "battery": {
                        "capacity_kWh": [h.battery_kwh for h in agents],
                        "peak_charge_kW": [h.battery_kw for h in agents],
                        "peak_discharge_kW": [h.battery_kw for h in agents],
                    },
                    "solar": {"peak_power_kW": [h.pv_kwp for h in agents]},
                },
            },
            # Fair LEG: ewz's published local-electricity-community tariff, and
            # therefore the status quo. The challenge is to beat what is
            # actually done today, not a toy.
            "reward": {"name": "leg_settlement", "payments": "fair_leg"},
        }
    )

    return Population(
        config=config,
        type_of_pq=resolved,
        inverter_id=inverter_id,
        distance_rank=tuple(distance_rank),
    )


def reference_scenario() -> Population:
    """The scenario every submission is scored on."""
    return assign_population(REFERENCE_POPULATION)


def step_duration_h() -> float:
    """Interval length in hours -- the kW/kWh conversion factor."""
    return 1.0 / STEPS_PER_DAY * 24.0


def peak_power_kw(population: Population) -> jnp.ndarray:
    """Per-agent inverter rating in kW, for reporting."""
    by_name = {t.name: t for t in REFERENCE_POPULATION}
    return jnp.asarray(
        [by_name[name].s_inv_max_kva for name in population.type_of_agent()],
        dtype=jnp.float32,
    )
