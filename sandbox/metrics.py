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

"""The jury. Fixed, and not editable by participants.

What a distribution network operator actually worries about, which is mostly
**not voltage**. Over-voltage is real on the ``rural`` feeder this challenge
runs on -- 8.7 % of bus-intervals above 1.05 pu, peaking at 1.107 -- and it
is reported (the ``>1.05`` column). But on the dense meshed network ewz
actually operates, meshing buys voltage stiffness and no thermal capacity
whatsoever, so voltage sits comfortably in band while the transformer, the
cables and the planning assumptions take the strain. The jury is weighted
for the constraints that bind on both: reverse flow, lost diversity, ramp.

Five families, and the third and fourth are the ones this challenge turns on.

**Network.** Transformer peak in both directions, how much of the week runs in
reverse, and losses. Reverse flow matters on its own: urban transformers,
their protection settings and their thermal models were largely specified for
power flowing one way.

**Economics.** What the community paid, against a do-nothing floor, plus what
was thrown away. Curtailed generation is real money, and a tariff that buys a
flat feeder by spilling a fifth of the solar has not solved anything.

**Diversity.** The coincidence factor -- peak of the sum over sum of the
peaks -- is the DSO's own planning quantity, the thing that lets a network be
built for far less than the sum of its connections. Synchronised control
destroys it, and destroying it invalidates the assumption the network was
sized under. It is also purely behavioural: identical at every feeder
impedance, so it measures the mechanism rather than the wiring.

Beside it, peak-to-average: how much of the week's work the feeder does in
its worst interval. It is the blunter of the two and the harder to argue
with, because it moves whenever a controller narrows the window without
changing the energy -- which is the usual way a household-optimal battery
harms the network.

**Ramp.** The steepest interval-to-interval swing at the transformer. Read it
as the tail of the same distribution peak-to-average summarises: it is a
single interval out of six hundred, so it is the sharpest number here and the
least robust one. Cite it with a companion, never alone.

**Synchrony.** The mechanism behind both of the above, read off the battery
rather than off the feeder: what share of the fleet's flexible movement is
common-mode. Read off the battery because the two series a household actually
touches -- the inverter power it requests, and the meter reading it is settled
on -- both score +0.87 for a fleet that coordinates nothing, the weather and
the working day being what they are. Alone among these it cannot be flattered
by a controller that simply acts less, because it is scale-free -- which is
also why it says nothing about magnitude and has to be read beside something
that does.

**Fairness.** What a kWh cost each of the four household types, broken out
rather than summarised, plus the share of the feeder's total import the
tenants carry. Computed over all connection points, never just the agents:
households without an inverter are absent from the reward array entirely, and
they are exactly the ones a badly designed tariff harms.

The import share is the one to watch, because it is the only place on this
scoreboard where the network's *cost base* appears. Self-supply shrinks what
a household buys, and a charge levied on what households buy therefore lands
on whoever is still buying. Nothing in the network family moves when that
happens; the feeder is indifferent to who pays for it.
"""

from dataclasses import asdict, dataclass

import jax.numpy as jnp
import numpy as np

from sandbox.rollout import Trajectory
from sandbox.scenarios import Population, step_duration_h

#: Planning trigger, not the statutory limit. EN 50160 allows 0.9-1.1 pu, but
#: that budget is shared with the medium-voltage network, so a DSO plans the
#: low-voltage share to a few percent. Reported, never gated: on a stiff urban
#: feeder it is simply never reached.
OVER_VOLTAGE_PU = 1.05

#: How far a submitted tariff may move the community's total settlement before
#: it stops being a tariff and starts being a subsidy. A mechanism that simply
#: pays everyone looks wonderful on every household metric.
REVENUE_TOLERANCE = 0.10


@dataclass(frozen=True)
class Score:
    """One episode, scored. Every field is a scalar; lower is better unless noted."""

    # -- network -----------------------------------------------------------
    transformer_draw_peak_kw: float
    transformer_export_peak_kw: float
    reverse_flow_share: float
    losses_share: float
    voltage_p99_pu: float
    over_voltage_share: float

    # -- diversity and ramp: the herding signature -------------------------
    coincidence_factor: float
    peak_to_average: float
    max_ramp_kw: float
    flex_synchrony: float

    # -- economics ---------------------------------------------------------
    community_settlement_chf: float
    curtailed_share: float
    self_consumption_share: float

    # -- fairness ----------------------------------------------------------
    cost_per_kwh_spread_chf: float
    tenant_cost_per_kwh_chf: float
    pv_only_cost_per_kwh_chf: float
    pv_battery_cost_per_kwh_chf: float
    large_flex_cost_per_kwh_chf: float
    owner_cost_per_kwh_chf: float
    tenant_import_share: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def _per_household_kw(trajectory: Trajectory) -> jnp.ndarray:
    """(T, num_pq) net power at every connection point, tenants included."""
    return trajectory.e_grid_kwh / step_duration_h()


def coincidence_factor(trajectory: Trajectory) -> float:
    """Peak of the sum divided by the sum of the peaks.

    The quantity a network is planned against. Eighteen households each
    peaking at 5 kW do not need a 90 kW connection, because historically they
    peaked at different moments; the factor by which they do not is what makes
    distribution economics work.

    A tariff that makes everyone act at once drives this toward 1 and quietly
    invalidates the assumption the feeder was built under -- which is a more
    expensive failure than a voltage excursion, and much harder to see.

    Purely behavioural: unchanged by the feeder's impedance, so it measures
    the mechanism and not the wiring.
    """
    per_household = jnp.abs(_per_household_kw(trajectory))
    return float(per_household.sum(-1).max() / per_household.max(0).sum())


def max_ramp_kw(trajectory: Trajectory) -> float:
    """Steepest interval-to-interval swing at the transformer.

    Where herding shows first. Batteries that all fill at the same moment stop
    absorbing at the same moment, and the feeder's flow steps rather than
    slides -- so a controller can improve the peak and make the ramp
    dramatically worse.
    """
    return float(jnp.abs(jnp.diff(trajectory.transformer_kw)).max())


def battery_kw(trajectory: Trajectory) -> jnp.ndarray:
    """(T, num_agents) the battery / flexible-load flow behind each inverter.

    Negative while absorbing, positive while discharging.

    An identity, not an estimate. The environment projects the requested
    inverter power onto what is feasible, then splits the result by
    dispatching as much solar as it can and handing the battery the residual,
    so ``p_inv_realized == sol_realized + bat_realized`` holds exactly (see
    ``gll_env.components.inverter``, "Internal dispatch"). Subtracting the
    realized solar recovers the battery term to float32.

    **Both terms are realized, not requested.** Where a controller asks for an
    inverter power the environment cannot deliver, what appears here is what
    happened, including the correction. The shipped controllers all call
    :func:`~sandbox.controller.clip_to_feasible`, so for them
    ``p_inv_set_kw == p_inv_realized_kw`` to the float and the distinction is
    invisible -- a submitted controller that does not clip is where it starts
    to matter, and where this stops being a clean read on that controller's
    intent.
    """
    return trajectory.p_inv_realized_kw - trajectory.pv_realized_kw


def flex_synchrony(trajectory: Trajectory) -> float:
    """Of the flexible energy that moved, how much moved together.

    Total covariance between households' battery flow, over the most that
    covariance could have been, so it is the share of the fleet's joint
    movement that is common-mode: 0 independent, 1 a single decision taken by
    everybody. Each pair enters weighted by how much those two households
    actually moved, so a household sitting still neither helps nor hurts and
    no threshold has to be invented to exclude it.

    **This is deliberately not the quantity a household chooses, nor the one
    it is billed on, and it is worth being clear about why.** A controller
    chooses ``p_inv_set_kw``, an inverter power. It is settled on
    ``p_grid_kw`` -- that inverter power minus its own load, the meter
    reading. Both are unusable here, and by a wide margin: a `passive` fleet,
    which coordinates nothing whatsoever, scores **+0.87 on either of them**,
    because twelve roofs share one sky and twelve households cook at the same
    time. Correlating what the household picks, or what it pays for, measures
    the weather and the working day. That is the premise of this whole
    challenge, and it makes those two series the wrong instrument for reading
    the one thing on top of them.

    The battery flow is what is left after the sun has been accounted for.
    That is why it is the series used here -- not because it is more
    fundamental, and not because the household chose it as such (the
    solar/battery split is the environment's dispatch rule, not a household
    decision). The same four runs read +0.21 passive, +0.60 self-consumption,
    and +0.32 once the charge windows are staggered -- which is the known fix
    for herding, correctly rewarded, at barely any loss of flexible energy
    moved.

    This metric replaced one that correlated changes in ``p_inv_set_kw`` and
    consequently ranked the do-nothing baseline as the most synchronised
    strategy on the board.

    Scale-free by design, which is the price of being immune to the other
    failure mode: a controller cannot flatter this by simply doing less. The
    flip side is that a fleet barely moving can still post a middling figure
    on a negligible flow, so read it beside something with units.
    """
    flow = np.asarray(battery_kw(trajectory))
    centred = flow - flow.mean(0, keepdims=True)
    covariance = (centred.T @ centred) / centred.shape[0]
    deviation = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    off_diagonal = ~np.eye(covariance.shape[0], dtype=bool)
    ceiling = float(np.outer(deviation, deviation)[off_diagonal].sum())
    if ceiling <= 1e-12:
        # Nobody moved at all. Not a synchronised fleet -- no fleet.
        return 0.0
    return float(covariance[off_diagonal].sum() / ceiling)


def curtailed_share(trajectory: Trajectory) -> float:
    """Generation thrown away, as a fraction of what the roofs could have made."""
    lost = jnp.maximum(trajectory.pv_available_kw - trajectory.pv_realized_kw, 0.0).sum()
    available = trajectory.pv_available_kw.sum()
    return float(jnp.where(available > 0, lost / available, 0.0))


def self_consumption_share(trajectory: Trajectory) -> float:
    """Generation used behind the meter rather than exported."""
    generated = float(trajectory.pv_realized_kw.sum())
    exported = float(jnp.maximum(trajectory.e_grid_kwh / step_duration_h(), 0.0).sum())
    if generated <= 0.0:
        return 0.0
    return float(max(0.0, 1.0 - exported / generated))


def cost_per_kwh(trajectory: Trajectory) -> np.ndarray:
    """(num_pq,) what each connection point paid per kWh it consumed.

    The comparable fairness quantity. Raw settlement is not: a household that
    exports heavily earns money and one that only consumes cannot, so
    comparing totals measures who owns a roof rather than who was treated
    well. Normalising by consumption asks the question that actually matters,
    which is what a kilowatt-hour cost you.

    Negative for a household that earned more than it spent.
    """
    settlement = np.asarray(trajectory.settlement_chf).sum(0)
    consumed = np.maximum(-np.asarray(trajectory.e_grid_kwh), 0.0).sum(0)
    consumed = np.where(consumed > 1e-6, consumed, np.nan)
    return -settlement / consumed


#: The four household types, in the order a fairness table reads best: the
#: households with nothing to change first, the ones with the most to change
#: last.
HOUSEHOLD_TYPES = ("tenant", "pv_only", "pv_battery", "large_flex")


def _mask_for(population: Population, *types: str) -> np.ndarray:
    mask = np.zeros(population.num_pq, dtype=bool)
    for name in types:
        mask |= population.mask_for(name)
    return mask


def _group_cost_per_kwh(trajectory: Trajectory, population: Population, *types: str) -> float:
    """What a kWh cost a group, pooling the group before dividing.

    Deliberately not the mean of :func:`cost_per_kwh` over the group. That
    average is a mean of ratios, and a household whose imports approach zero
    -- which is precisely what a battery under self-consumption does -- sends
    its own ratio to infinity and takes the group average with it. Pooling
    first asks the question a regulator asks, which is what the group paid for
    the energy the group actually took.

    Negative for a group that earned more than it spent.
    """
    mask = _mask_for(population, *types)
    settlement = float(np.asarray(trajectory.settlement_chf).sum(0)[mask].sum())
    consumed = float(np.maximum(-np.asarray(trajectory.e_grid_kwh), 0.0).sum(0)[mask].sum())
    return -settlement / consumed if consumed > 1e-6 else float("nan")


def import_share(trajectory: Trajectory, population: Population, *types: str) -> float:
    """A group's share of everything the feeder imported over the week.

    The cost base, in one number. Any network charge -- volumetric, capacity,
    or the fair-LEG baseline -- is ultimately recovered from households in
    rough proportion to this. When self-supply drains it, the charge does not
    go away; it lands on whoever is left, and the households left are the ones
    with no roof to self-supply from. That is the cost shift a tariff is
    supposed to be designed against, and it is invisible in every network
    metric on this scoreboard.
    """
    imported = np.maximum(-np.asarray(trajectory.e_grid_kwh), 0.0).sum(0)
    total = float(imported.sum())
    if total <= 1e-6:
        return float("nan")
    return float(imported[_mask_for(population, *types)].sum() / total)


def cost_per_kwh_by_type(trajectory: Trajectory, population: Population) -> dict[str, float]:
    """CHF per kWh consumed, for each of the four household types.

    The breakdown, rather than one spread, because who moved matters as much
    as how far: a tariff that lifts every group equally is not the same
    animal as one that leaves tenants exactly where they were.
    """
    return {name: _group_cost_per_kwh(trajectory, population, name) for name in HOUSEHOLD_TYPES}


def _spread(by_type: dict[str, float]) -> float:
    """How far apart the best- and worst-treated household types are.

    Across the four **types**, not across the eighteen households. The
    household-level spread is dominated by whichever single connection point
    imported least, which under any self-consumption strategy is a ratio
    approaching a division by zero rather than a fairness finding.
    """
    finite = [value for value in by_type.values() if np.isfinite(value)]
    return float(max(finite) - min(finite)) if finite else 0.0


def score(trajectory: Trajectory, population: Population) -> Score:
    """Score one episode.

    Args:
        trajectory: A single episode. Average the *metrics* over an ensemble,
            never the trajectories -- a peak is not linear, and the mean of
            two weeks' flows has a lower peak than either week. See
            :func:`sandbox.evaluate._mean_score`.
        population: Who lives on the feeder, for the fairness breakdown.
    """
    transformer = trajectory.transformer_kw
    flows = jnp.abs(_per_household_kw(trajectory)).sum(-1)
    served_kwh = float(jnp.abs(trajectory.e_grid_kwh).sum())
    by_type = cost_per_kwh_by_type(trajectory, population)

    return Score(
        transformer_draw_peak_kw=float(jnp.maximum(transformer.max(), 0.0)),
        transformer_export_peak_kw=float(jnp.maximum(-transformer.min(), 0.0)),
        reverse_flow_share=float(jnp.mean(transformer < 0.0)),
        losses_share=float(trajectory.losses_kw.sum() * step_duration_h() / max(served_kwh, 1e-9)),
        voltage_p99_pu=float(jnp.percentile(trajectory.voltage_pu, 99)),
        over_voltage_share=float(jnp.mean(trajectory.voltage_pu > OVER_VOLTAGE_PU)),
        coincidence_factor=coincidence_factor(trajectory),
        peak_to_average=float(flows.max() / jnp.maximum(flows.mean(), 1e-9)),
        max_ramp_kw=max_ramp_kw(trajectory),
        flex_synchrony=flex_synchrony(trajectory),
        community_settlement_chf=float(trajectory.settlement_chf.sum()),
        curtailed_share=curtailed_share(trajectory),
        self_consumption_share=self_consumption_share(trajectory),
        cost_per_kwh_spread_chf=_spread(by_type),
        tenant_cost_per_kwh_chf=by_type["tenant"],
        pv_only_cost_per_kwh_chf=by_type["pv_only"],
        pv_battery_cost_per_kwh_chf=by_type["pv_battery"],
        large_flex_cost_per_kwh_chf=by_type["large_flex"],
        owner_cost_per_kwh_chf=_group_cost_per_kwh(
            trajectory, population, "pv_only", "pv_battery", "large_flex"
        ),
        tenant_import_share=import_share(trajectory, population, "tenant"),
    )


def revenue_adequate(candidate: Score, reference: Score) -> bool:
    """Does this tariff still collect roughly what the reference collected?

    A pass/fail gate, not a metric, and the reason is that without it the
    scoreboard is trivially winnable: a tariff that simply pays everybody
    produces a delighted population and a bankrupt network operator. Being a
    *gate* means such a submission is disqualified rather than ranked first.
    """
    target = abs(reference.community_settlement_chf)
    if target < 1e-6:
        return True
    deviation = abs(candidate.community_settlement_chf - reference.community_settlement_chf)
    return bool(deviation / target <= REVENUE_TOLERANCE)


#: What a participant needs while iterating. The rest of the jury is real, and
#: is one `detail=True` away -- but eleven columns is not a thing anybody reads
#: between two edits.
HEADLINE = (
    "transformer_export_peak_kw",
    "peak_to_average",
    "max_ramp_kw",
    "coincidence_factor",
    "curtailed_share",
    "community_settlement_chf",
)


def compare(scores: dict[str, Score], detail: bool = True) -> str:
    """A readable table. `detail=False` shows only the five that matter most."""
    columns = [
        ("export_pk_kW", "transformer_export_peak_kw", "{:.1f}"),
        ("draw_pk_kW", "transformer_draw_peak_kw", "{:.1f}"),
        ("reverse", "reverse_flow_share", "{:.0%}"),
        ("coincid", "coincidence_factor", "{:.3f}"),
        ("pk/avg", "peak_to_average", "{:.2f}"),
        ("ramp_kW", "max_ramp_kw", "{:.1f}"),
        ("sync", "flex_synchrony", "{:+.2f}"),
        ("loss", "losses_share", "{:.2%}"),
        ("curtail", "curtailed_share", "{:.1%}"),
        ("selfcons", "self_consumption_share", "{:.0%}"),
        ("CHF", "community_settlement_chf", "{:.0f}"),
        (">1.05", "over_voltage_share", "{:.2%}"),
    ]
    if not detail:
        columns = [c for c in columns if c[1] in HEADLINE]
    width = max(len(name) for name in scores) + 2
    header = f"{'':<{width}}" + "".join(f"{label:>13}" for label, _, _ in columns)
    lines = [header, "-" * len(header)]
    for name, value in scores.items():
        cells = "".join(fmt.format(getattr(value, field)).rjust(13) for _, field, fmt in columns)
        lines.append(f"{name:<{width}}{cells}")
    return "\n".join(lines)


def fairness(scores: dict[str, Score]) -> str:
    """Who paid what, per household type, plus the cost base underneath it.

    Separate from :func:`compare` on purpose. Every column there is a feeder
    quantity, and a feeder quantity cannot tell you which households paid for
    the improvement -- a tariff can flatten the transformer beautifully by
    charging the people with no way to respond.

    ``CHF/kWh`` pools each group before dividing, so it is what that group
    paid for the energy it took; negative means the group earned more than it
    spent. ``tenant_import`` is the tenants' share of everything the feeder
    imported: the fraction of the network's cost base carried by the
    households that have no roof, no battery and no way to shrink it.
    """
    columns = [(name, f"{name}_cost_per_kwh_chf") for name in HOUSEHOLD_TYPES]
    width = max(len(name) for name in scores) + 2
    header = f"{'':<{width}}" + "".join(f"{label:>13}" for label, _ in columns)
    header += f"{'spread':>13}{'tenant_import':>15}"
    lines = ["CHF/kWh consumed, by household type", header, "-" * len(header)]
    for name, value in scores.items():
        cells = "".join(f"{getattr(value, field):+13.3f}" for _, field in columns)
        cells += f"{value.cost_per_kwh_spread_chf:13.2f}{value.tenant_import_share:15.0%}"
        lines.append(f"{name:<{width}}{cells}")
    return "\n".join(lines)
