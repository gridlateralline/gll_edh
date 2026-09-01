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
**not voltage**. On a dense, meshed urban network -- ewz's own -- meshing buys
voltage stiffness and no thermal capacity whatsoever, so voltage sits
comfortably in band while the transformer, the cables and the planning
assumptions take the strain. Over-voltage is a real constraint on long rural
feeders and a research question here, not the headline.

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

**Ramp.** The steepest interval-to-interval swing at the transformer. This is
where herding shows up first and most violently, and where a controller that
looks good on every household metric can be twice as bad as doing nothing.

**Fairness.** What each household paid per kWh it consumed, and how far apart
the best and worst treated are. Computed over all connection points, never
just the agents: households without an inverter are absent from the reward
array entirely, and they are exactly the ones a badly designed tariff harms.
"""

from dataclasses import asdict, dataclass
from typing import Optional

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
    action_correlation: float

    # -- economics ---------------------------------------------------------
    community_settlement_chf: float
    curtailed_share: float
    self_consumption_share: float

    # -- fairness ----------------------------------------------------------
    cost_per_kwh_spread_chf: float
    tenant_cost_per_kwh_chf: float
    owner_cost_per_kwh_chf: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def _per_household_kw(trajectory: Trajectory) -> jnp.ndarray:
    """(T, num_pq) net power at every connection point, tenants included."""
    return trajectory.meter_kwh / step_duration_h()


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


def action_correlation(trajectory: Trajectory) -> float:
    """Mean pairwise correlation of per-household power changes.

    The direct read on whether the school is turning as one. Near zero means
    households move independently; near one means a single decision is being
    taken by everybody.
    """
    changes = jnp.diff(trajectory.p_set_kw, axis=0)
    centred = changes - changes.mean(0, keepdims=True)
    deviation = jnp.sqrt((centred**2).mean(0)) + 1e-8
    correlation = (centred.T @ centred) / changes.shape[0] / jnp.outer(deviation, deviation)
    off_diagonal = ~jnp.eye(correlation.shape[0], dtype=bool)
    return float(correlation[off_diagonal].mean())


def curtailed_share(trajectory: Trajectory) -> float:
    """Generation thrown away, as a fraction of what the roofs could have made."""
    lost = jnp.maximum(trajectory.pv_available_kw - trajectory.pv_realized_kw, 0.0).sum()
    available = trajectory.pv_available_kw.sum()
    return float(jnp.where(available > 0, lost / available, 0.0))


def self_consumption_share(trajectory: Trajectory) -> float:
    """Generation used behind the meter rather than exported."""
    generated = float(trajectory.pv_realized_kw.sum())
    exported = float(jnp.maximum(trajectory.meter_kwh / step_duration_h(), 0.0).sum())
    if generated <= 0.0:
        return 0.0
    return float(max(0.0, 1.0 - exported / generated))


def cost_per_kwh(trajectory: Trajectory, population: Population) -> np.ndarray:
    """(num_pq,) what each connection point paid per kWh it consumed.

    The comparable fairness quantity. Raw settlement is not: a household that
    exports heavily earns money and one that only consumes cannot, so
    comparing totals measures who owns a roof rather than who was treated
    well. Normalising by consumption asks the question that actually matters,
    which is what a kilowatt-hour cost you.

    Negative for a household that earned more than it spent.
    """
    settlement = np.asarray(trajectory.settlement_chf).sum(0)
    consumed = np.maximum(-np.asarray(trajectory.meter_kwh), 0.0).sum(0)
    consumed = np.where(consumed > 1e-6, consumed, np.nan)
    del population
    return -settlement / consumed


def _mean_for(values: np.ndarray, population: Population, *types: str) -> float:
    mask = np.zeros(population.num_pq, dtype=bool)
    for name in types:
        mask |= population.mask_for(name)
    selected = values[mask]
    selected = selected[np.isfinite(selected)]
    return float(selected.mean()) if selected.size else float("nan")


def score(
    trajectory: Trajectory,
    population: Population,
    baseline: Optional[Score] = None,
) -> Score:
    """Score one episode.

    Args:
        trajectory: A single episode -- not a seed ensemble. Use
            :func:`score_ensemble` for those.
        population: Who lives on the feeder, for the fairness breakdown.
        baseline: Unused here, accepted so callers can pass the do-nothing
            reference through a uniform signature.
    """
    del baseline

    transformer = trajectory.transformer_kw
    flows = jnp.abs(_per_household_kw(trajectory)).sum(-1)
    served_kwh = float(jnp.abs(trajectory.meter_kwh).sum())
    per_kwh = cost_per_kwh(trajectory, population)
    finite = per_kwh[np.isfinite(per_kwh)]

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
        action_correlation=action_correlation(trajectory),
        community_settlement_chf=float(trajectory.settlement_chf.sum()),
        curtailed_share=curtailed_share(trajectory),
        self_consumption_share=self_consumption_share(trajectory),
        cost_per_kwh_spread_chf=float(finite.max() - finite.min()) if finite.size else 0.0,
        tenant_cost_per_kwh_chf=_mean_for(per_kwh, population, "tenant"),
        owner_cost_per_kwh_chf=_mean_for(
            per_kwh, population, "pv_only", "pv_battery", "large_flex"
        ),
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
        ("ramp_kW", "max_ramp_kw", "{:.1f}"),
        ("corr", "action_correlation", "{:+.2f}"),
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
