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

"""The other seam you edit: the grid tariff.

A tariff settles one interval and returns what every connection point owes or
earns, in CHF. It runs **after** the interval, with the realized power flow in
front of it -- every bus voltage, every flow, the transformer's throughput.
That is the whole asymmetry this challenge rests on:

============ ============================================ ==================
seam         sees                                         runs
============ ============================================ ==================
tariff       everything: the solved network               after the interval
controller   one household's own meter                    before, blind
============ ============================================ ==================

Retroactive means **unknowable when the household acted**, not dependent on
the future. A price computed from the power flow that just solved is fully
retroactive in the sense that matters -- nobody could have reacted to it --
and it is perfectly causal, which is why nothing here needs to see forward in
time. See ``docs/lookahead_rewards.md`` in `gll_env` if you ever want a
settlement that genuinely does.

Two rules your tariff has to respect, and one it should
------------------------------------------------------
It must publish a ``(num_pq,)`` settlement, not a ``(num_agents,)`` one. Six
of the eighteen connection points are tenants with no inverter and therefore
no agent; they are absent from the reward array entirely, and they are exactly
the households a badly designed tariff harms.

It must not print money. The jury gates on revenue adequacy, because a tariff
that simply pays everybody produces a delighted population and a bankrupt
network operator. Redistribution passes; subsidy does not.

And it should be worth anticipating. A price nobody can predict is noise, and
households tune against expected structure. If your tariff is unpredictable
even in distribution, no controller can respond to it and you have built a
lottery rather than a mechanism.
"""

from typing import TYPE_CHECKING, Any, Callable

import chex
import jax.numpy as jnp
from gll_env.components.prosumer import ProsumerDynamics
from gll_env.rewards.base import CausalReward
from gll_env.rewards.leg import LegSettlementReward, Payments
from gll_env.types import RewardObservation, RewardState

from sandbox.observation import GridView, to_grid_view

if TYPE_CHECKING:
    from gll_env.components.environment import EnvironmentDynamics, EnvironmentState


@chex.dataclass(frozen=True)
class TariffMemory:
    """The default tariff carry. Replace it with any fixed pytree -- a
    tariff's counterpart to a controller's `Memory`.

    Whatever a demand charge's running peak, a ratchet, or a *smoothed*
    (rather than instantaneous) congestion signal needs to remember lives
    here. `MyTariff`'s own default doesn't use it for anything beyond
    counting intervals; see ``TARIFF_COOKBOOK.md`` for what it's for.

    Attributes:
        intervals: Count of settled intervals, for anything phase- or
            billing-period-dependent.
    """

    intervals: chex.Array


def init_tariff_memory() -> TariffMemory:
    """The starting carry -- one per tariff, no connection-point axis."""
    return TariffMemory(intervals=jnp.int32(0))


def _assert_settles_every_connection_point(settlement_chf: chex.Array, num_pq: int) -> None:
    """The first rule, checked where it is broken rather than three frames later.

    Shapes are static at trace time, so this costs nothing at runtime. Without
    it the failure surfaces from inside ``lax.scan`` as "carry input and carry
    output must have equal types", which names neither the tariff nor the
    mistake.
    """
    try:
        chex.assert_shape(settlement_chf, (num_pq,))
    except AssertionError as mismatch:
        raise ValueError(
            f"{mismatch}\n\n"
            f"A tariff must settle all {num_pq} connection points. Six of them are "
            "tenants with no inverter and therefore no agent, so a settlement shaped "
            "like the agents silently drops exactly the households a bad tariff "
            "harms. Return something shaped like `grid.net_kwh` and this is right "
            "by construction. See TARIFF_COOKBOOK.md, 'The two rules'."
        ) from mismatch


@chex.dataclass(frozen=True)
class TariffState(RewardState):
    """What a tariff carries between intervals.

    Attributes:
        settlement_chf: The ``(num_pq,)`` settlement just computed. Every
            tariff must carry this, because it is how the per-connection-point
            result reaches the scorer -- the reward array cannot represent a
            household with no agent.
        collected_chf: Running total the network operator has taken in.
            Nothing reads it during an episode; it is here because a
            budget-balancing tariff needs exactly this kind of memory, and
            because carrying it demonstrates that a tariff *may*. State is
            what separates a tariff from a lookup table.
        carry: Whatever else the tariff wants to remember -- a fixed pytree,
            the tariff's counterpart to a controller's `carry`. Defaults to
            `TariffMemory`; replace it with any pytree the same way a
            controller replaces `Memory`.
    """

    settlement_chf: chex.Array
    collected_chf: chex.Array
    carry: Any


@chex.dataclass(frozen=True)
class TariffObservation(RewardObservation):
    """What the tariff publishes. Reaches the scorer as ``extras["reward"]``.

    Deliberately *not* routed into the controller's observation. Real
    settlement lags by days or months, past the end of an episode, so a
    household cannot see its bill in time to react to it -- which is the
    premise the whole challenge rests on.
    """

    settlement_chf: chex.Array  # (num_pq,) float32, signed
    is_normalized: bool = False

    def normalize(self, reward_dynamics: "CausalReward") -> "TariffObservation":
        del reward_dynamics
        return self


def base_payments() -> Payments:
    """ewz's published fair-LEG rates -- the tariff in force today."""
    from gll_env.factories import payments
    from omegaconf import OmegaConf

    return payments(OmegaConf.create({"payments": "fair_leg"}))


def base_tariff(prosumer: ProsumerDynamics) -> LegSettlementReward:
    """Fair LEG: the status quo, and the thing to beat.

    A local electricity community settling at preferential rates for whatever
    it matches internally and falling back to grid rates for the rest, with a
    peak/off-peak split. Real, published, and currently sold by ewz -- so a
    submission that beats it has said something about the world rather than
    about a toy.

    Note what it does *not* do: it carries no information about the network.
    Two households on the same feeder pay the same rate at the same hour
    whether one of them is at a congested node or not. Fixing that is the
    tariff pathway.
    """
    return LegSettlementReward(payments=base_payments(), prosumer=prosumer)


# ---------------------------------------------------------------------------
# Your tariff
# ---------------------------------------------------------------------------


class MyTariff(CausalReward):
    """A naive, deliberately flawed starting tariff.

    Out of the box it settles energy exactly as fair LEG does, then adds a
    congestion term driven by the feeder's own aggregate flow. When the whole
    feeder pushes past ``headroom_kwh`` in either direction, the households
    pushing it in that direction are charged in proportion to how much of the
    excess is theirs, and the proceeds are rebated equally across all
    eighteen connection points.

    "Fair LEG plus a surcharge" is one shape among many, chosen here because
    it is short. :meth:`settlement_from_view` returns the whole interval's
    settlement; override it (or hand a plain function to
    :func:`tariff_from_settlement`) to price the interval however you like,
    fair LEG included or not. See ``TARIFF_COOKBOOK.md``.

    Three properties worth keeping if you build on the default rather than
    replacing it outright:

    * **Retroactive.** It is computed from the flow that just happened, so no
      household knew it while acting.
    * **Revenue neutral.** The charge is redistributed rather than collected,
      so it passes the adequacy gate by construction. Break this and your
      tariff is a subsidy or a tax, not a price.
    * **Locational, at least in principle.** It charges contribution to a
      shared problem rather than consumption as such.

    The defaults are calibrated to the weakest setting that changes anybody's
    mind: at 1.00 CHF/kWh above 3 kWh the tuned household caps its exports,
    and below roughly that it does not. Worth knowing why the number is so
    much larger than the 0.14 CHF/kWh feed-in rate it competes with -- the
    charge is shared pro rata, so one household's *marginal* exposure is only
    its share of it, an order of magnitude less than the headline. A price
    that looks punitive at the top can be nearly invisible at the margin,
    which is the first thing to check when a tariff seems to do nothing.

    And here is what is wrong with it, which is the interesting part. It is
    an *aggregate* signal: every household on the feeder sees the same
    congestion condition, so every household tuned against it responds in the
    same interval. It prices the symptom rather than the location, and a
    population optimising against it may well synchronise harder, not less.
    The voltage and impedance data needed for a genuinely nodal price are all
    in `new_state`.

    Args:
        prosumer: The environment's prosumer dynamics, for the LEG settlement
            underneath and for ``num_pq``.
        headroom_kwh: Aggregate feeder flow, per interval and in either
            direction, that costs nothing. Beyond it, congestion is priced.
        price_chf_per_kwh: What a kWh of excess flow costs the household
            responsible for it.
        init_carry: Builds the tariff's starting carry -- state that survives
            across intervals. Defaults to `init_tariff_memory`; supply your
            own the way a controller supplies its own `init_carry`.
    """

    def __init__(
        self,
        prosumer: ProsumerDynamics,
        headroom_kwh: float = 3.0,
        price_chf_per_kwh: float = 1.00,
        init_carry: Callable[[], Any] = init_tariff_memory,
    ) -> None:
        self._leg = LegSettlementReward(payments=base_payments(), prosumer=prosumer)
        self._num_pq = prosumer.num_pq
        self._headroom_kwh = float(headroom_kwh)
        self._price = float(price_chf_per_kwh)
        self._init_carry = init_carry

    def reset(self, key: chex.PRNGKey) -> TariffState:
        del key
        return TariffState(
            settlement_chf=jnp.zeros((self._num_pq,), dtype=jnp.float32),
            collected_chf=jnp.float32(0.0),
            carry=self._init_carry(),
        )

    def settlement_from_view(self, grid: "GridView", carry: Any) -> tuple[chex.Array, Any]:
        """The WHOLE interval's settlement, ``(num_pq,)`` CHF, and the carry
        to bring to the next interval. Override this.

        This is the general seam. Any function of one settled interval plus
        whatever you chose to remember can go here: a flat rate, a
        time-of-use schedule, a locational price built from
        ``grid.voltage_pu``. ``grid.energy_chf`` is available if fair LEG's
        energy pricing is a useful place to start. The default builds on it
        with a redistributed surcharge and passes `carry` through untouched.
        """
        return grid.energy_chf - self.congestion_charge_from_view(grid), carry

    def congestion_charge_from_view(self, grid: "GridView") -> chex.Array:
        """Override this to price *just* the congestion term. See :class:`GridView`.

        Only relevant if you keep :meth:`settlement_from_view`'s default
        shape (fair LEG plus a surcharge). Redesigning the settlement itself
        means overriding :meth:`settlement_from_view` instead.
        """
        return self.congestion_charge(grid.net_kwh)

    def congestion_charge(self, e_pq_kwh: chex.Array) -> chex.Array:
        """CHF each connection point owes for this interval's congestion.

        Sums to zero: what the congested households pay, everybody shares.
        """
        aggregate_kwh = jnp.sum(e_pq_kwh)
        excess_kwh = jnp.maximum(jnp.abs(aggregate_kwh) - self._headroom_kwh, 0.0)

        # Only flow in the direction the feeder is already strained counts as
        # contributing to the strain. A household importing while everyone
        # else exports is helping.
        exporting = aggregate_kwh > 0.0
        contribution = jnp.where(exporting, jnp.maximum(e_pq_kwh, 0.0), jnp.maximum(-e_pq_kwh, 0.0))
        total = jnp.sum(contribution)
        share = jnp.where(total > 1e-9, contribution / total, 0.0)

        charge_chf = self._price * excess_kwh * share
        return charge_chf - jnp.mean(charge_chf)

    def settle(
        self,
        reward_state: RewardState,
        state: "EnvironmentState",
        new_state: "EnvironmentState",
        dynamics: "EnvironmentDynamics",
    ) -> tuple[TariffState, chex.Array]:
        leg_state, _ = self._leg.settle(reward_state, state, new_state, dynamics)
        energy_chf = jnp.asarray(leg_state.settlement_chf, dtype=jnp.float32)

        grid = to_grid_view(dynamics, new_state, energy_chf=energy_chf)
        settlement_chf, carry = self.settlement_from_view(grid, reward_state.carry)
        settlement_chf = jnp.asarray(settlement_chf, dtype=jnp.float32)

        _assert_settles_every_connection_point(settlement_chf, self._num_pq)

        inverter_id = jnp.asarray(dynamics.prosumer.inverter_id, dtype=jnp.int32)
        return (
            TariffState(
                settlement_chf=settlement_chf,
                collected_chf=reward_state.collected_chf - jnp.sum(settlement_chf),
                carry=carry,
            ),
            settlement_chf[inverter_id],
        )

    def observation(self, reward_state: RewardState) -> TariffObservation:
        return TariffObservation(
            settlement_chf=jnp.asarray(reward_state.settlement_chf, dtype=jnp.float32)
        )


def default_tariff(prosumer: ProsumerDynamics) -> MyTariff:
    """:class:`MyTariff` at its starting parameters -- a stand-in for "some
    submitted tariff" in tests and examples. Participants write their own in
    ``sandbox/my_idea.py`` instead of calling this directly."""
    return MyTariff(prosumer, headroom_kwh=3.0, price_chf_per_kwh=1.00)


def tariff_from_settlement(
    settlement_fn, params: dict, init_carry: Callable[[], Any] = init_tariff_memory
):
    """Turn a plain ``settlement(grid, carry, params) -> (chf, carry)`` into a tariff.

    This is the general pathway: `settlement_fn` returns the WHOLE interval's
    settlement. A flat rate, a time-of-use schedule, a fully nodal price, or
    fair LEG plus a congestion term are all just a different `settlement_fn`.
    `grid.energy_chf` carries fair LEG's own number for anyone who wants to
    build on it -- see :class:`GridView`.

    A participant writes one pure function over one view instead of
    subclassing anything. `carry` is threaded automatically between
    intervals, the tariff's counterpart to a controller's -- pass your own
    `init_carry` if you replace `TariffMemory` with something else, or ignore
    the argument entirely and return it unchanged for a stateless tariff.
    Returns a factory, which is what :func:`sandbox.rollout.build_env` takes.

    Revenue adequacy -- not handing out more than the network takes in -- is
    checked empirically against fair LEG's own total after the fact (see
    :func:`sandbox.metrics.revenue_adequate`), not enforced here. A settlement
    that happens to sum to zero every interval passes trivially; one that
    doesn't can still pass, as long as it stays within tolerance of what fair
    LEG collects.
    """

    class _FromSettlement(MyTariff):
        def settlement_from_view(self, grid: GridView, carry: Any) -> tuple[chex.Array, Any]:
            return settlement_fn(grid, carry, params)

    return lambda prosumer: _FromSettlement(prosumer, init_carry=init_carry)


def tariff_from_charge(charge_fn, params: dict):
    """Turn a plain ``charge(net_kwh, params) -> (num_pq,) CHF`` into a tariff.

    Narrower than :func:`tariff_from_settlement`, and a convenience when a
    redistributed congestion term over the existing tariff is exactly the
    idea: `charge_fn` is added on top of fair LEG's own settlement, so energy
    pricing stays as it is and you write only the term you care about.
    """

    class _FromCharge(MyTariff):
        def congestion_charge_from_view(self, grid: GridView) -> chex.Array:
            return charge_fn(grid, params)

    return lambda prosumer: _FromCharge(prosumer)
