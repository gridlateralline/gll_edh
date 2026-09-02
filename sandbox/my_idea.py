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

"""THE FAST PATH. Two functions, either or both, then::

    from sandbox.my_idea import check
    check()

Everything else in ``sandbox/`` is machinery you can ignore -- with one
exception. Both functions here are naive, working defaults you are meant to
overwrite, not a ceiling: `my_controller` and `my_tariff` are plain functions
because that covers most ideas with the least ceremony, but the tariff seam
underneath (`sandbox/tariff.py`) is a full settlement, not a bolt-on, and
supports redesigning it far past what fits in a single function -- see
``TARIFF_COOKBOOK.md`` and ``CONTROLLER_COOKBOOK.md`` when you want more room
than this file gives you.
"""

import jax.numpy as jnp

from sandbox.controller import clip_to_feasible, update_memory

# ---------------------------------------------------------------------------
# 1. THE HOUSEHOLD.  What one home does with its solar and its battery.
# ---------------------------------------------------------------------------

#: Numbers your controller reads. `check()` and the scorer can sweep these for
#: you -- list the ones worth searching in TUNE_OVER below.
CONTROLLER_PARAMS = {
    "export_cap_kw": 1.0e3,  # never push more than this to the grid (1e3 = no cap)
    "charge_after_hour": 0.0,  # leave the battery idle before this hour
}

TUNE_OVER = {
    "export_cap_kw": [1.0e3, 8.0, 4.0],
    "charge_after_hour": [0.0, 11.0, 13.0],
}


def my_controller(obs, memory, params, key):
    """ONE household, one interval. Return net inverter power in kW, + = exporting.

    `obs` is this household's own meter and nothing else -- no prices, no
    neighbours. Every field is a plain number:

        obs.voltage_pu             own bus, ~0.98 to 1.08
        obs.load_kw                what the house is drawing
        obs.load_forecast_kw       what it expects to draw next
        obs.pv_available_kw        what the roof could make next
        obs.soc_kwh                energy in the battery
        obs.bat_charge_max_kw      how fast it can still charge
        obs.bat_discharge_max_kw   how fast it can still discharge
        obs.p_min_kw, obs.p_max_kw what you are allowed to ask for
        obs.hour                   0 to 24 -- this is the clock
        obs.time_sin, obs.time_cos the clock again, smooth across midnight

    `memory` is yours, carried to the next interval. `key` is a random number
    seed -- useful if you want households to deliberately not act in unison.

    The default below is plain self-consumption: cover your own load, let the
    battery take the rest, export what it cannot hold. It is what every home
    battery ships with, and it is what causes the problem.
    """
    del key

    charging_allowed = obs.hour >= params["charge_after_hour"]

    surplus_kw = jnp.maximum(obs.pv_available_kw - obs.load_kw, 0.0)
    absorbable_kw = jnp.where(charging_allowed, obs.bat_charge_max_kw, 0.0)
    export_kw = jnp.minimum(jnp.maximum(surplus_kw - absorbable_kw, 0.0), params["export_cap_kw"])

    # === YOUR IDEA GOES HERE ===================================================
    # Ask for anything at all; it is clipped to what is physically possible, so
    # a controller cannot break the simulation. Some starting points:
    #   * act on obs.load_forecast_kw instead of obs.load_kw
    #   * hold battery capacity back for the evening using obs.hour
    #   * remember something in `memory` and react to a trend, not a level
    #   * use `key` to stagger against your neighbours
    # ===========================================================================

    p_set_kw = clip_to_feasible(obs.load_kw + export_kw, obs)
    return p_set_kw, update_memory(memory, obs, p_set_kw)


# ---------------------------------------------------------------------------
# 2. THE TARIFF.  What the network charges, once it can see what happened.
# ---------------------------------------------------------------------------

TARIFF_PARAMS = {
    "headroom_kwh": 3.0,  # feeder flow per interval that costs nothing
    "price_chf_per_kwh": 1.0,  # what a kWh beyond it costs whoever caused it
}


def my_tariff(grid, carry, params):
    """What each of the 18 connection points owes for this interval, in CHF.

    This is the WHOLE settlement, not a surcharge on top of one that is
    already decided -- return the final number each connection point pays or
    earns. You see the whole feeder, after the fact -- that is what being the
    network operator means. Everything is a plain array over the 18
    connection points:

        grid.net_kwh        what each household pushed (+) or drew (-)
        grid.net_kw         the same as a power
        grid.voltage_pu     voltage at each one -- but read the warning below
                            before pricing on it
        grid.transformer_kw  throughput at the substation, + = drawing
        grid.losses_kw       what the network itself burned
        grid.hour            0 to 24
        grid.energy_chf      what ewz's real fair-LEG rate would settle this
                             interval as -- a starting point, not a floor.
                             Add to it, replace pieces of it, or ignore it
                             completely and price energy from scratch.
        grid.has_inverter    who can act at all -- a static equipment fact,
                             not a live reading. Use it to say what you mean
                             directly (e.g. an unconditional tenant floor)
                             instead of inferring it from behaviour.

    `carry` is yours, carried to the next interval -- the tariff's
    counterpart to a controller's memory. The default below doesn't use it
    for anything beyond counting intervals; it is exactly where a demand
    charge's running peak, a ratchet, or a *smoothed* (rather than
    instantaneous) congestion signal would live. See `TariffMemory` in
    `sandbox/tariff.py` and `TARIFF_COOKBOOK.md`.

    **It must not print money.** Checked automatically after the fact against
    what fair LEG itself collects, within a tolerance -- see
    `revenue_adequate` in `sandbox/metrics.py`. You do not need to enforce
    this by hand (e.g. by forcing every interval to sum to exactly zero); a
    tariff that redistributes, or that collects a little more or less than
    fair LEG in aggregate, can still pass. One that hands out cash cannot.

    The default below keeps today's shape -- fair LEG's energy settlement,
    plus a congestion term shared by whoever is pushing the feeder past
    `headroom_kwh` -- and rebates the congestion proceeds equally. It works,
    and it is crude in two ways worth attacking:

      * It only ever adds to `grid.energy_chf`. A different rate structure
        entirely -- flat, time-of-use, subscription-plus-marginal -- is just
        a different return value; you do not need `grid.energy_chf` at all.
      * Its congestion term is an *aggregate* signal: every household on the
        feeder sees the same number, so a population tuned against it may
        well synchronise *harder*. A locational price -- see "exposure is
        not contribution" below before reaching for `grid.voltage_pu` -- is
        the obvious next step, and nothing here confines it to a surcharge.
    """
    net_kwh = grid.net_kwh
    aggregate_kwh = jnp.sum(net_kwh)
    excess_kwh = jnp.maximum(jnp.abs(aggregate_kwh) - params["headroom_kwh"], 0.0)

    # Only flow in the direction the feeder is already strained counts as
    # causing the strain; importing while everyone exports is helping.
    exporting = aggregate_kwh > 0.0
    contribution = jnp.where(exporting, jnp.maximum(net_kwh, 0.0), jnp.maximum(-net_kwh, 0.0))
    total = jnp.sum(contribution)
    share = jnp.where(total > 1e-9, contribution / total, 0.0)

    congestion_chf = params["price_chf_per_kwh"] * excess_kwh * share
    congestion_chf = congestion_chf - jnp.mean(congestion_chf)  # rebate: sums to zero
    carry = carry.replace(intervals=carry.intervals + 1)

    # === YOUR IDEA GOES HERE ===================================================
    # Replace any of this. Some starting points, roughly in order of ambition:
    #   * change headroom_kwh / price_chf_per_kwh -- still the same shape
    #   * make the congestion term locational instead of aggregate
    #   * add a time-of-use or demand-charge term alongside it
    #   * smooth the congestion signal through `carry` instead of pricing the
    #     instantaneous level -- the same anti-herding idea as the
    #     controller's `carry.voltage_ewma_pu`, on the price side this time
    #   * drop grid.energy_chf and price the interval from scratch
    # ===========================================================================

    return grid.energy_chf - congestion_chf, carry


# --- Exposure is not contribution -------------------------------------------
#
# The tempting locational tariff is "charge each household in proportion to the
# voltage at its own bus". Resist it, or at least know what it does.
#
# A household's bus voltage is mostly made by OTHER households. Someone at the
# end of a line where the neighbours export heavily sits at a high voltage
# whether or not they export anything themselves, so a price on the voltage
# LEVEL charges them for a condition they did not create -- and if they cut
# their own injection, the voltage barely moves, so the charge barely falls.
# It taxes position and gives almost no marginal incentive, which is close to
# the opposite of what a congestion price is for.
#
# The default above avoids this by construction: it charges each household its
# own share of the excess flow, which is a quantity that household actually
# caused and can actually change.
#
# A locational price done properly charges SENSITIVITY, not level -- how much
# does the binding quantity move per kW of *this* household's injection. That
# is a real and interesting thing to build, and grid.voltage_pu is an input to
# estimating it rather than the answer.
#
# Worth noting the contrast with fair LEG underneath, which is also
# interdependent -- your settlement depends on what everyone else did, through
# the community match ratio. The difference is that the match ratio is applied
# to everyone equally and pro rata, so interdependence there does not become a
# charge for where you happen to live.


# ---------------------------------------------------------------------------
# 3. TRY IT.  `check()` for a fast look, `score()` for the real thing.
# ---------------------------------------------------------------------------


def check(days: int = 1, detail: bool = False) -> None:
    """Run your idea against the reference and print what changed. Seconds.

    Use this while you work. When you are happy, run :func:`score`, which runs
    a full week over many weathers and is what the jury sees.
    """
    from sandbox.check import run_check

    run_check(days=days, detail=detail)


def score(detail: bool = True) -> None:
    """The full evaluation: a week, twenty weathers, the whole jury."""
    from sandbox.check import run_score

    run_score(detail=detail)
