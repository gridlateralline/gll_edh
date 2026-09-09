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

Everything else in ``sandbox/`` is machinery you can ignore. Both functions
here are naive, working defaults you are meant to overwrite. They are plain
functions because that covers most ideas with the least ceremony; when you
want more room than a single function gives you, ``CONTROLLER_COOKBOOK.md``
and ``TARIFF_COOKBOOK.md`` are the complete reference for each seam.

The one identity to keep in your head::

    p_inv_kw  (you choose)  -  p_load_kw  (you don't)  =  p_grid_kw  (you're billed on)

Your controller sets the **inverter**'s active power. The household's own load
sits behind the same meter, so what the feeder carries -- and what the tariff
settles -- is the difference. ``p_inv_kw = 0`` means an idle inverter and a
full import; ``p_inv_kw = obs.p_load_forecast_kw`` is what zeroes the exchange.

All three are active power: ``_kw`` is active, ``_kvar`` reactive, ``_kva``
apparent. A household only ever chooses active power here -- the Q(U) grid
code claims the reactive axis by law -- while a tariff can see both.
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

#: Optional. A zero-argument callable returning ONE household's starting
#: carry, if you want something richer than the default `Memory`
#: (``p_prev_kw``, ``voltage_ewma_pu``, ``intervals``). Uncomment and point it
#: at your own frozen chex dataclass -- fixed shape, fixed dtype.
#:
#: INIT_CARRY = lambda: MyCarry(offset_h=jnp.float32(0.0), intervals=jnp.int32(0))

#: The values `score()` sweeps -- the household's best response to your tariff.
#: Part of your submission, not a convenience: a good idea whose good
#: parameters are missing here gets scored at parameters nobody would choose.
#: Names must be keys of CONTROLLER_PARAMS; anything omitted keeps its default.
TUNE_OVER = {
    "export_cap_kw": [1.0e3, 8.0, 4.0],
    "charge_after_hour": [0.0, 11.0, 13.0],
}


def my_controller(obs, carry, params, key):
    """ONE household, one interval. Return the INVERTER's active power in kW.

    Positive = the inverter is producing. This is not the flow at the grid
    connection: your own load sits behind the same meter, so the feeder sees
    ``p_inv_kw - p_load_kw`` and that difference is what you are billed on.

    `obs` is this household's own meter and nothing else -- no prices, no
    neighbours. Every field is a plain number. The right-hand column says
    which interval it describes, and mixing those up is the commonest quiet
    mistake here::

        obs.hour                   0 to 24, start of the COMING interval
        obs.time_sin, obs.time_cos the clock again, smooth across midnight
        obs.voltage_pu             own bus, ~0.98 to 1.11        (last)
        obs.p_grid_kw              your net exchange with the grid,
                                   + = injecting -- what you are billed on
                                                                 (last)
        obs.p_load_kw                what the house drew           (last)
        obs.p_load_forecast_kw       what it will draw             (COMING)
        obs.pv_available_kw        what the roof could make      (COMING)
        obs.soc_kwh                energy in the battery
        obs.soc_headroom_kwh       room left in it
        obs.bat_charge_max_kw      how fast it can still charge
        obs.bat_discharge_max_kw   how fast it can still discharge
        obs.p_inv_min_kw           the least you may ask the inverter for
        obs.p_inv_max_kw           the most  (both are 0 for a tenant)

    `carry` is yours, carried to the next interval. `key` is a fresh JAX PRNG
    key, different for every household every interval -- the tool for making
    identical households deliberately not act in unison. See "Randomness" in
    ``CONTROLLER_COOKBOOK.md``.

    The default below is plain self-consumption: cover your own load, let the
    battery take the rest, export what it cannot hold. It is what every home
    battery ships with, and it is what causes the problem. Note that it acts
    on ``obs.p_load_kw`` -- last interval's load -- which is what a real battery
    servoing against its own meter does; see
    :func:`sandbox.controller.self_consumption` for what that lag costs.
    """
    del key

    charging_allowed = obs.hour >= params["charge_after_hour"]

    surplus_kw = jnp.maximum(obs.pv_available_kw - obs.p_load_kw, 0.0)
    absorbable_kw = jnp.where(charging_allowed, obs.bat_charge_max_kw, 0.0)
    export_kw = jnp.minimum(jnp.maximum(surplus_kw - absorbable_kw, 0.0), params["export_cap_kw"])

    # === YOUR IDEA GOES HERE ===================================================
    # Ask for anything at all; it is clipped to what is physically possible, so
    # a controller cannot break the simulation. Some starting points:
    #   * act on obs.p_load_forecast_kw instead of obs.p_load_kw -- the warm-up
    #   * hold battery capacity back for the evening using obs.hour
    #   * remember something in `carry` and react to a trend, not a level
    #   * subtract what you already know from obs.voltage_pu and act on the
    #     remainder -- that part is genuinely about your neighbourhood
    #   * use `key` to stagger against your neighbours
    # ===========================================================================

    p_inv_kw = clip_to_feasible(obs.p_load_kw + export_kw, obs)
    return p_inv_kw, update_memory(carry, obs, p_inv_kw)


# ---------------------------------------------------------------------------
# 2. THE TARIFF.  What the network charges, once it can see what happened.
# ---------------------------------------------------------------------------

TARIFF_PARAMS = {
    "headroom_kwh": 3.0,  # feeder flow per interval that costs nothing
    "price_chf_per_kwh": 1.0,  # what a kWh beyond it costs whoever caused it
}


def my_tariff(grid, carry, params):
    """What each of the 18 connection points owes for this interval, in CHF.

    Return the final number each connection point pays or earns -- the whole
    interval's settlement, signed, positive = the connection point is paid.
    You see the whole feeder, after the fact; that is what being the network
    operator means. Everything is a plain array over the 18 connection
    points::

        grid.e_grid_kwh     net ACTIVE energy each one exchanged with the
                            grid, + = pushed in. Inverter output minus
                            household load: the meter reading, not the
                            inverter's. This is what the baseline settles.
        grid.p_grid_kw      the same as a power
        grid.q_grid_kvar    its reactive counterpart -- visible to a tariff,
                            but set by the Q(U) grid code and the load's
                            power factor rather than chosen. Read "exposure
                            is not contribution" before pricing it.
        grid.voltage_pu     voltage at each one -- same warning
        grid.transformer_kw  active throughput at the substation, + = drawing
        grid.transformer_kvar  reactive throughput. A transformer is rated in
                            kVA, so hypot(kw, kvar) is what its limit sees.
        grid.losses_kw       what the network itself burned -- and where the
                            cost of reactive flow already shows up
        grid.hour            0 to 24
        grid.fair_leg_chf    the WHOLE settlement the fair-LEG baseline
                             would produce for this interval -- a finished
                             CHF figure, not a rate. An input you may build
                             on, take pieces of, or leave alone and price
                             the interval yourself. See `base_payments` in
                             `sandbox/tariff.py` for what fair LEG is.
        grid.has_inverter    who can act at all -- a static equipment fact,
                             not a live reading. Use it to say what you mean
                             directly (e.g. an unconditional tenant floor)
                             instead of inferring it from behaviour.

    `carry` is yours, carried to the next interval -- the tariff's
    counterpart to a controller's. The default below doesn't use it for
    anything beyond counting intervals; it is exactly where a demand
    charge's running peak, a ratchet, or a *smoothed* (rather than
    instantaneous) congestion signal would live. See `TariffMemory` in
    `sandbox/tariff.py` and `TARIFF_COOKBOOK.md`.

    **It must not print money.** Checked automatically after the fact against
    what fair LEG itself collects, within a tolerance -- see
    `revenue_adequate` in `sandbox/metrics.py`. You do not need to enforce
    this by hand (e.g. by forcing every interval to sum to exactly zero); a
    tariff that redistributes, or that collects a little more or less than
    fair LEG in aggregate, can still pass. One that hands out cash cannot.

    **A tariff changes nothing until a household re-tunes against it.** No
    controller sees a price during an episode, so `check()` -- which does not
    tune -- will show your tariff moving the settlement and *nothing else*.
    That is correct, not a bug. Use `score()`, which best-responds to your
    tariff before scoring it.

    The default below is fair LEG's settlement plus a congestion term
    shared by whoever is pushing the feeder past `headroom_kwh`, with the
    congestion proceeds rebated equally. It works, and it is crude in two
    ways worth attacking:

      * It only ever adds to `grid.fair_leg_chf`. A different rate structure
        entirely -- flat, time-of-use, subscription-plus-marginal -- is just
        a different return value from this function.
      * Its congestion term is an *aggregate* signal: every household on the
        feeder sees the same number, so a population tuned against it may
        well synchronise *harder*. A locational price is the obvious next
        step -- read "exposure is not contribution" below before reaching
        for `grid.voltage_pu`.
    """
    e_grid_kwh = grid.e_grid_kwh
    aggregate_kwh = jnp.sum(e_grid_kwh)
    excess_kwh = jnp.maximum(jnp.abs(aggregate_kwh) - params["headroom_kwh"], 0.0)

    # Only flow in the direction the feeder is already strained counts as
    # causing the strain; importing while everyone exports is helping.
    exporting = aggregate_kwh > 0.0
    contribution = jnp.where(exporting, jnp.maximum(e_grid_kwh, 0.0), jnp.maximum(-e_grid_kwh, 0.0))
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
    #   * drop grid.fair_leg_chf and price the interval from scratch
    # ===========================================================================

    return grid.fair_leg_chf - congestion_chf, carry


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


def check(days: int = 1, detail: bool = False, fast: bool = True) -> None:
    """Run your idea against the reference and print what changed. Seconds.

    Use this while you work. Pass ``fast=False`` when something breaks: it
    trades speed for concrete values, a working ``print`` and a traceback
    that points at your own line.

    **`check()` does not tune.** It runs both controllers at their current
    parameters on one week of weather. That makes it the right tool for a
    controller and a *misleading* one for a tariff: no household can see a
    price during an episode, so an untuned run shows your tariff changing the
    settlement column and nothing physical. Use :func:`score` to see whether
    a tariff works.

    When you are happy, run :func:`score`, which runs a full week over many
    weathers and is what the jury sees.
    """
    from sandbox.check import run_check

    run_check(days=days, detail=detail, fast=fast)


def score(detail: bool = True) -> None:
    """The full evaluation: a week, twenty weathers, the whole jury.

    Four rollouts with a tuning sweep inside each -- **about two minutes**,
    not the seconds :func:`check` takes. Nothing has hung.
    """
    from sandbox.check import run_score

    run_score(detail=detail)
