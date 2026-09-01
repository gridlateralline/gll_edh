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

"""THE ONLY FILE YOU EDIT.

Two functions. Change one or both, then::

    from sandbox.my_idea import check
    check()

Everything else in ``sandbox/`` is machinery you can ignore.
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


def my_congestion_charge(grid, params):
    """What each of the 18 connection points owes for this interval, in CHF.

    Added on top of ewz's real fair-LEG energy settlement, so you are pricing
    *congestion*, not electricity.

    You see the whole feeder, after the fact -- that is what being the network
    operator means. Everything is a plain array over the 18 connection points:

        grid.net_kwh        what each household pushed (+) or drew (-)
        grid.net_kw         the same as a power
        grid.voltage_pu     voltage at each one. THIS IS WHERE LOCATION LIVES:
                            two households exporting equally do not strain the
                            network equally, and this is the difference
        grid.transformer_kw  throughput at the substation, + = drawing
        grid.losses_kw       what the network itself burned
        grid.hour            0 to 24

    **It must sum to zero.** Money you take off one household you give back to
    the others. A tariff that simply pays everybody is a subsidy, and the
    scorer disqualifies it.

    The default charges whoever is pushing the feeder past `headroom_kwh`, in
    proportion to their share of the excess, and rebates the proceeds equally.
    It works, and it is crude: every household on the feeder sees the same
    congestion number, so a population tuned against it may well synchronise
    *harder*. A better tariff would distinguish *where* the strain is.
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

    # === YOUR IDEA GOES HERE ===================================================
    # The default ignores grid.voltage_pu entirely, which is why it is crude:
    # every household sees the same congestion number, so a population tuned
    # against it may synchronise harder rather than less. A price that leaned
    # on where the strain actually is would not have that problem.
    # ===========================================================================

    charge_chf = params["price_chf_per_kwh"] * excess_kwh * share
    return charge_chf - jnp.mean(charge_chf)  # rebate: sums to zero


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
