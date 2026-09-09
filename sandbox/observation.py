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

"""What one household can measure. Everything here is SI, and none of it is a price.

The three quantities, and why they are not the same number
----------------------------------------------------------
This is the single most important thing on the household side, and the one
most easily got wrong::

    p_inv_kw  (you choose)  -  p_load_kw  (you don't)  =  p_grid_kw  (you're billed on)

* **``p_inv_kw``** is what your controller **returns**: active power out of
  the *inverter*, positive when it is producing. Solar and battery both sit
  behind it, and the inverter serves solar first, so asking for more than
  the roof is making discharges the battery and asking for less charges it.
  It is bounded by ``p_inv_min_kw`` / ``p_inv_max_kw``.
* **``p_load_kw``** is the household's own consumption, sitting behind the same
  meter. Nobody controls it here -- it is a stochastic profile, and it is
  already known one interval ahead as ``p_load_forecast_kw``.
* **``p_grid_kw``** is the net exchange with the grid, ``p_inv_kw -
  p_load_kw``. **This is the quantity every tariff settles**, the quantity the
  feeder actually carries, and the quantity ``GridView`` publishes to the
  tariff as ``p_grid_kw`` / ``e_grid_kwh``. Your controller never sets it
  directly; it sets ``p_inv_kw`` and the subtraction happens for it.

So "export nothing this interval" is ``p_inv_kw = p_load_forecast_kw``, not
``p_inv_kw = 0`` -- returning zero means idling the inverter and importing
the entire load. The observation reports last interval's realized exchange
as ``p_grid_kw``, so a controller can see what its last request actually
came to.

There is no price in this observation
-------------------------------------
That is the design, not an omission. Real settlement lags by days or months
-- past the end of an episode -- so no controller can expect to see one in
time. Anticipation lives in the *parameters*, tuned across episodes; within
an episode a household acts on what its own meter and inverter can read.

Voltage: a thin signal, but a real one
--------------------------------------
The project is called *grid lateral line* partly for this: a fish reads the
water it is in through a line of local pressure sensors, and a household's
own bus voltage is the closest thing it has to the same organ. Local, cheap,
continuously available, and genuinely coupled to what the neighbours are
doing -- own bus voltage correlates with feeder export at **+0.99**.

The honest caveat is not that correlation, it is *redundancy*. Regress own
voltage on own PV, own load and the clock and about **86 %** of it is
already implied by things the household knew anyway. What is left -- the
part that genuinely describes the neighbourhood rather than this roof -- is:

======== ============= ==================================
feeder   voltage span  residual after own PV, load, clock
======== ============= ==================================
urban     2.88 %        0.25 % of nominal
suburban  8.12 %        0.67 %
rural    14.07 %        **1.01 %**
======== ============= ==================================

A Class 1 meter resolves roughly 0.5 % of nominal. On ``rural`` -- the feeder
this hackathon runs on, and the reason it does -- 1.01 % is twice that.
**The residual is a real, measurable signal and building a controller on it
is a legitimate design.** What the table warns against is only the *naive*
version: a threshold on the raw level mostly fires on "it is noon and my
roof is working", which every household knows without looking. Subtract what
you already know, and what is left is the neighbourhood.

And the residual is not a fixed budget. It is small partly *because* every
household currently runs the same rule off the same weather, so their
contributions to each other's voltage move together. A population that
deliberately differentiates -- staggered start times, hysteresis, randomised
timing off ``key`` -- makes its members' voltages less predictable from
their own state, which puts information back into the residual. Reading the
signal and creating something worth reading are the same project.

Regenerate the table with ``uv run python scripts/measure_voltage_residual.py``,
which carries the exact regression it comes from.

Why herding is hard
-------------------
Every local signal is correlated across the feeder -- one cloud shades the
whole neighbourhood, everyone cooks at seven -- and the one signal that is
genuinely about the neighbourhood is thin. There is very little
**idiosyncratic local information**. Households move together not by
accident but because the information structure gives them little to
differentiate on.

So the design space has two halves, and good submissions use both:
**read what is there** (the voltage residual, a trend rather than a level),
and **manufacture differentiation where the physics provides none** --
through the carry (memory, hysteresis, staggering), through ``key``
(deliberate desynchronisation), or, from the other seam, through a tariff
that creates locational distinctions the voltages do not.

Units, and how a field name tells you what it is
------------------------------------------------
Power in **kW**, energy in **kWh**, money in **CHF**, voltage in **per unit**.

The suffix carries the physics, the way it does on a datasheet: ``_kw`` is
**active** power, ``_kvar`` **reactive**, ``_kva`` **apparent**, with
``_kwh`` / ``_kvarh`` the matching energies. Where both halves exist at the
same terminal the name also carries a ``p_`` / ``q_`` prefix so the pair
reads together -- ``p_grid_kw`` beside ``q_grid_kvar``, ``p_inv_kw`` beside
the reactive power Q(U) sets. Quantities that are inherently active-only get
no prefix: a roof (``pv_available_kw``) and a battery (``soc_kwh``,
``bat_charge_max_kw``) have no reactive half to distinguish them from.

**The household seam is active power only, on purpose.** On a Swiss LV
connection the reactive axis is not the household's to choose: the Q(U) grid
code (NE7 4.3.2) sets the inverter's reactive power from the voltage at its
own bus, which is why the environment's action space is one-dimensional here
rather than two. What that leaves the controller is ``p_inv_kw``, and what
it costs is apparent-power headroom: ``p_inv_min_kw`` / ``p_inv_max_kw`` are
the active slice that survives once Q(U) has claimed its share. They sit
below the inverter's nameplate on about 91 % of intervals -- sometimes
because the roof and battery have nothing more to give, sometimes because
Q(U) has taken the headroom -- which is why an action is judged against them
and not against the datasheet. The tariff seam does see reactive power, as
``q_grid_kvar`` on :class:`GridView`; read its warning before pricing it.

`gll_env` works internally in energy-per-interval (kWh per 15 minutes), which
is right for a simulator and wrong for a person: a datasheet says 8 kWp and a
household says "charge at 3 kW". The conversion happens once, here. Voltage
stays per-unit because that is how EN 50160 states its limits, and it is the
one place where the normalized form *is* the natural unit.

Every field name carries its unit, because a silent factor-of-four between kW
and kWh is the single easiest mistake to make in this codebase.
"""

from typing import TYPE_CHECKING, Any, Optional

import chex
import jax.numpy as jnp

if TYPE_CHECKING:
    from gll_env.components.environment import (
        EnvironmentDynamics,
        EnvironmentObservation,
        EnvironmentState,
    )


@chex.dataclass(frozen=True)
class LocalObservation:
    """One household's view. Batched over agents outside a controller, scalar inside.

    A controller is written for a *single* household and ``jax.vmap``'d over
    the population, so inside your function every field below is a scalar.
    There is no agent axis to index, which is what makes it structurally
    impossible to peek at a neighbour.

    Every field describes either the interval that just **ended** (realized)
    or the one about to **begin** (forecast); the column is marked below,
    because acting on last interval's load when you were handed this
    interval's forecast is the commonest silent error here.

    Attributes:
        hour: Hour of the coming interval, 0 to 24, measured from midnight.
            **This is how you read the clock.** It comes straight off the
            environment's own `interval_start`, so it is exact and it is the
            interval you are about to act in.

            Do not try to reconstruct it from `time_sin`/`time_cos`. Those are
            deliberately a *pair* -- one alone is ambiguous, since a cosine is
            symmetric about noon -- and they describe the interval's MIDPOINT,
            so even a correct `atan2` of the two lands half an interval late.
        time_sin: Sine of the time of day. Together with `time_cos` this is a
            continuous clock with no discontinuity at midnight.
        time_cos: Cosine of the time of day.
        voltage_pu: Voltage magnitude at your own connection point, per unit
            of nominal. EN 50160 wants this inside 0.9-1.1; above ~1.05 your
            neighbourhood is exporting hard, below ~0.95 it is drawing hard.
        p_grid_kw: **Realized.** Net power exchanged with the grid over the
            interval that just ended -- ``p_inv_kw - p_load_kw``, positive when
            you were injecting. **This is the quantity you are billed on.**
            Your controller does not set it directly; it sets ``p_inv_kw``.
        p_load_kw: **Realized.** Your own consumption over the interval that
            just ended, behind the same meter. Not controllable.
        p_load_forecast_kw: **Forecast.** Your consumption over the *coming*
            interval -- the one you are about to act in. Returning
            ``p_inv_kw = p_load_forecast_kw`` is what drives ``p_grid_kw`` to
            zero; returning ``p_load_kw`` instead leaves whatever the load
            changed by, one interval late.
        pv_available_kw: **Forecast.** The most your roof can produce in the
            coming interval. Not a commitment -- unused generation is
            curtailed.
        soc_kwh: Energy currently stored in your battery. Zero, always, if you
            have none.
        soc_headroom_kwh: How much more your battery can absorb.
        bat_charge_max_kw: The fastest you may charge over the coming
            interval, already limited by both the battery's rating and how
            much room is left in it. Zero when full, or when you have none.
        bat_discharge_max_kw: The fastest you may discharge, likewise limited
            by rating and by what is actually stored.
        p_inv_min_kw: The least **inverter** active power you may request over
            the coming interval, negative when the inverter may absorb (charge
            the battery from the grid). Already accounts for your inverter
            rating, your grid connection, your battery's state, and the
            reactive power the Q(U) grid code has committed on your behalf.
        p_inv_max_kw: The most inverter active power you may request.

    ``p_inv_min_kw`` and ``p_inv_max_kw`` bound the **inverter** request your
    controller returns, not the grid exchange, and they are strictly narrower
    than the inverter's own nameplate. You do not have to respect them --
    anything you return is clipped and then projected onto the physically
    feasible set -- but a controller that ignores them is asking for something
    it will not get. A tenant has ``p_inv_min_kw == p_inv_max_kw == 0``: no
    inverter, no action, and every controller must survive that case.
    """

    hour: chex.Array
    time_sin: chex.Array
    time_cos: chex.Array
    voltage_pu: chex.Array
    p_grid_kw: chex.Array
    p_load_kw: chex.Array
    p_load_forecast_kw: chex.Array
    pv_available_kw: chex.Array
    soc_kwh: chex.Array
    soc_headroom_kwh: chex.Array
    bat_charge_max_kw: chex.Array
    bat_discharge_max_kw: chex.Array
    p_inv_min_kw: chex.Array
    p_inv_max_kw: chex.Array

    def as_dict(self) -> dict[str, chex.Array]:
        """Plain dict of named values, for controllers written in NumPy.

        Nobody should have to learn a type hierarchy to write a heuristic.
        """
        return {
            "hour": self.hour,
            "time_sin": self.time_sin,
            "time_cos": self.time_cos,
            "voltage_pu": self.voltage_pu,
            "p_grid_kw": self.p_grid_kw,
            "p_load_kw": self.p_load_kw,
            "p_load_forecast_kw": self.p_load_forecast_kw,
            "pv_available_kw": self.pv_available_kw,
            "soc_kwh": self.soc_kwh,
            "soc_headroom_kwh": self.soc_headroom_kwh,
            "bat_charge_max_kw": self.bat_charge_max_kw,
            "bat_discharge_max_kw": self.bat_discharge_max_kw,
            "p_inv_min_kw": self.p_inv_min_kw,
            "p_inv_max_kw": self.p_inv_max_kw,
        }


def to_local(
    model: "EnvironmentDynamics",
    observation: "EnvironmentObservation",
    state: "EnvironmentState",
) -> LocalObservation:
    """Slice the full environment observation down to what each meter sees.

    Takes the state as well as the observation because the feasible action
    interval lives on the state (``action_constraints``) rather than in the
    observation proper.

    Every gather here is by agent: grid quantities through ``agent_bus_id``,
    connection-point quantities through ``inverter_id``. Nothing global
    survives, which is the point.
    """
    step_h = float(model.time.step_duration_h)

    grid = observation.grid_observation
    prosumer = observation.prosumer_observation
    load = prosumer.load_observation
    inverter = prosumer.inverter_observation
    battery = inverter.battery_observation
    solar = inverter.solar_observation
    clock = observation.time_observation

    bus = model.agent_bus_id
    pq = jnp.asarray(model.prosumer.inverter_id, dtype=jnp.int32)
    num_agents = model.num_agents

    # bus_voltage_deviation is a kV offset from nominal, and pu_to_kv is a
    # plain scale, so dividing by the bus's own base recovers per unit exactly.
    base_kv = jnp.asarray(model.grid.base_v_kv, dtype=jnp.float32)
    voltage_pu = 1.0 + jnp.take(jnp.asarray(grid.bus_voltage_deviation) / base_kv, bus)

    # action_constraints live in normalized [-1, 1] space; action_scale carries
    # them back to kWh, and step_duration_h to kW.
    minimum, maximum = state.action_constraints.bounds()
    scale = jnp.asarray(model.action_scale) / step_h

    return LocalObservation(
        hour=jnp.broadcast_to(jnp.asarray(clock.interval_start), (num_agents,)),
        time_sin=jnp.broadcast_to(jnp.asarray(clock.time_sin), (num_agents,)),
        time_cos=jnp.broadcast_to(jnp.asarray(clock.time_cos), (num_agents,)),
        voltage_pu=voltage_pu,
        p_grid_kw=jnp.take(jnp.asarray(prosumer.p_pq_realized), pq) / step_h,
        p_load_kw=jnp.take(jnp.asarray(load.p_load_realized), pq) / step_h,
        p_load_forecast_kw=jnp.take(jnp.asarray(load.p_load_forecast), pq) / step_h,
        pv_available_kw=jnp.asarray(solar.sol_request_max) / step_h,
        soc_kwh=jnp.asarray(battery.bat_full),
        soc_headroom_kwh=jnp.asarray(battery.bat_free),
        # Battery flow is signed the way the inverter sees it: discharging
        # adds to the inverter's output, charging subtracts. So the most
        # negative admissible request is the fastest charge, and both bounds
        # already fold in the state of charge.
        bat_charge_max_kw=jnp.maximum(-jnp.asarray(battery.bat_request_min), 0.0) / step_h,
        bat_discharge_max_kw=jnp.maximum(jnp.asarray(battery.bat_request_max), 0.0) / step_h,
        p_inv_min_kw=minimum * scale,
        p_inv_max_kw=maximum * scale,
    )


def to_action(model: "EnvironmentDynamics", p_inv_kw: chex.Array) -> chex.Array:
    """Convert a controller's kW request into the normalized action the env takes.

    Clipped to ``[-1, 1]``: a controller may return anything at all, and the
    environment's own projection handles whatever survives the clip. There is
    no such thing as a crashing controller here.
    """
    step_h = float(model.time.step_duration_h)
    scale = jnp.asarray(model.action_scale) / step_h
    normalized = jnp.asarray(p_inv_kw, dtype=jnp.float32) / scale
    return jnp.clip(normalized, -1.0, 1.0).reshape(model.num_agents, model.action_dim)


# ---------------------------------------------------------------------------
# What the NETWORK sees. The tariff's counterpart to LocalObservation.
# ---------------------------------------------------------------------------


@chex.dataclass(frozen=True)
class GridView:
    """The whole feeder after an interval has been solved, in SI units.

    The tariff's view, and deliberately the mirror image of
    :class:`LocalObservation`: a household sees one meter and no prices, the
    network operator sees every connection point and every voltage, after the
    fact.

    It exists so that writing a tariff never requires reading `gll_env`.
    Everything a price is likely to depend on is here, flat and named, in
    kW / kWh / pu -- no per-unit conversions, no bus-versus-connection-point
    index hops, no environment state types.

    It is a fixed set, and deliberately not an exhaustive one. A tariff that
    needs something absent -- per-branch flows, the power-flow Jacobian,
    reactive power per node -- overrides
    :meth:`~sandbox.tariff.MyTariff.settle` instead and receives the full
    environment state and dynamics. That is the escape hatch, it is supported,
    and it is the one place where reading `gll_env` becomes necessary. Ask for
    a field here if you find yourself using it twice.

    Attributes:
        e_grid_kwh: (num_pq,) Net energy exchanged with the grid at each
            connection point over the interval -- inverter output minus
            household load, positive when that connection point pushed into
            the grid. It is the household side's ``p_grid_kw`` as an energy,
            the quantity the feeder actually carried, and the quantity a
            settlement is normally written against.
        p_grid_kw: (num_pq,) The same thing as a power.
        voltage_pu: (num_pq,) Voltage at each connection point.

            Careful with this one. A bus voltage is mostly made by OTHER
            households, so pricing the level charges exposure rather than
            contribution: a household at the end of a busy line pays for a
            condition it did not create, and cutting its own injection barely
            moves it. A locational price wants the SENSITIVITY of the binding
            quantity to that household's own injection; this is an input to
            estimating that, not the answer.
        q_grid_kvar: (num_pq,) Reactive power at each connection point, the
            reactive counterpart of `p_grid_kw`. Real, measured, and a real
            cost to the network -- reactive current heats the same conductors
            as active current, which is why it shows up inside `losses_kw`.

            **Read the same warning as `voltage_pu` before pricing it.** On a
            Swiss LV feeder a household does not choose its reactive power:
            the Q(U) grid code sets the inverter's share from the voltage at
            its own bus (NE7 4.3.2), and the rest is the load's own power
            factor. Charging a connection point for it is therefore charging
            exposure rather than contribution -- the household can only move
            it indirectly, by changing the active power that moves its
            voltage. If what you want is to price the cost reactive flow
            imposes, `losses_kw` already contains it and is caused by
            everybody together.

            Measured on the reference week: the agents' connection points
            carry more than 0.1 kvar on 90 % of intervals, averaging
            -1.03 kvar and reaching -6.4 kvar.
        transformer_kw: () Active throughput at the substation, positive when
            the feeder draws from the grid and negative when it exports.
        transformer_kvar: () Reactive throughput at the substation. A
            transformer is rated in kVA, so what its thermal limit actually
            sees is ``hypot(transformer_kw, transformer_kvar)`` -- worth
            knowing if you are pricing substation loading rather than active
            energy.
        losses_kw: () What the network itself consumed. Quadratic in flow, so
            synchronised behaviour costs more than its average suggests.
        hour: () Hour of the settled interval, 0 to 24.
        fair_leg_chf: (num_pq,) CHF, signed (+ = the connection point is paid).
            The **whole settlement** the fair-LEG baseline would produce for
            this interval -- not a rate, not a price per kWh: the finished
            number, energy and grid fee and public duties included. See
            :func:`sandbox.tariff.base_payments` for what fair LEG is and is
            not. A starting point, nothing more: a tariff is free to add to
            it, replace pieces of it, or ignore it completely and price the
            interval from scratch. It is here so that "design a tariff" does
            not silently mean "design a surcharge on top of this one" -- see
            :func:`sandbox.tariff.tariff_from_settlement`.
        has_inverter: (num_pq,) bool. Static equipment fact, not a live
            reading -- who has an inverter (and therefore can act at all)
            does not change during an episode. It is here so a tariff can say
            what it means directly (a tenant floor, a different rate class)
            instead of inferring it from behaviour. It is *not* an escape
            hatch for pricing what a household did this interval; use
            `e_grid_kwh` / `voltage_pu` for that, the way a real rate class
            never depends on this week's meter reading.
    """

    e_grid_kwh: chex.Array
    p_grid_kw: chex.Array
    q_grid_kvar: chex.Array
    voltage_pu: chex.Array
    transformer_kw: chex.Numeric
    transformer_kvar: chex.Numeric
    losses_kw: chex.Numeric
    hour: chex.Numeric
    fair_leg_chf: chex.Array
    has_inverter: chex.Array

    def as_dict(self) -> dict[str, chex.Array]:
        """Plain dict of named values, for tariffs written in NumPy.

        Nobody should have to learn a type hierarchy to write a heuristic.
        """
        return {
            "e_grid_kwh": self.e_grid_kwh,
            "p_grid_kw": self.p_grid_kw,
            "q_grid_kvar": self.q_grid_kvar,
            "voltage_pu": self.voltage_pu,
            "transformer_kw": self.transformer_kw,
            "transformer_kvar": self.transformer_kvar,
            "losses_kw": self.losses_kw,
            "hour": self.hour,
            "fair_leg_chf": self.fair_leg_chf,
            "has_inverter": self.has_inverter,
        }


def to_grid_view(
    env_model: Any, new_state: Any, fair_leg_chf: Optional[chex.Array] = None
) -> GridView:
    """Build the tariff's view from the environment state it just settled.

    `fair_leg_chf` is optional because computing it needs the reward state and
    dynamics a tariff carries, not just `new_state` -- callers without it get
    zeros, which is exactly right anywhere fair LEG itself is not in the loop
    (tests, the quickstart notebook).
    """
    grid = env_model.grid
    pq_id = jnp.asarray(grid.pq_id, dtype=jnp.int32)
    slack = jnp.asarray(grid.slack_id, dtype=jnp.int32)[0]
    injection_pu = new_state.grid_state.bus_power_injection_pu
    step_h = jnp.asarray(env_model.time.step_duration_h, dtype=jnp.float32)
    num_pq = jnp.asarray(pq_id).shape[0]
    inverter_id = jnp.asarray(env_model.prosumer.inverter_id, dtype=jnp.int32)

    s_grid_kvah = new_state.prosumer_state.s_pq_realized_kvah
    e_grid_kwh = jnp.real(s_grid_kvah)
    return GridView(
        e_grid_kwh=e_grid_kwh,
        p_grid_kw=e_grid_kwh / step_h,
        q_grid_kvar=jnp.imag(s_grid_kvah) / step_h,
        voltage_pu=jnp.abs(new_state.grid_state.bus_voltage_pu)[pq_id],
        transformer_kw=grid.pu_to_kw(jnp.real(injection_pu)[slack]),
        transformer_kvar=grid.pu_to_kw(jnp.imag(injection_pu)[slack]),
        losses_kw=grid.pu_to_kw(jnp.sum(jnp.real(injection_pu))),
        fair_leg_chf=(
            jnp.zeros_like(e_grid_kwh)
            if fair_leg_chf is None
            else jnp.asarray(fair_leg_chf, dtype=jnp.float32)
        ),
        has_inverter=jnp.zeros((num_pq,), dtype=bool).at[inverter_id].set(True),
        hour=env_model.time.observation(new_state.time_state).interval_start,
    )
