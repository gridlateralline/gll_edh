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

There is no price in this observation, and that is the design, not an
omission. Real settlement lags by days or months -- past the end of an
episode -- so no controller can expect to see one in time. Anticipation lives
in the *parameters*, tuned across episodes; within an episode a household acts
on what its own meter and inverter can read.

Voltage looks like the price proxy, and is nearly useless as one
--------------------------------------------------------------
A nodal price is high exactly when the local feeder is congested, and
congestion is measurable at the household's own terminal: own bus voltage
correlates with feeder export at **+0.99** on every feeder here. So far so
good.

But measure how much of that a household did not already know. Regress own
voltage on own PV, own load and the clock, and roughly **90 % of it is already
explained**. What is left -- the part that genuinely describes the
neighbourhood rather than this roof -- is:

======== ============= ==================================
feeder   voltage span  residual after own PV, load, clock
======== ============= ==================================
urban     2.43 %        **0.20 %** of nominal
suburban  7.04 %        0.58 %
rural    12.05 %        0.86 %
======== ============= ==================================

Regenerate with ``uv run python scripts/measure_voltage_residual.py``, which
carries the exact regression these come from. They move whenever the weather
model, the population or the feeder set does.

A Class 1 meter resolves roughly 0.5 % of nominal. So on ``urban`` the
neighbourhood-only content of the voltage signal sits well below what a real
meter could see: a controller reading voltage there is reading a noisy
restatement of "it is noon and my roof is working". It is marginal on
``suburban`` and genuinely readable on ``rural``, which is why the hackathon
runs there.

Which is the deepest thing this sandbox has to say
--------------------------------------------------
It explains *why* herding is hard rather than merely that it happens. Every
signal above is correlated across the feeder -- one cloud shades the whole
neighbourhood, everyone cooks at seven -- and the one signal that is
genuinely about the neighbourhood carries almost no information a household
did not already have. There is nearly **no idiosyncratic local information**.
Households move together not by accident but because the information
structure leaves them nothing to differentiate on.

So the design space is not "read the local signal better". It is to
**manufacture differentiation where the physics provides none**: through the
carry (memory, hysteresis, staggering), through `key` (deliberate
desynchronisation), or -- from the other seam -- through a tariff that creates
locational distinctions the voltages do not.

Units
-----
Power in **kW**, energy in **kWh**, money in **CHF**, voltage in **per unit**.

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
        meter_kw: Net power at your grid connection over the interval that
            just ended. Positive means you were injecting.
        load_kw: Your own consumption over the interval that just ended.
        load_forecast_kw: Your expected consumption over the coming interval.
        pv_available_kw: The most your roof can produce in the coming
            interval. Not a commitment -- unused generation is curtailed.
        soc_kwh: Energy currently stored in your battery. Zero, always, if you
            have none.
        soc_headroom_kwh: How much more your battery can absorb.
        bat_charge_max_kw: The fastest you may charge over the coming
            interval, already limited by both the battery's rating and how
            much room is left in it. Zero when full, or when you have none.
        bat_discharge_max_kw: The fastest you may discharge, likewise limited
            by rating and by what is actually stored.
        p_min_kw: The least active power you may put out over the coming
            interval, negative when you may draw. Already accounts for your
            inverter rating, your grid connection, your battery's state, and
            the reactive power the Q(U) grid code has committed on your
            behalf.
        p_max_kw: The most active power you may put out.

    ``p_min_kw`` and ``p_max_kw`` are the interval an action is actually
    judged against, and they are strictly narrower than the inverter's own
    rating. You do not have to respect them -- anything you return is clipped
    and projected -- but a controller that ignores them is asking for
    something it will not get.
    """

    hour: chex.Array
    time_sin: chex.Array
    time_cos: chex.Array
    voltage_pu: chex.Array
    meter_kw: chex.Array
    load_kw: chex.Array
    load_forecast_kw: chex.Array
    pv_available_kw: chex.Array
    soc_kwh: chex.Array
    soc_headroom_kwh: chex.Array
    bat_charge_max_kw: chex.Array
    bat_discharge_max_kw: chex.Array
    p_min_kw: chex.Array
    p_max_kw: chex.Array

    def as_dict(self) -> dict[str, chex.Array]:
        """Plain dict of named values, for controllers written in NumPy.

        Nobody should have to learn a type hierarchy to write a heuristic.
        """
        return {
            "hour": self.hour,
            "time_sin": self.time_sin,
            "time_cos": self.time_cos,
            "voltage_pu": self.voltage_pu,
            "meter_kw": self.meter_kw,
            "load_kw": self.load_kw,
            "load_forecast_kw": self.load_forecast_kw,
            "pv_available_kw": self.pv_available_kw,
            "soc_kwh": self.soc_kwh,
            "soc_headroom_kwh": self.soc_headroom_kwh,
            "bat_charge_max_kw": self.bat_charge_max_kw,
            "bat_discharge_max_kw": self.bat_discharge_max_kw,
            "p_min_kw": self.p_min_kw,
            "p_max_kw": self.p_max_kw,
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
        meter_kw=jnp.take(jnp.asarray(prosumer.p_pq_realized), pq) / step_h,
        load_kw=jnp.take(jnp.asarray(load.p_load_realized), pq) / step_h,
        load_forecast_kw=jnp.take(jnp.asarray(load.p_load_forecast), pq) / step_h,
        pv_available_kw=jnp.asarray(solar.sol_request_max) / step_h,
        soc_kwh=jnp.asarray(battery.bat_full),
        soc_headroom_kwh=jnp.asarray(battery.bat_free),
        # Battery flow is signed the way the inverter sees it: discharging
        # adds to the inverter's output, charging subtracts. So the most
        # negative admissible request is the fastest charge, and both bounds
        # already fold in the state of charge.
        bat_charge_max_kw=jnp.maximum(-jnp.asarray(battery.bat_request_min), 0.0) / step_h,
        bat_discharge_max_kw=jnp.maximum(jnp.asarray(battery.bat_request_max), 0.0) / step_h,
        p_min_kw=minimum * scale,
        p_max_kw=maximum * scale,
    )


def to_action(model: "EnvironmentDynamics", p_set_kw: chex.Array) -> chex.Array:
    """Convert a controller's kW request into the normalized action the env takes.

    Clipped to ``[-1, 1]``: a controller may return anything at all, and the
    environment's own projection handles whatever survives the clip. There is
    no such thing as a crashing controller here.
    """
    step_h = float(model.time.step_duration_h)
    scale = jnp.asarray(model.action_scale) / step_h
    normalized = jnp.asarray(p_set_kw, dtype=jnp.float32) / scale
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
        net_kwh: (num_pq,) Net energy at each connection point over the
            interval, positive when that household pushed into the grid.
        net_kw: (num_pq,) The same thing as a power.
        voltage_pu: (num_pq,) Voltage at each connection point.

            Careful with this one. A bus voltage is mostly made by OTHER
            households, so pricing the level charges exposure rather than
            contribution: a household at the end of a busy line pays for a
            condition it did not create, and cutting its own injection barely
            moves it. A locational price wants the SENSITIVITY of the binding
            quantity to that household's own injection; this is an input to
            estimating that, not the answer.
        transformer_kw: () Throughput at the substation, positive when the
            feeder draws from the grid and negative when it exports.
        losses_kw: () What the network itself consumed. Quadratic in flow, so
            synchronised behaviour costs more than its average suggests.
        hour: () Hour of the settled interval, 0 to 24.
        energy_chf: (num_pq,) What fair LEG -- ewz's published rate, in force
            today -- would settle this interval as. A starting point, nothing
            more: a tariff is free to add to it, replace pieces of it, or
            ignore it completely and price the interval from scratch. It is
            here so that "design a tariff" does not silently mean "design a
            surcharge on top of this one" -- see
            :func:`sandbox.tariff.tariff_from_settlement`.
        has_inverter: (num_pq,) bool. Static equipment fact, not a live
            reading -- who has an inverter (and therefore can act at all)
            does not change during an episode. It is here so a tariff can say
            what it means directly (a tenant floor, a different rate class)
            instead of inferring it from behaviour. It is *not* an escape
            hatch for pricing what a household did this interval; use
            `net_kwh` / `voltage_pu` for that, the way a real rate class
            never depends on this week's meter reading.
    """

    net_kwh: chex.Array
    net_kw: chex.Array
    voltage_pu: chex.Array
    transformer_kw: chex.Numeric
    losses_kw: chex.Numeric
    hour: chex.Numeric
    energy_chf: chex.Array
    has_inverter: chex.Array

    def as_dict(self) -> dict[str, chex.Array]:
        """Plain dict of named values, for tariffs written in NumPy.

        Nobody should have to learn a type hierarchy to write a heuristic.
        """
        return {
            "net_kwh": self.net_kwh,
            "net_kw": self.net_kw,
            "voltage_pu": self.voltage_pu,
            "transformer_kw": self.transformer_kw,
            "losses_kw": self.losses_kw,
            "hour": self.hour,
            "energy_chf": self.energy_chf,
            "has_inverter": self.has_inverter,
        }


def to_grid_view(
    env_model: Any, new_state: Any, energy_chf: Optional[chex.Array] = None
) -> GridView:
    """Build the tariff's view from the environment state it just settled.

    `energy_chf` is optional because computing it needs the reward state and
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

    net_kwh = jnp.real(new_state.prosumer_state.s_pq_realized_kvah)
    return GridView(
        net_kwh=net_kwh,
        net_kw=net_kwh / step_h,
        voltage_pu=jnp.abs(new_state.grid_state.bus_voltage_pu)[pq_id],
        transformer_kw=grid.pu_to_kw(jnp.real(injection_pu)[slack]),
        losses_kw=grid.pu_to_kw(jnp.sum(jnp.real(injection_pu))),
        energy_chf=(
            jnp.zeros_like(net_kwh)
            if energy_chf is None
            else jnp.asarray(energy_chf, dtype=jnp.float32)
        ),
        has_inverter=jnp.zeros((num_pq,), dtype=bool).at[inverter_id].set(True),
        hour=env_model.time.observation(new_state.time_state).interval_start,
    )
