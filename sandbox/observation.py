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

**Voltage is the price proxy.** A nodal price is high exactly when the local
feeder is congested, and congestion is directly measurable at the household's
own terminal. That is the whole intellectual content of the household pathway:
hand teams a price and they write a threshold rule, never discovering that
their own bus voltage told them the same thing, earlier and for free.

And every one of these signals is *correlated across the feeder* -- one cloud
shades the whole neighbourhood, everyone cooks at seven. So the same
information that lets a household anticipate makes every household infer the
same thing at the same moment. The anticipation mechanism and the herding
mechanism are the same mechanism. That is the challenge.

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

from typing import TYPE_CHECKING

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
