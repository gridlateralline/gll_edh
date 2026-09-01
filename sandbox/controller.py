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

"""One of the two seams you edit: the household controller.

A controller decides one number per interval -- **net active power at the
inverter, in kW, positive when injecting** -- for **one** household. The
harness ``jax.vmap``s it over the population.

That is not only a speed trick. Because your function is written for a single
household, there is no agent axis inside it and you *cannot* reference a
neighbour, even by accident. vmap is the fairness contract.

The signature::

    def controller(obs, carry, params, key) -> (p_set_kw, carry)

* ``obs``    -- :class:`~sandbox.observation.LocalObservation`, all scalars.
* ``carry``  -- your household's memory, carried between intervals. Must be a
  fixed-shape, fixed-dtype pytree: no growing lists, no changing dtypes.
* ``params`` -- your tunable numbers, shared across the population. Tune them
  across episodes; there is no price to read within one.
* ``key``    -- a fresh PRNG key per household per interval. Use it if you
  want to break symmetry -- see the note on desynchronization below.

Return anything you like; the harness clips to ``[p_min_kw, p_max_kw]`` and
the environment projects whatever survives onto the physically feasible set.
A controller cannot crash the simulation.

On the carry, and desynchronization
-----------------------------------
The carry is more useful than it looks precisely *because* there is no price.
It is where a load forecast, a voltage trend, or "how long since I last
curtailed" lives. It is also where **deliberate staggering** lives: a
household that remembers what it just did can offset itself against its
neighbours. Hysteresis and randomized start times are legitimate answers to
herding, and they are pure carry mechanisms.
"""

from typing import Any, Callable, Protocol

import chex
import jax.numpy as jnp

from sandbox.observation import LocalObservation


@chex.dataclass(frozen=True)
class Memory:
    """The default carry. Replace it with your own -- any fixed pytree works.

    Attributes:
        p_prev_kw: What this household set last interval.
        voltage_ewma_pu: Exponentially weighted mean of its own bus voltage,
            a cheap read on whether local congestion is building rather than
            merely present.
        intervals: Count of intervals elapsed, for anything phase-dependent.
    """

    p_prev_kw: chex.Array
    voltage_ewma_pu: chex.Array
    intervals: chex.Array


def init_memory() -> Memory:
    """A single household's starting memory -- no agent axis."""
    return Memory(
        p_prev_kw=jnp.float32(0.0),
        voltage_ewma_pu=jnp.float32(1.0),
        intervals=jnp.int32(0),
    )


class ControllerFn(Protocol):
    """``(obs, carry, params, key) -> (p_set_kw, carry)`` for one household."""

    def __call__(
        self,
        obs: LocalObservation,
        carry: Any,
        params: Any,
        key: chex.PRNGKey,
    ) -> tuple[chex.Array, Any]: ...


@chex.dataclass(frozen=True)
class Controller:
    """A controller bundled with the two things the harness needs to run it.

    Attributes:
        name: Label used in results and plots.
        fn: The per-household decision function.
        params: Tunable parameters, shared across the population.
        init_carry: Builds one household's starting memory.

    Parameters are shared deliberately. Heterogeneity lives in the *state* --
    a tenant has ``p_min_kw == p_max_kw == 0`` and the same parameters produce
    no action from them -- so one tuning run covers a mixed population and
    there is no per-agent best-response game to chase. Supply parameters with
    a leading agent axis if you want per-household values anyway.
    """

    name: str
    fn: ControllerFn
    params: Any
    init_carry: Callable[[], Any]


def clip_to_feasible(p_set_kw: chex.Array, obs: LocalObservation) -> chex.Array:
    """Clamp a request into this household's own feasible interval.

    The harness does this for you. It is exported because doing it *inside* a
    controller is often what you want: a rule that saturates should know it
    saturated, and can then record that in its carry.
    """
    return jnp.clip(p_set_kw, obs.p_min_kw, obs.p_max_kw)


def update_memory(carry: Memory, obs: LocalObservation, p_set_kw: chex.Array) -> Memory:
    """Roll the default carry forward. ``voltage_ewma_pu`` uses a 24-interval
    (six hour) time constant -- long enough to describe the neighbourhood
    rather than this instant, short enough to move within a day."""
    alpha = 1.0 / 24.0
    return Memory(
        p_prev_kw=p_set_kw,
        voltage_ewma_pu=(1.0 - alpha) * carry.voltage_ewma_pu + alpha * obs.voltage_pu,
        intervals=carry.intervals + 1,
    )


# ---------------------------------------------------------------------------
# The base controller
# ---------------------------------------------------------------------------


def hour_of_day(obs: "LocalObservation") -> chex.Array:
    """Hour of the day, 0 to 24, from the clock in the observation.

    The clock arrives as a sine/cosine pair so that it is continuous across
    midnight -- but that means you cannot read an hour off either one alone.
    ``12 * (1 - time_cos)`` looks like it works and does not: it peaks at 24 at
    noon and is symmetric about it, so "after 13:00" also matches 11:00.
    """
    angle = jnp.arctan2(obs.time_sin, obs.time_cos) % (2.0 * jnp.pi)
    return angle * (24.0 / (2.0 * jnp.pi))


def self_consumption(
    obs: LocalObservation,
    carry: Memory,
    params: dict[str, chex.Array],
    key: chex.PRNGKey,
) -> tuple[chex.Array, Memory]:
    """Cover your own load, bank the rest. What every home battery does by default.

    Asking the inverter for exactly the household's own consumption drives the
    meter to zero, and the inverter dispatches solar before battery, so the
    battery takes up whatever is left -- charging on surplus, discharging on
    shortfall.

    The second term is what stops that quietly curtailing. Once the battery
    cannot absorb any more, surplus generation has nowhere to go and the
    inverter throws it away rather than exporting it. A real household
    exports, so ask for the part of the surplus the battery cannot take.

    At its default parameters this is exactly the out-of-the-box behaviour of
    essentially every residential storage product -- and it is **the thing
    that causes the problem**. Every roof peaks at noon, so every battery
    charges at noon and is full by early afternoon, at which point the whole
    feeder exports at once. Every household cooks at seven, so every battery
    empties together. No price is involved: the correlation is in the weather
    and the working day.

    Two knobs, both no-ops by default, both the obvious first thing to tune:

    ``export_cap_kw``
        Never push more than this into the grid. Surplus above it goes to the
        battery, and once that is full, is curtailed. Blunt: it buys a flat
        feeder by throwing energy away.
    ``charge_after_hour``
        Leave the battery idle before this hour of the day. Counter-intuitive
        and much more interesting than the cap -- a battery that waits is
        still absorbing during the afternoon export peak instead of having
        filled up at eleven. It costs nothing in energy, only in timing.
    """
    del key
    charging_allowed = hour_of_day(obs) >= params["charge_after_hour"]

    surplus_kw = jnp.maximum(obs.pv_available_kw - obs.load_kw, 0.0)
    absorbable_kw = jnp.where(charging_allowed, obs.bat_charge_max_kw, 0.0)
    export_kw = jnp.minimum(jnp.maximum(surplus_kw - absorbable_kw, 0.0), params["export_cap_kw"])

    p_set_kw = clip_to_feasible(obs.load_kw + export_kw, obs)
    return p_set_kw, update_memory(carry, obs, p_set_kw)


def self_consumption_params() -> dict[str, chex.Array]:
    """Defaults that reproduce plain greedy self-consumption exactly."""
    return {
        "export_cap_kw": jnp.float32(1.0e3),
        "charge_after_hour": jnp.float32(0.0),
    }


#: The space a tuner explores when a tariff needs a household to best-respond
#: to it. Small on purpose: a grid search should finish while somebody is
#: watching, and two legible knobs beat six opaque ones.
TUNING_GRID: dict[str, list[float]] = {
    "export_cap_kw": [1.0e3, 8.0, 4.0, 2.0],
    "charge_after_hour": [0.0, 9.0, 11.0, 13.0],
}


def base_controller() -> Controller:
    """The reference household every submitted tariff is scored against.

    Tunable, and it has to be: with no price visible during an episode, a
    tariff reaches a household only by changing what that household would
    have wanted to do. Score a submitted tariff against a household that
    cannot re-tune and you measure redistribution and nothing else.
    """
    return Controller(
        name="self_consumption",
        fn=self_consumption,
        params=self_consumption_params(),
        init_carry=init_memory,
    )


def passive(
    obs: LocalObservation,
    carry: Memory,
    params: dict[str, chex.Array],
    key: chex.PRNGKey,
) -> tuple[chex.Array, Memory]:
    """Never use the battery: whatever the roof makes goes straight to the grid.

    The do-nothing anchor for the economics metrics. A tariff so punitive that
    nobody moves lands here, which is how the scorer tells "solved the
    problem" apart from "suppressed all activity".
    """
    del params, key
    p_set_kw = clip_to_feasible(obs.pv_available_kw, obs)
    return p_set_kw, update_memory(carry, obs, p_set_kw)


def passive_controller() -> Controller:
    return Controller(
        name="passive",
        fn=passive,
        params=self_consumption_params(),
        init_carry=init_memory,
    )


# ---------------------------------------------------------------------------
# Your controller
# ---------------------------------------------------------------------------


def my_controller(
    obs: LocalObservation,
    carry: Memory,
    params: dict[str, chex.Array],
    key: chex.PRNGKey,
) -> tuple[chex.Array, Memory]:
    """**EDIT ME.** Starts as greedy self-consumption with a voltage nudge.

    The nudge is the smallest possible gesture at the real question: back off
    injecting when your own terminal voltage is already high, since that is
    the local, measurable signature of a congested feeder. It is deliberately
    crude -- it reacts to *this* interval's voltage, so every household backs
    off in the same interval, which is exactly the herding this challenge is
    about. Beating it means anticipating instead.

    Ideas that need no price: use ``load_forecast_kw`` rather than ``load_kw``;
    hold charge back for the evening peak using the clock; watch
    ``carry.voltage_ewma_pu`` for a trend rather than a level; stagger against
    your neighbours using ``key``.
    """
    del key
    # Start from working self-consumption, then trim exports when this
    # household's own terminal voltage says the neighbourhood is already
    # pushing hard. Everything above is what a real battery does; only the
    # `allowance_kw` line is the design.
    surplus_kw = jnp.maximum(obs.pv_available_kw - obs.load_kw, 0.0)
    export_kw = jnp.maximum(surplus_kw - obs.bat_charge_max_kw, 0.0)

    # Voltage droop, trimming the standing allowance. Note the scale: on a
    # stiff feeder `excess_pu` is a few thousandths, so `droop_kw_per_pu` has
    # to be in the hundreds before it changes anything -- and if the allowance
    # it trims never binds in the first place, nothing happens at any gain.
    excess_pu = jnp.maximum(obs.voltage_pu - params["voltage_setpoint_pu"], 0.0)
    allowance_kw = jnp.maximum(params["export_cap_kw"] - params["droop_kw_per_pu"] * excess_pu, 0.0)

    p_set_kw = clip_to_feasible(obs.load_kw + jnp.minimum(export_kw, allowance_kw), obs)
    return p_set_kw, update_memory(carry, obs, p_set_kw)


def my_controller_params() -> dict[str, chex.Array]:
    """Starting parameters. These are what you tune across episodes."""
    return {
        # The allowance a droop of zero leaves in place. Start it where it
        # does not bind, so the default really is plain self-consumption.
        "export_cap_kw": jnp.float32(1.0e3),
        # Where "my neighbourhood is pushing hard" begins. On the urban feeder
        # voltage barely reaches 1.02, so a setpoint above that never fires --
        # check the range you are actually working in before picking one.
        "voltage_setpoint_pu": jnp.float32(1.005),
        # kW of allowance surrendered per pu of excess. Zero by default: the
        # example does nothing until you or a tuner make it.
        "droop_kw_per_pu": jnp.float32(0.0),
    }


#: What a tuner sweeps for :func:`my_controller`. Yours should name whichever
#: of your own parameters are worth searching; anything omitted keeps its
#: default.
MY_TUNING_GRID: dict[str, list[float]] = {
    "export_cap_kw": [1.0e3, 6.0, 3.0],
    "voltage_setpoint_pu": [1.000, 1.010],
    "droop_kw_per_pu": [0.0, 300.0],
}


def my_controller_bundle() -> Controller:
    return Controller(
        name="my_controller",
        fn=my_controller,
        params=my_controller_params(),
        init_carry=init_memory,
    )


__all__ = [
    "MY_TUNING_GRID",
    "TUNING_GRID",
    "Controller",
    "ControllerFn",
    "LocalObservation",
    "Memory",
    "base_controller",
    "clip_to_feasible",
    "hour_of_day",
    "init_memory",
    "my_controller",
    "my_controller_bundle",
    "my_controller_params",
    "passive",
    "passive_controller",
    "self_consumption",
    "self_consumption_params",
    "update_memory",
]
