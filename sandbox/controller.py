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

A controller decides one number per interval, for **one** household::

    def controller(obs, carry, params, key) -> (p_inv_kw, carry)

``p_inv_kw`` is **active power out of the inverter, in kW, positive when the
inverter is producing.** It is *not* the flow at the grid connection. The
household's own load sits behind the same meter and nobody controls it, so
what the feeder sees -- and what every tariff settles -- is::

    p_grid_kw = p_inv_kw - p_load_kw

Returning ``p_inv_kw = 0`` idles the inverter and imports the whole load;
returning ``p_inv_kw = obs.p_load_forecast_kw`` is what drives the grid
exchange to zero. :class:`~sandbox.observation.LocalObservation` carries the
whole picture: ``p_load_forecast_kw`` for the coming interval, and
``p_grid_kw`` for what last interval's request actually came to.

The harness ``jax.vmap``s the function over the population. That is not only
a speed trick: because it is written for a single household there is no agent
axis inside it and you *cannot* reference a neighbour, even by accident. vmap
is the fairness contract.

* ``obs``    -- :class:`~sandbox.observation.LocalObservation`, all scalars.
* ``carry``  -- your household's memory, carried between intervals. Must be a
  fixed-shape, fixed-dtype pytree: no growing lists, no changing dtypes.
* ``params`` -- your tunable numbers, shared across the population. Tune them
  across episodes; there is no price to read within one.
* ``key``    -- a fresh PRNG key per household per interval. Use it if you
  want to break symmetry -- see the note on desynchronization below.

Return anything you like; the harness clips to ``[p_inv_min_kw,
p_inv_max_kw]`` and the environment projects whatever survives onto the
physically feasible set. A controller cannot crash the simulation.

On the carry, and desynchronization
-----------------------------------
The carry is more useful than it looks precisely *because* there is no price.
It is where a load forecast, a voltage trend, or "how long since I last
curtailed" lives. It is also where **deliberate staggering** lives: a
household that remembers what it just did can offset itself against its
neighbours. Hysteresis and randomized start times are legitimate answers to
herding, and they are pure carry mechanisms.

Randomness, and why ``key`` is a key
------------------------------------
``key`` is a JAX PRNG key, freshly split per household per interval, so
identical households draw *different* numbers -- which is the entire point.
JAX has no hidden global RNG state: you pass a key in and get numbers out,
and the same key always gives the same numbers, which is what keeps a scored
run reproducible. Draw from it directly and you need no state at all::

    jitter_h = jax.random.uniform(key, minval=-1.0, maxval=1.0)  # per household
    start_h = params["charge_after_hour"] + params["jitter_h"] * jitter_h

That draws afresh every interval, which is right for adding noise to an
action and wrong for a *start time* -- a household whose start hour rerolls
every fifteen minutes has no start hour. For a per-household constant, draw
once and keep it in the carry::

    offset = jnp.where(carry.intervals == 0,
                       jax.random.uniform(key, minval=0.0, maxval=params["spread_h"]),
                       carry.offset_h)

Two rules and nothing more: never reuse a key for two different draws (split
it -- ``k1, k2 = jax.random.split(key)``), and remember that any parameter
you randomize over is still tuned as a *distribution* across episodes, not
picked per run.
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
    """``(obs, carry, params, key) -> (p_inv_kw, carry)`` for one household."""

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
    a tenant has ``p_inv_min_kw == p_inv_max_kw == 0`` and the same parameters produce
    no action from them -- so one tuning run covers a mixed population and
    there is no per-agent best-response game to chase. Supply parameters with
    a leading agent axis if you want per-household values anyway.
    """

    name: str
    fn: ControllerFn
    params: Any
    init_carry: Callable[[], Any]


def clip_to_feasible(p_inv_kw: chex.Array, obs: LocalObservation) -> chex.Array:
    """Clamp a request into this household's own feasible interval.

    The harness does this for you. It is exported because doing it *inside* a
    controller is often what you want: a rule that saturates should know it
    saturated, and can then record that in its carry.
    """
    return jnp.clip(p_inv_kw, obs.p_inv_min_kw, obs.p_inv_max_kw)


def update_memory(carry: Memory, obs: LocalObservation, p_inv_kw: chex.Array) -> Memory:
    """Roll the default carry forward. ``voltage_ewma_pu`` uses a 24-interval
    (six hour) time constant -- long enough to describe the neighbourhood
    rather than this instant, short enough to move within a day."""
    alpha = 1.0 / 24.0
    return Memory(
        p_prev_kw=p_inv_kw,
        voltage_ewma_pu=(1.0 - alpha) * carry.voltage_ewma_pu + alpha * obs.voltage_pu,
        intervals=carry.intervals + 1,
    )


# ---------------------------------------------------------------------------
# The base controller
# ---------------------------------------------------------------------------


def self_consumption(
    obs: LocalObservation,
    carry: Memory,
    params: dict[str, chex.Array],
    key: chex.PRNGKey,
) -> tuple[chex.Array, Memory]:
    """Cover your own load, bank the rest. What every home battery does by default.

    Asking the inverter for exactly the household's own consumption drives the
    grid exchange to zero, and the inverter dispatches solar before battery, so
    the battery takes up whatever is left -- charging on surplus, discharging
    on shortfall.

    The second term is what stops that quietly curtailing. Once the battery
    cannot absorb any more, surplus generation has nowhere to go and the
    inverter throws it away rather than exporting it. A real household
    exports, so ask for the part of the surplus the battery cannot take.

    **It deliberately uses ``obs.p_load_kw``, last interval's load, and not
    ``obs.p_load_forecast_kw``, the coming one.** That is a real product, not a
    typo: a home battery servos against the net reading its meter is
    reporting *now*, so it always chases the load by one control period.
    Modelling it with the forecast would flatter the installed base and hide
    a headroom you are meant to be able to take. What the lag costs, measured
    over twenty weeks: unintended grid exchange of 0.445 kWh per
    agent-interval where the forecast leaves 0.436, self-consumption share
    29.4 % where the forecast reaches 30.3 %, and about 3 CHF a week across
    the community. Swapping in ``p_load_forecast_kw`` is the cheapest correct
    edit available -- and, as those numbers say, worth a couple of per cent
    and *nothing at all* on peak, ramp or coincidence. It is a warm-up, not a
    strategy.

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
    charging_allowed = obs.hour >= params["charge_after_hour"]

    surplus_kw = jnp.maximum(obs.pv_available_kw - obs.p_load_kw, 0.0)
    absorbable_kw = jnp.where(charging_allowed, obs.bat_charge_max_kw, 0.0)
    export_kw = jnp.minimum(jnp.maximum(surplus_kw - absorbable_kw, 0.0), params["export_cap_kw"])

    p_inv_kw = clip_to_feasible(obs.p_load_kw + export_kw, obs)
    return p_inv_kw, update_memory(carry, obs, p_inv_kw)


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
    p_inv_kw = clip_to_feasible(obs.pv_available_kw, obs)
    return p_inv_kw, update_memory(carry, obs, p_inv_kw)


def passive_controller() -> Controller:
    return Controller(
        name="passive",
        fn=passive,
        params=self_consumption_params(),
        init_carry=init_memory,
    )


__all__ = [
    "Controller",
    "ControllerFn",
    "LocalObservation",
    "Memory",
    "base_controller",
    "clip_to_feasible",
    "init_memory",
    "passive",
    "passive_controller",
    "self_consumption",
    "self_consumption_params",
    "update_memory",
]
