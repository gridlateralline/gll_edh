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

"""Regenerate the residual table in :mod:`sandbox.observation`.

    uv run python scripts/measure_voltage_residual.py

The claim being measured: own bus voltage tracks feeder congestion almost
perfectly, but a household already knows most of it from its own PV, its own
load and the clock. What is left over -- the part genuinely about the
neighbourhood -- is what a controller could actually learn from reading
voltage, and on a stiff feeder it is smaller than a real meter can resolve.

The regression is deliberately the simplest reading of that sentence: per
household, ordinary least squares of own ``voltage_pu`` on own
``pv_available_kw``, own ``load_kw`` and the clock (``time_sin``,
``time_cos``), with an intercept. Residuals are pooled across households and
reported as a percentage of nominal voltage, next to the ~0.5 % of nominal a
Class 1 meter resolves.

This script exists because the numbers it produces are quoted in
``sandbox/observation.py``, ``sandbox/scenarios.py``, the README and
``CONTROLLER_COOKBOOK.md``, and they went stale once before: the figures
predating the persistent-weather change were roughly a third narrower than
the model now produces, and two files disagreed about the rural residual for
some time before anyone noticed. Re-run this whenever the weather model, the
population or the feeder set changes, and update those four places together.

It runs the eager rollout path, because a household's own load is a field of
``LocalObservation`` and never reaches ``Trajectory``. That takes a few
minutes for the full table.
"""

import jax
import jax.numpy as jnp
import numpy as np

from sandbox.controller import base_controller
from sandbox.observation import to_local
from sandbox.rollout import build_env, to_action
from sandbox.scenarios import EPISODE_STEPS, FEEDER_STRENGTHS, reference_scenario

#: What a Class 1 meter resolves, as a percentage of nominal. A residual below
#: this is not a signal a real household could act on.
METER_RESOLUTION_PCT = 0.5

#: Which fields a household is credited with already knowing.
REGRESSORS = ("pv_available_kw", "load_kw", "time_sin", "time_cos")


def collect(population, impedance_scale, key, n_steps):
    """Every household's own observation, interval by interval, as (T, num_agents)."""
    env = build_env(population, time_limit=n_steps, impedance_scale=impedance_scale)
    model = env.environment
    controller = base_controller()
    decide = jax.vmap(controller.fn, in_axes=(0, 0, None, 0))

    reset_key, loop_key = jax.random.split(key)
    state, timestep = env.reset(reset_key)
    carry = jax.tree_util.tree_map(
        lambda leaf: jnp.broadcast_to(leaf, (model.num_agents, *jnp.shape(leaf))),
        controller.init_carry(),
    )

    rows = []
    for _ in range(n_steps):
        loop_key, decide_key = jax.random.split(loop_key)
        local = to_local(model, timestep.observation, state)
        rows.append({name: np.asarray(value) for name, value in local.as_dict().items()})
        keys = jax.random.split(decide_key, model.num_agents)
        p_set_kw, carry = decide(local, carry, controller.params, keys)
        state, timestep = env.step(state, to_action(model, p_set_kw))

    return {name: np.stack([row[name] for row in rows]) for name in rows[0]}


def residual_pct(observations):
    """Fraction of own voltage explained, and what is left, per cent of nominal."""
    voltage = observations["voltage_pu"]
    features = np.stack([observations[name] for name in REGRESSORS], axis=-1)

    residuals, centred = [], []
    for agent in range(voltage.shape[1]):
        y = voltage[:, agent]
        design = np.column_stack([np.ones(len(y)), features[:, agent, :]])
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        residuals.append(y - design @ beta)
        centred.append(y - y.mean())

    residuals = np.concatenate(residuals)
    centred = np.concatenate(centred)
    explained = 1.0 - (residuals @ residuals) / (centred @ centred)
    span = float(voltage.max() - voltage.min()) * 100.0
    return explained * 100.0, residuals.std() * 100.0, span


def main() -> None:
    population = reference_scenario()
    key = jax.random.PRNGKey(0)

    print(f"{'feeder':<10} {'span':>8} {'explained':>11} {'residual':>10}   vs meter")
    print("-" * 58)
    for feeder, scale in FEEDER_STRENGTHS.items():
        observations = collect(population, scale, key, EPISODE_STEPS)
        explained, residual, span = residual_pct(observations)
        verdict = "above" if residual > METER_RESOLUTION_PCT else "BELOW"
        print(
            f"{feeder:<10} {span:7.2f}% {explained:10.1f}% {residual:9.2f}%   "
            f"{verdict} the {METER_RESOLUTION_PCT}% a Class 1 meter resolves"
        )


if __name__ == "__main__":
    main()
