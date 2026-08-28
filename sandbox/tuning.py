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

"""How a tariff actually reaches a household: by being tuned against.

No household sees a price during an episode -- real settlement lags past the
end of one. So a tariff changes nothing about behaviour on its own, and
scoring a submitted tariff against a *fixed* controller measures redistribution
and nothing else: identical peaks, identical ramps, identical everything
physical. Verified, not assumed; it is what the harness does if you skip this
step.

What a tariff does is change what a household would *want* to have done, which
shows up when the household re-tunes. So the tariff pathway is scored by
re-tuning the base controller against the submitted tariff first. That makes
the evaluation a Stackelberg game -- the network operator moves, households
best-respond -- which is the correct frame for mechanism design and also the
only one in which a tariff can be said to have worked.

Two rules
---------
**The tuner maximises the household's own bill, never the grid score.** A tuner
that optimises network welfare assumes benevolent households and measures what
a central planner could achieve rather than what a price can induce. The gap
between what a household wants and what the network needs *is* the mechanism
design problem; closing it is what designing a tariff means.

**Parameters are shared across the population.** Heterogeneity lives in the
state -- a tenant has ``p_min_kw == p_max_kw == 0``, so the same parameters
produce no action from them -- which keeps this one optimisation rather than a
twelve-player game with a fixed point to chase.
"""

from itertools import product
from typing import Any, Optional

import chex
import jax
import jax.numpy as jnp
import numpy as np

from sandbox.controller import Controller
from sandbox.rollout import TariffFactory, Trajectory, build_env, rollout
from sandbox.scenarios import EPISODE_STEPS, Population

#: Seeds for tuning. Fewer than scoring uses: a tuner only needs to rank
#: candidates, and the ranking is stable long before the estimate is precise.
TUNING_SEEDS = 4


def household_return(trajectory: Trajectory) -> chex.Array:
    """Mean per-agent settlement over the episode, in CHF.

    What a household is trying to maximise, and deliberately nothing else. It
    knows nothing of voltages, peaks or its neighbours' bills.
    """
    return trajectory.reward_chf.sum(axis=0).mean()


def parameter_grid(candidates: dict[str, Any]) -> list[dict[str, Any]]:
    """Cartesian product of per-parameter candidate lists.

    Grid search rather than anything cleverer, on purpose: it is explicable to
    a room, reproducible exactly, and trivially parallel under ``vmap``. Swap
    in CEM if a submission needs a bigger space.
    """
    names = list(candidates)
    combinations = product(*(candidates[name] for name in names))
    return [dict(zip(names, values, strict=True)) for values in combinations]


def _stack(entries: list[dict[str, Any]]) -> dict[str, chex.Array]:
    return {
        name: jnp.stack([jnp.asarray(entry[name], dtype=jnp.float32) for entry in entries])
        for name in entries[0]
    }


def tune(
    controller: Controller,
    population: Population,
    candidates: dict[str, Any],
    tariff: Optional[TariffFactory] = None,
    n_steps: int = EPISODE_STEPS,
    seeds: int = TUNING_SEEDS,
    key: Optional[chex.PRNGKey] = None,
) -> tuple[dict[str, Any], np.ndarray]:
    """Best response: the parameters a household would choose under `tariff`.

    Args:
        controller: The controller whose parameters are being tuned.
        population: Who lives on the feeder.
        candidates: Parameter name to the values to try. Every combination is
            evaluated.
        tariff: The tariff to best-respond to. ``None`` means the population's
            own configured tariff, i.e. fair LEG.
        n_steps: Episode length.
        seeds: Episodes per candidate. A candidate that only wins on one
            week's weather has not won.
        key: Seeds the seeds. Fixed by default, so a tuning run is repeatable.

    Returns:
        The winning parameters, and the mean household return of every
        candidate in the order :func:`parameter_grid` produced them.
    """
    entries = parameter_grid(candidates)
    if not entries:
        raise ValueError("no candidate parameters to tune over")

    env = build_env(population, time_limit=n_steps, tariff=tariff)
    keys = jax.random.split(key if key is not None else jax.random.PRNGKey(0), seeds)

    def one(params: dict[str, chex.Array], seed: chex.PRNGKey) -> chex.Array:
        # Merged, not substituted: a sweep over two of a controller's three
        # parameters must leave the third alone rather than delete it.
        return household_return(
            rollout(
                controller,
                population,
                seed,
                n_steps=n_steps,
                params={**controller.params, **params},
                env=env,
            )
        )

    # Candidates on the outer axis, seeds on the inner: one compile, every
    # candidate evaluated on identical weather.
    returns = jax.vmap(jax.vmap(one, in_axes=(None, 0)), in_axes=(0, None))(_stack(entries), keys)
    mean_return = np.asarray(returns.mean(axis=1))
    return entries[int(mean_return.argmax())], mean_return


def best_response(
    controller: Controller,
    population: Population,
    candidates: dict[str, Any],
    tariff: Optional[TariffFactory] = None,
    **kwargs: Any,
) -> Controller:
    """The controller a household would run under `tariff`, ready to score."""
    params, _ = tune(controller, population, candidates, tariff=tariff, **kwargs)
    return controller.replace(
        name=f"{controller.name}@{'my_tariff' if tariff else 'fair_leg'}",
        params={**controller.params, **params},
    )
