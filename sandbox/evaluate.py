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

"""Scoring a submission: four rollouts, not one.

ewz publishes a tariff; it does not choose anybody's controller. Households
do that, in their own interest -- so if the price is any good, the controller
that serves their interest is the one the submission proposes. A submitted
controller is a claim about the follower's best response, not a product to be
installed.

Both cells under a submitted tariff therefore best-respond to it. They differ
in what they may best-respond *with*: the strategy households run today, or
the strategy the submission proposes. A household cannot best-respond into a
strategy its firmware cannot express, which makes the first the short-run
answer and the second the equilibrium the price is steering toward.

The response is bounded by design. :func:`~sandbox.tuning.tune` searches the
parameters in ``Submission.candidates`` within the controller of the cell it
is scoring, not every controller anyone could write. An unconstrained
follower would leave the two cells sharing no basis for comparison, and the
premium nothing to measure.

Four cells, at the cost of one extra pair of rollouts:

===================== ===================== ==========================
                      today's controller    your controller
===================== ===================== ==========================
**fair LEG**          reference floor       does yours help *today*?
**your tariff**       the short run: the    the equilibrium: your
                      installed base        price at its best response
===================== ===================== ==========================

The gap between the two bottom cells is the **co-design premium**: how much of
the result waits on controllers of the proposed shape existing at all.
Reported, not gated -- it measures how far the market has to move before the
mechanism pays in full, which is a judgement for the jury rather than a
threshold.

Every cell involving a submitted tariff re-tunes its controller first, the
bottom-left one included. That is not a control to be corrected for: the
household's best response *is* the follower move the Stackelberg game is built
on. Without it a tariff changes nothing physical whatsoever, because nobody
can see a price during an episode, and the row would measure redistribution
alone.
"""

from dataclasses import dataclass
from typing import Any, Optional

import jax
import numpy as np

from sandbox.controller import Controller, base_controller
from sandbox.metrics import Score, compare, revenue_adequate, score
from sandbox.rollout import TariffFactory, build_env, rollout
from sandbox.scenarios import EPISODE_STEPS, Population
from sandbox.tuning import tune

#: Episodes every cell is scored over. One week of weather decides nothing;
#: the ensemble is what stops a submission winning on luck. Cheap, because a
#: rollout is a pure function of its key.
SCORING_SEEDS = 20


@dataclass(frozen=True)
class Submission:
    """What a team hands in. Either half may be left at the reference.

    Attributes:
        controller: The household controller, or ``None`` to use the base.
        tariff: The tariff, or ``None`` to use fair LEG.
        candidates: Parameter values to tune over before scoring under a
            submitted tariff. Names must exist in the controller's own
            parameters; anything not swept keeps its current value. Required
            if `tariff` is given, since otherwise the tariff has no way to
            reach anybody at all.
    """

    controller: Optional[Controller] = None
    tariff: Optional[TariffFactory] = None
    candidates: Optional[dict[str, Any]] = None


@dataclass(frozen=True)
class Evaluation:
    """The four cells, plus what they imply.

    Attributes:
        cells: The scored grid, keyed ``tariff/controller``.
        revenue_adequate: Whether the submitted tariff redistributes rather
            than subsidises -- see :attr:`revenue_check`.
        revenue_check: The submitted tariff scored against the base controller
            at its *untuned* parameters. Not part of the displayed grid: it
            exists solely to isolate the tariff's own effect on revenue from
            the effect of households changing what they do. A tariff that
            makes households export less collects less, and that is the
            tariff working, not printing money.
        co_design_premium: How much of the improvement depends on the
            households running precisely the submitted controller.
    """

    cells: dict[str, Score]
    revenue_adequate: bool
    revenue_check: Optional[Score]
    co_design_premium: dict[str, float]

    def __str__(self) -> str:
        gate = "PASS" if self.revenue_adequate else "FAIL"
        if self.revenue_check is not None:
            gate += (
                f" (tariff collects {self.revenue_check.community_settlement_chf:.0f} CHF "
                f"at fixed behaviour vs {self.cells['fair_leg/base'].community_settlement_chf:.0f})"
            )
        premium = "  ".join(
            f"{name} {value:+.1%}" for name, value in self.co_design_premium.items()
        )
        return (
            f"{compare(self.cells)}\n\n"
            f"revenue adequacy: {gate}\n"
            f"co-design premium: {premium or 'n/a (no tariff submitted)'}"
        )


def _mean_score(
    controller: Controller,
    population: Population,
    tariff: Optional[TariffFactory],
    n_steps: int,
    seeds: int,
    key: jax.Array,
) -> Score:
    """Score across an ensemble, averaging the metrics rather than the episodes.

    Averaging trajectories first would be wrong: a peak is not linear, and the
    mean of two weeks' flows has a lower peak than either week.
    """
    env = build_env(population, time_limit=n_steps, tariff=tariff)
    scores = [
        score(rollout(controller, population, seed, n_steps=n_steps, env=env), population)
        for seed in jax.random.split(key, seeds)
    ]
    fields = scores[0].to_dict()
    return Score(**{name: float(np.mean([getattr(s, name) for s in scores])) for name in fields})


def evaluate(
    submission: Submission,
    population: Population,
    n_steps: int = EPISODE_STEPS,
    seeds: int = SCORING_SEEDS,
    key: Optional[jax.Array] = None,
) -> Evaluation:
    """Score a submission over the full four cells."""
    key = key if key is not None else jax.random.PRNGKey(0)
    base = base_controller()
    submitted = submission.controller or base

    def controller_for(tariff: Optional[TariffFactory], controller: Controller) -> Controller:
        """Re-tune before scoring under a submitted tariff.

        Skipped for fair LEG, whose reference controller is by definition
        already the one households run today.
        """
        if tariff is None or not submission.candidates or not controller.params:
            return controller
        params, _ = tune(
            controller,
            population,
            submission.candidates,
            tariff=tariff,
            n_steps=n_steps,
        )
        return controller.replace(params={**controller.params, **params})

    cells: dict[str, Score] = {}
    for tariff_name, tariff in (("fair_leg", None), ("submitted", submission.tariff)):
        if tariff is None and tariff_name == "submitted":
            continue
        for controller_name, controller in (("base", base), ("submitted", submitted)):
            if controller is base and controller_name == "submitted":
                continue
            cells[f"{tariff_name}/{controller_name}"] = _mean_score(
                controller_for(tariff, controller),
                population,
                tariff,
                n_steps,
                seeds,
                key,
            )

    reference = cells["fair_leg/base"]

    # Gate the TARIFF with behaviour held fixed: the base controller at its
    # default parameters, which is exactly the behaviour behind the reference
    # cell. Every other cell has re-tuned, so a revenue difference there
    # reflects households changing their minds -- which is the tariff working,
    # not the tariff printing money. Comparing those would fail every tariff
    # that succeeded.
    check: Optional[Score] = None
    adequate = True
    if submission.tariff is not None:
        check = _mean_score(base, population, submission.tariff, n_steps, seeds, key)
        adequate = revenue_adequate(check, reference)

    premium: dict[str, float] = {}
    if "submitted/submitted" in cells and "submitted/base" in cells:
        combo = cells["submitted/submitted"]
        naive = cells["submitted/base"]
        for field in ("transformer_export_peak_kw", "max_ramp_kw", "coincidence_factor"):
            baseline = getattr(reference, field)
            alone = baseline - getattr(naive, field)
            together = baseline - getattr(combo, field)
            # Only meaningful when the tariff helped on this metric at all and
            # the combination helped more. When they pull in opposite
            # directions the ratio is a large number that means nothing, so
            # the field is omitted rather than reported as a spurious figure.
            if together > 1e-6 and alone >= 0.0:
                premium[field] = 1.0 - alone / together

    return Evaluation(
        cells=cells,
        revenue_adequate=adequate,
        revenue_check=check,
        co_design_premium=premium,
    )
