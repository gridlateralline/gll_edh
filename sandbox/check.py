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

"""The feedback loop: edit, `check()`, repeat.

Wires whatever is currently in :mod:`sandbox.my_idea` into the harness and
prints what changed, so nobody has to assemble a Controller, a Submission and
a tuning grid before seeing a number.
"""

from typing import Optional

import jax
import jax.numpy as jnp

from sandbox.controller import Controller, base_controller, init_memory
from sandbox.evaluate import Submission, evaluate
from sandbox.metrics import compare, score
from sandbox.rollout import build_env, rollout
from sandbox.scenarios import STEPS_PER_DAY, reference_scenario
from sandbox.tariff import tariff_from_settlement


def my_controller_as_bundle() -> Controller:
    """The function in ``my_idea`` packaged the way the harness expects."""
    from sandbox import my_idea

    return Controller(
        name="yours",
        fn=my_idea.my_controller,
        params={k: jnp.float32(v) for k, v in my_idea.CONTROLLER_PARAMS.items()},
        init_carry=init_memory,
    )


def my_tariff_factory():
    """The settlement function in ``my_idea``, wrapped as a tariff."""
    from sandbox import my_idea

    return tariff_from_settlement(my_idea.my_tariff, my_idea.TARIFF_PARAMS)


def run_check(days: int = 1, detail: bool = False, key: Optional[jax.Array] = None) -> None:
    """Reference versus yours, on one weather. Fast and rough, for iterating."""
    population = reference_scenario()
    steps = days * STEPS_PER_DAY
    key = key if key is not None else jax.random.PRNGKey(0)

    rows = {}
    for label, controller, tariff in (
        ("reference", base_controller(), None),
        ("yours", my_controller_as_bundle(), my_tariff_factory()),
    ):
        env = build_env(population, time_limit=steps, tariff=tariff)
        rows[label] = score(rollout(controller, population, key, steps, env=env), population)

    print(compare(rows, detail=detail))
    _verdict(rows["reference"], rows["yours"])
    print("\n  one weather, {} day(s) -- rough. `score()` runs the real thing.".format(days))


def run_score(detail: bool = True, seeds: int = 20) -> None:
    """The evaluation the jury sees: a full week over many weathers."""
    from sandbox import my_idea

    population = reference_scenario()
    submission = Submission(
        controller=my_controller_as_bundle(),
        tariff=my_tariff_factory(),
        candidates=my_idea.TUNE_OVER,
    )
    evaluation = evaluate(submission, population, seeds=seeds)
    print(evaluation if detail else compare(evaluation.cells, detail=False))
    _verdict(evaluation.cells["fair_leg/base"], evaluation.cells["submitted/submitted"])


def _verdict(reference, yours) -> None:
    """Four numbers, in plain words, with the direction that counts as better."""
    print()
    lines = [
        ("solar exported at the worst moment", "transformer_export_peak_kw", "kW", -1),
        ("worst sudden swing (herding)", "max_ramp_kw", "kW", -1),
        ("everyone acting at once", "coincidence_factor", "", -1),
        ("solar thrown away", "curtailed_share", "%", -1),
        ("what the households earned", "community_settlement_chf", "CHF", +1),
    ]
    for label, field, unit, better in lines:
        was, now = getattr(reference, field), getattr(yours, field)
        if unit == "%":
            was, now = was * 100, now * 100
        change = now - was
        # Relative, because "unchanged" arrives as a float difference of 1e-6.
        negligible = abs(change) <= 0.005 * max(abs(was), 1e-9)
        mark = "  same" if negligible else (" better" if change * better > 0 else " worse")
        print(f"  {label:<36} {was:8.2f} -> {now:8.2f} {unit:<4}{mark}")
