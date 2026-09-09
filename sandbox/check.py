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

import inspect
from typing import Optional

import jax
import jax.numpy as jnp

from sandbox.controller import Controller, base_controller, init_memory
from sandbox.evaluate import Submission, evaluate
from sandbox.metrics import compare, fairness, score
from sandbox.rollout import build_env, rollout
from sandbox.scenarios import STEPS_PER_DAY, reference_scenario
from sandbox.tariff import tariff_from_settlement


def my_controller_as_bundle() -> Controller:
    """The controller in ``my_idea`` packaged the way the harness expects.

    ``@numpy_controller`` already returns a :class:`Controller`, so the NumPy
    tier is passed straight through and ``check()`` works on it unchanged.

    Define ``INIT_CARRY`` in ``my_idea`` -- a zero-argument callable returning
    one household's starting carry -- to replace the default
    :class:`~sandbox.controller.Memory` with your own pytree. Without it the
    default is used, which is the right answer for most submissions.
    """
    from sandbox import my_idea

    if isinstance(my_idea.my_controller, Controller):
        return my_idea.my_controller

    return Controller(
        name="yours",
        fn=my_idea.my_controller,
        params={k: jnp.float32(v) for k, v in my_idea.CONTROLLER_PARAMS.items()},
        init_carry=getattr(my_idea, "INIT_CARRY", init_memory),
    )


def my_tariff_factory():
    """The settlement function in ``my_idea``, wrapped as a tariff.

    ``@numpy_tariff`` already returns the factory ``build_env`` takes -- one
    argument, the prosumer model -- where a settlement function takes three
    (``grid, carry, params``). Telling them apart by arity is what lets the
    NumPy tier work through ``check()`` like any other.
    """
    from sandbox import my_idea

    fn = my_idea.my_tariff
    if len(inspect.signature(fn).parameters) == 1:
        return fn

    return tariff_from_settlement(fn, my_idea.TARIFF_PARAMS)


def run_check(
    days: int = 1,
    detail: bool = False,
    key: Optional[jax.Array] = None,
    fast: bool = True,
) -> None:
    """Reference versus yours, on one weather. Fast and rough, for iterating.

    ``fast=False`` runs the eager path: a Python loop instead of ``scan``, so
    values are concrete, ``print`` works and a traceback points at your own
    line. Much slower, and the first thing to reach for when something breaks.

    No tuning happens here -- both controllers run at their current
    parameters. That is what makes this fast, and it is why a *tariff* cannot
    show its effect in a ``check()``: nothing in an episode reacts to a price.
    """
    population = reference_scenario()
    steps = days * STEPS_PER_DAY
    key = key if key is not None else jax.random.PRNGKey(0)

    rows = {}
    for label, controller, tariff in (
        ("reference", base_controller(), None),
        ("yours", my_controller_as_bundle(), my_tariff_factory()),
    ):
        env = build_env(population, time_limit=steps, tariff=tariff)
        trajectory = rollout(controller, population, key, steps, env=env, fast=fast)
        rows[label] = score(trajectory, population)

    print(compare(rows, detail=detail))
    print()
    print(fairness(rows))
    _verdict(rows["reference"], rows["yours"])
    print(
        f"\n  one weather, {days} day(s), no tuning -- rough. `score()` runs the real thing."
        "\n  A tariff cannot show its effect here: nothing in an episode reacts to a"
        "\n  price, so only `score()`, which re-tunes households first, can price it."
    )


def run_score(detail: bool = True, seeds: int = 20) -> None:
    """The evaluation the jury sees: a full week over many weathers."""
    from sandbox import my_idea

    population = reference_scenario()
    controller = my_controller_as_bundle()

    # A name in TUNE_OVER that the controller does not actually read would be
    # swept, change nothing, and quietly leave the submission scored at
    # whatever the tuner happened to return first. Say so instead.
    unknown = [name for name in my_idea.TUNE_OVER if name not in (controller.params or {})]
    if unknown:
        print(
            f"  WARNING: TUNE_OVER names {unknown}, which your controller's "
            "parameters do not\n  contain. Those entries are ignored. Check "
            "CONTROLLER_PARAMS in my_idea.py.\n"
        )

    submission = Submission(
        controller=controller,
        tariff=my_tariff_factory(),
        candidates=my_idea.TUNE_OVER,
    )
    evaluation = evaluate(submission, population, seeds=seeds)
    print(evaluation if detail else compare(evaluation.cells, detail=False))
    if not detail:
        print()
        print(fairness(evaluation.cells))
    _verdict(evaluation.cells["fair_leg/base"], evaluation.cells["submitted/submitted"])


#: Rows of the plain-words verdict: label, Score field, unit, the sign of
#: "better", and whether the row is *physical*.
#:
#: The two fairness rows are here rather than only in the table because they
#: are the rows a *tariff* moves. Everything above them is a feeder quantity
#: that a controller edit changes and a price alone cannot, so a tariff author
#: reading only the top of the list would conclude nothing had happened.
#:
#: They are flagged non-physical for one reason: the "nothing has moved yet"
#: notice below asks whether the *simulation* is unchanged, and the shipped
#: default tariff redistributes from the first run. Counting fairness into
#: that check would suppress a notice a first-time reader needs.
_VERDICT_ROWS = (
    ("solar exported at the worst moment", "transformer_export_peak_kw", "kW", -1, True),
    ("work squeezed into the peak interval", "peak_to_average", "", -1, True),
    ("worst sudden swing (herding)", "max_ramp_kw", "kW", -1, True),
    ("everyone acting at once", "coincidence_factor", "", -1, True),
    ("solar thrown away", "curtailed_share", "%", -1, True),
    ("what the households earned", "community_settlement_chf", "CHF", +1, True),
    ("what a tenant paid per kWh", "tenant_cost_per_kwh_chf", "CHF", -1, False),
    ("cost base left to tenants", "tenant_import_share", "%", -1, False),
)


def _verdict(reference, yours) -> None:
    """The headline numbers in plain words, with the direction that counts as better."""
    print()
    unchanged = 0
    physical = 0
    for label, field, unit, better, is_physical in _VERDICT_ROWS:
        was, now = getattr(reference, field), getattr(yours, field)
        if unit == "%":
            was, now = was * 100, now * 100
        change = now - was
        # Relative, because "unchanged" arrives as a float difference of 1e-6.
        negligible = abs(change) <= 0.005 * max(abs(was), 1e-9)
        physical += is_physical
        unchanged += negligible and is_physical
        mark = "  same" if negligible else (" better" if change * better > 0 else " worse")
        print(f"  {label:<38} {was:8.2f} -> {now:8.2f} {unit:<4}{mark}")

    if unchanged == physical:
        # The shipped defaults reproduce the reference physically, so a first
        # run prints identical rows down to the fairness pair. Saying so is
        # the difference between "the harness works" and "something is
        # broken" -- and the fairness rows, which do move, are the first
        # demonstration that a tariff is a redistribution before it is
        # anything else.
        print(
            "\n  Every physical row identical. Expected on a fresh checkout: the\n"
            "  default `my_controller` IS the reference controller, so nothing on\n"
            "  the feeder changed. The fairness rows still move, because the\n"
            "  default `my_tariff` rebates its congestion term equally over all\n"
            "  eighteen connection points -- that is a tariff redistributing\n"
            "  without changing a single kilowatt. Edit sandbox/my_idea.py and\n"
            "  run this again."
        )
