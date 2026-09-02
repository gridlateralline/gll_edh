# What can a school of fish teach us about grid incentives?

A sandbox for the [Energy Data Hackdays 2026](https://www.energydatahackdays.ch/challenges/what-can-a-school-of-fish-teach-us-about-grid-incentives-2)
challenge from **ewz**.

Eighteen households on a real low-voltage feeder. You edit **two functions** — a
grid tariff and a household controller — and everything else is fixed: the power
flow, the feasibility projection, the rollout, the scoring.

## Start here

The one prerequisite is [**uv**](https://docs.astral.sh/uv/getting-started/installation/),
which fetches the right Python and every dependency for you:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

On Windows, `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`.
Homebrew, pipx and standalone installers are all on the
[installation page](https://docs.astral.sh/uv/getting-started/installation/).

Then:

```bash
git clone https://github.com/gridlateralline/gll_edh.git
cd gll_edh
uv sync
uv run jupyter lab notebooks/00_quickstart.ipynb
```

`uv sync` builds the environment and pulls in `gll_env` — the simulator
underneath — as a git dependency that `uv.lock` pins to an exact commit, so
everyone runs identical physics however that repository moves during the event.

Now open **[`sandbox/my_idea.py`](sandbox/my_idea.py)**: two plain functions, a
household controller and a grid tariff, and the only file you need to touch.
Both ship as naive but working defaults, meant to be overwritten. Iterate from
the notebook or any Python prompt:

```python
from sandbox.my_idea import check, score

check()               # seconds: your idea vs the reference, on one day
check(fast=False)     # slower, but concrete values and a working `print`
score()               # ~2 min: a full week, twenty weathers, the whole jury
```

Edit, `check()`, repeat — and reach for `check(fast=False)` the moment
something breaks, because it drops the compilation and puts your own line in
the traceback. `score()` runs four rollouts with a tuning sweep inside each, so
give it a couple of minutes; it has not hung.

When a single function stops being enough room, the
[controller](CONTROLLER_COOKBOOK.md) and [tariff](TARIFF_COOKBOOK.md) cookbooks
are the complete reference for each seam.

**You never need to read the simulator.** Both seams are handed plain SI views
of it — `obs` for one household, `grid` for the whole feeder — so neither
function ever mentions an environment type, a per-unit conversion or a bus
index.

Everything below is context you can read later.

---

## The problem in one paragraph

Give every household a battery and let each one do the obvious thing — store
your own solar, use it later — and each household is better off on every
measure it can see. Lower peak, lower losses, more self-consumption, a smaller
bill. And the feeder gets **worse**, because every roof peaks at noon so every
battery fills at noon, and every battery therefore stops absorbing at the same
moment. No price is involved. The correlation is in the weather and the working
day.

Measured on the reference week:

| controller | export peak | ramp | coincidence | bill |
|---|---|---|---|---|
| do nothing | 64.6 kW | **6.3 kW** | 0.805 | 276 CHF |
| self-consumption | 62.9 kW | **17.1 kW** | 0.788 | 299 CHF |

The steepest swing at the transformer gets **2.7× worse**, while every number a
household can see improves.

Better for each household, worse for the system they share. That is the school
of fish: every fish using only what it can see from where it is, and the school
still turning as one.

## The two seams

| | sees | runs |
|---|---|---|
| **[tariff](sandbox/tariff.py)** | everything — the solved network, every voltage and flow | *after* the interval |
| **[controller](sandbox/controller.py)** | one household's own meter | *before*, blind |

**The controller never sees a price.** Not a delayed one — none. Real
settlement lags by days or months, past the end of an episode, so no household
can react to a price in time. Anticipation lives in the *parameters*, tuned
across episodes.

### And the local signal is thinner than it looks

Own bus voltage correlates with feeder congestion at **+0.99**, so it looks
like the obvious proxy for a nodal price. Then measure how much of it a
household did not already know: regress it on own PV, own load and the clock
and roughly **90% is already explained**. The residual — the part genuinely
about the neighbourhood — is **0.86% of nominal** on the default feeder, just
above what a Class 1 smart meter resolves. It is why the sandbox runs on
`rural`: on ewz's own `urban` network that residual is 0.20%, well *below*
meter resolution, and a controller "reading voltage" there is reading a noisy
clock. (`scripts/measure_voltage_residual.py` regenerates all of these.)

That is the real finding here, and it explains *why* herding is hard rather
than just that it happens. Every local signal is correlated across the feeder,
and the one that is genuinely about the neighbourhood carries almost nothing
new. There is nearly **no idiosyncratic local information** — households move
together because the information structure leaves them nothing to
differentiate on.

So the design space is not "read the signal better". It is to **manufacture
differentiation where the physics provides none**: memory and hysteresis in the
carry, deliberate desynchronisation via `key`, or a tariff that creates
locational distinctions the voltages do not.

**The controller never sees a neighbour, either.** It is written for one
household and `vmap`'d over the population, so there is no agent axis inside it
to index. vmap is the fairness contract, not just a speed trick.

## Four pathways

1. **Design the price** — edit `my_tariff` in [`sandbox/my_idea.py`](sandbox/my_idea.py).
   What you return is the interval's whole settlement; see
   [`TARIFF_COOKBOOK.md`](TARIFF_COOKBOOK.md) for how far that goes.
2. **Design the household** — edit `my_controller` in
   [`sandbox/my_idea.py`](sandbox/my_idea.py). See
   [`CONTROLLER_COOKBOOK.md`](CONTROLLER_COOKBOOK.md).
3. **Audit it** — `sandbox.export.to_dataframe()` gives you a tidy pandas frame. No JAX required.
4. **Show it** — same frame; the feeder has coordinates for a map.

The two design pathways are equally central, and a submission may touch either,
both, or neither and fall back to the reference. The controller pathology is
what makes the challenge exist; the tariff is what closes it.

## Writing a controller

```python
def my_controller(obs, carry, params, key):
    """One household. Returns net active power at the inverter, in kW."""
    surplus = jnp.maximum(obs.pv_available_kw - obs.load_kw, 0.0)
    export = jnp.maximum(surplus - obs.bat_charge_max_kw, 0.0)
    return clip_to_feasible(obs.load_kw + export, obs), carry
```

Return anything you like — it is clipped to `[p_min_kw, p_max_kw]` and then
projected onto the physically feasible set. **A controller cannot crash the
simulation.** Full reference: [`CONTROLLER_COOKBOOK.md`](CONTROLLER_COOKBOOK.md).

Everything is SI: **kW**, **kWh**, **CHF**, **per-unit** voltage. Every field
name carries its unit, because a silent factor of four between kW and kWh is
the easiest mistake here to make.

### Three ways to write it, all interchangeable

| tier | how | you get | cost |
|---|---|---|---|
| **eager** | `rollout(..., fast=False)` | readable tracebacks, working `print` | 14× slower |
| **numpy** | [`@numpy_controller`](sandbox/numpy_bridge.py) | real `if`, real loops, real SciPy | 1.4× slower |
| **jax** | plain `jnp` | instant seed sweeps | — |

The rollout cannot tell them apart, and neither can `check()` or `score()`.
Write it eager, keep it in NumPy if you like, and score it either way — the
results are identical, and there is a test asserting that. The same three tiers
are open to a tariff.

## Writing a tariff

```python
def my_tariff(grid, carry, params):
    """The WHOLE feeder, one interval. Returns (num_pq,) CHF, signed, and carry."""
    return grid.energy_chf - my_congestion_term(grid, params), carry
```

What you return **is** what each of the eighteen connection points pays or
earns for the interval. `grid.energy_chf` is one of the fields you are handed —
what ewz's published fair-LEG rate would have settled — and the default builds
on it, but a flat rate, a time-of-use schedule, a demand charge or a fully
nodal price written from scratch are all just different return values from the
same function.

`grid.has_inverter` is static rate-class metadata rather than a live reading,
so a tariff can say what it means directly — a tenant floor — instead of
inferring identity from behaviour. `carry` threads to the next interval,
exactly like a controller's: a demand charge's running peak, a ratchet, or a
*smoothed* congestion signal all live there.

Two things are checked for you rather than by hand: `settlement_chf` must be
shaped `(num_pq,)`, and revenue adequacy is gated empirically against what fair
LEG itself collects — a tariff that pays everybody is disqualified, one that
redistributes is not. Full reference: [`TARIFF_COOKBOOK.md`](TARIFF_COOKBOOK.md).

## Scoring

Four rollouts, not one:

|  | today's controller | your controller |
|---|---|---|
| **fair LEG** | reference floor | does yours help *today*? |
| **your tariff** | the short run: the installed base | the equilibrium: your price at its best response |

**Tailoring a controller to your tariff is the point, not a trick.** Every cell
involving a submitted tariff **re-tunes** the controller first, because without
that a tariff changes nothing physical at all — nobody can see it during an
episode, so it would only redistribute. The tuner maximises the *household's
own bill*, never the grid score: the gap between what a household wants and
what the network needs is the mechanism design problem, and closing it is what
designing a tariff means.

What the four cells separate is *when*. ewz publishes a tariff; it does not
choose anybody's controller. Households do that, in their own interest — and
if your price is any good, the controller that serves their interest is the
one you submitted. That is what a best response *is*, and it is why the
controller pathway sits beside the tariff rather than under it: your
controller is your claim about what households will end up running once your
price is in force.

So both bottom cells best-respond to your tariff. What differs is what they
are allowed to best-respond *with*. The left one tunes the control strategy
households run today; the right one tunes yours. A household cannot
best-respond into a strategy its firmware cannot express — today's batteries
ship self-consumption logic and not much else — so the left cell is the
short-run answer, what your price extracts from the installed base, and the
right cell is the equilibrium it is steering toward once controllers of the
shape you propose exist.

The gap between them is the **co-design premium**. Reported, not gated: it is
not a penalty but a statement of how far the market has to move before your
mechanism pays in full.

The jury is [`sandbox/metrics.py`](sandbox/metrics.py), fixed and not editable:

- **Network** — transformer peak both ways, reverse-flow share, losses.
- **Diversity** — the coincidence factor, the quantity distribution networks
  are actually planned against.
- **Ramp** — where herding shows up first and most violently.
- **Economics** — settlement against a do-nothing floor, and curtailment. A
  tariff that flattens the feeder by spilling a fifth of the solar has not
  solved anything.
- **Fairness** — cost per kWh, over all eighteen connection points. Six of them
  are tenants with no inverter, absent from anything indexed by agent, and
  exactly who a bad tariff harms.

One hard gate: **revenue adequacy**. A tariff that simply pays everybody
produces a delighted population and a bankrupt network operator, so it is
disqualified rather than ranked.

## Handing it in

Fork this repository, work in your fork, and open a pull request back here when
you are done. You do not need to be given access to anything — a fork and a PR
is the whole process.

```bash
gh repo fork gridlateralline/gll_edh --clone     # or use the Fork button
```

Work on a branch, commit `sandbox/my_idea.py` along with anything else your
idea needed, and open the PR:

```bash
git checkout -b our-team-name
git commit -am "our tariff and controller"
git push -u origin our-team-name
gh pr create
```

In the PR description, say what you tried and what the numbers did — paste the
`score()` output, and tell us what you expected that did not happen. A
submission that explains a negative result is worth more than one that only
shows the run that worked.

## Which feeder, and why it matters

The default is `rural` — a long feeder, end-of-line impedance 0.91 Ω, about
twice IEC 60725's reference. Voltage crosses the 1.05 pu planning trigger on
about 9% of bus-intervals — the `>1.05` column `score()` prints — and peaks at
1.10, right at the EN 50160 limit. That is a genuine standards violation rather
than a modelling artefact, and it is what gives a household something local to
read.

`FEEDER_STRENGTHS` also ships **`urban`** — ewz's own network, unmodified.
There over-voltage simply never happens, because a dense meshed feeder is stiff
and never leaves 1.02. But meshing buys voltage stiffness and no thermal
capacity, so what binds instead is throughput: the feeder runs backwards for
47% of the week, its export peak is several times its largest draw, and the
diversity the network was planned under is gone. Same population, same jury,
different binding constraint — and which one bites where is worth a submission
on its own.

Turning the grid code off instead of weakening the feeder was measured and
does almost nothing: Q(U) only acts outside its deadband, and on a stiff
feeder voltage never gets there.

## The feeder

CIGRE low-voltage, 19 buses, 18 connection points, 15-minute intervals, seven
days. Twelve households have an inverter and are agents; six are tenants who
cannot respond to anything.

| type | n | roof | battery | inverter |
|---|---|---|---|---|
| tenant | 6 | — | — | — |
| pv_only | 5 | 9 kWp | — | 7 kVA |
| pv_battery | 5 | 12 kWp | 13 kWh | 10 kVA |
| large_flex | 2 | 15 kWp | 20 kWh | 13 kVA |

Inverters are deliberately smaller than the roof — DC/AC ≈ 1.2, which is what
real installations use and which produces about 2% clipping. It is also
load-bearing: sizing inverters to the array was measured to collapse control
authority to zero, because a household that can export everything never needs
its battery.

A full week runs in about **one second**; a 20-seed ensemble in **eight**.

## Layout

```
sandbox/
├── my_idea.py       ← START HERE (both pathways, plain functions)
├── check.py           what check() and score() run      [do not edit]
├── tariff.py          the tariff seam underneath      [pathway 1, advanced]
├── controller.py      the controller seam underneath  [pathway 2, advanced]
├── numpy_bridge.py    write your controller or tariff in NumPy instead
├── scenarios.py       who lives on the feeder
├── observation.py     what one household can measure, and what the grid can
├── rollout.py         the loop           [do not edit]
├── tuning.py          household best response
├── metrics.py         the jury           [do not edit]
├── evaluate.py        the four cells     [do not edit]
└── export.py          tidy frames, no JAX needed

scripts/
└── measure_voltage_residual.py   regenerates the residual table above
```

`tariff.py` and `controller.py` are where `my_idea.py`'s two functions actually
get wired in, and where the escape hatches live for anyone who outgrows a plain
function — a stateful tariff (`MyTariff`), a `@numpy_controller`. Most
submissions never need to open either.

The repo shares `gll_env`'s pre-commit setup (ruff, `ty`, whitespace, licence
headers, conventional commits). It is not required to participate:

```bash
uv run pre-commit install    # optional
```

## Licence

Apache 2.0.
