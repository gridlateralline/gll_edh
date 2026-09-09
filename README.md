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

check()               # ~10 s: your idea vs the reference, on one day
check(fast=False)     # ~80 s: concrete values and a working `print`
score()               # ~2 min: a full week, twenty weathers, the whole jury
```

Edit, `check()`, repeat — and reach for `check(fast=False)` the moment
something breaks, because it drops the compilation and puts your own line in
the traceback.

**On a fresh checkout `check()` prints five identical rows, and that is
correct.** The shipped `my_controller` *is* the reference controller and the
shipped `my_tariff` only redistributes, so nothing has moved yet. It says so.

**And `check()` cannot show a tariff working.** It does not tune, and no
household sees a price during an episode, so a tariff-only edit changes the
settlement and nothing physical. That is what a tariff *is* here — see
"Scoring" below. Use `score()` to judge one.

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
measure it can see. Lower peak, more self-consumption, a bigger cheque. And
the feeder gets **worse**, because every roof peaks at noon so every battery
fills at noon, and every battery therefore stops absorbing at the same moment.
No price is involved. The correlation is in the weather and the working day.

One week, seed 0 — what cell 4 of the quickstart prints:

| controller | export peak | ramp | coincidence | community CHF |
|---|---|---|---|---|
| do nothing | 69.4 kW | **6.9 kW** | 0.805 | 299 |
| self-consumption | 66.8 kW | **16.1 kW** | 0.773 | 332 |

The steepest swing at the transformer gets **2.3× worse**, while every number a
household can see improves.

Better for each household, worse for the system they share. That is the school
of fish: every fish using only what it can see from where it is, and the school
still turning as one.

## The three quantities

The single most useful thing to fix in your head before writing anything:

```
p_inv_kw  (you choose)  -  p_load_kw  (you don't)  =  p_grid_kw  (you're billed on)
```

A controller sets the **inverter**'s active power. The household's own load
sits behind the same meter and nobody controls it, so what the feeder carries —
and what every tariff settles, as `grid.e_grid_kwh` — is the difference.
Returning `0.0` idles the inverter and imports the whole load; returning
`obs.p_load_forecast_kw` is what drives the grid exchange to zero.

Solar and battery both sit behind that one inverter, and it serves solar
first. Ask for more than the roof is making and the battery discharges; ask
for less and the surplus charges it. You never address the battery directly.

Everything is SI: **kW**, **kWh**, **CHF**, **per-unit** voltage. Every field
name carries its unit, because a silent factor of four between kW and kWh is
the easiest mistake here to make. The suffix also carries the physics: `_kw`
is **active** power, `_kvar` **reactive**, `_kva` **apparent**, with a
`p_`/`q_` prefix wherever both halves exist at the same terminal
(`p_grid_kw` beside `q_grid_kvar`). Fields with no reactive half — a roof, a
battery — carry no prefix.

**A controller chooses active power only, and that is the law rather than a
simplification.** On a Swiss LV connection the Q(U) grid code (NE7 §4.3.2)
sets the inverter's reactive power from the voltage at its own bus, so the
action space is one-dimensional. A household feels reactive power only as
lost headroom: `p_inv_min_kw` / `p_inv_max_kw` are the active slice left once
Q(U) has taken its share. A *tariff* does see reactive flow — `q_grid_kvar`
per connection point, `transformer_kvar` at the substation — because a network
operator measures it. Pricing it is allowed and has a trap in it; see
[`TARIFF_COOKBOOK.md`](TARIFF_COOKBOOK.md).

## The two seams

| | sees | runs |
|---|---|---|
| **[tariff](sandbox/tariff.py)** | everything — the solved network, every voltage and flow | *after* the interval |
| **[controller](sandbox/controller.py)** | one household's own meter | *before*, blind |

**The controller never sees a price.** Not a delayed one — none. Real
settlement lags by days or months, past the end of an episode, so no household
can react to a price in time. Anticipation lives in the *parameters*, tuned
across episodes.

**The controller never sees a neighbour, either.** It is written for one
household and `vmap`'d over the population, so there is no agent axis inside it
to index. vmap is the fairness contract, not just a speed trick.

### The local signal is thin, and it is real

The project is called *grid lateral line* for a reason. A fish reads the water
it is in through a line of local pressure sensors; a household's own bus
voltage is the closest thing it has to the same organ — local, free,
continuously available, and genuinely coupled to what the neighbours are
doing. It correlates with feeder congestion at **+0.99**.

The caveat is not that correlation. It is **redundancy**: regress own voltage
on own PV, own load and the clock and about **86 %** is already implied by
things the household knew anyway. The residual — the part genuinely about the
neighbourhood — is **1.01 % of nominal** on this feeder, twice the ~0.5 % a
Class 1 smart meter resolves.

So reading voltage is a **legitimate design**, and the sandbox runs on the
feeder where it is legitimate. What fails is the naive version: a threshold on
the raw level mostly fires on "it is noon and my roof is working", which every
household on the feeder learns at the same instant. Subtract what you already
know — or watch the trend rather than the level — and what is left is the
neighbourhood.

And the residual is **not a fixed budget**. It is small partly *because* every
household currently runs the same rule off the same weather. A population that
deliberately differentiates — staggered starts, hysteresis, randomised timing
off `key` — makes its members' voltages less predictable from their own state,
which puts information back into the residual. Reading the signal and creating
something worth reading are the same project.

That is the finding this sandbox has to offer, and it explains *why* herding is
hard rather than just that it happens. Every local signal is correlated across
the feeder, and the one genuinely about the neighbourhood is thin. There is
very little **idiosyncratic local information** — households move together
because the information structure gives them little to differentiate on.

So the design space has two halves, and good submissions use both: **read what
is there**, and **manufacture differentiation where the physics provides
none** — through the carry, through `key`, or through a tariff that creates
locational distinctions the voltages do not.
(`scripts/measure_voltage_residual.py` regenerates all of these numbers.)

## Four pathways

1. **Design the price** — edit `my_tariff` in [`sandbox/my_idea.py`](sandbox/my_idea.py).
   What you return is the interval's whole settlement; see
   [`TARIFF_COOKBOOK.md`](TARIFF_COOKBOOK.md) for how far that goes.
2. **Design the household** — edit `my_controller` in
   [`sandbox/my_idea.py`](sandbox/my_idea.py). See
   [`CONTROLLER_COOKBOOK.md`](CONTROLLER_COOKBOOK.md).
3. **Audit it** — `sandbox.export.to_dataframe()` gives you a tidy pandas frame,
   all eighteen connection points, tenants included. No JAX required.
4. **Show it** — the same frame carries `distance_rank`, so a plot can be
   ordered by electrical distance from the transformer. (There are no map
   coordinates: the grid asset's `position` field is pandapower plotting
   geodata, incomplete, and unused here.)

The two design pathways are equally central, and a submission may touch either,
both, or neither and fall back to the reference. The controller pathology is
what makes the challenge exist; the tariff is what closes it.

## Writing a controller

```python
def my_controller(obs, carry, params, key):
    """One household. Returns the INVERTER's active power, in kW."""
    surplus = jnp.maximum(obs.pv_available_kw - obs.p_load_forecast_kw, 0.0)
    export = jnp.maximum(surplus - obs.bat_charge_max_kw, 0.0)
    p_inv_kw = clip_to_feasible(obs.p_load_forecast_kw + export, obs)
    return p_inv_kw, update_memory(carry, obs, p_inv_kw)
```

Return anything you like — it is clipped to `[p_inv_min_kw, p_inv_max_kw]` and
then projected onto the physically feasible set. **A controller cannot crash
the simulation.** Full reference: [`CONTROLLER_COOKBOOK.md`](CONTROLLER_COOKBOOK.md).

### Three ways to write it — and why you want the JAX one

| tier | how | you get | one week | a tuning sweep | a `score()` |
|---|---|---|---|---|---|
| **jax** | plain `jnp` | the fast path | 2.0 s | 17 s | ~2 min |
| **numpy** | [`@numpy_controller`](sandbox/numpy_bridge.py) | real `if`, loops, SciPy | 6.9 s | 128 s | ~20 min |
| **eager** | `rollout(..., fast=False)` | tracebacks, working `print` | ~100× slower | — | — |

The rollout cannot tell them apart, and neither can `check()` or `score()` —
the results are identical, and there is a test asserting it. **But write the
finished thing in `jnp` if you possibly can.** `score()` runs four cells with
a tuning sweep inside each; under JAX that sweep is one compile `vmap`'d
across every candidate and seed at once, while the NumPy tier pays a host
round trip per household per interval and cannot batch at all. Two minutes
against twenty is the difference between ten experiments and one. Prototype
in NumPy if it helps, then port. The same three tiers are open to a tariff.

## Writing a tariff

```python
def my_tariff(grid, carry, params):
    """The WHOLE feeder, one interval. Returns (num_pq,) CHF, signed, and carry."""
    return grid.fair_leg_chf - my_congestion_term(grid, params), carry
```

What you return **is** what each of the eighteen connection points pays or
earns for the interval. `grid.fair_leg_chf` is one of the fields you are
handed — the finished CHF settlement the fair-LEG baseline would produce, not
a rate — and the default builds on it, but a flat rate, a time-of-use
schedule, a demand charge or a fully nodal price written from scratch are all
just different return values from the same function.

`grid.has_inverter` is static rate-class metadata rather than a live reading,
so a tariff can say what it means directly — a tenant floor — instead of
inferring identity from behaviour. `carry` threads to the next interval,
exactly like a controller's: a demand charge's running peak, a ratchet, or a
*smoothed* congestion signal all live there.

Two things are checked for you rather than by hand: `settlement_chf` must be
shaped `(num_pq,)`, and revenue adequacy is gated empirically against what fair
LEG itself collects — a tariff that pays everybody is disqualified, one that
redistributes is not. Full reference: [`TARIFF_COOKBOOK.md`](TARIFF_COOKBOOK.md).

### What fair LEG is, and is not

**Fair LEG is not a product ewz sells.** It is assembled from ewz's real,
published 2026 rate components — the EEA feed-in tariff, ewz.natur energy,
grid usage, public duties — with one construction on top: trading inside a
local electricity community waives 40 % of the grid usage fee, and fair LEG
splits that saving **evenly** between injector and consumer. Symmetric by
construction, hence the name.

ewz's actual LEG product is **Solarquartier**. It grants the consumer the
whole rebate but charges a flat 13 Rp./kWh LEG energy rate, which leaves a
pure consumer about **5.7 Rp./kWh worse off** off-peak inside the community
than outside it — the producer captures nearly all of the saving. That would
be the easier baseline and the wrong one: beating a tariff that penalises
consumers says nothing about the network. So the status quo modelled here is
the *fair* version, already balanced between the two sides, and beating it has
to mean saying something about congestion, diversity and ramp.

## Scoring

Four rollouts, not one:

|  | today's controller | your controller |
|---|---|---|
| **fair LEG** | reference floor | does yours help *today*? |
| **your tariff** | the short run: the installed base | the equilibrium: your price at its best response |

**Tailoring a controller to your tariff is the point, not a trick.** Every cell
involving a submitted tariff **re-tunes** the controller first, because without
that a tariff changes nothing physical at all — nobody can see it during an
episode, so it would only redistribute. Each controller is tuned over its own
parameter grid: the installed base over `TUNING_GRID` in
[`sandbox/controller.py`](sandbox/controller.py), yours over `TUNE_OVER` in
`my_idea.py`. The tuner maximises the *household's own bill*, never the grid
score: the gap between what a household wants and what the network needs is
the mechanism design problem, and closing it is what designing a tariff means.

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
mechanism pays in full. It can come out **negative** on a metric, and that is
informative rather than an error — it means today's controller happens to
serve that metric better under your price than the one you propose, which is
usually a sign your controller is trading that metric for another.

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
disqualified rather than ranked. It is checked with behaviour held fixed — a
tariff that makes households export less collects less, and that is the tariff
working, not printing money.

### Reading the output

`score()` prints one row per cell, keyed `tariff/controller`:

| column | meaning | better |
|---|---|---|
| `export_pk_kW` / `draw_pk_kW` | transformer peak, exporting / drawing | lower |
| `reverse` | share of the week the feeder runs backwards | lower |
| `coincid` | coincidence factor — everyone acting at once | lower |
| `ramp_kW` | steepest interval-to-interval swing at the transformer | lower |
| `corr` | mean pairwise correlation of household power changes | lower |
| `loss` | network losses as a share of energy served | lower |
| `curtail` | generation thrown away | lower |
| `selfcons` | generation used behind the meter | higher |
| `CHF` | what the community earned, summed over all 18 points | higher |
| `>1.05` | share of bus-intervals above the planning trigger | lower |

Then `revenue adequacy: PASS/FAIL` and the co-design premium, then five of
those columns restated in plain words.

**Read the whole row.** The shipped default tariff halves the export peak and
improves the ramp — by making the tuned household cap its exports and curtail
a quarter of the week's generation, which collapses the community's earnings.
`curtail` and `CHF` are in the jury precisely so that cannot pass quietly.

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

## The feeder

CIGRE low-voltage, 19 buses, 18 connection points, 15-minute intervals, seven
days. Twelve households have an inverter and are agents; six are tenants who
cannot respond to anything.

| type | n | bus idx | roof | battery | inverter |
|---|---|---|---|---|---|
| tenant | 6 | 1–6 | — | — | — |
| pv_only | 2 | 7, 11 | 9 kWp | — | 7 kVA |
| pv_battery | 6 | 8, 9, 10, 12, 16, 17 | 12 kWp | 13 kWh | 10 kVA |
| large_flex | 4 | 13, 14, 15, 18 | 15 kWp | 20 kWh | 13 kVA |

Bus 0 is the slack (the transformer); buses 1–18 are the 18 PQ connection
points, placed by electrical distance from the slack — nearest first, so
`tenant` sits closest and `large_flex`/`pv_battery` sit at the far end where
a nodal injection moves voltage the most. Placement is deterministic
(`sandbox.scenarios.assign_population`), not random, so every submission is
scored on the same feeder.

Inverters are deliberately smaller than the roof — DC/AC ≈ 1.2, which is what
real installations use and which produces about 2% clipping. It is also
load-bearing: sizing inverters to the array was measured to collapse control
authority to zero, because a household that can export everything never needs
its battery.

A full week runs in about **two seconds**.

### Which feeder — and no, you cannot change it

**The hackathon runs on `rural`**, a long feeder with an end-of-line impedance
of 0.91 Ω, about twice IEC 60725's reference. Voltage crosses the 1.05 pu
planning trigger on about 8.7 % of bus-intervals — the `>1.05` column
`score()` prints — and peaks at 1.107, right at the EN 50160 limit.

The reason is the household seam. On ewz's own `urban` network the
neighbourhood-only residual in the voltage signal is 0.25 % of nominal, below
what a real meter resolves, so a controller reading voltage there is reading a
noisy clock and the controller pathway would be a dead end by construction.
`rural` lifts that residual to 1.01 % — twice meter resolution — and gives a
household something genuine to read.

What does *not* change with the feeder is the pathology itself: reverse flow
sits at 42 % of the week and the coincidence factor at 0.79 on all three
strengths in `FEEDER_STRENGTHS`. Diversity is purely behavioural, so the
herding this challenge is about is the same problem on a stiff meshed city
network as on a long rural line — only the export peak and the voltage move.
`urban` and `suburban` remain in the code because the comparison is
instructive and because `scripts/measure_voltage_residual.py` sweeps all
three, not because they are options: every submission is scored on `rural`.

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
└── measure_voltage_residual.py   regenerates the residual numbers above
```

`tariff.py` and `controller.py` are where `my_idea.py`'s two functions actually
get wired in, and where the escape hatches live for anyone who outgrows a plain
function — a stateful tariff (`MyTariff`), a `@numpy_controller`. Most
submissions never need to open either.

The tests are the executable version of the claims above:

```bash
uv run pytest
```

The repo shares `gll_env`'s pre-commit setup (ruff, `ty`, whitespace, licence
headers, conventional commits). It is not required to participate:

```bash
uv run pre-commit install    # optional
```

## Licence

Apache 2.0.
