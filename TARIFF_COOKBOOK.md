# Tariff cookbook

Everything you need to know to write a tariff, and nothing you do not.

## The signature

```python
def my_tariff(grid, carry, params) -> (settlement_chf, carry):
    """The WHOLE feeder, one interval. Returns (num_pq,) CHF, signed:
       positive = the connection point is paid, negative = it owes."""
```

Your function runs **once per interval**, after the power flow has solved,
over the whole feeder at once. There is no household-by-household loop to
write and no agent axis to index — every field below is already an array
over all 18 connection points.

What you return is the interval's settlement: the final number each connection
point pays or earns. A flat rate, a time-of-use schedule, a demand charge, a
nodal price built from voltage sensitivity, or fair LEG plus a congestion term
are all just different return values from this one function.

`carry` is yours, threaded to the next interval — the tariff's counterpart to
a controller's `carry`. It defaults to `TariffMemory` (see "carrying state"
below); a stateless tariff just returns it unchanged.

## What you are settling

`e_grid_kwh` — the net **active** energy exchanged with the grid at each
connection point over the interval. That is **inverter output minus household
load**: the meter reading, not the inverter's own production and not the
household's own consumption. A household chooses its inverter setpoint and
is billed on what that came to once its own load was served. (The household
side calls the same quantity `p_grid_kw`; see
[`CONTROLLER_COOKBOOK.md`](CONTROLLER_COOKBOOK.md).)

Units carry the physics: `_kw` is active, `_kvar` reactive, `_kva` apparent,
`_kwh`/`_kvarh` the matching energies, with a `p_`/`q_` prefix wherever both
halves exist at the same terminal. Reactive flow is visible to a tariff — see
below — but it is not what the baseline settles.

## What you can see

`grid` is a [`GridView`](sandbox/observation.py) — plain SI, no environment
state, no per-unit conversions, no bus-versus-connection-point index hops.

| field | shape | unit | meaning |
|---|---|---|---|
| `e_grid_kwh` | `(18,)` | kWh | net **active** exchange with the grid, + = pushed in |
| `p_grid_kw` | `(18,)` | kW | the same, as a power |
| `q_grid_kvar` | `(18,)` | kvar | its **reactive** counterpart — see the warning below |
| `voltage_pu` | `(18,)` | pu | voltage at each connection point — see the warning below |
| `transformer_kw` | scalar | kW | substation active throughput, + = the feeder drawing, - = exporting |
| `transformer_kvar` | scalar | kvar | substation reactive throughput |
| `losses_kw` | scalar | kW | what the network itself burned — quadratic in flow |
| `hour` | scalar | h | 0–24, the settled interval |
| `fair_leg_chf` | `(18,)` | CHF | the **whole settlement** fair LEG would produce this interval |
| `has_inverter` | `(18,)` bool | — | who can act at all — a static equipment fact, not a live reading |

**`fair_leg_chf` is an input, not a base you have to build on.** It is a
finished CHF figure, not a rate — energy, grid fee and public duties already
folded in — and it is there because pricing energy from scratch is work you
may not want to redo. The shipped default does build on it. Use it whole,
use the parts you want, or compute the interval's settlement without touching
it.

**`has_inverter` is who they are, not what they did this interval.** It
comes from the population's fixed asset mix and never changes within an
episode, the same way a real rate class doesn't depend on this week's meter
reading. It exists so a tariff can say what it means directly — an
unconditional floor for tenants, a different rate class for `large_flex` —
instead of reverse-engineering an identity from behaviour (a tenant already
reveals itself every interval anyway: no inverter, no battery, `e_grid_kwh`
never driven by anything but load). Keep it to that use. Pricing what a
connection point *did* — its flow, its voltage — belongs to `e_grid_kwh` /
`voltage_pu`; `has_inverter` is not a channel for smuggling behavioural
information a tariff isn't supposed to have.

## Reactive power: you can see it, be careful pricing it

Unlike a household, a network operator measures reactive flow, so `GridView`
gives you `q_grid_kvar` per connection point and `transformer_kvar` at the
substation. It is not a rounding error here — the agents' connection points
carry more than 0.1 kvar on **90 %** of intervals, averaging −1.03 kvar and
reaching −6.4 kvar.

**But it is almost entirely not chosen.** On a Swiss LV connection the Q(U)
grid code (NE7 §4.3.2) sets each inverter's reactive power from the voltage
at its own bus, and the remainder is the load's own power factor. A household
can move it only indirectly, by changing the active power that moves its
voltage. So a charge on `q_grid_kvar` is the reactive version of a charge on
`voltage_pu`: it prices **exposure, not contribution**, and gives almost no
marginal incentive. Read "Exposure is not contribution" below — it applies
here word for word.

Two places where reactive is genuinely worth reaching for:

- **Losses.** Reactive current heats the same conductors as active current,
  so the cost reactive flow imposes is already inside `losses_kw` — and
  `losses_kw` is caused by everybody together, which is the honest thing to
  share out.
- **Substation loading.** A transformer is rated in **kVA**, not kW, so what
  its thermal limit actually sees is `hypot(transformer_kw,
  transformer_kvar)`. A demand charge written against apparent power is more
  faithful to the binding constraint than one written against active power
  alone.

The baseline does not bill reactive, and neither do real Swiss LV residential
tariffs — reactive charges appear on larger commercial connections. Your
tariff returns a whole settlement, so it *may* price whatever it can see. It
just has to be able to say what behaviour it is trying to change.

## What fair LEG is, and is not

**Fair LEG is not a product ewz sells.** It is assembled from ewz's real,
published 2026 rate components — the EEA feed-in tariff, ewz.natur energy,
grid usage, public duties — with one construction on top. Trading inside a
local electricity community waives 40 % of the grid usage fee, and fair LEG
splits that saving **evenly** between injector and consumer: every member
gains the same 1.19 Rp./kWh off-peak, 2.38 Rp./kWh peak, against its own
fallback rate. Symmetric by construction, hence the name.

ewz's actual LEG product is **Solarquartier**, and `gll_env` ships it too
(`payments: solarquartier`). It grants the consumer the whole 40 % rebate but
charges a flat 13 Rp./kWh LEG energy rate — well above what ewz.natur charges
off-peak. Net effect for a pure consumer: about **5.7 Rp./kWh worse**
off-peak inside the community than outside it, and about 1.1 Rp./kWh better
at peak. The producer captures nearly all of the community's saving.

Solarquartier would be the easier baseline and the wrong one. Beating a tariff
that actively penalises consumers is a trivial win that says nothing about
the network. So the status quo modelled here is the *fair* version — a LEG
whose incentives between the two sides are already balanced — and beating it
has to mean saying something about congestion, diversity and ramp, which is
the question the challenge is actually about.

## `check()` will not show your tariff working

This trips up almost everyone. **No household can see a price during an
episode**, so running your tariff against an unchanged controller changes the
settlement and *nothing physical* — identical peaks, identical ramps,
identical everything. That is not your tariff failing; it is what a tariff
is.

What `check()` *can* show you is the second table it prints, `CHF/kWh
consumed, by household type`. Redistribution needs no behavioural response to
be visible, so the incidence of your tariff — who is now paying for the
network, and whether the households with no way to respond are carrying it —
is legible in ten seconds. That is worth iterating on well before you spend
two minutes on a `score()`.

`check()` does not tune. **`score()` does**, and it is the only thing that can
show a tariff moving the *feeder*. Budget about two minutes for it.

## Exposure is not contribution

The tempting locational tariff is "charge each connection point in
proportion to the voltage at its own bus." Resist it, or at least know what
it does.

A connection point's bus voltage is mostly made by *other* households.
Someone at the end of a line where the neighbours export heavily sits at a
high voltage whether or not they export anything themselves, so a price on
the voltage **level** charges them for a condition they did not create — and
if they cut their own injection, the voltage barely moves, so the charge
barely falls. It taxes position and gives almost no marginal incentive,
which is close to the opposite of what a congestion price is for.

A locational price done properly charges **sensitivity**, not level — how
much the binding quantity (voltage, transformer throughput, losses) moves
per kW of *this* connection point's own injection. `voltage_pu` is an input
to estimating that, not the answer on its own.

Worth noting the contrast with fair LEG's `fair_leg_chf`, which is also
interdependent — your settlement depends on what everyone else did, through
the community match ratio. The difference is that the match ratio applies to
everyone equally and pro rata, so interdependence there does not become a
charge for where you happen to live.

## The two rules, and how they're checked

**It must publish `(num_pq,)`, not `(num_agents,)`.** Six of the eighteen
connection points are tenants with no inverter and therefore no agent; they
are absent from the reward array entirely, and they are exactly the
households a badly designed tariff harms. `GridView` is already shaped
`(num_pq,)` everywhere, so returning something the same shape as
`grid.e_grid_kwh` gets this right automatically.

**It must not print money.** Checked *empirically*, after the fact, against
what fair LEG itself collects — see `revenue_adequate` in
[`sandbox/metrics.py`](sandbox/metrics.py). You do not need to force every
interval to sum to exactly zero by hand. A tariff that redistributes, or
that collects a little more or less than fair LEG across the whole episode,
can still pass; the tolerance is `REVENUE_TOLERANCE = 0.10` of the reference
total. One that simply pays everybody — a subsidy dressed as a price — cannot,
and is disqualified rather than ranked.

The gate is run with **behaviour held fixed**, against the base controller at
its default parameters. That is deliberate: a tariff that makes households
export less collects less, and that is the tariff working, not the tariff
printing money. Comparing re-tuned cells would fail every tariff that
succeeded.

It should also be **worth anticipating**: households tune against expected
structure across episodes, not against any one price. If your tariff is
unpredictable even in distribution, no controller can respond to it and
you've built a lottery, not a mechanism.

## The shipped default is a bad tariff. Here is how it fails.

`sandbox/my_idea.py`'s `my_tariff` ships as fair LEG's settlement plus a
crude aggregate congestion surcharge. Run `score()` on a fresh checkout and
it looks like a triumph on the headline columns — and it is not. It is the
worked example of the failure mode the jury exists to catch:

| | export peak | ramp | coincid | curtail | CHF |
|---|---|---|---|---|---|
| `fair_leg/base` | 66.2 kW | 16.5 kW | 0.777 | 1.8 % | 311 |
| `submitted/submitted` | **38.6 kW** | **10.8 kW** | 0.855 | **26.2 %** | **156** |

The export peak drops by 42 % and the ramp by a third — **because** the tuned
household simply caps its exports, so a quarter of the week's generation is
thrown away and the community's earnings halve. And the coincidence factor
gets *worse*, from 0.777 to 0.855: the aggregate signal has synchronised the
population harder than no signal at all.

A tariff that flattens the feeder by throwing solar away has not solved
anything. It has bought two metrics with three others, and `curtailed_share`,
`community_settlement_chf` and `coincidence_factor` are in the jury precisely
so it cannot do that quietly. Read the whole row, not the first two columns.

The second thing wrong with it is structural: the congestion term is an
**aggregate** signal. Every household on the feeder sees the same number
every interval, which is the herding problem restated as a price — a
population tuned against a common signal can synchronise *harder*, not less.
Watch `coincid` when you run it.

## From the default to something ambitious

Directions from there, roughly in order of how much of the default survives:

1. **Retune the same shape.** Change `headroom_kwh` / `price_chf_per_kwh`.
   Cheapest experiment, tells you if the mechanism is even in the right
   ballpark — see "scale" below. Start by asking what price stops short of
   inducing curtailment.
2. **Make the congestion term locational** instead of aggregate, so that
   households at different points on the feeder face different prices and
   have a reason to act at different times.
3. **Add a second term** alongside it — a time-of-use schedule on
   `grid.hour`, a demand charge on `grid.transformer_kw`, whatever your idea
   needs.
4. **Price the interval from scratch.** Nothing in the harness assumes fair
   LEG underneath. A flat rate, a two-part tariff, a fully nodal price built
   from voltage sensitivity — compute the settlement and return it.

For (4), the plain-function signature is still enough — you never need to
touch `sandbox/tariff.py` unless you want state that survives across
intervals (see "carrying state" below).

## Wiring it in

`check()` and `score()` pick up whatever `my_tariff` currently is and wrap it
via [`tariff_from_settlement`](sandbox/tariff.py) with `TARIFF_PARAMS`, so
editing the function and its parameters in `my_idea.py` is all that is
required. To build one by hand — for a notebook, or a sweep of your own:

```python
from sandbox.tariff import tariff_from_settlement
tariff = tariff_from_settlement(my_tariff, TARIFF_PARAMS)
```

There is also [`tariff_from_charge`](sandbox/tariff.py), a narrower
convenience: give it a function returning a surcharge and it adds that to fair
LEG's settlement for you, leaving energy pricing alone. Reach for it when a
redistributed charge on top of the existing tariff is exactly the idea;
`tariff_from_settlement` is the general pathway.

## Carrying state across intervals

By default every interval is priced from what just happened and nothing more —
`carry` starts at `TariffMemory`'s default and a stateless tariff returns it
unchanged, as the shipped `my_tariff` does. But state is a first-class part of
the signature, not an escape hatch: use it for a running total towards a demand
charge, a ratchet, or — the interesting one — a *smoothed* congestion signal
instead of an instantaneous one, the same anti-herding idea as a controller's
`carry.voltage_ewma_pu`, applied to the price instead of the household:

```python
@chex.dataclass(frozen=True)
class MyCarry:
    congestion_ewma_kwh: chex.Array

def my_tariff(grid, carry, params):
    aggregate_kwh = jnp.sum(grid.e_grid_kwh)
    smoothed = 0.9 * carry.congestion_ewma_kwh + 0.1 * jnp.abs(aggregate_kwh)
    excess = jnp.maximum(smoothed - params["headroom_kwh"], 0.0)
    ...
    return settlement_chf, carry.replace(congestion_ewma_kwh=smoothed)

tariff = tariff_from_settlement(
    my_tariff, TARIFF_PARAMS,
    init_carry=lambda: MyCarry(congestion_ewma_kwh=jnp.float32(0.0)),
)
```

`TariffMemory` (the default `carry` type, just an `intervals` counter) is
only a placeholder — replace it with any fixed pytree you like via
`init_carry`, exactly the way a controller passes its own `init_carry` in
place of `Memory`. The same rule survives from the controller side: **fixed
shape, fixed dtype**, because the harness has to declare the carry's shape
before the episode runs.

If you need more than one interval's `GridView` and a carry can give you —
per-branch flows, the power-flow Jacobian, anything living in `dynamics` or
`new_state` itself — that's the actual boundary: subclass
[`MyTariff`](sandbox/tariff.py) and override
`settlement_from_view(self, grid, carry)` directly, or drop `MyTariff`
altogether and write your own `CausalReward`. Ask for a field on `GridView`
if you find yourself reaching past it more than once.

## Five JAX rules

`my_tariff` is jnp code under the same `jit`/`scan` as a controller, not a
plain Python function that happens to run once per interval instead of once
per household. Skip this section entirely if you use `@numpy_tariff` below.

**No `if` on a traced value.**
```python
p = jnp.where(grid.hour > 18.0, 0.20, 0.10)   # yes
if grid.hour > 18.0: ...                       # no
```

**No item assignment.** `x = x.at[i].set(v)`, never `x[i] = v`.

**No `.item()`, `float()`, `bool()`** on a traced value.

**No Python loops over connection points.** They are already an array; act on
all eighteen at once.

**Clip, do not assert.** `jnp.clip` and `jnp.where` instead of raising.

## When it breaks

Run **`check(fast=False)`** first, always: it drops the compilation, so values
are concrete, `print` works, and the traceback points at your own line.

| what you see | what it means |
|---|---|
| `A tariff must settle all 18 connection points` | you returned `(num_agents,)`, or a scalar. Return something shaped like `grid.e_grid_kwh`. |
| `TracerBoolConversionError` | a Python `if` on a value that depends on `grid`. Use `jnp.where`, or move to the NumPy tier. |
| `scan body function carry ... must have equal types` | your carry changed shape or dtype between intervals. It must be fixed in both. |

## Writing it in NumPy — and why you probably shouldn't ship it

Simpler than the controller's `@numpy_controller`: a tariff already runs
once per interval over the whole feeder, not once per household, so there is
no agent axis to fake under `vmap` — just a bare host round trip.

```python
from sandbox.numpy_bridge import numpy_tariff

@numpy_tariff
def my_tariff(grid, carry, params):
    if grid["hour"] > 18:                      # a real branch
        return -0.20 * grid["e_grid_kwh"], carry
    return -0.10 * grid["e_grid_kwh"], carry
```

`grid` arrives as a dict of plain NumPy arrays — see
[`GridView.as_dict`](sandbox/observation.py). Decorate `my_tariff` in
`my_idea.py` in place and `check()` and `score()` keep working; pass your own
numbers with `@numpy_tariff(params=TARIFF_PARAMS)`. See
[`sandbox/numpy_bridge.py`](sandbox/numpy_bridge.py) for the two rules that
survive into this tier — fixed-shape carry, no side effects — identical to the
controller's NumPy tier.

**Write the finished tariff in `jnp` if you possibly can.** The host round
trip is cheap on a single rollout and expensive on the scoring path, where
`score()` runs four cells with a tuning sweep inside each. Under JAX that
sweep is one compile `vmap`'d across every candidate and seed at once; the
NumPy tier cannot batch and pays a callback per interval per rollout.
Measured on a controller: 3.4× on a single week, **8×** on a tuning sweep,
and a `score()` that runs in about two minutes under `jnp` takes roughly
twenty. Prototype in NumPy if it helps; port before you iterate.

## It's a Stackelberg game, and that's the point

The tariff moves first: you commit to a settlement rule, then
[`tune()`](sandbox/tuning.py) finds the household's best response to it —
whatever parameters maximise the household's own bill under your tariff,
knowing nothing of your intent. You are the *leader*; `tune()` computes the
*follower*. That's why tariff parameters are never searched automatically
the way `TUNE_OVER` searches a controller's — doing so would hand the
leader's move to another optimiser and remove the actual mechanism-design
question, which is the gap between what your tariff *intends* and what a
rational household *does* with it.

Both bottom cells of the scorer ([`sandbox/evaluate.py`](sandbox/evaluate.py))
best-respond to your tariff — the follower move is never skipped, because a
tariff nobody responds to has changed nothing physical. What differs is
*which* controller is doing the responding, and each is tuned over **its own**
parameter grid: the installed base over `TUNING_GRID` in
[`sandbox/controller.py`](sandbox/controller.py), yours over `TUNE_OVER`.

That distinction matters because you are only ever the leader. ewz publishes
a tariff; households then run whatever serves their own bill. You cannot put
a controller in anyone's basement — you can only make one worth building. So
when you submit a controller alongside a tariff, you are not shipping
firmware: you are stating what you believe the household's best response to
your price turns out to be, and the right-hand cell scores your tariff
against it. That is the Stackelberg evaluation proper — the leader's payoff
at the follower's best response.

The left cell is the same question asked of the installed base, which can only
best-respond within the strategy it already implements. The gap, the
**co-design premium**, is how much of your result needs the market to supply
the controller your price is designed to make worth building.

"Best response" is bounded on purpose: `tune()` searches a declared parameter
grid inside whichever controller the cell is running. It is not an
unconstrained search over every controller anyone could write — that would be
a research project rather than a scoring step, and it would leave the two
bottom cells with nothing in common to compare. Pinning the follower's
strategy space is what makes the premium mean something. It also means
[`TUNE_OVER`](sandbox/my_idea.py) is part of your submission: a controller
whose good parameters are not in the sweep will be scored at parameters
nobody would actually choose.

## Scale, and why nothing happened

The single most common way a tariff does nothing: the price is the wrong
order of magnitude relative to what it competes with. The default's
`price_chf_per_kwh: 1.00` looks large next to the ~0.14 CHF/kWh feed-in rate
it's up against — until you notice the charge is shared pro rata across
everyone contributing to the excess, so one household's *marginal* exposure
is an order of magnitude below the headline. A price that looks punitive in
aggregate can be nearly invisible at the margin, which is the first thing to
check when a tariff seems to change nothing.

The failure at the other end is just as easy and less obvious: a price that
*does* bite makes curtailment the household's cheapest response, and you buy
a flat feeder by spilling generation. Both edges are one `score()` apart.

Print `grid.*` and your intermediate terms under `check(fast=False)` before
concluding your idea does not move anybody.

## Checklist

- [ ] Returns `(settlement_chf, carry)`, `settlement_chf` shaped `(num_pq,)`
      matching `grid.e_grid_kwh`
- [ ] Carry has fixed shape and dtype
- [ ] Runs under `check(fast=False)` without an exception
- [ ] Judged with `score()`, not `check()` — `check()` cannot show a tariff working
- [ ] `score()` reports `revenue adequacy: PASS` (or you understand exactly why not)
- [ ] `curtailed_share` has not blown up — a flat feeder bought by spilling
      solar is not a solution
- [ ] `coincid` has not got worse — an aggregate signal can synchronise the
      population harder than no signal at all
- [ ] Parameters are the right order of magnitude — verified against the
      marginal exposure, not the headline number
- [ ] Tenants (`p_inv_min_kw == p_inv_max_kw == 0`, absent from any
      agent-indexed array) are not silently harmed by a rule written with
      prosumers in mind
- [ ] The fairness table reads defensibly: `tenant` CHF/kWh has not gone up to
      buy the network columns, and `spread` between the four household types
      has not widened much. A flat feeder paid for by the six households with
      no roof is a result you have to be willing to defend to a regulator
- [ ] **Would all five parties sign it?** The four household types and the
      network operator. Go through them one at a time against the fairness
      table. If one would refuse, either fix it or narrow the tariff to a
      voluntary rate class that everyone inside it would join — and then show
      the customers outside it are no worse off. Both are legitimate answers;
      a universal tariff that quietly loses somebody is not
- [ ] A household could **anticipate** your price from what it knows in
      advance — the clock, the weather, its own load and state of charge. If
      responding correctly requires knowing what the neighbours did, the
      tariff redistributes after the fact and steers nothing
- [ ] `tenant_import` — the tenants' share of everything the feeder imports —
      is the cost base your charge is recovered from. If your tariff lets
      self-supplying households shrink out of it, say what happens to the
      households left behind
- [ ] `has_inverter`, if used, only ever expresses static rate-class intent —
      never a stand-in for a live reading `e_grid_kwh`/`voltage_pu` already
      gives you
