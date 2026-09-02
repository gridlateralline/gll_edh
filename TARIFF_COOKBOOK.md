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

## What you can see

`grid` is a [`GridView`](sandbox/observation.py) — plain SI, no environment
state, no per-unit conversions, no bus-versus-connection-point index hops.

| field | shape | unit | meaning |
|---|---|---|---|
| `net_kwh` | `(18,)` | kWh | what each connection point pushed (+) or drew (-) this interval |
| `net_kw` | `(18,)` | kW | the same, as a power |
| `voltage_pu` | `(18,)` | pu | voltage at each connection point — see the warning below |
| `transformer_kw` | scalar | kW | substation throughput, + = the feeder drawing, - = exporting |
| `losses_kw` | scalar | kW | what the network itself burned — quadratic in flow |
| `hour` | scalar | h | 0–24, the settled interval |
| `energy_chf` | `(18,)` | CHF | what ewz's real, published fair-LEG rate would settle this interval as |
| `has_inverter` | `(18,)` bool | — | who can act at all — a static equipment fact, not a live reading |

**`energy_chf` is an input, not a base you have to build on.** It is there
because pricing energy from scratch is work you may not want to redo, and the
shipped default does build on it. Use it whole, use the parts you want, or
compute the interval's settlement without touching it.

**`has_inverter` is who they are, not what they did this interval.** It
comes from the population's fixed asset mix and never changes within an
episode, the same way a real rate class doesn't depend on this week's meter
reading. It exists so a tariff can say what it means directly — an
unconditional floor for tenants, a different rate class for `large_flex` —
instead of reverse-engineering an identity from behaviour (a tenant already
reveals itself every interval anyway: no inverter, no battery, `net_kwh`
never driven by anything but load). Keep it to that use. Pricing what a
connection point *did* — its flow, its voltage — belongs to `net_kwh` /
`voltage_pu`; `has_inverter` is not a channel for smuggling behavioural
information a tariff isn't supposed to have.

### Exposure is not contribution

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

Worth noting the contrast with fair LEG's `energy_chf`, which is also
interdependent — your settlement depends on what everyone else did, through
the community match ratio. The difference is that the match ratio applies to
everyone equally and pro rata, so interdependence there does not become a
charge for where you happen to live.

## The two rules, and how they're checked

**It must publish `(num_pq,)`, not `(num_agents,)`.** Six of the eighteen
connection points are tenants with no inverter and therefore no agent; they
are absent from the reward array entirely, and they are exactly the
households a badly designed tariff harms. `GridView` is already shaped
`(num_pq,)` everywhere, so returning something the same shape as `grid.net_kwh`
gets this right automatically.

**It must not print money.** Checked *empirically*, after the fact, against
what fair LEG itself collects — see `revenue_adequate` in
[`sandbox/metrics.py`](sandbox/metrics.py). You do not need to force every
interval to sum to exactly zero by hand. A tariff that redistributes, or
that collects a little more or less than fair LEG across the whole episode,
can still pass; the tolerance is `REVENUE_TOLERANCE = 0.10` of the reference
total. One that simply pays everybody — a subsidy dressed as a price — cannot,
and is disqualified rather than ranked.

It should also be **worth anticipating**: households tune against expected
structure across episodes, not against any one price. If your tariff is
unpredictable even in distribution, no controller can respond to it and
you've built a lottery, not a mechanism.

## From the default to something ambitious

`sandbox/my_idea.py`'s `my_tariff` ships with fair LEG's `energy_chf` plus a
crude, aggregate congestion surcharge. Treat it as tier zero. Some directions
from there, roughly in order of how much of the default survives:

1. **Retune the same shape.** Change `headroom_kwh` / `price_chf_per_kwh`.
   Cheapest experiment, tells you if the mechanism is even in the right
   ballpark — see "scale" below.
2. **Make the congestion term locational** instead of aggregate. Right now
   every household sees the same congestion number every interval, which is
   exactly the herding problem restated as a price: a population tuned
   against an aggregate signal can synchronise *harder*, not less.
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
via [`tariff_from_settlement`](sandbox/tariff.py), so editing the function in
`my_idea.py` is all that is required. To build one by hand — for a notebook,
or a sweep of your own:

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
    aggregate_kwh = jnp.sum(grid.net_kwh)
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
| `A tariff must settle all 18 connection points` | you returned `(num_agents,)`, or a scalar. Return something shaped like `grid.net_kwh`. |
| `TracerBoolConversionError` | a Python `if` on a value that depends on `grid`. Use `jnp.where`, or move to the NumPy tier. |
| `scan body function carry ... must have equal types` | your carry changed shape or dtype between intervals. It must be fixed in both. |

## Writing it in NumPy

Simpler than the controller's `@numpy_controller`: a tariff already runs
once per interval over the whole feeder, not once per household, so there is
no agent axis to fake under `vmap` — just a bare host round trip.

```python
from sandbox.numpy_bridge import numpy_tariff

@numpy_tariff
def my_tariff(grid, carry, params):
    if grid["hour"] > 18:                      # a real branch
        return -0.20 * grid["net_kwh"], carry
    return -0.10 * grid["net_kwh"], carry
```

`grid` arrives as a dict of plain NumPy arrays — see
[`GridView.as_dict`](sandbox/observation.py). Decorate `my_tariff` in
`my_idea.py` in place and `check()` and `score()` keep working; pass your own
numbers with `@numpy_tariff(params=TARIFF_PARAMS)`. See
[`sandbox/numpy_bridge.py`](sandbox/numpy_bridge.py) for the two rules that
survive into this tier — fixed-shape carry, no side effects — identical to the
controller's NumPy tier.

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
*which* controller is doing the responding: the one households run today, or
the one you wrote.

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

## Scale, and why nothing happened

The single most common way a tariff does nothing: the price is the wrong
order of magnitude relative to what it competes with. The default's
`price_chf_per_kwh: 1.00` looks large next to the ~0.14 CHF/kWh feed-in rate
it's up against — until you notice the charge is shared pro rata across
everyone contributing to the excess, so one household's *marginal* exposure
is an order of magnitude below the headline. A price that looks punitive in
aggregate can be nearly invisible at the margin, which is the first thing to
check when a tariff seems to change nothing.

Print `grid.*` and your intermediate terms under `check(fast=False)` before
concluding your idea does not move anybody.

## Checklist

- [ ] Returns `(settlement_chf, carry)`, `settlement_chf` shaped `(num_pq,)`
      matching `grid.net_kwh`
- [ ] Carry has fixed shape and dtype
- [ ] Runs under `check(fast=False)` without an exception
- [ ] `score()` reports `revenue adequacy: PASS` (or you understand exactly why
      not) — `check()` is too short to run the gate
- [ ] Parameters are the right order of magnitude — verified against the
      marginal exposure, not the headline number
- [ ] Tenants (`p_min_kw == p_max_kw == 0`, absent from any agent-indexed
      array) are not silently harmed by a rule written with prosumers in mind
- [ ] `has_inverter`, if used, only ever expresses static rate-class intent —
      never a stand-in for a live reading `net_kwh`/`voltage_pu` already give you
