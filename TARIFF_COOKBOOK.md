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

This is the whole settlement, not a surcharge on one that is already
decided. Nothing requires you to build on fair LEG, add a term to it, or even
look at it.

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

**`energy_chf` is a starting point, not a floor.** It exists so "design a
tariff" does not silently mean "design a surcharge on top of this one." Use
it as a base and add a term to it (the default does exactly this), use only
the parts you want, or ignore it completely and price the interval from
scratch — a flat rate, a time-of-use schedule, a demand charge, whatever you
like. All three are just different return values from the same function.

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

## From a naive default to something ambitious

`sandbox/my_idea.py`'s `my_tariff` ships with fair LEG's `energy_chf` plus a
crude, aggregate congestion surcharge — the same shape as before this
cookbook existed. Treat it as tier zero. Some directions from there, roughly
in order of how much of the naive default survives:

1. **Retune the same shape.** Change `headroom_kwh` / `price_chf_per_kwh`.
   Cheapest experiment, tells you if the mechanism is even in the right
   ballpark — see "scale" below.
2. **Make the congestion term locational** instead of aggregate. Right now
   every household sees the same congestion number every interval, which is
   exactly the herding problem restated as a price: a population tuned
   against an aggregate signal can synchronise *harder*, not less.
3. **Add a second term** alongside it — a time-of-use schedule on
   `grid.hour`, a demand charge on `grid.transformer_kw`, whatever your idea
   needs. Still additive to `grid.energy_chf`; just more than one term.
4. **Replace `grid.energy_chf` entirely.** Nothing about the harness assumes
   fair LEG underneath. A flat rate, a two-part tariff, a fully nodal price
   built from voltage sensitivity — write the interval's settlement from
   scratch and return it.

For (4), the plain-function signature is still enough — you never need to
touch `sandbox/tariff.py` unless you want state that survives across
intervals (see "carrying state" below).

## Wiring it in

`sandbox/check.py` wraps whatever `my_tariff` currently is via
[`tariff_from_settlement`](sandbox/tariff.py):

```python
from sandbox.tariff import tariff_from_settlement
tariff = tariff_from_settlement(my_tariff, TARIFF_PARAMS)
```

`tariff_from_settlement` is the general pathway — your function's return
value **is** the settlement. There is also
[`tariff_from_charge`](sandbox/tariff.py), narrower and kept only for anyone
who wants exactly the old shape: a stateless function that returns a
surcharge which gets added to fair LEG's own settlement for them, rather
than replacing it.

## Carrying state across intervals

Every interval by default is priced from what just happened, nothing more —
`carry` starts at `TariffMemory`'s default and a stateless tariff just
returns it unchanged, same as the naive `my_tariff` does. But state is a
first-class part of the signature, not an escape hatch: use it for a running
total towards a demand charge, a ratchet, or — the interesting one — a
*smoothed* congestion signal instead of an instantaneous one, the same
anti-herding idea as a controller's `carry.voltage_ewma_pu`, applied to the
price instead of the household:

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

Exactly the [same five as the controller](CONTROLLER_COOKBOOK.md#five-jax-rules)
— `my_tariff` is jnp code under the same `jit`/`scan`, not a plain Python
function that happens to run once per interval instead of once per
household. In particular:

**No `if` on a traced value.**
```python
p = jnp.where(grid.hour > 18.0, 0.20, 0.10)   # yes
if grid.hour > 18.0: ...                       # no
```

**No item assignment, no `.item()`/`float()`/`bool()`, no Python loops over
connection points, clip-don't-assert.** Use `@numpy_tariff` (below) and none
of this applies.

When something breaks, run `rollout(..., fast=False)` first. Values become
concrete, `print` works, and the traceback points at your line.

## Writing it in NumPy

Simpler than the controller's `@numpy_controller`: a tariff already runs
once per interval over the whole feeder, not once per household, so there is
no agent axis to fake under `vmap` — just a bare host round trip.

```python
from sandbox.numpy_bridge import numpy_tariff

@numpy_tariff
def my_rule(grid, carry, params):
    if grid["hour"] > 18:                      # a real branch
        return -0.20 * grid["net_kwh"], carry
    return -0.10 * grid["net_kwh"], carry
```

`grid` arrives as a dict of plain NumPy arrays — see
[`GridView.as_dict`](sandbox/observation.py). Returns a tariff factory
directly, ready for `build_env(tariff=...)`. See
[`sandbox/numpy_bridge.py`](sandbox/numpy_bridge.py) for the full docstring
and the two rules that survive into this tier (fixed-shape carry, no side
effects) — identical to the controller's NumPy tier.

## It's a Stackelberg game, and that's the point

The tariff moves first: you commit to a settlement rule, then
[`tune()`](sandbox/tuning.py) finds the household's best response to it —
whatever parameters maximise the household's own bill under your tariff,
knowing nothing of your intent. You are the *leader*; `tune()` computes the
*follower*. That's why tariff parameters are never searched automatically
the way `TUNE_OVER` searches a controller's — doing so would hand the
leader's move to another optimiser and remove the actual mechanism-design
question, which is the gap between what your tariff *intends* and what a
rational household *does* with it. The scorer's four cells
([`sandbox/evaluate.py`](sandbox/evaluate.py)) are exactly this game played
out: reference-leader/reference-follower, your-leader/naive-follower, and
your-leader/best-responding-follower. The last two apart is the "co-design
premium" — how much of your result needs the household to be running
precisely the response you designed for, rather than something naive.

## Scale, and why nothing happened

The single most common way a tariff does nothing: the price is the wrong
order of magnitude relative to what it competes with. The default's
`price_chf_per_kwh: 1.00` looks large next to the ~0.14 CHF/kWh feed-in rate
it's up against — until you notice the charge is shared pro rata across
everyone contributing to the excess, so one household's *marginal* exposure
is an order of magnitude below the headline. A price that looks punitive in
aggregate can be nearly invisible at the margin, which is the first thing to
check when a tariff seems to change nothing.

Print `grid.*` and your intermediate terms in eager mode (`fast=False`)
before concluding your idea does not move anybody.

## Checklist

- [ ] Returns `(settlement_chf, carry)`, `settlement_chf` shaped `(num_pq,)`
      matching `grid.net_kwh`
- [ ] Carry has fixed shape and dtype
- [ ] Runs under `rollout(..., fast=False)` without an exception
- [ ] `check()` shows `revenue adequacy: PASS` (or you understand exactly why not)
- [ ] Parameters are the right order of magnitude — verified against the
      marginal exposure, not the headline number
- [ ] Tenants (`p_min_kw == p_max_kw == 0`, absent from any agent-indexed
      array) are not silently harmed by a rule written with prosumers in mind
- [ ] `has_inverter`, if used, only ever expresses static rate-class intent —
      never a stand-in for a live reading `net_kwh`/`voltage_pu` already give you
