# Controller cookbook

Everything you need to know to write a controller, and nothing you do not.

## The signature

```python
def my_controller(obs, carry, params, key) -> (p_inv_kw, carry):
    """ONE household. Returns the INVERTER's active power, kW.
       Positive = the inverter is producing."""
```

Your function runs for a **single** household and is `vmap`'d over the
population. Every field of `obs` is a scalar inside it. There is no agent axis,
so you cannot look at a neighbour even by accident.

Edit it in place in [`sandbox/my_idea.py`](sandbox/my_idea.py); `check()` and
`score()` pick up whatever is there.

## The one identity to keep in your head

```
p_inv_kw  (you choose)  -  p_load_kw  (you don't)  =  p_grid_kw  (you're billed on)
```

You set the **inverter**'s active power. Your own load sits behind the same
meter and nobody controls it, so what the feeder carries — and what every
tariff settles — is the difference.

All three are **active** power, in kW. That is what the `p_` says: `_kw` is
active, `_kvar` reactive, `_kva` apparent, and a `p_`/`q_` prefix appears
wherever both halves exist at the same terminal. `pv_available_kw` and the
battery fields carry no prefix because they have no reactive half.

| you want | you return |
|---|---|
| export nothing, import nothing | `obs.p_load_forecast_kw` |
| export 3 kW to the grid | `obs.p_load_forecast_kw + 3.0` |
| idle the inverter (import the whole load) | `0.0` |
| charge the battery from the grid at 2 kW | `obs.p_load_forecast_kw - 2.0` |

Solar and battery both sit behind that one inverter, and it serves solar
first: ask for **more** than the roof is making and the battery discharges to
cover the gap; ask for **less** and the surplus charges it. You never address
the battery directly — you choose one number and the split follows.

`obs.p_grid_kw` reports what last interval's request actually came to, so a
controller can see whether it hit what it aimed at.

## What you can see

`obs` is one household's own meter and nothing else. Half these fields
describe the interval that just **ended** and half the one about to
**begin** — mixing those up is the commonest quiet mistake here.

| field | unit | interval | meaning |
|---|---|---|---|
| `hour` | h | coming | **the clock.** 0–24, start of the interval you act in |
| `time_sin`, `time_cos` | — | coming | the same clock, smooth across midnight |
| `voltage_pu` | pu | last | your own bus, ~0.98 to 1.11 — see below |
| `p_grid_kw` | kW | last | your net *active* exchange with the grid, + = injecting. **The billed quantity.** |
| `p_load_kw` | kW | last | what the house drew |
| `p_load_forecast_kw` | kW | **coming** | what it will draw |
| `pv_available_kw` | kW | **coming** | the most your roof can make |
| `soc_kwh`, `soc_headroom_kwh` | kWh | now | stored, and room left |
| `bat_charge_max_kw`, `bat_discharge_max_kw` | kW | coming | already limited by state of charge |
| `p_inv_min_kw`, `p_inv_max_kw` | kW | coming | what you may ask the inverter for |

**No price.** Real settlement lags past the end of an episode. Tune your
parameters across episodes instead — that is what "anticipate it" means.

`p_inv_min_kw` / `p_inv_max_kw` bound the **inverter request**, not the grid
exchange. They already fold in your inverter rating, your grid connection,
your battery's state and the reactive power Q(U) has committed on your behalf,
so they are narrower than your nameplate — below it on about 91 % of
intervals — and they are what an action is judged against. Six of the
eighteen connection points are tenants with
`p_inv_min_kw == p_inv_max_kw == 0`; your controller runs for them too and
must survive it.

**One number, not two.** You choose active power and nothing else. On a Swiss
LV connection the reactive axis is not yours: the Q(U) grid code (NE7 §4.3.2)
sets your inverter's reactive power from your own bus voltage, which is why
the action space here is one-dimensional. You feel it only as headroom —
reactive power consumes apparent-power capacity, and what is left on the
active axis is exactly `p_inv_min_kw` … `p_inv_max_kw`. Nothing else about it
is yours to design. (A tariff *can* see reactive flow; see
[`TARIFF_COOKBOOK.md`](TARIFF_COOKBOOK.md).)

## Voltage: a thin signal, but a real one

The project is called *grid lateral line* partly for this. A fish reads the
water it is in through a line of local pressure sensors; your own bus voltage
is the closest thing a household has to the same organ. It is local, free,
continuously available, and genuinely coupled to what the neighbours are
doing — it correlates with feeder congestion at **+0.99**.

That correlation is not the caveat. **Redundancy** is. Regress own voltage on
your own PV, your own load and the clock and about **86 %** of it is already
implied by things you knew anyway. What is left — the part genuinely about
your neighbourhood — is **1.01 % of nominal** on the `rural` feeder this
hackathon runs on, twice the ~0.5 % a Class 1 smart meter resolves.

So: **the residual is a real, measurable signal and building on it is a
legitimate design.** What fails is the naive version — a threshold on the raw
level, which mostly fires on "it is noon and my roof is working", something
every household on the feeder learns at the same instant. Two ways to do
better:

```python
# subtract what you already knew; act on the surprise, not the level
expected_pu = 1.0 + params["k_pv"] * obs.pv_available_kw - params["k_load"] * obs.p_load_kw
surprise_pu = obs.voltage_pu - expected_pu

# or use the trend rather than the level -- carry.voltage_ewma_pu is free
rising = obs.voltage_pu - carry.voltage_ewma_pu
```

And the residual is **not a fixed budget**. It is small partly *because*
every household currently runs the same rule off the same weather, so their
contributions to each other's voltage move together. A population that
deliberately differentiates — staggered starts, hysteresis, randomised timing
off `key` — makes its members' voltages less predictable from their own
state, which puts information back into the residual. Reading the signal and
creating something worth reading are the same project.

(For the record: on ewz's stiff `urban` network the same residual is 0.25 %,
below meter resolution, which is exactly why the hackathon runs on `rural`
instead. The feeder is fixed — see the README.)

## Getting hours out of the clock

```python
obs.hour        # 0 .. 24, start of the interval you are about to act in
```

That is all. It comes straight off the environment's own clock, so it is exact.

`time_sin` / `time_cos` are there too, for anyone who wants a feature that is
smooth across midnight. **Do not reconstruct the hour from them.** They are a
pair because either alone is ambiguous — a cosine is symmetric about noon, so
`12 * (1 - time_cos)` reads 24 at midday and quietly matches 11:00 as well as
13:00 — and they describe the interval's *midpoint*, so even a correct
`atan2` of the two lands half an interval late.

## Randomness: using `key`

`key` is a JAX PRNG key, freshly split **per household per interval**, so
twelve identical households draw twelve different numbers. That is the whole
point of it: symmetry-breaking that costs nothing in energy.

JAX has no hidden global RNG. You pass a key in, you get numbers out, the same
key always gives the same numbers — which is what keeps a scored run
reproducible. Two rules and no more: **never use one key for two draws**
(split it first), and remember that anything you randomise is still tuned as a
*distribution* across episodes, not picked per run.

**Jitter an action** — a fresh draw every interval is right here:

```python
import jax

noise = jax.random.uniform(key, minval=-1.0, maxval=1.0)
p_inv_kw = target_kw + params["jitter_kw"] * noise
```

**A per-household constant** — a start time that rerolls every fifteen minutes
is not a start time. Draw once and keep it in the carry:

```python
offset_h = jnp.where(
    carry.intervals == 0,
    jax.random.uniform(key, minval=0.0, maxval=params["spread_h"]),
    carry.offset_h,          # add offset_h to your carry to hold it
)
charging_allowed = obs.hour >= params["charge_after_hour"] + offset_h
```

**Two draws in one interval** — split, never reuse:

```python
k_start, k_jitter = jax.random.split(key)
```

**A biased coin** — `jax.random.bernoulli(key, p)` returns a traced boolean;
feed it to `jnp.where`, not to a Python `if`.

## Five JAX rules

Only if you write in `jnp`. Use `@numpy_controller` (below) and none of this
applies.

**No `if` on a traced value.**
```python
p = jnp.where(obs.voltage_pu > 1.02, 0.0, obs.p_load_kw)   # yes
if obs.voltage_pu > 1.02: ...                            # no
```

**No item assignment.** `x = x.at[i].set(v)`, never `x[i] = v`.

**No `.item()`, `float()`, `bool()`** on a traced value.

**No Python loops over agents.** There is only one household in scope.

**Clip, do not assert.** `jnp.clip` and `jnp.where` instead of raising.

## When it breaks

Run **`check(fast=False)`** first, always. It drops the compilation, so values
are concrete, `print` works, and the traceback points at your own line instead
of somewhere inside `scan`. It costs about 80 seconds against `check()`'s ten.

The two errors you are most likely to meet, and what they actually mean:

| what you see | what it means |
|---|---|
| `TracerBoolConversionError: Attempted boolean conversion of traced array` | a Python `if` on a value that depends on `obs`. Use `jnp.where`, or move to the NumPy tier. |
| `scan body function carry input and carry output must have equal types` | your carry changed shape or dtype between intervals. It must be fixed in both. |

A rule that works under `fast=False` and fails under `fast=True` is almost
always the first row: eager mode has real numbers, so the `if` you wrote runs
fine right up until it is compiled.

## Three tiers — and why you want the JAX one

All three produce identical results and the rollout cannot tell them apart.
What differs is wall clock, and it differs by a lot once you reach `score()`:

| tier | how | you get | one week | a `tune()` sweep | a `score()` |
|---|---|---|---|---|---|
| **jax** | plain `jnp` | the fast path | 2.0 s | 17 s | ~2 min |
| **numpy** | [`@numpy_controller`](sandbox/numpy_bridge.py) | real `if`, loops, SciPy | 6.9 s (3.4×) | 128 s (**8×**) | ~20 min |
| **eager** | `rollout(..., fast=False)` | readable tracebacks, working `print` | ~100× per interval | not usable | not usable |

**Write the finished thing in `jnp` if you possibly can.** The 3.4× on a
single rollout understates it: `score()` runs four cells with a tuning sweep
inside each, and under JAX that sweep is one compile `vmap`'d across every
candidate and seed at once, while the NumPy tier forces a host round trip per
household per interval and cannot batch at all. Two minutes against twenty
is, at a hackathon, the difference between ten experiments and one.

Use the eager tier for debugging and the NumPy tier to get an idea working
when the JAX form is not obvious. Then port it. `jnp.where` replaces most
`if`s mechanically, and the two tiers are asserted equal in
`tests/test_harness.py`.

## Writing it in NumPy

If the rule you want is easier to express with real branches and real loops,
write it as actual NumPy and let the harness call it from inside the compiled
rollout:

```python
from sandbox.numpy_bridge import numpy_controller

@numpy_controller(params={"threshold_pu": 1.02})
def my_controller(obs, carry, params):        # note: no `key` in this tier
    if obs["voltage_pu"] > params["threshold_pu"]:   # a real branch
        return 0.0, carry
    return float(obs["p_load_kw"]), carry
```

`obs` arrives as a dict of plain NumPy scalars — see
[`LocalObservation.as_dict`](sandbox/observation.py). Decorate `my_controller`
in `my_idea.py` in place and `check()` and `score()` keep working on it.

Two rules survive into this tier: the carry must still be a **fixed-shape,
fixed-dtype** pytree, and the function must have **no side effects** —
everything comes back through the return value. There is no `key` argument;
seed your own randomness from `params` or the carry.

## The carry

Your household's memory, carried between intervals. **Fixed shape, fixed
dtype** — no growing lists, no changing types. That one rule survives into the
NumPy tier too, because the host callback has to declare what it returns.

The default `Memory` carries `p_prev_kw`, `voltage_ewma_pu` and `intervals`.
Replace it with any fixed pytree you like: define `INIT_CARRY` in
`my_idea.py` — a zero-argument callable returning **one** household's starting
carry — and `check()` and `score()` pick it up.

```python
import chex, jax.numpy as jnp

@chex.dataclass(frozen=True)
class MyCarry:
    offset_h: chex.Array
    voltage_ewma_pu: chex.Array
    intervals: chex.Array

INIT_CARRY = lambda: MyCarry(
    offset_h=jnp.float32(0.0),
    voltage_ewma_pu=jnp.float32(1.0),
    intervals=jnp.int32(0),
)
```

The harness broadcasts that across the population for you, so write it for a
single household with no agent axis. Note that `update_memory` only knows how
to roll the default `Memory` forward — with your own carry you return it
yourself.

The carry is where the interesting answers live. Not just forecasts and
trends — **staggering**. A household that remembers what it just did can offset
itself against its neighbours, and hysteresis or randomised timing is a
legitimate answer to herding that costs nothing in energy.

## Ideas that need no price

- **Wait.** A battery that starts charging at 13:00 instead of 09:00 is still
  absorbing during the afternoon export peak instead of having filled up at
  eleven. Costs nothing in energy, only in timing. This is the single
  highest-leverage knob in the default controller.
- **Stagger.** Use `key` (above). If all twelve households do the same thing
  at the same moment, they are one household with twelve times the power.
- **Watch the trend, not the level.** `carry.voltage_ewma_pu` moves on a
  six-hour time constant. Everyone sees the same *level* at the same instant;
  that is the trap.
- **Subtract what you already knew** from `obs.voltage_pu` and act on the
  remainder — see "Voltage" above.
- **Hold back capacity for the evening ramp** using the clock.
- Use `p_load_forecast_kw` rather than `p_load_kw`. Honest accounting: this is
  a correctness fix, not a strategy. Measured over twenty weeks it moves
  self-consumption from 29.4 % to 30.3 % and the community bill by about
  3 CHF a week, and it moves peak, ramp and coincidence by nothing at all.
  Do it, then go and do something that matters.

## Scale, and why nothing happened

The single most common way a controller does nothing: the parameter is the
wrong order of magnitude. Voltage is the usual culprit, because it is
per-unit and its *deviations* are what your rule actually sees.

Voltage on this feeder runs roughly 0.98 to 1.11, so a droop written as
`gain * (voltage_pu - 1.02)` has an input of a few hundredths and needs a gain
in the tens before it moves a kW. And if the quantity it trims never binds in
the first place — an `export_cap_kw` sitting at 1e3, say — then no gain helps
at all.

Print your intermediate values in eager mode (`check(fast=False)`) before
concluding your idea does not work.

## Tuning

There is no price to react to, so a controller is tuned across episodes. List
the parameters worth searching in `TUNE_OVER` in `my_idea.py` and `score()`
sweeps them for you.

`TUNE_OVER` is part of your submission, not a convenience. It defines the
household's best response — the tuner searches those values inside your
controller, and nothing outside them — so a good idea whose good parameters
are missing from the sweep gets scored at parameters no household would
actually choose. Names must be keys of `CONTROLLER_PARAMS`; names your
controller does not have are ignored.

Every combination is evaluated, so the sweep is the product of the lists:
three values each for two parameters is nine candidates, each run over four
seeds. Keep it in the low tens.

To run the same search by hand:

```python
from sandbox.check import my_controller_as_bundle, my_tariff_factory
from sandbox.my_idea import TUNE_OVER
from sandbox.scenarios import reference_scenario
from sandbox.tuning import tune

params, table = tune(
    my_controller_as_bundle(),
    reference_scenario(),
    TUNE_OVER,
    tariff=my_tariff_factory(),
)
```

The tuner maximises **your own bill** and knows nothing about voltages or your
neighbours' costs — which is the point. Making the household's interest line up
with the network's is the tariff's job, not the controller's.

## Checklist

- [ ] Runs under `check(fast=False)` without an exception
- [ ] Same answer with `check()` (the compiled path)
- [ ] Carry has fixed shape and dtype
- [ ] Returns the **inverter** setpoint, not the intended grid flow
- [ ] Parameters are the right order of magnitude — verified, not assumed
- [ ] `TUNE_OVER` names parameters your controller actually reads, and brackets
      the values you believe are good
- [ ] Does something a tenant with `p_inv_min_kw == p_inv_max_kw == 0` survives
