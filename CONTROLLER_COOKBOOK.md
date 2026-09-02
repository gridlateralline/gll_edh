# Controller cookbook

Everything you need to know to write a controller, and nothing you do not.

## The signature

```python
def my_controller(obs, carry, params, key) -> (p_set_kw, carry):
    """ONE household. Returns net active power at the inverter, kW.
       Positive = injecting into the grid."""
```

Your function runs for a **single** household and is `vmap`'d over the
population. Every field of `obs` is a scalar inside it. There is no agent axis,
so you cannot look at a neighbour even by accident.

Edit it in place in [`sandbox/my_idea.py`](sandbox/my_idea.py); `check()` and
`score()` pick up whatever is there.

## What you can see

| field | unit | meaning |
|---|---|---|
| `hour` | h | **the clock.** 0–24, start of the coming interval |
| `time_sin`, `time_cos` | — | the same clock, smooth across midnight |
| `voltage_pu` | pu | your own bus. Correlates +0.99 with congestion — but see below |
| `meter_kw` | kW | your net flow last interval, + = injecting |
| `load_kw` | kW | your consumption last interval |
| `load_forecast_kw` | kW | your expected consumption, coming interval |
| `pv_available_kw` | kW | the most your roof can make, coming interval |
| `soc_kwh`, `soc_headroom_kwh` | kWh | stored, and room left |
| `bat_charge_max_kw`, `bat_discharge_max_kw` | kW | already limited by state of charge |
| `p_min_kw`, `p_max_kw` | kW | what you may actually do |

**No price.** Real settlement lags past the end of an episode. Tune your
parameters across episodes instead — that is what "anticipate it" means.

**And do not over-trust voltage.** It tracks congestion almost perfectly, but
roughly 90% of it is already implied by your own PV, your own load and the
clock. What is left — the part genuinely about your neighbourhood — is 0.86% of
nominal on the default `rural` feeder, which sits just above the ~0.5% a Class
1 smart meter resolves and is worth using. On `urban` it is 0.20%, well *below*
meter resolution: a voltage threshold there is an expensive way to read a
clock. Check which feeder you are on before building a rule around it.

`p_min_kw` / `p_max_kw` already fold in your inverter rating, your grid
connection, your battery's state and the reactive power Q(U) committed on your
behalf. They are narrower than your nameplate, and they are what an action is
judged against.

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

## Five JAX rules

Only if you write in `jnp`. Use `@numpy_controller` (below) and none of this
applies.

**No `if` on a traced value.**
```python
p = jnp.where(obs.voltage_pu > 1.02, 0.0, obs.load_kw)   # yes
if obs.voltage_pu > 1.02: ...                            # no
```

**No item assignment.** `x = x.at[i].set(v)`, never `x[i] = v`.

**No `.item()`, `float()`, `bool()`** on a traced value.

**No Python loops over agents.** There is only one household in scope.

**Clip, do not assert.** `jnp.clip` and `jnp.where` instead of raising.

## When it breaks

Run **`check(fast=False)`** first, always. It drops the compilation, so values
are concrete, `print` works, and the traceback points at your own line instead
of somewhere inside `scan`.

The two errors you are most likely to meet, and what they actually mean:

| what you see | what it means |
|---|---|
| `TracerBoolConversionError: Attempted boolean conversion of traced array` | a Python `if` on a value that depends on `obs`. Use `jnp.where`, or move to the NumPy tier. |
| `scan body function carry input and carry output must have equal types` | your carry changed shape or dtype between intervals. It must be fixed in both. |

A rule that works under `fast=False` and fails under `fast=True` is almost
always the first row: eager mode has real numbers, so the `if` you wrote runs
fine right up until it is compiled.

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
    return float(obs["load_kw"]), carry
```

`obs` arrives as a dict of plain NumPy scalars — see
[`LocalObservation.as_dict`](sandbox/observation.py). Decorate `my_controller`
in `my_idea.py` in place and `check()` and `score()` keep working on it.

Two rules survive into this tier: the carry must still be a **fixed-shape,
fixed-dtype** pytree, and the function must have **no side effects** —
everything comes back through the return value. Cost is about 1.4× wall clock.

## The carry

Your household's memory, carried between intervals. **Fixed shape, fixed
dtype** — no growing lists, no changing types. That one rule survives into the
NumPy tier too, because the host callback has to declare what it returns.

The default `Memory` carries `p_prev_kw`, `voltage_ewma_pu` and `intervals`.
Replace it with any pytree you like.

The carry is where the interesting answers live. Not just forecasts and
trends — **staggering**. A household that remembers what it just did can offset
itself against its neighbours, and hysteresis or randomised timing is a
legitimate answer to herding that costs nothing in energy.

## Ideas that need no price

- Use `load_forecast_kw` rather than `load_kw` — act on the coming interval.
- **Wait.** A battery that starts charging at 13:00 instead of 09:00 is still
  absorbing during the afternoon export peak instead of having filled up at
  eleven. Costs nothing in energy, only in timing.
- Watch `carry.voltage_ewma_pu` for a *trend* rather than a level. Everyone
  sees the same level at the same moment; that is the trap.
- Use `key` to stagger. If all twelve households do the same thing at the same
  time, they are one household with twelve times the power.
- Hold back capacity for the evening ramp using the clock.

## Scale, and why nothing happened

The single most common way a controller does nothing: the parameter is the
wrong order of magnitude. Voltage is the usual culprit, because it is
per-unit and its *deviations* are what your rule actually sees.

On the default `rural` feeder voltage runs roughly 0.98 to 1.10, so a droop
written as `gain * (voltage_pu - 1.02)` has an input of a few hundredths and
needs a gain in the tens before it moves a kW. On `urban`, where voltage barely
reaches 1.02, the same rule sees *thousandths* and would need a gain in the
hundreds — and if the quantity it trims never binds in the first place, no gain
will help at all.

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
actually choose.

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
- [ ] Parameters are the right order of magnitude — verified, not assumed
- [ ] Does something a tenant with `p_min_kw == p_max_kw == 0` survives
