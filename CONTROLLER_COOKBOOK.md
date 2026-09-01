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
82% of it is already implied by your own PV, your own load and the clock. The
part that is genuinely about your neighbourhood is 0.13% of nominal on the
urban feeder — below what a real smart meter resolves. On `rural` it is 0.68%
and worth using. Check which feeder you are on before building a rule around
it; a voltage threshold on the urban feeder is an expensive way to read a
clock.

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

Only if you write in `jnp`. Use [`@numpy_controller`](sandbox/numpy_bridge.py)
and none of this applies.

**No `if` on a traced value.**
```python
p = jnp.where(obs.voltage_pu > 1.02, 0.0, obs.load_kw)   # yes
if obs.voltage_pu > 1.02: ...                            # no
```

**No item assignment.** `x = x.at[i].set(v)`, never `x[i] = v`.

**No `.item()`, `float()`, `bool()`** on a traced value.

**No Python loops over agents.** There is only one household in scope.

**Clip, do not assert.** `jnp.clip` and `jnp.where` instead of raising.

When something breaks, run `rollout(..., fast=False)` first. Values become
concrete, `print` works, and the traceback points at your line.

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
wrong order of magnitude.

On the urban feeder `voltage_pu` barely reaches **1.02**. A droop written as
`gain * (voltage_pu - setpoint)` therefore has an input of a few *thousandths*,
so `gain` has to be in the hundreds before it moves a kW. And if the quantity
it trims never binds in the first place, no gain will help.

Print your intermediate values in eager mode before concluding your idea does
not work.

## Tuning

There is no price to react to, so a controller is tuned across episodes:

```python
from sandbox.tuning import tune
params, table = tune(my_bundle, population, MY_TUNING_GRID, tariff=my_tariff)
```

The tuner maximises **your own bill** and knows nothing about voltages or your
neighbours' costs — which is the point. Making the household's interest line up
with the network's is the tariff's job, not the controller's.

## Checklist

- [ ] Runs under `rollout(..., fast=False)` without an exception
- [ ] Same answer with `fast=True`
- [ ] Carry has fixed shape and dtype
- [ ] Parameters are the right order of magnitude — verified, not assumed
- [ ] Does something a tenant with `p_min_kw == p_max_kw == 0` survives
