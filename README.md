# What can a school of fish teach us about grid incentives?

A sandbox for the [Energy Data Hackdays 2026](https://www.energydatahackdays.ch/challenges/what-can-a-school-of-fish-teach-us-about-grid-incentives-2)
challenge from **ewz**.

Eighteen households on a real low-voltage feeder. You edit **two functions** — a
grid tariff and a household controller — and everything else is fixed: the power
flow, the feasibility projection, the rollout, the scoring.

```bash
uv sync
uv run jupyter lab notebooks/00_quickstart.ipynb
```

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
| do nothing | 46.6 kW | **6.8 kW** | 0.595 | 68 CHF |
| self-consumption | 44.2 kW | **11.8 kW** | 0.571 | 91 CHF |

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

That is not a limitation, it is the interesting part. **Voltage is the price
proxy**: a nodal price is high exactly when the local feeder is congested, and
congestion is measurable at your own terminal. But every such signal is
correlated across the feeder — one cloud shades the whole neighbourhood — so
the mechanism that lets you anticipate is the same one that makes everybody
move together. Designing around that is the challenge.

**The controller never sees a neighbour, either.** It is written for one
household and `vmap`'d over the population, so there is no agent axis inside it
to index. vmap is the fairness contract, not just a speed trick.

## Four pathways

1. **Design the price** — edit `MyTariff` in [`sandbox/tariff.py`](sandbox/tariff.py).
2. **Design the household** — edit `my_controller` in [`sandbox/controller.py`](sandbox/controller.py).
3. **Audit it** — `sandbox.export.to_dataframe()` gives you a tidy pandas frame. No JAX required.
4. **Show it** — same frame; the feeder has coordinates for a map.

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
simulation.**

Everything is SI: **kW**, **kWh**, **CHF**, **per-unit** voltage. Every field
name carries its unit, because a silent factor of four between kW and kWh is
the easiest mistake here to make.

### Three ways to write it, all interchangeable

| tier | how | you get | cost |
|---|---|---|---|
| **eager** | `rollout(..., fast=False)` | readable tracebacks, working `print` | 14× slower |
| **numpy** | [`@numpy_controller`](sandbox/numpy_bridge.py) | real `if`, real loops, real SciPy | 1.4× slower |
| **jax** | plain `jnp` | instant seed sweeps | — |

The rollout cannot tell them apart. Write it eager, keep it in NumPy if you
like, and score it either way — the results are identical, and there is a test
asserting that.

## Scoring

Four rollouts, not one:

|  | base controller | your controller |
|---|---|---|
| **fair LEG** | reference floor | does yours help *today*? |
| **your tariff** | does yours help a *naive* household? | your combination |

The gap between the last two is the **co-design premium** — how much of your
result needs households to be running precisely the controller you shipped.
Reported, not gated.

Every cell involving a submitted tariff **re-tunes** the controller first.
Without that a tariff changes nothing physical at all, because nobody can see
it during an episode; it would only redistribute. The tuner maximises the
*household's own bill*, never the grid score — the gap between what a household
wants and what the network needs is the mechanism design problem, and closing
it is what designing a tariff means.

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

## What is *not* the constraint

Over-voltage. On this feeder — ewz's own, meshed and stiff — voltage never
leaves the band no matter what anybody does, and the Swiss Q(U) grid code holds
it there by absorbing reactive power. Meshing buys voltage stiffness; it buys
no thermal capacity. What binds is reverse flow at three times the forward
peak for 42% of the week, and the loss of the diversity the network was sized
under.

`FEEDER_STRENGTHS` in [`sandbox/scenarios.py`](sandbox/scenarios.py) also ships
`suburban` (the IEC 60725 reference impedance) and `rural`, where voltage
*does* bind. Which constraint bites where is itself worth a submission.

## The feeder

CIGRE low-voltage, 19 buses, 18 connection points, 15-minute intervals, seven
days. Twelve households have an inverter and are agents; six are tenants who
cannot respond to anything.

| type | n | roof | battery |
|---|---|---|---|
| tenant | 6 | — | — |
| pv_only | 5 | 6 kWp | — |
| pv_battery | 5 | 8 kWp | 13 kWh |
| large_flex | 2 | 10 kWp | 20 kWh |

A full week runs in about **one second**; a 20-seed ensemble in **eight**.

## Install

```bash
uv sync
```

`gll_env` — the simulator underneath — is pinned in `pyproject.toml`. If your
machine blocks PyPI, a devcontainer is provided.

## Layout

```
sandbox/
├── tariff.py        ← EDIT ME (pathway 1)
├── controller.py    ← EDIT ME (pathway 2)
├── numpy_bridge.py    write it in NumPy instead
├── scenarios.py       who lives on the feeder
├── observation.py     what one household can measure
├── rollout.py         the loop           [do not edit]
├── tuning.py          household best response
├── metrics.py         the jury           [do not edit]
├── evaluate.py        the four cells     [do not edit]
└── export.py          tidy frames, no JAX needed
```

## Licence

Apache 2.0.
