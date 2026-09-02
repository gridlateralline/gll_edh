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

"""Write your controller -- or your tariff -- in plain NumPy.

There is no shim that makes arbitrary NumPy run under ``vmap`` and ``jit``.
`jax.numpy` is already almost API-compatible with NumPy; what breaks is
control flow, mutation and side effects -- ``if x > 0``, ``arr[i] = v``,
Python loops, ``.item()``. None of that is fixable by an import, and it
applies to a tariff exactly as much as a controller: `my_tariff` is jnp code
under the same ``jit``/``scan``, not a plain Python function that happens to
be called less often.

So instead this runs your function as *actual* NumPy, on the host, from inside
the compiled rollout. You get real ``if``, real loops, real SciPy, and working
``print`` and ``breakpoint`` inside a jitted scan.

::

    @numpy_controller(init_carry=my_carry)
    def my_rule(obs, carry, params):
        if obs["voltage_pu"] > 1.02:          # a real branch
            return 0.0, carry
        return float(obs["load_kw"]), carry

    @numpy_tariff
    def my_price(grid, carry, params):
        if grid["hour"] > 18:                 # a real branch, over the feeder
            return -0.20 * grid["net_kwh"], carry
        return -0.10 * grid["net_kwh"], carry

`obs`/`grid` arrive as a dict of plain NumPy values -- ``obs["voltage_pu"]``,
``grid["net_kwh"]`` and so on; see
:meth:`sandbox.observation.LocalObservation.as_dict` and
:meth:`sandbox.observation.GridView.as_dict` for the full sets.

`numpy_tariff` is the simpler of the two: a tariff already runs once per
interval over the whole feeder, not once per household, so there is no agent
axis to fake under ``vmap`` -- it is a bare host round trip, no
``vmap_method`` to reason about.

Three tiers, all interchangeable, for either seam
--------------------------------------------------
The rollout cannot tell these apart, so a submission may mix them freely and
only wall-clock differs:

* **eager** -- ``rollout(..., fast=False)``. A Python loop, no vmap. Readable
  tracebacks while you are still working things out.
* **numpy** -- this module. Real NumPy inside the fast path.
* **jax** -- a native controller or tariff. Instant seed sweeps.

Two rules survive into this tier
--------------------------------
**The carry must be a fixed-shape, fixed-dtype pytree.** The host callback has
to declare what it returns before it runs, so your carry cannot grow a list or
change dtype between intervals. This is the one JAX rule NumPy does not buy
you out of.

**No side effects.** ``pure_callback`` is pure by contract; a mutated global
may be dropped or reordered. Everything comes back through the carry.

The cost is a host round trip per interval (per household, for a controller).
On the reference scenario a full scoring sweep is seconds rather than
milliseconds, which is a fine trade for keeping your own code.
"""

from typing import Any, Callable

import chex
import jax
import jax.numpy as jnp
import numpy as np

from sandbox.controller import Controller, Memory, init_memory
from sandbox.observation import GridView, LocalObservation
from sandbox.tariff import TariffMemory, init_tariff_memory, tariff_from_settlement

#: ``(obs_dict, carry, params) -> (p_set_kw, carry)``, in plain NumPy.
NumpyControllerFn = Callable[[dict[str, np.ndarray], Any, Any], tuple[float, Any]]

#: ``(grid_dict, carry, params) -> (settlement_chf, carry)``, in plain NumPy.
NumpyTariffFn = Callable[[dict[str, np.ndarray], Any, Any], tuple[np.ndarray, Any]]


def _spec_like(tree: Any) -> Any:
    """Shape and dtype of every leaf -- what the callback must promise."""
    return jax.tree_util.tree_map(
        lambda leaf: jax.ShapeDtypeStruct(jnp.shape(leaf), jnp.asarray(leaf).dtype), tree
    )


def numpy_controller(
    fn: NumpyControllerFn | None = None,
    *,
    init_carry: Callable[[], Any] = init_memory,
    name: str | None = None,
    params: Any = None,
) -> Any:
    """Wrap a NumPy controller so the harness can run it like any other.

    Usable bare or with arguments::

        @numpy_controller
        def rule(obs, carry, params): ...

        @numpy_controller(init_carry=my_carry, name="thermostat")
        def rule(obs, carry, params): ...

    Returns a :class:`~sandbox.controller.Controller`, ready for
    :func:`~sandbox.rollout.rollout` and for tuning.
    """

    def decorate(inner: NumpyControllerFn) -> Controller:
        carry_spec = _spec_like(init_carry())
        action_spec = jax.ShapeDtypeStruct((), jnp.float32)

        def host(obs: dict[str, np.ndarray], carry: Any, params: Any) -> tuple[np.ndarray, Any]:
            action, new_carry = inner(obs, carry, params)
            # Coerced here rather than trusted: the callback's declared return
            # types are a promise to the compiler, and breaking it silently
            # corrupts the trace instead of raising.
            return np.asarray(action, dtype=np.float32), jax.tree_util.tree_map(
                lambda leaf, spec: np.asarray(leaf, dtype=spec.dtype), new_carry, carry_spec
            )

        def wrapped(
            obs: LocalObservation, carry: Any, params: Any, key: chex.PRNGKey
        ) -> tuple[chex.Array, Any]:
            del key  # a NumPy controller may use `params` for its own randomness
            # "sequential" is what makes this behave the way a NumPy author
            # expects under vmap: one call per household, with scalars.
            return jax.pure_callback(
                host,
                (action_spec, carry_spec),
                obs.as_dict(),
                carry,
                params,
                vmap_method="sequential",
            )

        wrapped.__name__ = inner.__name__
        wrapped.__doc__ = inner.__doc__
        return Controller(
            name=name or inner.__name__,
            fn=wrapped,
            params={} if params is None else params,
            init_carry=init_carry,
        )

    return decorate if fn is None else decorate(fn)


def numpy_tariff(
    fn: NumpyTariffFn | None = None,
    *,
    init_carry: Callable[[], Any] = init_tariff_memory,
    params: Any = None,
) -> Any:
    """Write your tariff in plain NumPy.

    Simpler than :func:`numpy_controller`: a tariff already runs once per
    interval over the whole feeder, not once per household, so there is no
    agent axis to fake under ``vmap`` -- this is a bare host round trip.

    Usable bare or with arguments::

        @numpy_tariff
        def my_rule(grid, carry, params): ...

        @numpy_tariff(init_carry=my_carry, params=MY_PARAMS)
        def my_rule(grid, carry, params): ...

    `grid` arrives as a dict of plain NumPy arrays -- ``grid["net_kwh"]``,
    ``grid["voltage_pu"]`` and so on; see :meth:`sandbox.observation.GridView.as_dict`
    for the full set. The same two rules as :func:`numpy_controller` apply:
    the carry must be a fixed-shape, fixed-dtype pytree, and there are no
    side effects -- everything comes back through the return value.

    Returns a tariff factory, ready for ``build_env(tariff=...)`` or
    ``tune(tariff=...)`` -- the same thing :func:`sandbox.tariff.tariff_from_settlement`
    produces, because that is what builds it underneath.
    """

    def decorate(inner: NumpyTariffFn) -> Any:
        carry_spec = _spec_like(init_carry())

        def host(
            grid: dict[str, np.ndarray], carry: Any, tariff_params: Any
        ) -> tuple[np.ndarray, Any]:
            settlement_chf, new_carry = inner(grid, carry, tariff_params)
            return np.asarray(settlement_chf, dtype=np.float32), jax.tree_util.tree_map(
                lambda leaf, spec: np.asarray(leaf, dtype=spec.dtype), new_carry, carry_spec
            )

        def wrapped(grid: GridView, carry: Any, tariff_params: Any) -> tuple[chex.Array, Any]:
            num_pq = grid.net_kwh.shape[-1]
            settlement_spec = jax.ShapeDtypeStruct((num_pq,), jnp.float32)
            return jax.pure_callback(
                host, (settlement_spec, carry_spec), grid.as_dict(), carry, tariff_params
            )

        wrapped.__name__ = inner.__name__
        wrapped.__doc__ = inner.__doc__
        return tariff_from_settlement(
            wrapped, {} if params is None else params, init_carry=init_carry
        )

    return decorate if fn is None else decorate(fn)


__all__ = ["Memory", "TariffMemory", "numpy_controller", "numpy_tariff"]
