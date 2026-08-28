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

"""Write your controller in plain NumPy.

There is no shim that makes arbitrary NumPy run under ``vmap`` and ``jit``.
`jax.numpy` is already almost API-compatible with NumPy; what breaks is
control flow, mutation and side effects -- ``if x > 0``, ``arr[i] = v``,
Python loops, ``.item()``. None of that is fixable by an import.

So instead this runs your function as *actual* NumPy, on the host, from inside
the compiled rollout. You get real ``if``, real loops, real SciPy, and working
``print`` and ``breakpoint`` inside a jitted scan.

::

    @numpy_controller(init_carry=my_carry)
    def my_rule(obs, carry, params):
        if obs["voltage_pu"] > 1.02:          # a real branch
            return 0.0, carry
        return float(obs["load_kw"]), carry

`obs` arrives as a dict of plain NumPy scalars -- ``obs["voltage_pu"]``,
``obs["p_max_kw"]`` and so on; see
:meth:`sandbox.observation.LocalObservation.as_dict` for the full set.

Three tiers, all interchangeable
--------------------------------
The rollout cannot tell these apart, so a submission may mix them freely and
only wall-clock differs:

* **eager** -- ``rollout(..., fast=False)``. A Python loop, no vmap. Readable
  tracebacks while you are still working things out.
* **numpy** -- this module. Real NumPy inside the fast path.
* **jax** -- a native controller. Instant seed sweeps.

Two rules survive into this tier
--------------------------------
**The carry must be a fixed-shape, fixed-dtype pytree.** The host callback has
to declare what it returns before it runs, so your carry cannot grow a list or
change dtype between intervals. This is the one JAX rule NumPy does not buy
you out of.

**No side effects.** ``pure_callback`` is pure by contract; a mutated global
may be dropped or reordered. Everything comes back through the carry.

The cost is a host round trip per household per interval. On the reference
scenario a full scoring sweep is seconds rather than milliseconds, which is
a fine trade for keeping your own code.
"""

from typing import Any, Callable

import chex
import jax
import jax.numpy as jnp
import numpy as np

from sandbox.controller import Controller, Memory, init_memory
from sandbox.observation import LocalObservation

#: ``(obs_dict, carry, params) -> (p_set_kw, carry)``, in plain NumPy.
NumpyControllerFn = Callable[[dict[str, np.ndarray], Any, Any], tuple[float, Any]]


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


__all__ = ["Memory", "numpy_controller"]
