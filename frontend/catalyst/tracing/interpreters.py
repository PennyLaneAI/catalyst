# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Module with JAX interpreters (in the tracing framework) developed for Catalyst."""

import functools

import jax
import jax._src.api_util as au
import jax._src.linear_util as lu
import jax.flatten_util

from catalyst.jax_primitives import AbstractQbit, AbstractQreg


class ADTrace(jax.core.Trace):
    def pure(self, x):
        return ADTracer(self, x, active=False)

    def lift(self, x):
        return ADTracer(self, x, active=False)

    def process_primitive(self, primitive, in_tracers, params):
        # Propagate differentiability information. This should be the conservative action.
        # TODO: Improve the algorithm by taking into account semantics of individual primitives.
        is_active = any(tracer.active for tracer in in_tracers)

        # Identity transform: Unbox tracers to send them one level down.
        lowered_tracers = [tracer.value for tracer in in_tracers]
        out_tracers = primitive.bind(*lowered_tracers, **params)
        out_tracers = (out_tracers,) if not isinstance(out_tracers, (list, tuple)) else out_tracers

        # Since qubit values (tracers) are cached by Catalyst's tracing machinery, it's possible
        # to produce leaked tracers from any jaxpr transform launched within trace_quantum_function.
        # Excluding these values from being boxed up in ADTracers will prevent such leaks.
        results = []
        for tracer in out_tracers:
            if isinstance(jax.core.get_aval(tracer), (AbstractQreg, AbstractQbit)):
                results.append(tracer)
            else:
                results.append(ADTracer(self, tracer, is_active))

        return results if primitive.multiple_results else results[0]

    def process_call(self, call_primitive, f: lu.WrappedFun, in_tracers, params):
        # Wrap the callee to inject our ADTrace. Note f is a JAX WrappedFun.
        argnums = [idx for idx, x in enumerate(in_tracers) if x.active]
        f_wrapped = lu.WrappedFun(
            trace_diffargs(f.f, argnums), f.transforms, f.stores, f.params, f.in_type, f.debug_info
        )

        lowered_tracers = [tracer.value for tracer in in_tracers]
        out_tracers = call_primitive.bind(f_wrapped, *lowered_tracers, **params)

        # TODO: Extract the differentiability of the call results.
        assert call_primitive.multiple_results
        return [ADTracer(self, x, bool(argnums)) for x in out_tracers]


class ADTracer(jax.core.Tracer):
    __slots__ = ["value", "active"]

    def __init__(self, trace, value, active):
        self._trace = trace
        self.value = value
        self.active = active

    @property
    def aval(self):
        return jax.core.get_aval(self.value)

    def full_lower(self):
        if not self.active:
            return jax.core.full_lower(self.value)
        return self


# Needed so that PennyLane doesn't think we are a new type of ML interface.
ADTracer.__module__ = ADTracer.__module__.replace("catalyst", "jax", 1)


def trace_diffargs(fun, argnums):

    @functools.wraps(fun)
    def diffargs_transform_wrapper(*args, **kwargs):
        args_flat, args_tree = jax.tree_flatten((args, kwargs))
        fun_flat, res_tree = flatten_fun(fun, args_tree)

        with jax.core.new_main(ADTrace) as main:
            trace = ADTrace(main, jax.core.cur_sublevel())

            # TODO: optimize by not instantiating ADTracers for non diff args, means the
            #       ADTrace.process_primitive can be skipped for ops not acting on any ADTracers
            in_tracers = [ADTracer(trace, arg, i in argnums) for i, arg in enumerate(args_flat)]
            res_flat = fun_flat(*in_tracers)
            out_tracers = [trace.full_raise(res) for res in res_flat]

        results = [tracer.value for tracer in out_tracers]
        return jax.tree_unflatten(res_tree(), results)

    return diffargs_transform_wrapper


# utils (should be available in JAX, but they differ somewhat from the simple Autodidax version)
def flatten_fun(f, in_tree):
    store = Store()

    def flat_fun(*args_flat):
        pytree_args, pytree_kwargs = jax.tree_unflatten(in_tree, args_flat)
        out = f(*pytree_args, **pytree_kwargs)
        out_flat, out_tree = jax.tree_flatten(out)
        store.set_value(out_tree)
        return out_flat

    return flat_fun, store


class Store:
    val = None

    def set_value(self, val):
        assert self.val is None
        self.val = val

    def __call__(self):
        return self.val
