# Copyright 2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This module contains utility functions related to JAX tracing
"""
from __future__ import annotations

import collections
import functools
import gc
import inspect
import itertools as it
import math
import operator
import threading
import types
import warnings
from collections import namedtuple
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial, partialmethod, total_ordering
from operator import attrgetter
from typing import (
    Any,
    Callable,
    ClassVar,
    DefaultDict,
    Dict,
    FrozenSet,
    Generator,
    Generic,
    Hashable,
    Iterable,
    Iterator,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
    overload,
)
from weakref import ref

import numpy as np
from jax._src import (state, util)
from jax._src.config import FLAGS, config
from jax._src.core import (
    ShapedArray, Trace, _update_thread_local_jit_state, thread_local_state,
    ClosedJaxpr, Jaxpr, JaxprEqn, MainTrace, Primitive as JaxprPrimitive,
    gensym
)
from jax._src.errors import (
    ConcretizationTypeError,
    TracerArrayConversionError,
    TracerIntegerConversionError,
    UnexpectedTracerError,
)
from jax._src.interpreters.partial_eval import (
    DynamicJaxprTracer,
    _input_type_to_tracers,
    make_jaxpr_effects,
    convert_constvars_jaxpr,
)
from jax._src.lax.control_flow import (_initial_style_jaxpr, _initial_style_open_jaxpr)
from jax._src.lax.lax import _abstractify
from jax._src.lib import jax_jit
from jax._src.typing import Array, DimSize, Shape
from jax._src.util import (
    HashableFunction,
    HashableWrapper,
    as_hashable_function,
    cache,
    curry,
    partition_list,
    safe_map,
    safe_zip,
    toposort,
    tuple_delete,
    tuple_insert,
    unzip3,
    weakref_lru_cache,
    wrap_name,
)

from jax._src.linear_util import (wrap_init, annotate)
from jax._src.tree_util import (
    PyTreeDef,
    tree_flatten,
    tree_structure,
    tree_unflatten,
    treedef_is_leaf,
)
from jax._src.api_util import (
    flatten_fun,
    shaped_abstractify,
)

map, unsafe_map = safe_map, map


@contextmanager
def new_main2(
    trace_type: Type[Trace],
    dynamic: bool = False,
    main: Optional[MainTrace(level, trace_type, **payload)] = None,
    **payload,
) -> Generator[MainTrace, None, None]:
    """A verison of JAX `new_main` function that knows how to re-use an already existing `MainTrace`
    object"""

    stack = thread_local_state.trace_state.trace_stack
    level = stack.next_level() if main is None else main.level
    main = MainTrace(level, trace_type, **payload) if main is None else main
    stack.push(main)
    if dynamic:
        prev_dynamic, stack.dynamic = stack.dynamic, main
        _update_thread_local_jit_state(stack.dynamic)

    try:
        yield main
    finally:
        stack.pop()
        if dynamic:
            stack.dynamic = prev_dynamic
            _update_thread_local_jit_state(stack.dynamic)


def sort_eqns(eqns: List[JaxprEqn], forced_order_primitives: Set[JaxprPrimitive]) -> List[JaxprEqn]:
    """Topologically sort JAXRR equations in a list, based on their input/output variables."""

    # FIXME: The functions might emit different correct results, depending on id(eqns). One need to
    # make this function stable. Moreover, some equation (`qdevice` ones) do not depend on others
    # but need to be at the top of the output. We force this order for now. Stable sorting might
    # also allow us to remove this conditioning.
    class Box:
        def __init__(self, e):
            self.e: JaxprEqn = e
            self.parents: Set["Box"] = {}

    boxes = [Box(e) for e in eqns]
    qdevices = [(i, b) for (i, b) in enumerate(boxes) if b.e.primitive in forced_order_primitives]
    origin: Dict[int, Box] = {}
    for b in boxes:
        origin.update({ov.count: b for ov in b.e.outvars})
    for b in boxes:
        b.parents = {origin[v.count] for v in b.e.invars if v.count in origin}
    for i, q in qdevices:
        for b in boxes[i + 1 :]:
            b.parents.add(q)
    return [b.e for b in toposort(boxes)]


def initial_style_jaxprs_with_common_consts1(
    funs: Sequence[Callable], in_tree, in_avals, primitive_name: str
):
    """This function is the head (shorter) part of the original
    `lax.control_flow.common._initial_style_jaxprs_with_common_consts` of JAX. The algorithm is the
    same at the time of this writing, we use this function only to avoid conflicts with future
    versions of JAX.
    """
    jaxprs, all_consts, all_out_trees = unzip3(
        _initial_style_open_jaxpr(fun, in_tree, in_avals, primitive_name) for fun in funs
    )
    closed_jaxprs, consts = initial_style_jaxprs_with_common_consts2(jaxprs, all_consts)
    return closed_jaxprs, consts, all_out_trees


def initial_style_jaxprs_with_common_consts2(jaxprs, all_consts):
    """This function is the tail (largest) part of the
    `lax.control_flow.common._initial_style_jaxprs_with_common_consts` of JAX. The JAX version traces
    argument Python functions in order to determine signatures to be unified. Here we rely on the fact
    that the tracing was already done elsewhere - and this is the only difference.
    """

    all_const_avals = [map(_abstractify, consts) for consts in all_consts]
    for consts, consts_avals in zip(all_consts, all_const_avals):
        for c, aval in zip(consts, consts_avals):
            if isinstance(aval, state.AbstractRef):
                assert isinstance(c, DynamicJaxprTracer)
    canonical_ref_indices = []
    canonical_refs: List[Any] = []
    tracer_id_to_canonical_id = {}
    all_nonref_consts = []
    canonical_ref_avals = []
    all_nonref_const_avals = []
    for consts, consts_avals in zip(all_consts, all_const_avals):
        ref_indices = []
        nonref_consts = []
        nonref_const_avals = []
        for c, aval in zip(consts, consts_avals):
            if isinstance(aval, state.AbstractRef):
                tracer_id = id(c)
                if tracer_id not in tracer_id_to_canonical_id:
                    canonical_id = len(canonical_refs)
                    canonical_refs.append(c)
                    tracer_id_to_canonical_id[tracer_id] = canonical_id
                    canonical_ref_avals.append(aval)
                canonical_id = tracer_id_to_canonical_id[tracer_id]
                ref_indices.append(canonical_id)
            else:
                nonref_consts.append(c)
                nonref_const_avals.append(aval)
        all_nonref_consts.append(nonref_consts)
        all_nonref_const_avals.append(nonref_const_avals)
        canonical_ref_indices.append(ref_indices)

    newvar = gensym(jaxprs, suffix="_")
    unused_ref_const_vars = map(newvar, canonical_ref_avals)
    unused_const_vars = [map(newvar, const_avals) for const_avals in all_nonref_const_avals]

    def pad_jaxpr_constvars(i, jaxpr):
        is_ref = [isinstance(v.aval, state.AbstractRef) for v in jaxpr.constvars]
        nonref_constvars, ref_constvars = partition_list(is_ref, jaxpr.constvars)
        padded_ref_constvars = unused_ref_const_vars[:]
        for canonical_id, ref_var in zip(canonical_ref_indices[i], ref_constvars):
            padded_ref_constvars[canonical_id] = ref_var
        const_prefix = util.concatenate(unused_const_vars[:i])
        const_suffix = util.concatenate(unused_const_vars[i + 1 :])
        constvars = [*padded_ref_constvars, *const_prefix, *nonref_constvars, *const_suffix]
        jaxpr = jaxpr.replace(constvars=constvars)
        effects = make_jaxpr_effects(jaxpr.constvars, jaxpr.invars, jaxpr.outvars, jaxpr.eqns)
        jaxpr = jaxpr.replace(effects=effects)
        return jaxpr

    consts = [*canonical_refs, *util.concatenate(all_nonref_consts)]
    jaxprs = tuple(pad_jaxpr_constvars(i, jaxpr) for i, jaxpr in enumerate(jaxprs))
    closed_jaxprs = [ClosedJaxpr(convert_constvars_jaxpr(jaxpr), ()) for jaxpr in jaxprs]
    return closed_jaxprs, consts


def deduce_avals(f: Callable, args, kwargs):
    flat_args, in_tree = tree_flatten((args, kwargs))
    wf = wrap_init(f)
    in_avals, keep_inputs = list(map(shaped_abstractify, flat_args)), [True] * len(flat_args)
    in_type = tuple(zip(in_avals, keep_inputs))
    wff, out_tree_promise = flatten_fun(wf, in_tree)
    wffa = annotate(wff, in_type)
    return wffa, in_avals, out_tree_promise


