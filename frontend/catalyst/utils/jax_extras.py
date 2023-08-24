from __future__ import annotations

import collections
from collections import namedtuple
from contextlib import contextmanager
from dataclasses import dataclass
import functools
from functools import partial, partialmethod, total_ordering
import gc
import inspect
import itertools as it
import math
import operator
from operator import attrgetter
import threading
import types
from typing import (Any, Callable, ClassVar, DefaultDict, Dict, FrozenSet,
                    Generator, Generic, Hashable, Iterable, Iterator, List,
                    NamedTuple, Optional, Sequence, Set, Tuple, Type, TypeVar,
                    Union, cast, overload)
import warnings
from weakref import ref

import numpy as np

from jax._src import dtypes
from jax._src import config as jax_config
from jax._src import effects
from jax._src.config import FLAGS, config
from jax._src.errors import (
    ConcretizationTypeError, TracerArrayConversionError,
    TracerIntegerConversionError, UnexpectedTracerError)
from jax._src import linear_util as lu

from jax._src import source_info_util
from jax._src.util import (safe_zip, safe_map, curry, tuple_insert,
                           tuple_delete, as_hashable_function,
                           HashableFunction, HashableWrapper, weakref_lru_cache,
                           partition_list)
import jax._src.pretty_printer as pp
from jax._src.lib import jax_jit
from jax._src import traceback_util
from jax._src.typing import Array, DimSize, Shape
from jax._src import typing

from jax._src.core import thread_local_state, Trace, MainTrace, _update_thread_local_jit_state
from jax._src.util import wrap_name, toposort
from jax._src.lax.lax import _abstractify
from jax._src import state
from jax._src import core
from jax._src import util
from jax._src.util import (cache, weakref_lru_cache, safe_map, unzip3,
                           partition_list)
from jax._src.interpreters import partial_eval as pe
from jax._src.lax.control_flow.common import _initial_style_open_jaxpr

map, unsafe_map = safe_map, map


@contextmanager
def new_main2(trace_type: Type[Trace],
              dynamic: bool = False,
              main:Optional[MainTrace(level, trace_type, **payload)] = None,
              **payload) -> Generator[MainTrace, None, None]:
  """ A verison of JAX `new_main` function that knows how to re-use an already existing `MainTrace`
  object """

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


def sort_eqns(eqns:List[JaxprEqn])->List[JaxprEqn]:
    """ Topologically sort JAXRR equations in a list, based on their input/output variables. """
    # FIXME: The functions might emit different correct results, depending on id(eqns). One need to
    # make this function stable. Moreover, some equation (`qdevice` ones) do not depend on others
    # but need to be at the top of the output. We force this order for now. Stable sorting might
    # also allow us to remove this conditioning.
    class Box:
        def __init__(self, e):
            self.e:JaxprEqn = e
            self.parents:Set["Box"] = {}
    boxes = [Box(e) for e in eqns]
    qdevices = [b for b in boxes if b.e.primitive.name == "qdevice"]
    origin:Dict[int,Box] = {}
    for b in boxes:
        origin.update({ov.count:b for ov in b.e.outvars})
    for b in boxes:
        b.parents = {origin[v.count] for v in b.e.invars if v.count in origin}
    for b in boxes:
        if b not in qdevices:
            b.parents.update(qdevices)
    return [b.e for b in toposort(boxes)]


def initial_style_jaxprs_with_common_consts1(
        funs: Sequence[Callable], in_tree, in_avals, primitive_name: str):
  """ This function is the head (shorter) part of the original
  `lax.control_flow.common._initial_style_jaxprs_with_common_consts` of JAX. The algorithm is the
  same at the time of this writing, we use this function only to avoid conflicts with future
  versions of JAX.
  """
  jaxprs, all_consts, all_out_trees = \
      unzip3(_initial_style_open_jaxpr(fun, in_tree, in_avals, primitive_name)
             for fun in funs)
  closed_jaxprs, consts = initial_style_jaxprs_with_common_consts2(jaxprs, all_consts)
  return closed_jaxprs, consts, all_out_trees

def initial_style_jaxprs_with_common_consts2(jaxprs, all_consts):
  """ This function is the tail (largest) part of the
  `lax.control_flow.common._initial_style_jaxprs_with_common_consts` of JAX. The JAX version traces
  argument Python functions in order to determine signatures to be unified. Here we rely on the fact
  that the tracing was already done elsewhere - and this is the only difference.
  """

  all_const_avals = [map(_abstractify, consts) for consts in all_consts]
  for consts, consts_avals in zip(all_consts, all_const_avals):
    for c, aval in zip(consts, consts_avals):
      if isinstance(aval, state.AbstractRef):
        assert isinstance(c, pe.DynamicJaxprTracer)
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

  newvar = core.gensym(jaxprs, suffix='_')
  unused_ref_const_vars = map(newvar, canonical_ref_avals)
  unused_const_vars = [map(newvar, const_avals)
                       for const_avals in all_nonref_const_avals]
  def pad_jaxpr_constvars(i, jaxpr):
    is_ref = [isinstance(v.aval, state.AbstractRef) for v in jaxpr.constvars]
    nonref_constvars, ref_constvars = partition_list(is_ref, jaxpr.constvars)
    padded_ref_constvars = unused_ref_const_vars[:]
    for canonical_id, ref_var in zip(canonical_ref_indices[i], ref_constvars):
      padded_ref_constvars[canonical_id] = ref_var
    const_prefix = util.concatenate(unused_const_vars[:i])
    const_suffix = util.concatenate(unused_const_vars[i + 1:])
    constvars = [*padded_ref_constvars, *const_prefix, *nonref_constvars,
                 *const_suffix]
    jaxpr = jaxpr.replace(constvars=constvars)
    effects = pe.make_jaxpr_effects(jaxpr.constvars, jaxpr.invars,
                                    jaxpr.outvars, jaxpr.eqns)
    jaxpr = jaxpr.replace(effects=effects)
    return jaxpr

  consts = [*canonical_refs, *util.concatenate(all_nonref_consts)]
  jaxprs = tuple(pad_jaxpr_constvars(i, jaxpr) for i, jaxpr in enumerate(jaxprs))
  closed_jaxprs = [core.ClosedJaxpr(pe.convert_constvars_jaxpr(jaxpr), ())
                   for jaxpr in jaxprs]
  return closed_jaxprs, consts


