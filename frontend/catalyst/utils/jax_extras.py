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

@contextmanager
def new_main2(trace_type: Type[Trace],
              dynamic: bool = False,
              main:Optional[MainTrace(level, trace_type, **payload)] = None,
              **payload) -> Generator[MainTrace, None, None]:
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
    class Box:
        def __init__(self, e):
            self.e = e
            self.parents = {}
    boxes = [Box(e) for e in eqns]
    origin:Dict[int,Box] = {}
    for b in boxes:
        origin.update({ov.count:b for ov in b.e.outvars})
    for b in boxes:
        b.parents = {origin[v.count] for v in b.e.invars if v.count in origin}
    return [b.e for b in toposort(boxes)]

