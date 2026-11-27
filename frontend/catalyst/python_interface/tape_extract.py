# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A compile-time, one-way exit from a ``qjit``-compiled program to a PennyLane tape.

The entry point is :func:`flatten` (exposed in PennyLane as ``qp.flatten``). The program is
captured with all of its arguments treated as compile-time constants, lowered to MLIR, and run
through the quantum compilation stage of its ``CompilePipeline`` (e.g. ``cancel_inverses``,
``merge_rotations``). The resulting MLIR is then *evaluated at compile time* by
:class:`_TapeInterpreter`: classical values are computed concretely, loops are unrolled,
branches on known values are resolved, and branches conditioned on mid-circuit measurement
outcomes become :class:`~.Conditional` operations tied to their :class:`~.MidMeasure`. The
quantum instructions, the terminal measurements and the shots are collected into a
:class:`~.QuantumScript`.

Anything that cannot be resolved at compile time (e.g. a ``while`` loop whose condition depends on
a mid-circuit measurement), or that has no PennyLane analogue (e.g. operations of the ``pbc``
dialect after ``to_ppr``) raises a :class:`FlattenError`.
"""

from __future__ import annotations

import contextlib
import copy
import functools
import math as pymath
import uuid
from collections.abc import Callable
from typing import Any

import numpy as np
import pennylane as qp
from pennylane.ops import MidMeasure
from pennylane.ops.mid_measure import MeasurementValue
from pennylane.tape import QuantumScript
from xdsl.dialects.builtin import (
    ArrayAttr,
    BoolAttr,
    ComplexType,
    DenseArrayBase,
    DenseIntOrFPElementsAttr,
    FloatAttr,
    IndexType,
    IntegerAttr,
    IntegerType,
    ModuleOp,
    StringAttr,
    SymbolRefAttr,
    TensorType,
    UnregisteredOp,
)
from xdsl.ir import Block, Operation, Region, SSAValue

from catalyst.utils.exceptions import CompileError

__all__ = ["FlattenError", "flatten", "tape_from_mlir"]

# Upper bound for the number of iterations of a single ``while`` loop that is unrolled at compile
# time. It guards against programs that never terminate.
MAX_WHILE_ITERATIONS = 100_000


class FlattenError(CompileError):
    """Raised when a compiled program cannot be represented as a flat PennyLane tape."""


######################################################
### Public API
######################################################


def flatten(qjit_fn) -> Callable[..., QuantumScript]:
    """Return a function that produces the compiled tape of a ``qjit``-compiled program.

    The returned function takes the same arguments as ``qjit_fn``. All of them (positional and
    keyword) are treated as static, compile-time constants. The program is lowered to MLIR, its
    ``CompilePipeline`` is applied, and the compiled MLIR is converted into a
    :class:`~.QuantumScript` holding the quantum instructions, the terminal measurements and the
    shots. Execution configuration such as the MCM method or the differentiation method is not
    part of the tape.

    The result is a one-way exit point: it can be transformed, executed and drawn with the
    PennyLane (classic) tape tooling, but it cannot be fed back into ``qjit``.

    Args:
        qjit_fn (QJIT): a function decorated with :func:`~.qjit` that executes a single QNode.

    Returns:
        Callable[..., QuantumScript]: a function returning the compiled tape.

    Raises:
        FlattenError: if the compiled program contains dynamic behaviour that cannot be resolved at
            compile time, or instructions without a PennyLane analogue.

    **Example**

    .. code-block:: python

        @qp.qjit(capture=True)
        @qp.transforms.cancel_inverses
        @qp.qnode(qp.device("lightning.qubit", wires=2))
        def circuit(n):
            qp.H(0)
            qp.H(0)

            @qp.for_loop(0, n)
            def loop(i):
                qp.RX(0.1 * i, 1)

            loop()
            return qp.expval(qp.Z(1))

    >>> tape = qp.flatten(circuit)(3)
    >>> tape.operations
    [RX(0.0, wires=[1]), RX(0.1, wires=[1]), RX(0.2, wires=[1])]
    """
    # pylint: disable-next=import-outside-toplevel
    from catalyst.jit import QJIT

    if not isinstance(qjit_fn, QJIT):
        raise TypeError(
            f"flatten can only be applied to a function decorated with qjit, got {qjit_fn!r}."
        )

    @functools.wraps(qjit_fn)
    def wrapper(*args, **kwargs) -> QuantumScript:
        module = _compile_to_quantum_stage(qjit_fn, args, kwargs)
        return tape_from_mlir(module)

    return wrapper


def tape_from_mlir(mlir: str | ModuleOp, *, wire_labels=None) -> QuantumScript:
    """Convert compiled MLIR (value-semantics ``quantum`` dialect) into a flat tape.

    Args:
        mlir (str | xdsl.dialects.builtin.ModuleOp): the module, either as a generic-form MLIR
            string or as a parsed xDSL module. Its first public function is the entry point.
        wire_labels (Sequence | None): device wire labels, indexed by the qubit indices of the IR.

    Returns:
        QuantumScript: the tape.
    """
    if isinstance(mlir, str):
        # pylint: disable-next=import-outside-toplevel
        from catalyst.python_interface.conversion import parse_generic_to_xdsl_module

        mlir = parse_generic_to_xdsl_module(mlir)

    return _TapeInterpreter(mlir, wire_labels=wire_labels).run()


######################################################
### Compilation
######################################################


def _compile_to_quantum_stage(qjit_fn, args, kwargs) -> ModuleOp:
    """Capture ``qjit_fn`` with static ``args``/``kwargs`` and apply its quantum compilation
    stage. Returns the resulting xDSL module."""
    # pylint: disable=import-outside-toplevel
    from catalyst.compiler import to_mlir_opt
    from catalyst.python_interface.conversion import parse_generic_to_xdsl_module

    try:
        from catalyst.device.python_device import disable_bridge_rewrites
    except ImportError:  # the Python device bridge is not available
        disable_bridge_rewrites = contextlib.nullcontext

    user_function = qjit_fn.user_function

    # All arguments become compile-time constants: the traced function takes no arguments and
    # closes over the concrete values. Unlike ``static_argnums`` this also works for unhashable
    # values such as arrays.
    @functools.wraps(user_function)
    def static_program():
        return user_function(*args, **kwargs)

    # Work on a shallow copy so that the user's QJIT object (its options, cached compiled
    # functions and IR) is left untouched.
    tracer = copy.copy(qjit_fn)
    tracer.user_function = static_program
    tracer.compile_options = copy.deepcopy(qjit_fn.compile_options)
    tracer.compile_options.static_argnums = ()
    tracer.compile_options.static_argnames = None
    tracer.jaxpr = None
    tracer.mlir_module = None
    tracer._placement_cache = None  # pylint: disable=protected-access

    # Decomposition rules are only needed by graph-based decomposition passes; compiling them is
    # expensive, so only do it when the pipeline actually contains such a pass.
    user_collect = tracer.compile_options.collect_decomp_rules
    tracer.compile_options.collect_decomp_rules = False
    with disable_bridge_rewrites():
        mlir_module = _generate_ir(tracer)
        if user_collect and _uses_decomposition_pass(mlir_module):
            tracer.compile_options.collect_decomp_rules = True
            mlir_module = _generate_ir(tracer)

    # The device's own transform sequence (e.g. the ``dynamic-one-shot`` MCM method) belongs to
    # the execution configuration, which is not part of the tape.
    _clear_transform_sequence(mlir_module, "__transform_device")

    options = copy.deepcopy(tracer.compile_options)
    options.pipelines = [_quantum_compilation_stage(options)]
    options.lower_to_llvm = False
    options.keep_intermediate = False
    options.checkpoint_stage = ""

    using_python_compiler = tracer.compiler.is_using_python_compiler(mlir_module)
    ir = mlir_module.operation.get_asm(
        binary=False, print_generic_op_form=True, assume_verified=True
    )
    compiled = to_mlir_opt(
        "--mlir-print-op-generic",
        stdin=ir,
        options=options,
        using_python_compiler=using_python_compiler,
        stderr_return=False,
    )
    return parse_generic_to_xdsl_module(compiled)


def _generate_ir(tracer):
    tracer.jaxpr, *_ = tracer.capture(())
    return tracer.generate_ir()


def _quantum_compilation_stage(options):
    """The stage of the pipeline that applies the quantum ``CompilePipeline``."""
    stages = options.get_stages()
    for stage in stages:
        if stage[0] == "QuantumCompilationStage":
            return stage
    return stages[0]


def _transform_sequences(mlir_module):
    """Yield all ``transform.named_sequence`` operations of a jaxlib MLIR module."""
    stack = [mlir_module.operation]
    while stack:
        op = stack.pop()
        if op.name == "transform.named_sequence":
            yield op
        for region in op.regions:
            for block in region.blocks:
                stack.extend(o.operation for o in block.operations)


def _uses_decomposition_pass(mlir_module) -> bool:
    for seq in _transform_sequences(mlir_module):
        for block in seq.regions[0].blocks:
            for op in block.operations:
                if op.operation.name == "transform.apply_registered_pass":
                    name = str(op.operation.attributes["pass_name"])
                    if "decompos" in name:
                        return True
    return False


def _clear_transform_sequence(mlir_module, sym_name):
    for seq in _transform_sequences(mlir_module):
        if str(seq.attributes["sym_name"]).strip('"') != sym_name:
            continue
        for block in seq.regions[0].blocks:
            ops = [op.operation for op in block.operations][:-1]  # keep the terminator
            for op in reversed(ops):
                op.erase()


######################################################
### Interpreter values
######################################################


class _Dynamic:
    """A classical value that is not known at compile time."""

    def __init__(self, reason: str):
        self.reason = reason

    def __repr__(self):
        return f"<dynamic: {self.reason}>"


class _Qubit:
    """A qubit SSA value, identified by its (device) wire index."""

    __slots__ = ("wire",)

    def __init__(self, wire: int):
        self.wire = wire


class _Register:
    """A quantum register SSA value."""

    __slots__ = ("size",)

    def __init__(self, size: int):
        self.size = size


class _CompBasis:
    """The ``quantum.compbasis`` pseudo-observable. ``wires is None`` means all wires."""

    def __init__(self, wires):
        self.wires = wires


class _MCMObs:
    """The ``quantum.mcmobs`` pseudo-observable (statistics of mid-circuit measurements)."""

    def __init__(self, mvs):
        self.mvs = mvs


class _FuncReturn(Exception):
    """Internal control flow: ``func.return`` reached."""

    def __init__(self, values):
        super().__init__()
        self.values = values


def _is_symbolic(value) -> bool:
    return isinstance(value, (_Dynamic, MeasurementValue))


######################################################
### Helpers
######################################################


def _name(op: Operation) -> str:
    if isinstance(op, UnregisteredOp):
        return op.op_name.data
    return op.name


def _attr(op: Operation, name: str):
    attr = op.properties.get(name)
    if attr is None:
        attr = op.attributes.get(name)
    return attr


def _int_array_attr(attr) -> list[int]:
    if attr is None:
        return []
    if isinstance(attr, DenseArrayBase):
        return [int(v) for v in attr.get_values()]
    if isinstance(attr, DenseIntOrFPElementsAttr):
        return [int(v) for v in attr.get_values()]
    if isinstance(attr, ArrayAttr):
        return [int(a.value.data) for a in attr.data]
    raise FlattenError(f"Unexpected attribute {attr}.")


def _int_attr(attr) -> int:
    if isinstance(attr, IntegerAttr):
        return int(attr.value.data)
    raise FlattenError(f"Unexpected attribute {attr}.")


def _dtype_of(xtype):
    """Numpy dtype of an MLIR element or tensor type."""
    if isinstance(xtype, TensorType):
        xtype = xtype.element_type
    if isinstance(xtype, IndexType):
        return np.int64
    if isinstance(xtype, IntegerType):
        width = xtype.width.data
        if width == 1:
            return np.bool_
        return {8: np.int8, 16: np.int16, 32: np.int32, 64: np.int64}.get(width, np.int64)
    if isinstance(xtype, ComplexType):
        return np.complex128 if _dtype_of(xtype.element_type) == np.float64 else np.complex64
    name = getattr(xtype, "name", "")
    return {"f16": np.float16, "f32": np.float32, "f64": np.float64}.get(name, np.float64)


def _shape_of(xtype):
    if isinstance(xtype, TensorType):
        return tuple(int(d) for d in xtype.get_shape())
    return ()


def _as_array(value, xtype) -> np.ndarray:
    return np.asarray(value, dtype=_dtype_of(xtype)).reshape(_shape_of(xtype))


def _to_python_scalar(value):
    """Convert a concrete 0-d (or single element) array to a Python number."""
    arr = np.asarray(value)
    if arr.size != 1:
        raise FlattenError(f"Expected a scalar value, got an array of shape {arr.shape}.")
    item = arr.reshape(()).item()
    return item


def _dense_attr_values(attr, xtype) -> np.ndarray:
    if isinstance(attr, DenseIntOrFPElementsAttr):
        if isinstance(attr.get_element_type(), ComplexType):
            values = [complex(*v) if isinstance(v, tuple) else complex(v) for v in attr.get_values()]
        else:
            values = list(attr.get_values())
        dtype = _dtype_of(xtype)
        shape = _shape_of(xtype)
        arr = np.asarray(values, dtype=dtype)
        if arr.size == 1 and int(np.prod(shape, dtype=int)) != 1:
            return np.full(shape, arr.reshape(()), dtype=dtype)  # splat
        return arr.reshape(shape)
    if isinstance(attr, IntegerAttr):
        return np.asarray(attr.value.data, dtype=_dtype_of(xtype))
    if isinstance(attr, FloatAttr):
        return np.asarray(attr.value.data, dtype=_dtype_of(xtype))
    if isinstance(attr, BoolAttr):
        return np.asarray(bool(attr.value.data))
    raise FlattenError(f"Unsupported constant attribute: {attr}")


def _comparison(op: Operation) -> str:
    for key in ("comparison_direction", "predicate"):
        attr = _attr(op, key)
        if attr is None:
            continue
        if isinstance(attr, IntegerAttr):  # arith predicates
            return _int_attr(attr)
        data = getattr(attr, "data", None)
        text = str(getattr(data, "value", data if data is not None else attr))
        for direction in ("EQ", "NE", "GE", "GT", "LE", "LT"):
            if direction in text.upper().split()[-1] or text.upper().endswith(direction):
                return direction
    raise FlattenError(f"Could not determine the comparison direction of {_name(op)}.")


_COMPARE = {
    "EQ": np.equal,
    "NE": np.not_equal,
    "LT": np.less,
    "LE": np.less_equal,
    "GT": np.greater,
    "GE": np.greater_equal,
}

# arith.cmpi / arith.cmpf predicates, by integer value
_CMPI = ["EQ", "NE", "LT", "LE", "GT", "GE", "LT", "LE", "GT", "GE"]
_CMPF = [
    None,
    "EQ",
    "GT",
    "GE",
    "LT",
    "LE",
    "NE",
    None,
    None,
    "EQ",
    "GT",
    "GE",
    "LT",
    "LE",
    "NE",
    None,
]


def _region_has_quantum(region_or_op) -> bool:
    for op in region_or_op.walk():
        name = _name(op)
        if name.startswith(("quantum.", "qref.", "pbc.", "mbqc.", "qec")):
            return True
        if name in ("func.call", "catalyst.launch_kernel"):
            return True
    return False


######################################################
### Classical op evaluation (pure numpy functions)
######################################################

_UNARY = {
    "abs": np.abs,
    "negate": np.negative,
    "negf": np.negative,
    "sine": np.sin,
    "sin": np.sin,
    "cosine": np.cos,
    "cos": np.cos,
    "tan": np.tan,
    "tanh": np.tanh,
    "exponential": np.exp,
    "exp": np.exp,
    "exponential_minus_one": np.expm1,
    "log": np.log,
    "log_plus_one": np.log1p,
    "sqrt": np.sqrt,
    "rsqrt": lambda x: 1 / np.sqrt(x),
    "cbrt": np.cbrt,
    "floor": np.floor,
    "ceil": np.ceil,
    "sign": np.sign,
    "is_finite": np.isfinite,
    "real": np.real,
    "imag": np.imag,
    "round_nearest_even": np.round,
    "round_nearest_afz": lambda x: np.sign(x) * np.floor(np.abs(x) + 0.5),
    "logistic": lambda x: 1 / (1 + np.exp(-x)),
    "atan": np.arctan,
    "asin": np.arcsin,
    "acos": np.arccos,
}


def _not(x):
    x = np.asarray(x)
    return np.logical_not(x) if x.dtype == np.bool_ else np.invert(x)


_BINARY = {
    "add": np.add,
    "addi": np.add,
    "addf": np.add,
    "subtract": np.subtract,
    "subi": np.subtract,
    "subf": np.subtract,
    "multiply": np.multiply,
    "muli": np.multiply,
    "mulf": np.multiply,
    "divf": np.divide,
    "power": np.power,
    "powf": np.power,
    "maximum": np.maximum,
    "maxsi": np.maximum,
    "maxui": np.maximum,
    "maximumf": np.maximum,
    "maxnumf": np.maximum,
    "minimum": np.minimum,
    "minsi": np.minimum,
    "minui": np.minimum,
    "minimumf": np.minimum,
    "minnumf": np.minimum,
    "and": np.bitwise_and,
    "andi": np.bitwise_and,
    "or": np.bitwise_or,
    "ori": np.bitwise_or,
    "xor": np.bitwise_xor,
    "xori": np.bitwise_xor,
    "atan2": np.arctan2,
    "complex": lambda a, b: a + 1j * b,
    "shift_left": np.left_shift,
    "shli": np.left_shift,
}


def _divide(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if np.issubdtype(a.dtype, np.integer):
        return np.trunc(a / b).astype(a.dtype)
    return a / b


def _remainder(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return np.fmod(a, b)


def _reduce_fn(body: Block) -> Callable | None:
    """Recognize the reduction computed by the body of a ``stablehlo.reduce``."""
    ops = [op for op in body.ops if _name(op) not in ("stablehlo.return",)]
    if len(ops) != 1:
        return None
    kind = _name(ops[0]).split(".")[-1]
    return {
        "add": np.add,
        "multiply": np.multiply,
        "maximum": np.maximum,
        "minimum": np.minimum,
        "and": np.logical_and,
        "or": np.logical_or,
    }.get(kind)


def _eval_classical(op: Operation, operands: list) -> list | None:
    """Evaluate a pure classical operation on concrete numpy operands.

    Returns ``None`` if the operation is not a known classical operation.
    """
    # pylint: disable=too-many-return-statements,too-many-branches,too-many-statements
    name = _name(op)
    dialect, _, kind = name.partition(".")
    res_types = [r.type for r in op.results]

    def out(*values):
        return [_as_array(v, t) for v, t in zip(values, res_types)]

    if dialect not in ("arith", "stablehlo", "tensor", "math", "chlo", "index"):
        return None

    if kind == "constant":
        return [_dense_attr_values(_attr(op, "value"), res_types[0])]

    if kind in _UNARY and len(operands) == 1:
        return out(_UNARY[kind](operands[0]))
    if kind in ("not",):
        return out(_not(operands[0]))
    if kind in _BINARY and len(operands) == 2:
        return out(_BINARY[kind](operands[0], operands[1]))
    if kind in ("divide", "divsi", "divui", "floordivsi", "ceildivsi"):
        if kind == "floordivsi":
            return out(np.floor_divide(operands[0], operands[1]))
        if kind == "ceildivsi":
            return out(-np.floor_divide(-np.asarray(operands[0]), operands[1]))
        return out(_divide(operands[0], operands[1]))
    if kind in ("remainder", "remsi", "remui", "remf"):
        return out(_remainder(operands[0], operands[1]))

    if kind in (
        "convert",
        "index_cast",
        "index_castui",
        "sitofp",
        "uitofp",
        "fptosi",
        "fptoui",
        "extui",
        "extsi",
        "trunci",
        "extf",
        "truncf",
        "bitcast_convert",
        "reshape",
    ):
        return out(np.asarray(operands[0]).reshape(_shape_of(res_types[0])))

    if kind in ("compare", "cmpi", "cmpf"):
        direction = _comparison(op)
        if isinstance(direction, int):
            direction = (_CMPI if kind == "cmpi" else _CMPF)[direction]
            if direction is None:
                raise FlattenError(f"Unsupported predicate for {name}.")
        return out(_COMPARE[direction](operands[0], operands[1]))

    if kind == "select":
        return out(np.where(operands[0], operands[1], operands[2]))

    if kind == "clamp":
        return out(np.clip(operands[1], operands[0], operands[2]))

    if kind == "broadcast_in_dim":
        dims = _int_array_attr(_attr(op, "broadcast_dimensions"))
        shape = _shape_of(res_types[0])
        src = np.asarray(operands[0])
        expanded = [1] * len(shape)
        for i, d in enumerate(dims):
            expanded[d] = src.shape[i]
        return out(np.broadcast_to(src.reshape(expanded), shape))

    if kind == "slice":
        start = _int_array_attr(_attr(op, "start_indices"))
        limit = _int_array_attr(_attr(op, "limit_indices"))
        strides = _int_array_attr(_attr(op, "strides")) or [1] * len(start)
        index = tuple(slice(s, l, st) for s, l, st in zip(start, limit, strides))
        return out(np.asarray(operands[0])[index])

    if kind == "dynamic_slice":
        src = np.asarray(operands[0])
        sizes = _int_array_attr(_attr(op, "slice_sizes"))
        starts = [int(np.clip(_to_python_scalar(s), 0, src.shape[i] - sizes[i])) for i, s in enumerate(operands[1:])]
        index = tuple(slice(s, s + n) for s, n in zip(starts, sizes))
        return out(src[index])

    if kind == "dynamic_update_slice":
        src = np.array(operands[0])
        upd = np.asarray(operands[1])
        starts = [int(np.clip(_to_python_scalar(s), 0, src.shape[i] - upd.shape[i])) for i, s in enumerate(operands[2:])]
        index = tuple(slice(s, s + n) for s, n in zip(starts, upd.shape))
        src[index] = upd
        return out(src)

    if kind == "concatenate":
        dim = _int_attr(_attr(op, "dimension"))
        return out(np.concatenate([np.asarray(o) for o in operands], axis=dim))

    if kind == "transpose":
        perm = _int_array_attr(_attr(op, "permutation"))
        return out(np.transpose(operands[0], perm))

    if kind == "iota":
        dim = _int_attr(_attr(op, "iota_dimension"))
        shape = _shape_of(res_types[0])
        idx = np.arange(shape[dim]).reshape([-1 if i == dim else 1 for i in range(len(shape))])
        return out(np.broadcast_to(idx, shape))

    if kind == "reverse":
        dims = _int_array_attr(_attr(op, "dimensions"))
        return out(np.flip(operands[0], axis=tuple(dims)))

    if kind == "dot_general" or kind == "dot":
        a, b = np.asarray(operands[0]), np.asarray(operands[1])
        if kind == "dot" or (a.ndim <= 2 and b.ndim <= 2):
            return out(np.dot(a, b))
        return None

    if kind == "reduce":
        n = len(op.results)
        fn = _reduce_fn(op.regions[0].blocks[0])
        if fn is None or n != 1:
            return None
        dims = tuple(_int_array_attr(_attr(op, "dimensions")))
        src, init = np.asarray(operands[0]), np.asarray(operands[1])
        return out(fn.reduce(src, axis=dims, initial=init.reshape(())))

    if dialect == "tensor":
        if kind == "extract":
            src = np.asarray(operands[0])
            idx = tuple(int(_to_python_scalar(i)) for i in operands[1:])
            return out(src[idx] if idx else src.reshape(()))
        if kind == "from_elements":
            return out(np.asarray([_to_python_scalar(o) for o in operands]))
        if kind == "insert":
            dest = np.array(operands[1])
            idx = tuple(int(_to_python_scalar(i)) for i in operands[2:])
            dest[idx] = _to_python_scalar(operands[0])
            return out(dest)
        if kind == "empty":
            return out(np.zeros(_shape_of(res_types[0])))
        if kind == "splat":
            return out(np.full(_shape_of(res_types[0]), _to_python_scalar(operands[0])))
        if kind == "cast":
            return out(operands[0])
        if kind == "dim":
            return out(np.asarray(operands[0]).shape[int(_to_python_scalar(operands[1]))])
        return None

    return None


def _lift_to_measurement_value(fn: Callable, operands: list, n_results: int):
    """Evaluate a pure function whose operands include ``MeasurementValue``s.

    The result is a ``MeasurementValue`` over all mid-circuit measurements involved, whose
    processing function applies ``fn`` branch-wise.
    """
    measurements = []
    for operand in operands:
        if isinstance(operand, MeasurementValue):
            for m in operand.measurements:
                if m not in measurements:
                    measurements.append(m)
    measurements.sort(key=lambda m: m.meas_uid)

    def branch_values(bits):
        concrete = dict(zip(measurements, bits))
        args = [
            np.asarray(o.concretize(concrete)) if isinstance(o, MeasurementValue) else o
            for o in operands
        ]
        return fn(args)

    # Only scalar results can be carried by a MeasurementValue
    probe = branch_values((0,) * len(measurements))
    if probe is None or any(np.asarray(v).size != 1 for v in probe):
        return None

    results = []
    for i in range(n_results):

        def processing_fn(*bits, _i=i):
            return _to_python_scalar(branch_values(bits)[_i])

        results.append(_simplify_mv(MeasurementValue(list(measurements), processing_fn)))
    return results


# Raw measurement values, by the ``meas_uid`` of their mid-circuit measurement
_RAW_MEASUREMENT_VALUES: dict[str, MeasurementValue] = {}


def _simplify_mv(mv: MeasurementValue) -> MeasurementValue:
    """Return the raw measurement value if ``mv`` is the identity function of one measurement.

    Lowering ``qp.cond(m, ...)`` introduces casts and comparisons (e.g. ``m != 0``); mapping them
    back to ``m`` itself gives the same tape as the non-compiled PennyLane pathway.
    """
    if len(mv.measurements) == 1:
        raw = _RAW_MEASUREMENT_VALUES.get(mv.measurements[0].meas_uid)
        branches = mv.branches
        if raw is not None and all(
            not isinstance(v, float) and v == b[0] for b, v in branches.items()
        ):
            return raw
    return mv


######################################################
### Tape interpreter
######################################################


# pylint: disable=too-many-instance-attributes
class _TapeInterpreter:
    """Evaluates a compiled module at compile time, collecting the quantum instructions."""

    def __init__(self, module: ModuleOp, wire_labels=None):
        self.module = module
        self.wire_labels = wire_labels
        self.operations = []
        self.measurements = []
        self.shots = None
        self.seen_device = False
        self.device_released = False
        self.conditions: list[MeasurementValue] = []
        # ``scf.if`` operations that implement the reset of a mid-circuit measurement
        self.resets: set[Operation] = set()
        self._symbols = self._collect_symbols(module)

    # --------------------------------------------------------------- entry
    def run(self) -> QuantumScript:
        _RAW_MEASUREMENT_VALUES.clear()
        try:
            return self._run()
        finally:
            _RAW_MEASUREMENT_VALUES.clear()

    def _run(self) -> QuantumScript:
        entry = self._entry_point()
        n_args = len(entry.body.blocks[0].args)
        if n_args:
            raise FlattenError(
                "The entry point of the program has dynamic arguments; flatten requires all "
                "arguments to be static."
            )
        self.call_function(entry, [])
        if not self.seen_device:
            raise FlattenError("The program does not execute a QNode.")
        return QuantumScript(self.operations, self.measurements, shots=self.shots)

    def _entry_point(self):
        for op in self.module.body.ops:
            if _name(op) == "func.func" and _is_public(op):
                return op
        raise FlattenError("Could not find the entry point of the program.")

    @staticmethod
    def _collect_symbols(module) -> dict:
        symbols = {}

        def visit(mod, prefix):
            for op in mod.body.ops:
                if isinstance(op, ModuleOp):
                    sym = op.properties.get("sym_name") or op.attributes.get("sym_name")
                    visit(op, prefix + ((sym.data,) if sym is not None else ()))
                elif _name(op) == "func.func":
                    sym = _attr(op, "sym_name").data
                    symbols[prefix + (sym,)] = op
                    symbols.setdefault((sym,), op)

        visit(module, ())
        return symbols

    def _lookup(self, ref: SymbolRefAttr):
        path = (ref.root_reference.data,) + tuple(r.data for r in ref.nested_references.data)
        func = self._symbols.get(path) or self._symbols.get(path[-1:])
        if func is None:
            raise FlattenError(f"Could not find function {'::'.join(path)}.")
        return func

    # --------------------------------------------------------------- execution
    def call_function(self, func_op, args):
        block = func_op.body.blocks[0]
        env = dict(zip(block.args, args))
        try:
            self.run_block(block, env)
        except _FuncReturn as ret:
            return ret.values
        return []

    def run_region(self, region: Region, env: dict, args=()):
        """Run a single-block region, returning the operands of its terminator."""
        block = region.blocks[0]
        local = dict(env)
        local.update(zip(block.args, args))
        return self.run_block(block, local)

    def run_block(self, block: Block, env: dict):
        for op in block.ops:
            name = _name(op)
            if name in ("scf.yield", "scf.condition", "stablehlo.return", "quantum.yield"):
                return [self.get(env, v) for v in op.operands]
            if name == "func.return":
                raise _FuncReturn([self.get(env, v) for v in op.operands])
            results = self.run_op(op, env)
            if results is None:
                results = [_Dynamic(f"result of '{name}'")] * len(op.results)
            if len(results) != len(op.results):
                raise FlattenError(f"Internal error while evaluating {name}.")
            env.update(zip(op.results, results))
        return []

    @staticmethod
    def get(env, ssa: SSAValue):
        try:
            return env[ssa]
        except KeyError as e:
            raise FlattenError(f"Undefined value {ssa} in compiled program.") from e

    # pylint: disable=too-many-return-statements,too-many-branches
    def run_op(self, op: Operation, env: dict):
        name = _name(op)
        operands = [self.get(env, v) for v in op.operands]

        handler = _HANDLERS.get(name)
        if handler is not None:
            return handler(self, op, operands, env)

        dialect = name.split(".")[0]
        if dialect in ("quantum", "qref", "pbc", "mbqc", "qecl", "qecp", "ion", "mitigation"):
            raise FlattenError(
                f"The compiled program contains the operation '{name}', which has no PennyLane "
                "analogue and cannot be represented in a tape."
            )

        if any(isinstance(o, (_Qubit, _Register)) for o in operands):
            raise FlattenError(
                f"The compiled program contains the operation '{name}' acting on qubits, which "
                "cannot be represented in a tape."
            )

        return self.eval_pure(op, operands)

    def eval_pure(self, op, operands):
        """Evaluate a side-effect free classical op, propagating symbolic values."""
        dynamic = next((o for o in operands if isinstance(o, _Dynamic)), None)
        if dynamic is not None:
            return [dynamic] * len(op.results)

        if op.regions and _name(op) != "stablehlo.reduce":
            if _region_has_quantum(op):
                raise FlattenError(
                    f"The compiled program contains the operation '{_name(op)}' with quantum "
                    "instructions in its regions, which cannot be represented in a tape."
                )
            return [_Dynamic(f"result of '{_name(op)}'")] * len(op.results)

        if any(isinstance(o, MeasurementValue) for o in operands):
            lifted = _lift_to_measurement_value(
                lambda args: _eval_classical(op, args), operands, len(op.results)
            )
            if lifted is None:
                return [_Dynamic("value derived from a mid-circuit measurement")] * len(
                    op.results
                )
            return lifted

        return _eval_classical(op, operands)

    # --------------------------------------------------------------- quantum helpers
    def wire(self, qubit) -> Any:
        if not isinstance(qubit, _Qubit):
            raise FlattenError(f"Expected a qubit, got {qubit!r}.")
        if self.wire_labels is not None:
            return self.wire_labels[qubit.wire]
        return qubit.wire

    def static_scalar(self, value, what: str):
        if isinstance(value, _Dynamic):
            raise FlattenError(
                f"{what} could not be resolved at compile time ({value.reason}). All arguments "
                "are static; dynamic values can only come from runtime results."
            )
        if isinstance(value, MeasurementValue):
            raise FlattenError(
                f"{what} depends on a mid-circuit measurement outcome, which cannot be "
                "represented in a tape. Only branches (qp.cond) conditioned on mid-circuit "
                "measurements are supported."
            )
        return _to_python_scalar(value)

    def static_array(self, value, what: str):
        if _is_symbolic(value):
            self.static_scalar(value, what)
        return np.asarray(value)

    def add_operation(self, op):
        if self.device_released:
            raise FlattenError("flatten only supports programs that execute a single QNode.")
        if self.conditions:
            condition = self.conditions[0]
            for c in self.conditions[1:]:
                condition = condition & c
            op = qp.ops.Conditional(condition, op)
        self.operations.append(op)

    def add_measurement(self, mp):
        if self.conditions:
            raise FlattenError("Terminal measurements cannot be conditioned on MCM outcomes.")
        self.measurements.append(mp)


def _is_public(func_op) -> bool:
    vis = _attr(func_op, "sym_visibility")
    return vis is None or vis.data == "public"


def _make_op(fn, *args, **kwargs):
    """Construct a PennyLane operator without queuing or capturing it."""
    with qp.QueuingManager.stop_recording():
        if qp.capture.enabled():
            with qp.capture.pause():
                return fn(*args, **kwargs)
        return fn(*args, **kwargs)


######################################################
### Handlers
######################################################

_HANDLERS: dict[str, Callable] = {}


def _handles(*names):
    def decorator(fn):
        for n in names:
            _HANDLERS[n] = fn
        return fn

    return decorator


# ---------------------------------------------------------------- functions


@_handles("func.call", "catalyst.launch_kernel")
def _call(interp: _TapeInterpreter, op, operands, _env):
    func = interp._lookup(_attr(op, "callee"))  # pylint: disable=protected-access
    if func.body.blocks == () or not func.body.blocks:
        # External declaration (e.g. a runtime helper): its results are unknown
        return [_Dynamic(f"result of external function {_attr(op, 'callee')}")] * len(op.results)
    return interp.call_function(func, operands)


@_handles("catalyst.callback_call", "catalyst.custom_call", "catalyst.print", "catalyst.assert")
def _classical_side_effect(_interp, op, operands, _env):
    if any(isinstance(o, (_Qubit, _Register)) for o in operands):
        raise FlattenError(f"'{_name(op)}' acting on qubits cannot be represented in a tape.")
    return [_Dynamic(f"result of '{_name(op)}'")] * len(op.results)


# ---------------------------------------------------------------- scf control flow


@_handles("scf.for")
def _scf_for(interp: _TapeInterpreter, op, operands, env):
    lb, ub, step, *inits = operands
    body = op.regions[0]
    if any(_is_symbolic(v) for v in (lb, ub, step)):
        if _region_has_quantum(body):
            interp.static_scalar(next(v for v in (lb, ub, step) if _is_symbolic(v)), "A for loop bound")
        return [_Dynamic("result of a dynamic for loop")] * len(op.results)

    lb, ub, step = (int(_to_python_scalar(v)) for v in (lb, ub, step))
    iv_type = body.blocks[0].args[0].type
    values = inits
    for i in range(lb, ub, step):
        values = interp.run_region(body, env, [_as_array(i, iv_type), *values])
    return values


@_handles("scf.while")
def _scf_while(interp: _TapeInterpreter, op, operands, env):
    before, after = op.regions
    values = operands
    for _ in range(MAX_WHILE_ITERATIONS):
        cond, *forwarded = interp.run_region(before, env, values)
        if _is_symbolic(cond):
            if _region_has_quantum(op):
                interp.static_scalar(cond, "The condition of a while loop")
            return [_Dynamic("result of a dynamic while loop")] * len(op.results)
        if not bool(_to_python_scalar(cond)):
            return forwarded
        values = interp.run_region(after, env, forwarded)
    raise FlattenError(
        f"A while loop did not terminate after {MAX_WHILE_ITERATIONS} iterations at compile time."
    )


@_handles("scf.if")
def _scf_if(interp: _TapeInterpreter, op, operands, env):
    cond = operands[0]
    true_region, false_region = op.regions[0], op.regions[1]

    if isinstance(cond, _Dynamic):
        if _region_has_quantum(op):
            interp.static_scalar(cond, "The condition of a branch")
        return [cond] * len(op.results)

    if not isinstance(cond, MeasurementValue):
        region = true_region if bool(_to_python_scalar(cond)) else false_region
        return interp.run_region(region, env) if region.blocks else []

    if op in interp.resets:
        # Already represented by the ``reset`` of the mid-circuit measurement
        return [interp.get(env, op.operands[0].owner.out_qubit)]

    # Branch conditioned on mid-circuit measurement outcomes -> Conditional operations
    for region in (true_region, false_region):
        for inner in region.walk():
            if _name(inner) in ("quantum.measure", "qref.measure"):
                raise FlattenError(
                    "Mid-circuit measurements inside branches conditioned on mid-circuit "
                    "measurement outcomes cannot be represented in a tape."
                )

    results = []
    for region, branch_cond in ((true_region, cond), (false_region, ~cond)):
        if not region.blocks:
            results.append([])
            continue
        interp.conditions.append(_simplify_mv(branch_cond))
        try:
            results.append(interp.run_region(region, env))
        finally:
            interp.conditions.pop()

    true_vals, false_vals = results
    merged = []
    for t, f in zip(true_vals, false_vals):
        merged.append(_merge_branch_values(cond, t, f))
    return merged


def _is_reset(if_op) -> bool:
    """Whether an ``scf.if`` is the lowering of the ``reset`` option of a mid-circuit measurement.

    The branch is conditioned directly on the ``i1`` outcome of a ``quantum.measure`` (a user
    ``qp.cond`` goes through a comparison instead) and only flips the measured qubit.
    """
    if _name(if_op) != "scf.if":
        return False
    measure = if_op.operands[0].owner
    if not isinstance(measure, Operation) or _name(measure) != "quantum.measure":
        return False
    true_ops = list(if_op.regions[0].block.ops) if if_op.regions[0].blocks else []
    false_ops = list(if_op.regions[1].block.ops) if if_op.regions[1].blocks else []
    if len(true_ops) != 2 or len(false_ops) > 1 or len(if_op.results) != 1:
        return False
    flip, yield_op = true_ops
    return (
        _name(flip) == "quantum.custom"
        and _attr(flip, "gate_name").data == "PauliX"
        and not flip.params
        and not flip.in_ctrl_qubits
        and tuple(flip.in_qubits) == (measure.out_qubit,)
        and tuple(yield_op.operands) == tuple(flip.out_qubits)
        and (not false_ops or tuple(false_ops[0].operands) == (measure.out_qubit,))
    )


def _merge_branch_values(cond: MeasurementValue, t, f):
    if isinstance(t, _Qubit) and isinstance(f, _Qubit):
        if t.wire != f.wire:
            raise FlattenError("Branches conditioned on MCM outcomes yield different qubits.")
        return t
    if isinstance(t, _Register) and isinstance(f, _Register):
        return t
    if _is_symbolic(t) or _is_symbolic(f):
        return _Dynamic("value selected by a mid-circuit measurement")
    if np.array_equal(np.asarray(t), np.asarray(f)):
        return t
    lifted = _lift_to_measurement_value(
        lambda args: [np.where(args[0], args[1], args[2])], [cond, t, f], 1
    )
    return lifted[0] if lifted else _Dynamic("value selected by a mid-circuit measurement")


@_handles("scf.index_switch")
def _scf_index_switch(interp: _TapeInterpreter, op, operands, env):
    arg = operands[0]
    if _is_symbolic(arg):
        if _region_has_quantum(op):
            interp.static_scalar(arg, "The index of a switch")
        return [_Dynamic("result of a dynamic switch")] * len(op.results)
    cases = _int_array_attr(_attr(op, "cases"))
    idx = int(_to_python_scalar(arg))
    # regions: default region first, then one per case
    region = op.regions[0]
    if idx in cases:
        region = op.regions[1 + cases.index(idx)]
    return interp.run_region(region, env)


# ---------------------------------------------------------------- quantum: bookkeeping


@_handles("quantum.init", "quantum.finalize")
def _noop(_interp, _op, _operands, _env):
    return []


@_handles("quantum.device")
def _device(interp: _TapeInterpreter, op, operands, _env):
    if interp.seen_device:
        raise FlattenError("flatten only supports programs that execute a single QNode.")
    interp.seen_device = True
    shots = operands[0] if operands else 0
    shots = int(interp.static_scalar(shots, "The number of shots"))
    interp.shots = shots or None
    return []


@_handles("quantum.device_release")
def _device_release(interp: _TapeInterpreter, _op, _operands, _env):
    interp.device_released = True
    return []


@_handles("quantum.alloc")
def _alloc(interp: _TapeInterpreter, op, operands, _env):
    if _attr(op, "nqubits_attr") is not None:
        size = _int_attr(_attr(op, "nqubits_attr"))
    else:
        size = int(interp.static_scalar(operands[0], "The number of allocated qubits"))
    return [_Register(size)]


@_handles("quantum.dealloc")
def _dealloc(_interp, _op, _operands, _env):
    return []


@_handles("quantum.extract")
def _extract(interp: _TapeInterpreter, op, operands, _env):
    if _attr(op, "idx_attr") is not None:
        idx = _int_attr(_attr(op, "idx_attr"))
    else:
        idx = int(interp.static_scalar(operands[1], "A wire index"))
    return [_Qubit(idx)]


@_handles("quantum.insert")
def _insert(_interp, _op, operands, _env):
    return [operands[0]]


@_handles("quantum.alloc_qb", "quantum.dealloc_qb")
def _dynamic_alloc(_interp, op, _operands, _env):
    raise FlattenError(
        f"Dynamic qubit allocation ('{_name(op)}') cannot be represented in a tape yet."
    )


# ---------------------------------------------------------------- quantum: gates


def _gate_class(name: str):
    # pylint: disable-next=import-outside-toplevel
    from catalyst.python_interface.inspection.xdsl_conversion import from_str_to_PL_gate

    cls = from_str_to_PL_gate.get(name) or getattr(qp, name, None)
    if cls is None:
        raise FlattenError(f"The gate '{name}' has no PennyLane analogue.")
    return cls


def _segments(op, operands, *names):
    """Split flat operands according to the named operand segments of an xDSL op."""
    out = {}
    for n in names:
        vals = getattr(op, n)
        if isinstance(vals, SSAValue):
            vals = (vals,)
        elif vals is None:
            vals = ()
        out[n] = list(vals)
    return out


def _apply_modifiers(interp: _TapeInterpreter, op, gate, env):
    if _attr(op, "adjoint") is not None:
        gate = _make_op(qp.adjoint, gate)
    ctrl_qubits = [interp.get(env, v) for v in getattr(op, "in_ctrl_qubits", ())]
    if ctrl_qubits:
        ctrl_values = [
            bool(interp.static_scalar(interp.get(env, v), "A control value"))
            for v in op.in_ctrl_values
        ]
        gate = _make_op(
            qp.ctrl,
            gate,
            control=[interp.wire(q) for q in ctrl_qubits],
            control_values=ctrl_values,
        )
    interp.add_operation(gate)


def _gate_results(interp: _TapeInterpreter, op, env):
    """Out qubits carry the wires of the corresponding in qubits."""
    ins = [interp.get(env, v) for v in getattr(op, "in_qubits", ())]
    ctrls = [interp.get(env, v) for v in getattr(op, "in_ctrl_qubits", ())]
    return ins + ctrls


@_handles("quantum.custom")
def _custom(interp: _TapeInterpreter, op, _operands, env):
    name = _attr(op, "gate_name").data
    params = [interp.static_scalar(interp.get(env, p), f"A parameter of {name}") for p in op.params]
    wires = [interp.wire(interp.get(env, q)) for q in op.in_qubits]
    gate = _make_op(_gate_class(name), *params, wires=wires)
    _apply_modifiers(interp, op, gate, env)
    return _gate_results(interp, op, env)


@_handles("quantum.gphase")
def _gphase(interp: _TapeInterpreter, op, _operands, env):
    phi = interp.static_scalar(interp.get(env, op.angle), "The global phase")
    gate = _make_op(qp.GlobalPhase, phi)
    _apply_modifiers(interp, op, gate, env)
    return _gate_results(interp, op, env)


@_handles("quantum.multirz")
def _multirz(interp: _TapeInterpreter, op, _operands, env):
    theta = interp.static_scalar(interp.get(env, op.theta), "The angle of MultiRZ")
    wires = [interp.wire(interp.get(env, q)) for q in op.in_qubits]
    _apply_modifiers(interp, op, _make_op(qp.MultiRZ, theta, wires=wires), env)
    return _gate_results(interp, op, env)


@_handles("quantum.pcphase")
def _pcphase(interp: _TapeInterpreter, op, _operands, env):
    theta = interp.static_scalar(interp.get(env, op.theta), "The angle of PCPhase")
    dim = _int_attr(_attr(op, "dim"))
    wires = [interp.wire(interp.get(env, q)) for q in op.in_qubits]
    _apply_modifiers(interp, op, _make_op(qp.PCPhase, theta, dim=dim, wires=wires), env)
    return _gate_results(interp, op, env)


@_handles("quantum.paulirot")
def _paulirot(interp: _TapeInterpreter, op, _operands, env):
    theta = interp.static_scalar(interp.get(env, op.angle), "The angle of PauliRot")
    word = "".join(str(a.data) for a in op.pauli_product.data)
    wires = [interp.wire(interp.get(env, q)) for q in op.in_qubits]
    _apply_modifiers(interp, op, _make_op(qp.PauliRot, theta, word, wires=wires), env)
    return _gate_results(interp, op, env)


@_handles("quantum.unitary")
def _unitary(interp: _TapeInterpreter, op, _operands, env):
    matrix = interp.static_array(interp.get(env, op.matrix), "The matrix of QubitUnitary")
    wires = [interp.wire(interp.get(env, q)) for q in op.in_qubits]
    _apply_modifiers(interp, op, _make_op(qp.QubitUnitary, matrix, wires=wires), env)
    return _gate_results(interp, op, env)


@_handles("quantum.set_state")
def _set_state(interp: _TapeInterpreter, op, _operands, env):
    state = interp.static_array(interp.get(env, op.in_state), "The state of StatePrep")
    wires = [interp.wire(interp.get(env, q)) for q in op.in_qubits]
    interp.add_operation(_make_op(qp.StatePrep, state, wires=wires))
    return _gate_results(interp, op, env)


@_handles("quantum.set_basis_state")
def _set_basis_state(interp: _TapeInterpreter, op, _operands, env):
    state = interp.static_array(interp.get(env, op.basis_state), "The state of BasisState")
    wires = [interp.wire(interp.get(env, q)) for q in op.in_qubits]
    interp.add_operation(_make_op(qp.BasisState, state.astype(int), wires=wires))
    return _gate_results(interp, op, env)


@_handles("quantum.measure")
def _measure(interp: _TapeInterpreter, op, _operands, env):
    qubit = interp.get(env, op.in_qubit)
    postselect = _attr(op, "postselect")
    postselect = None if postselect is None else _int_attr(postselect)
    # ``qp.measure(w, reset=True)`` is lowered to a measurement followed by a flip of the measured
    # qubit when the outcome is 1.
    resets = [use.operation for use in op.mres.uses if _is_reset(use.operation)]
    interp.resets.update(resets)
    mp = _make_op(
        MidMeasure,
        wires=[interp.wire(qubit)],
        reset=bool(resets),
        postselect=postselect,
        meas_uid=str(uuid.uuid4()),
    )
    mv = MeasurementValue([mp])
    _RAW_MEASUREMENT_VALUES[mp.meas_uid] = mv
    interp.add_operation(mp)
    return [mv, qubit]


# ---------------------------------------------------------------- quantum: observables


@_handles("quantum.compbasis")
def _compbasis(interp: _TapeInterpreter, op, _operands, env):
    if op.qreg is not None:
        return [_CompBasis(None)]
    return [_CompBasis([interp.wire(interp.get(env, q)) for q in op.qubits])]


_NAMED_OBS = {
    "PauliX": qp.X,
    "PauliY": qp.Y,
    "PauliZ": qp.Z,
    "Hadamard": qp.Hadamard,
    "Identity": qp.Identity,
}


@_handles("quantum.namedobs")
def _namedobs(interp: _TapeInterpreter, op, _operands, env):
    kind = op.type.data
    kind = getattr(kind, "value", str(kind))
    return [_make_op(_NAMED_OBS[kind], wires=interp.wire(interp.get(env, op.qubit)))]


@_handles("quantum.hermitian")
def _hermitian(interp: _TapeInterpreter, op, _operands, env):
    matrix = interp.static_array(interp.get(env, op.matrix), "The matrix of Hermitian")
    wires = [interp.wire(interp.get(env, q)) for q in op.qubits]
    return [_make_op(qp.Hermitian, matrix, wires=wires)]


@_handles("quantum.tensor")
def _tensor(interp: _TapeInterpreter, op, _operands, env):
    return [_make_op(qp.prod, *[interp.get(env, t) for t in op.terms])]


@_handles("quantum.hamiltonian")
def _hamiltonian(interp: _TapeInterpreter, op, _operands, env):
    coeffs = interp.static_array(interp.get(env, op.coeffs), "The coefficients of a Hamiltonian")
    terms = [interp.get(env, t) for t in op.terms]
    return [_make_op(qp.Hamiltonian, [float(np.real(c)) for c in coeffs.reshape(-1)], terms)]


@_handles("quantum.mcmobs")
def _mcmobs(interp: _TapeInterpreter, op, _operands, env):
    return [_MCMObs([interp.get(env, m) for m in op.mcms])]


# ---------------------------------------------------------------- quantum: measurements


def _measurement(interp: _TapeInterpreter, op, env, fn, **kwargs):
    obs = interp.get(env, op.obs)
    if isinstance(obs, _CompBasis):
        mp = _make_op(fn, wires=obs.wires, **kwargs) if obs.wires else _make_op(fn, **kwargs)
    elif isinstance(obs, _MCMObs):
        mvs = obs.mvs if len(obs.mvs) > 1 else obs.mvs[0]
        mp = _make_op(fn, op=mvs, **kwargs)
    else:
        mp = _make_op(fn, obs, **kwargs)
    interp.add_measurement(mp)
    return [_Dynamic(f"result of the terminal measurement {mp}")] * len(op.results)


@_handles("quantum.expval")
def _expval(interp, op, _operands, env):
    return _measurement(interp, op, env, qp.expval)


@_handles("quantum.var")
def _var(interp, op, _operands, env):
    return _measurement(interp, op, env, qp.var)


@_handles("quantum.probs")
def _probs(interp, op, _operands, env):
    return _measurement(interp, op, env, qp.probs)


@_handles("quantum.sample")
def _sample(interp, op, _operands, env):
    return _measurement(interp, op, env, qp.sample)


@_handles("quantum.counts")
def _counts(interp, op, _operands, env):
    return _measurement(interp, op, env, qp.counts)


@_handles("quantum.state")
def _state(interp, op, _operands, env):
    obs = interp.get(env, op.obs)
    if not isinstance(obs, _CompBasis):
        raise FlattenError("quantum.state requires a computational basis observable.")
    if obs.wires is not None:
        raise FlattenError("quantum.state is only supported on all wires.")
    mp = _make_op(qp.state)
    interp.add_measurement(mp)
    return [_Dynamic("result of the terminal measurement state()")] * len(op.results)
