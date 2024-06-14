# Copyright 2022-2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This module contains JAX-compatible quantum primitives to support the lowering
of quantum operations, measurements, and observables to JAXPR.
"""

import sys
from dataclasses import dataclass
from itertools import chain
from typing import Dict, Iterable, List, Union

import jax
import numpy as np
import pennylane as qml
from jax._src import api_util, core, source_info_util, util
from jax._src.lib.mlir import ir
from jax.core import AbstractValue
from jax.interpreters import mlir
from jax.tree_util import PyTreeDef, tree_unflatten
from jaxlib.hlo_helpers import shape_dtype_to_ir_type
from jaxlib.mlir.dialects.arith import (
    AddIOp,
    CeilDivSIOp,
    ConstantOp,
    ExtUIOp,
    IndexCastOp,
    MulIOp,
    SubIOp,
)
from jaxlib.mlir.dialects.func import CallOp
from jaxlib.mlir.dialects.scf import ConditionOp, ForOp, IfOp, WhileOp, YieldOp
from jaxlib.mlir.dialects.stablehlo import ConstantOp as StableHLOConstantOp
from jaxlib.mlir.dialects.stablehlo import ConvertOp as StableHLOConvertOp
from mlir_quantum.dialects.catalyst import PrintOp, PythonCallOp
from mlir_quantum.dialects.gradient import GradOp, JVPOp, VJPOp
from mlir_quantum.dialects.mitigation import ZneOp
from mlir_quantum.dialects.quantum import (
    AdjointOp,
    AllocOp,
    ComputationalBasisOp,
    CountsOp,
    CustomOp,
    DeallocOp,
    DeviceInitOp,
    DeviceReleaseOp,
    ExpvalOp,
    ExtractOp,
    GlobalPhaseOp,
    HamiltonianOp,
    HermitianOp,
    InsertOp,
    MeasureOp,
    MultiRZOp,
    NamedObsOp,
    ProbsOp,
    QubitUnitaryOp,
    SampleOp,
    StateOp,
    TensorOp,
    VarianceOp,
)
from mlir_quantum.dialects.quantum import YieldOp as QYieldOp

from catalyst.compiler import get_lib_path
from catalyst.jax_extras import (
    ClosedJaxpr,
    DynshapePrimitive,
    cond_expansion_strategy,
    for_loop_expansion_strategy,
    infer_output_type_jaxpr,
    while_loop_expansion_strategy,
)
from catalyst.utils.calculate_grad_shape import Signature, calculate_grad_shape
from catalyst.utils.extra_bindings import FromElementsOp, TensorExtractOp
from catalyst.utils.types import convert_shaped_arrays_to_tensors

# pylint: disable=unused-argument,abstract-method,too-many-lines,too-many-statements
# pylint: disable=too-many-arguments

#########
# Types #
#########


#
# qbit
#
class AbstractQbit(AbstractValue):
    """Abstract Qbit"""

    hash_value = hash("AbstractQubit")

    def __eq__(self, other):  # pragma: nocover
        return isinstance(other, AbstractQbit)

    def __hash__(self):  # pragma: nocover
        return self.hash_value


class ConcreteQbit(AbstractQbit):
    """Concrete Qbit."""


def _qbit_lowering(aval):
    assert isinstance(aval, AbstractQbit)
    return (ir.OpaqueType.get("quantum", "bit"),)


#
# qreg
#
class AbstractQreg(AbstractValue):
    """Abstract quantum register."""

    hash_value = hash("AbstractQreg")

    def __eq__(self, other):
        return isinstance(other, AbstractQreg)

    def __hash__(self):
        return self.hash_value


class ConcreteQreg(AbstractQreg):
    """Concrete quantum register."""


def _qreg_lowering(aval):
    assert isinstance(aval, AbstractQreg)
    return (ir.OpaqueType.get("quantum", "reg"),)


#
# observable
#
class AbstractObs(AbstractValue):
    """Abstract observable."""

    def __init__(self, num_qubits=None, primitive=None):
        self.num_qubits = num_qubits
        self.primitive = primitive

    def __eq__(self, other):  # pragma: nocover
        if not isinstance(other, AbstractObs):
            return False

        return self.num_qubits == other.num_qubits and self.primitive == other.primitive

    def __hash__(self):  # pragma: nocover
        return hash(self.primitive) + self.num_qubits


class ConcreteObs(AbstractObs):
    """Concrete observable."""


def _obs_lowering(aval):
    assert isinstance(aval, AbstractObs)
    return (ir.OpaqueType.get("quantum", "obs"),)


#
# registration
#
core.raise_to_shaped_mappings[AbstractQbit] = lambda aval, _: aval
mlir.ir_type_handlers[AbstractQbit] = _qbit_lowering

core.raise_to_shaped_mappings[AbstractQreg] = lambda aval, _: aval
mlir.ir_type_handlers[AbstractQreg] = _qreg_lowering

core.raise_to_shaped_mappings[AbstractObs] = lambda aval, _: aval
mlir.ir_type_handlers[AbstractObs] = _obs_lowering


##############
# Primitives #
##############

zne_p = core.Primitive("zne")
zne_p.multiple_results = True
qdevice_p = core.Primitive("qdevice")
qdevice_p.multiple_results = True
qalloc_p = core.Primitive("qalloc")
qdealloc_p = core.Primitive("qdealloc")
qdealloc_p.multiple_results = True
qextract_p = core.Primitive("qextract")
qinsert_p = core.Primitive("qinsert")
gphase_p = core.Primitive("gphase")
gphase_p.multiple_results = True
qinst_p = core.Primitive("qinst")
qinst_p.multiple_results = True
qunitary_p = core.Primitive("qunitary")
qunitary_p.multiple_results = True
qmeasure_p = core.Primitive("qmeasure")
qmeasure_p.multiple_results = True
compbasis_p = core.Primitive("compbasis")
namedobs_p = core.Primitive("namedobs")
hermitian_p = core.Primitive("hermitian")
tensorobs_p = core.Primitive("tensorobs")
hamiltonian_p = core.Primitive("hamiltonian")
sample_p = core.Primitive("sample")
counts_p = core.Primitive("counts")
counts_p.multiple_results = True
expval_p = core.Primitive("expval")
var_p = core.Primitive("var")
probs_p = core.Primitive("probs")
state_p = core.Primitive("state")
cond_p = DynshapePrimitive("cond")
cond_p.multiple_results = True
while_p = DynshapePrimitive("while_loop")
while_p.multiple_results = True
for_p = DynshapePrimitive("for_loop")
for_p.multiple_results = True
grad_p = core.Primitive("grad")
grad_p.multiple_results = True
func_p = core.CallPrimitive("func")
func_p.multiple_results = True
jvp_p = core.Primitive("jvp")
jvp_p.multiple_results = True
vjp_p = core.Primitive("vjp")
vjp_p.multiple_results = True
adjoint_p = jax.core.Primitive("adjoint")
adjoint_p.multiple_results = True
print_p = jax.core.Primitive("debug_print")
print_p.multiple_results = True
python_callback_p = core.Primitive("python_callback")
python_callback_p.multiple_results = True


def _assert_jaxpr_without_constants(jaxpr: ClosedJaxpr):
    assert len(jaxpr.consts) == 0, (
        "Abstract evaluation is not defined for Jaxprs with non-empty constants because these are "
        "not available at the time of the creation of output tracers."
    )


@python_callback_p.def_abstract_eval
def _python_callback_abstract_eval(*avals, callback, results_aval):
    """Abstract evaluation"""
    return results_aval


@python_callback_p.def_impl
def _python_callback_def_impl(*avals, callback, results_aval):  # pragma: no cover
    """Concrete evaluation"""
    raise NotImplementedError()


def _python_callback_lowering(jax_ctx: mlir.LoweringRuleContext, *args, callback, results_aval):
    """Callback lowering"""

    sys.path.append(get_lib_path("runtime", "RUNTIME_LIB_DIR"))
    import catalyst_callback_registry as registry  # pylint: disable=import-outside-toplevel

    callback_id = registry.register(callback)

    ctx = jax_ctx.module_context.context
    i64_type = ir.IntegerType.get_signless(64, ctx)
    identifier = ir.IntegerAttr.get(i64_type, callback_id)

    mlir_ty = list(convert_shaped_arrays_to_tensors(results_aval))
    return PythonCallOp(mlir_ty, args, identifier, number_original_arg=len(args)).results


#
# print
#
@print_p.def_abstract_eval
def _print_abstract_eval(*args, string=None, memref=False):
    return ()


@print_p.def_impl
def _print_def_impl(*args, string=None, memref=False):  # pragma: no cover
    raise NotImplementedError()


def _print_lowering(jax_ctx: mlir.LoweringRuleContext, *args, string=None, memref=False):
    val = args[0] if args else None
    return PrintOp(val=val, const_val=None, print_descriptor=memref).results


#
# func
#
mlir_fn_cache: Dict["catalyst.jax_tracer.Function", str] = {}


@func_p.def_impl
def _func_def_impl(ctx, *args, call_jaxpr, fn, call=True):  # pragma: no cover
    raise NotImplementedError()


def _func_def_lowering(ctx, fn, call_jaxpr) -> str:
    """Create a func::FuncOp from JAXPR."""
    if isinstance(call_jaxpr, core.Jaxpr):
        call_jaxpr = core.ClosedJaxpr(call_jaxpr, ())
    func_op = mlir.lower_jaxpr_to_fun(ctx, fn.__name__, call_jaxpr, tuple())

    if isinstance(fn, qml.QNode):
        func_op.attributes["qnode"] = ir.UnitAttr.get()
        # "best", the default option in PennyLane, chooses backprop on the device
        # if supported and parameter-shift otherwise. Emulating the same behaviour
        # would require generating code to query the device.
        # For simplicity, Catalyst instead defaults to parameter-shift.
        diff_method = "parameter-shift" if fn.diff_method == "best" else str(fn.diff_method)
        func_op.attributes["diff_method"] = ir.StringAttr.get(diff_method)

    return func_op.name.value


def _func_call_lowering(symbol_name, avals_out, *args):
    """Create a func::CallOp from JAXPR."""
    output_types = list(map(mlir.aval_to_ir_types, avals_out))
    flat_output_types = util.flatten(output_types)
    call = CallOp(
        flat_output_types,
        ir.FlatSymbolRefAttr.get(symbol_name),
        mlir.flatten_lowering_ir_args(args),
    )
    out_nodes = util.unflatten(call.results, map(len, output_types))
    return out_nodes


def _func_lowering(ctx, *args, call_jaxpr, fn, call=True):
    """Lower a quantum function into MLIR in a two step process.
    The first step is the compilation of the definition of the function fn.
    The second step is compiling a call to function fn.

    Args:
      ctx: the MLIR context
      args: list of arguments or abstract arguments to the function
      name: name of the function
      call_jaxpr: the jaxpr representation of the fn
      fn: the function being compiled
    """
    if fn in mlir_fn_cache:
        symbol_name = mlir_fn_cache[fn]
    else:
        symbol_name = _func_def_lowering(ctx.module_context, fn, call_jaxpr)
        mlir_fn_cache[fn] = symbol_name

    if not call:
        return None

    out_nodes = _func_call_lowering(
        symbol_name,
        ctx.avals_out,
        *args,
    )
    return out_nodes


#
# grad
#
@dataclass
class GradParams:
    """Common gradient parameters. The parameters are expected to be checked before the creation of
    this structure"""

    method: str
    scalar_out: bool
    h: float
    argnum: Union[int, List]
    scalar_argnum: bool = None
    expanded_argnum: List[int] = None
    with_value: bool = False  # if true it calls value_and_grad instead of grad


@grad_p.def_impl
def _grad_def_impl(ctx, *args, jaxpr, fn, grad_params):  # pragma: no cover
    raise NotImplementedError()


@grad_p.def_abstract_eval
def _grad_abstract(*args, jaxpr, fn, grad_params):
    """This function is called with abstract arguments for tracing."""
    signature = Signature(jaxpr.consts + jaxpr.in_avals, jaxpr.out_avals)
    offset = len(jaxpr.consts)
    new_argnum = [num + offset for num in grad_params.expanded_argnum]
    transformed_signature = calculate_grad_shape(signature, new_argnum)
    return tuple(transformed_signature.get_results())


def _grad_lowering(ctx, *args, jaxpr, fn, grad_params):
    """Lowering function to gradient.
    Args:
        ctx: the MLIR context
        args: the points in the function in which we are to calculate the derivative
        jaxpr: the jaxpr representation of the grad op
        fn(Grad): the function to be differentiated
        method: the method used for differentiation
        h: the difference for finite difference. May be None when fn is not finite difference.
        argnum: argument indices which define over which arguments to
            differentiate.
    """
    method, h, argnum = grad_params.method, grad_params.h, grad_params.expanded_argnum
    mlir_ctx = ctx.module_context.context
    finiteDiffParam = None
    if h:
        f64 = ir.F64Type.get(mlir_ctx)
        finiteDiffParam = ir.FloatAttr.get(f64, h)
    offset = len(jaxpr.consts)
    new_argnum = [num + offset for num in argnum]
    argnum_numpy = np.array(new_argnum)
    diffArgIndices = ir.DenseIntElementsAttr.get(argnum_numpy)

    _func_lowering(ctx, *args, call_jaxpr=jaxpr.eqns[0].params["call_jaxpr"], fn=fn, call=False)
    symbol_name = mlir_fn_cache[fn]
    output_types = list(map(mlir.aval_to_ir_types, ctx.avals_out))
    flat_output_types = util.flatten(output_types)

    # ``ir.DenseElementsAttr.get()`` constructs a dense elements attribute from an array of
    # element values. This doesn't support ``jaxlib.xla_extension.Array``, so we have to cast
    # such constants to numpy array types.

    constants = []
    for const in jaxpr.consts:
        const_type = shape_dtype_to_ir_type(const.shape, const.dtype)
        nparray = np.asarray(const)
        attr = ir.DenseElementsAttr.get(nparray, type=const_type)
        constantVals = StableHLOConstantOp(attr).results
        constants.append(constantVals)
    args_and_consts = constants + list(args)

    return GradOp(
        flat_output_types,
        ir.StringAttr.get(method),
        ir.FlatSymbolRefAttr.get(symbol_name),
        mlir.flatten_lowering_ir_args(args_and_consts),
        diffArgIndices=diffArgIndices,
        finiteDiffParam=finiteDiffParam,
    ).results


#
# vjp/jvp
#
@jvp_p.def_impl
def _jvp_def_impl(ctx, *args, jaxpr, fn, grad_params):  # pragma: no cover
    raise NotImplementedError()


@jvp_p.def_abstract_eval
def _jvp_abstract(*args, jaxpr, fn, grad_params):  # pylint: disable=unused-argument
    """This function is called with abstract arguments for tracing.
    Note: argument names must match these of `_jvp_lowering`."""
    return jaxpr.out_avals + jaxpr.out_avals


def _jvp_lowering(ctx, *args, jaxpr, fn, grad_params):
    """
    Returns:
        MLIR results
    """
    args = list(args)
    method, h, argnum = grad_params.method, grad_params.h, grad_params.expanded_argnum
    mlir_ctx = ctx.module_context.context
    new_argnum = np.array([len(jaxpr.consts) + num for num in argnum])

    output_types = list(map(mlir.aval_to_ir_types, ctx.avals_out))
    flat_output_types = util.flatten(output_types)
    constants = [
        StableHLOConstantOp(ir.DenseElementsAttr.get(np.asarray(const))).results
        for const in jaxpr.consts
    ]
    consts_and_args = constants + args
    func_call_jaxpr = jaxpr.eqns[0].params["call_jaxpr"]
    func_args = consts_and_args[: len(func_call_jaxpr.invars)]
    tang_args = consts_and_args[len(func_call_jaxpr.invars) :]

    _func_lowering(
        ctx,
        *func_args,
        call_jaxpr=func_call_jaxpr,
        fn=fn,
        call=False,
    )

    assert (
        len(flat_output_types) % 2 == 0
    ), f"The total number of result tensors is expected to be even, not {len(flat_output_types)}"
    return JVPOp(
        flat_output_types[: len(flat_output_types) // 2],
        flat_output_types[len(flat_output_types) // 2 :],
        ir.StringAttr.get(method),
        ir.FlatSymbolRefAttr.get(mlir_fn_cache[fn]),
        mlir.flatten_lowering_ir_args(func_args),
        mlir.flatten_lowering_ir_args(tang_args),
        diffArgIndices=ir.DenseIntElementsAttr.get(new_argnum),
        finiteDiffParam=ir.FloatAttr.get(ir.F64Type.get(mlir_ctx), h) if h else None,
    ).results


@vjp_p.def_impl
def _vjp_def_impl(ctx, *args, jaxpr, fn, grad_params):  # pragma: no cover
    raise NotImplementedError()


@vjp_p.def_abstract_eval
# pylint: disable=unused-argument
def _vjp_abstract(*args, jaxpr, fn, grad_params):
    """This function is called with abstract arguments for tracing."""
    return jaxpr.out_avals + [jaxpr.in_avals[i] for i in grad_params.expanded_argnum]


def _vjp_lowering(ctx, *args, jaxpr, fn, grad_params):
    """
    Returns:
        MLIR results
    """
    args = list(args)
    method, h, argnum = grad_params.method, grad_params.h, grad_params.expanded_argnum
    mlir_ctx = ctx.module_context.context
    new_argnum = np.array([len(jaxpr.consts) + num for num in argnum])

    output_types = list(map(mlir.aval_to_ir_types, ctx.avals_out))
    flat_output_types = util.flatten(output_types)
    constants = [
        StableHLOConstantOp(ir.DenseElementsAttr.get(np.asarray(const))).results
        for const in jaxpr.consts
    ]
    consts_and_args = constants + args
    func_call_jaxpr = jaxpr.eqns[0].params["call_jaxpr"]
    func_args = consts_and_args[: len(func_call_jaxpr.invars)]
    cotang_args = consts_and_args[len(func_call_jaxpr.invars) :]
    func_result_types = flat_output_types[: len(flat_output_types) - len(argnum)]
    vjp_result_types = flat_output_types[len(flat_output_types) - len(argnum) :]

    _func_lowering(
        ctx,
        *func_args,
        call_jaxpr=func_call_jaxpr,
        fn=fn,
        call=False,
    )

    return VJPOp(
        func_result_types,
        vjp_result_types,
        ir.StringAttr.get(method),
        ir.FlatSymbolRefAttr.get(mlir_fn_cache[fn]),
        mlir.flatten_lowering_ir_args(func_args),
        mlir.flatten_lowering_ir_args(cotang_args),
        diffArgIndices=ir.DenseIntElementsAttr.get(new_argnum),
        finiteDiffParam=ir.FloatAttr.get(ir.F64Type.get(mlir_ctx), h) if h else None,
    ).results


#
# zne
#


@zne_p.def_impl
def _zne_def_impl(ctx, *args, jaxpr, fn):  # pragma: no cover
    raise NotImplementedError()


@zne_p.def_abstract_eval
def _zne_abstract_eval(*args, jaxpr, fn):  # pylint: disable=unused-argument
    shape = list(args[-1].shape)
    if len(jaxpr.out_avals) > 1:
        shape.append(len(jaxpr.out_avals))
    return [core.ShapedArray(shape, jaxpr.out_avals[0].dtype)]


def _zne_lowering(ctx, *args, jaxpr, fn):
    """Lowering function to the ZNE opearation.
    Args:
        ctx: the MLIR context
        args: the arguments with scale factors as last
        jaxpr: the jaxpr representation of the circuit
        fn: the function to be mitigated
    """
    _func_lowering(ctx, *args, call_jaxpr=jaxpr.eqns[0].params["call_jaxpr"], fn=fn, call=False)
    symbol_name = mlir_fn_cache[fn]
    output_types = list(map(mlir.aval_to_ir_types, ctx.avals_out))
    flat_output_types = util.flatten(output_types)
    return ZneOp(
        flat_output_types,
        ir.FlatSymbolRefAttr.get(symbol_name),
        mlir.flatten_lowering_ir_args(args[0:-1]),
        args[-1],
    ).results


#
# qdevice
#
@qdevice_p.def_impl
def _qdevice_def_impl(ctx, rtd_lib, rtd_name, rtd_kwargs):  # pragma: no cover
    raise NotImplementedError()


@qdevice_p.def_abstract_eval
def _qdevice_abstract_eval(rtd_lib, rtd_name, rtd_kwargs):
    return ()


def _qdevice_lowering(jax_ctx: mlir.LoweringRuleContext, rtd_lib, rtd_name, rtd_kwargs):
    ctx = jax_ctx.module_context.context
    ctx.allow_unregistered_dialects = True
    DeviceInitOp(
        ir.StringAttr.get(rtd_lib), ir.StringAttr.get(rtd_name), ir.StringAttr.get(rtd_kwargs)
    )

    return ()


#
# qalloc
#
@qalloc_p.def_impl
def _qalloc_def_impl(ctx, size_value):  # pragma: no cover
    raise NotImplementedError()


@qalloc_p.def_abstract_eval
def _qalloc_abstract_eval(size):
    """This function is called with abstract arguments for tracing."""
    return AbstractQreg()


def _qalloc_lowering(jax_ctx: mlir.LoweringRuleContext, size_value: ir.Value):
    ctx = jax_ctx.module_context.context
    ctx.allow_unregistered_dialects = True

    assert size_value.owner.name == "stablehlo.constant"
    size_value_attr = size_value.owner.attributes["value"]
    assert ir.DenseIntElementsAttr.isinstance(size_value_attr)
    size = ir.DenseIntElementsAttr(size_value_attr)[0]

    qreg_type = ir.OpaqueType.get("quantum", "reg", ctx)
    i64_type = ir.IntegerType.get_signless(64, ctx)
    size_attr = ir.IntegerAttr.get(i64_type, size)

    return AllocOp(qreg_type, nqubits_attr=size_attr).results


#
# qdealloc
#
@qdealloc_p.def_impl
def _qdealloc_def_impl(ctx, size_value):  # pragma: no cover
    raise NotImplementedError()


@qdealloc_p.def_abstract_eval
def _qdealloc_abstract_eval(qreg):
    return ()


def _qdealloc_lowering(jax_ctx: mlir.LoweringRuleContext, qreg):
    ctx = jax_ctx.module_context.context
    ctx.allow_unregistered_dialects = True
    DeallocOp(qreg)
    DeviceReleaseOp()  # end of qnode
    return ()


#
# qextract
#
@qextract_p.def_impl
def _qextract_def_impl(ctx, qreg, qubit_idx):  # pragma: no cover
    raise NotImplementedError()


@qextract_p.def_abstract_eval
def _qextract_abstract_eval(qreg, qubit_idx):
    """This function is called with abstract arguments for tracing."""
    assert isinstance(qreg, AbstractQreg), f"Expected AbstractQreg(), got {qreg}"
    return AbstractQbit()


def _qextract_lowering(jax_ctx: mlir.LoweringRuleContext, qreg: ir.Value, qubit_idx: ir.Value):
    ctx = jax_ctx.module_context.context
    ctx.allow_unregistered_dialects = True

    assert ir.OpaqueType.isinstance(qreg.type), qreg.type
    assert ir.OpaqueType(qreg.type).dialect_namespace == "quantum"
    assert ir.OpaqueType(qreg.type).data == "reg"

    if ir.RankedTensorType.isinstance(qubit_idx.type):
        baseType = ir.RankedTensorType(qubit_idx.type).element_type
        if ir.RankedTensorType(qubit_idx.type).shape == []:
            qubit_idx = TensorExtractOp(baseType, qubit_idx, []).result
        elif ir.RankedTensorType(qubit_idx.type).shape == [1]:
            c0 = ConstantOp(ir.IndexType.get(), 0)
            qubit_idx = TensorExtractOp(baseType, qubit_idx, [c0]).result
    assert ir.IntegerType.isinstance(qubit_idx.type), "Scalar integer required for extract op!"

    if ir.IntegerType(qubit_idx.type).width < 64:
        qubit_idx = ExtUIOp(ir.IntegerType.get_signless(64), qubit_idx).result
    assert ir.IntegerType(qubit_idx.type).width == 64, "64-bit integer required for extract op!"

    qubit_type = ir.OpaqueType.get("quantum", "bit", ctx)
    return ExtractOp(qubit_type, qreg, idx=qubit_idx).results


#
# qinsert
#
@qinsert_p.def_impl
def _qinsert_def_impl(ctx, qreg_old, qubit_idx, qubit):  # pragma: no cover
    raise NotImplementedError()


@qinsert_p.def_abstract_eval
def _qinsert_abstract_eval(qreg_old, qubit_idx, qubit):
    """This function is called with abstract arguments for tracing."""
    assert isinstance(qreg_old, AbstractQreg)
    assert isinstance(qubit, AbstractQbit)
    return AbstractQreg()


def _qinsert_lowering(
    jax_ctx: mlir.LoweringRuleContext, qreg_old: ir.Value, qubit_idx: ir.Value, qubit: ir.Value
):
    ctx = jax_ctx.module_context.context
    ctx.allow_unregistered_dialects = True

    assert ir.OpaqueType.isinstance(qreg_old.type)
    assert ir.OpaqueType(qreg_old.type).dialect_namespace == "quantum"
    assert ir.OpaqueType(qreg_old.type).data == "reg"

    if ir.RankedTensorType.isinstance(qubit_idx.type):
        baseType = ir.RankedTensorType(qubit_idx.type).element_type
        if ir.RankedTensorType(qubit_idx.type).shape == []:
            qubit_idx = TensorExtractOp(baseType, qubit_idx, []).result
        elif ir.RankedTensorType(qubit_idx.type).shape == [1]:
            c0 = ConstantOp(ir.IndexType.get(), 0)
            qubit_idx = TensorExtractOp(baseType, qubit_idx, [c0]).result
    assert ir.IntegerType.isinstance(qubit_idx.type), "Scalar integer required for insert op!"

    if ir.IntegerType(qubit_idx.type).width < 64:
        qubit_idx = ExtUIOp(ir.IntegerType.get_signless(64), qubit_idx).result
    assert ir.IntegerType(qubit_idx.type).width == 64, "64-bit integer required for insert op!"

    qreg_type = ir.OpaqueType.get("quantum", "reg", ctx)
    return InsertOp(qreg_type, qreg_old, qubit, idx=qubit_idx).results


#
# gphase
#
@gphase_p.def_abstract_eval
def _gphase_abstract_eval(*qubits_or_params, op=None, ctrl_len: int = 0):
    # The signature here is: (using * to denote zero or more)
    # param, ctrl_qubits*, ctrl_values*
    # since gphase has no target qubits.
    param = qubits_or_params[0]
    assert not isinstance(param, AbstractQbit)
    ctrl_qubits = qubits_or_params[-2 * ctrl_len : -ctrl_len]
    for idx in range(ctrl_len):
        qubit = ctrl_qubits[idx]
        assert isinstance(qubit, AbstractQbit)
    return (AbstractQbit(),) * (ctrl_len)


@qinst_p.def_impl
def _gphase_abstract_eval(*qubits_or_params, op=None, ctrl_len: int = 0):
    """Not implemented"""
    raise NotImplementedError()


def _gphase_lowering(
    jax_ctx: mlir.LoweringRuleContext,
    *qubits_or_params,
    op=None,
    ctrl_len: int = 0,
):
    ctx = jax_ctx.module_context.context
    ctx.allow_unregistered_dialects = True

    param = qubits_or_params[0]
    ctrl_qubits = qubits_or_params[1 : 1 + ctrl_len]
    ctrl_values = qubits_or_params[1 + ctrl_len :]

    if ir.RankedTensorType.isinstance(param.type) and ir.RankedTensorType(param.type).shape == []:
        baseType = ir.RankedTensorType(param.type).element_type

    if not ir.F64Type.isinstance(baseType):
        baseType = ir.F64Type.get()
        resultTensorType = ir.RankedTensorType.get((), baseType)
        param = StableHLOConvertOp(resultTensorType, param).results

    param = TensorExtractOp(baseType, param, []).result

    assert ir.F64Type.isinstance(
        param.type
    ), "Only scalar double parameters are allowed for quantum gates!"

    ctrl_values_i1 = []
    for v in ctrl_values:
        p = TensorExtractOp(ir.IntegerType.get_signless(1), v, []).result
        ctrl_values_i1.append(p)

    GlobalPhaseOp(
        params=param,
        out_ctrl_qubits=[qubit.type for qubit in ctrl_qubits],
        in_ctrl_qubits=ctrl_qubits,
        in_ctrl_values=ctrl_values_i1,
    )
    return ctrl_qubits


#
# qinst
#
@qinst_p.def_abstract_eval
def _qinst_abstract_eval(
    *qubits_or_params, op=None, qubits_len: int = 0, params_len: int = 0, ctrl_len: int = 0
):
    # The signature here is: (using * to denote zero or more)
    # qubits*, params*, ctrl_qubits*, ctrl_values*
    qubits = qubits_or_params[:qubits_len]
    ctrl_qubits = qubits_or_params[-2 * ctrl_len : -ctrl_len]
    all_qubits = qubits + ctrl_qubits
    for idx in range(qubits_len + ctrl_len):
        qubit = all_qubits[idx]
        assert isinstance(qubit, AbstractQbit)
    return (AbstractQbit(),) * (qubits_len + ctrl_len)


@qinst_p.def_impl
def _qinst_def_impl(
    ctx, *qubits_or_params, op, qubits_len, params_len, ctrl_len
):  # pragma: no cover
    raise NotImplementedError()


def _qinst_lowering(
    jax_ctx: mlir.LoweringRuleContext,
    *qubits_or_params: tuple,
    op=None,
    qubits_len: int = 0,
    params_len: int = 0,
    ctrl_len: int = 0,
):
    ctx = jax_ctx.module_context.context
    ctx.allow_unregistered_dialects = True

    qubits = qubits_or_params[:qubits_len]
    params = qubits_or_params[qubits_len : qubits_len + params_len]
    ctrl_qubits = qubits_or_params[qubits_len + params_len : qubits_len + params_len + ctrl_len]
    ctrl_values = qubits_or_params[qubits_len + params_len + ctrl_len :]

    for qubit in qubits:
        assert ir.OpaqueType.isinstance(qubit.type)
        assert ir.OpaqueType(qubit.type).dialect_namespace == "quantum"
        assert ir.OpaqueType(qubit.type).data == "bit"

    float_params = []
    for p in params:
        if ir.RankedTensorType.isinstance(p.type) and ir.RankedTensorType(p.type).shape == []:
            baseType = ir.RankedTensorType(p.type).element_type

        if not ir.F64Type.isinstance(baseType):
            baseType = ir.F64Type.get()
            resultTensorType = ir.RankedTensorType.get((), baseType)
            p = StableHLOConvertOp(resultTensorType, p).results

        p = TensorExtractOp(baseType, p, []).result

        assert ir.F64Type.isinstance(
            p.type
        ), "Only scalar double parameters are allowed for quantum gates!"

        float_params.append(p)

    ctrl_values_i1 = []
    for v in ctrl_values:
        p = TensorExtractOp(ir.IntegerType.get_signless(1), v, []).result
        ctrl_values_i1.append(p)

    name_attr = ir.StringAttr.get(op)
    name_str = str(name_attr)
    name_str = name_str.replace('"', "")

    if name_str == "MultiRZ":
        assert len(float_params) == 1, "MultiRZ takes one float parameter"
        float_param = float_params[0]
        return MultiRZOp(
            out_qubits=[qubit.type for qubit in qubits],
            out_ctrl_qubits=[qubit.type for qubit in ctrl_qubits],
            theta=float_param,
            in_qubits=qubits,
            in_ctrl_qubits=ctrl_qubits,
            in_ctrl_values=ctrl_values_i1,
        ).results

    return CustomOp(
        out_qubits=[qubit.type for qubit in qubits],
        out_ctrl_qubits=[qubit.type for qubit in ctrl_qubits],
        params=float_params,
        in_qubits=qubits,
        gate_name=name_attr,
        in_ctrl_qubits=ctrl_qubits,
        in_ctrl_values=ctrl_values_i1,
    ).results


#
# qubit unitary operation
#
@qunitary_p.def_abstract_eval
def _qunitary_abstract_eval(matrix, *qubits, qubits_len: int = 0, ctrl_len: int = 0):
    for idx in range(qubits_len + ctrl_len):
        qubit = qubits[idx]
        assert isinstance(qubit, AbstractQbit)
    return (AbstractQbit(),) * (qubits_len + ctrl_len)


@qunitary_p.def_impl
def _qunitary_def_impl(
    ctx, matrix, qubits, qubits_len: int = None, ctrl_len: int = 0
):  # pragma: no cover
    raise NotImplementedError()


def _qunitary_lowering(
    jax_ctx: mlir.LoweringRuleContext,
    matrix: ir.Value,
    *qubits_or_controlled: tuple,
    qubits_len: int = 0,
    ctrl_len: int = 0,
):
    qubits = qubits_or_controlled[:qubits_len]
    ctrl_qubits = qubits_or_controlled[qubits_len : qubits_len + ctrl_len]
    ctrl_values = qubits_or_controlled[qubits_len + ctrl_len :]

    ctx = jax_ctx.module_context.context
    ctx.allow_unregistered_dialects = True

    for q in qubits:
        assert ir.OpaqueType.isinstance(q.type)
        assert ir.OpaqueType(q.type).dialect_namespace == "quantum"
        assert ir.OpaqueType(q.type).data == "bit"

    matrix_type = matrix.type
    is_tensor = ir.RankedTensorType.isinstance(matrix_type)
    shape = ir.RankedTensorType(matrix_type).shape if is_tensor else None
    is_2d_tensor = len(shape) == 2 if is_tensor else False
    if not is_2d_tensor:
        raise TypeError("QubitUnitary must be a 2 dimensional tensor.")

    possibly_complex_type = ir.RankedTensorType(matrix_type).element_type
    is_complex = ir.ComplexType.isinstance(possibly_complex_type)
    is_f64_type = False

    if is_complex:
        complex_type = ir.ComplexType(possibly_complex_type)
        possibly_f64_type = complex_type.element_type
        is_f64_type = ir.F64Type.isinstance(possibly_f64_type)

    is_complex_f64_type = is_complex and is_f64_type
    if not is_complex_f64_type:
        f64_type = ir.F64Type.get()
        complex_f64_type = ir.ComplexType.get(f64_type)
        tensor_complex_f64_type = ir.RankedTensorType.get(shape, complex_f64_type)
        matrix = StableHLOConvertOp(tensor_complex_f64_type, matrix).results

    ctrl_values_i1 = []
    for v in ctrl_values:
        p = TensorExtractOp(ir.IntegerType.get_signless(1), v, []).result
        ctrl_values_i1.append(p)

    return QubitUnitaryOp(
        out_qubits=[q.type for q in qubits],
        out_ctrl_qubits=[q.type for q in ctrl_qubits],
        matrix=matrix,
        in_qubits=qubits,
        in_ctrl_qubits=ctrl_qubits,
        in_ctrl_values=ctrl_values_i1,
    ).results


#
# qmeasure
#
@qmeasure_p.def_abstract_eval
def _qmeasure_abstract_eval(qubit, postselect: int = None):
    assert isinstance(qubit, AbstractQbit)
    return core.ShapedArray((), bool), qubit


@qmeasure_p.def_impl
def _qmeasure_def_impl(ctx, qubit, postselect: int = None):  # pragma: no cover
    raise NotImplementedError()


def _qmeasure_lowering(jax_ctx: mlir.LoweringRuleContext, qubit: ir.Value, postselect: int = None):
    ctx = jax_ctx.module_context.context
    ctx.allow_unregistered_dialects = True

    assert ir.OpaqueType.isinstance(qubit.type)
    assert ir.OpaqueType(qubit.type).dialect_namespace == "quantum"
    assert ir.OpaqueType(qubit.type).data == "bit"

    # Prepare postselect attribute
    if postselect is not None:
        i32_type = ir.IntegerType.get_signless(32, ctx)
        postselect = ir.IntegerAttr.get(i32_type, postselect)

    result_type = ir.IntegerType.get_signless(1)

    result, new_qubit = MeasureOp(result_type, qubit.type, qubit, postselect=postselect).results

    result_from_elements_op = ir.RankedTensorType.get((), result.type)
    from_elements_op = FromElementsOp(result_from_elements_op, result)

    return (
        from_elements_op.results[0],
        new_qubit,
    )


#
# compbasis observable
#
@compbasis_p.def_abstract_eval
def _compbasis_abstract_eval(*qubits):
    for qubit in qubits:
        assert isinstance(qubit, AbstractQbit)
    return AbstractObs(len(qubits), compbasis_p)


@compbasis_p.def_impl
def _compbasis_def_impl(ctx, *qubits):  # pragma: no cover
    raise NotImplementedError()


def _compbasis_lowering(jax_ctx: mlir.LoweringRuleContext, *qubits: tuple):
    ctx = jax_ctx.module_context.context
    ctx.allow_unregistered_dialects = True

    for qubit in qubits:
        assert ir.OpaqueType.isinstance(qubit.type)
        assert ir.OpaqueType(qubit.type).dialect_namespace == "quantum"
        assert ir.OpaqueType(qubit.type).data == "bit"

    result_type = ir.OpaqueType.get("quantum", "obs")

    return ComputationalBasisOp(result_type, qubits).results


#
# named observable
#
@namedobs_p.def_impl
def _namedobs_def_impl(qubit, kind):  # pragma: no cover
    raise NotImplementedError()


@namedobs_p.def_abstract_eval
def _namedobs_abstract_eval(qubit, kind):
    assert isinstance(qubit, AbstractQbit)
    return AbstractObs()


def _named_obs_attribute(ctx, kind: str):
    return ir.OpaqueAttr.get(
        "quantum", ("named_observable " + kind).encode("utf-8"), ir.NoneType.get(ctx), ctx
    )


def _named_obs_lowering(jax_ctx: mlir.LoweringRuleContext, qubit: ir.Value, kind: str):
    ctx = jax_ctx.module_context.context
    ctx.allow_unregistered_dialects = True

    assert ir.OpaqueType.isinstance(qubit.type)
    assert ir.OpaqueType(qubit.type).dialect_namespace == "quantum"
    assert ir.OpaqueType(qubit.type).data == "bit"

    obsId = _named_obs_attribute(ctx, kind)
    result_type = ir.OpaqueType.get("quantum", "obs", ctx)

    return NamedObsOp(result_type, qubit, obsId).results


#
# hermitian observable
#
@hermitian_p.def_abstract_eval
def _hermitian_abstract_eval(matrix, *qubits):
    for q in qubits:
        assert isinstance(q, AbstractQbit)
    return AbstractObs()


@hermitian_p.def_impl
def _hermitian_def_impl(ctx, matrix, *qubits):  # pragma: no cover
    raise NotImplementedError()


def _hermitian_lowering(jax_ctx: mlir.LoweringRuleContext, matrix: ir.Value, *qubits: tuple):
    ctx = jax_ctx.module_context.context
    ctx.allow_unregistered_dialects = True

    result_type = ir.OpaqueType.get("quantum", "obs", ctx)

    return HermitianOp(result_type, matrix, qubits).results


#
# tensor observable
#
@tensorobs_p.def_impl
def _tensorobs_def_impl(ctx, *terms):  # pragma: no cover
    raise NotImplementedError()


@tensorobs_p.def_abstract_eval
def _tensorobs_abstract_eval(*terms):
    for o in terms:
        assert isinstance(o, AbstractObs)
    return AbstractObs()


def _tensor__obs_lowering(jax_ctx: mlir.LoweringRuleContext, *terms: tuple):
    ctx = jax_ctx.module_context.context
    ctx.allow_unregistered_dialects = True

    result_type = ir.OpaqueType.get("quantum", "obs")

    return TensorOp(result_type, terms).results


#
# hamiltonian observable
#
@hamiltonian_p.def_abstract_eval
def _hamiltonian_abstract_eval(coeffs, *terms):
    for o in terms:
        assert isinstance(o, AbstractObs)
    return AbstractObs()


@hamiltonian_p.def_impl
def _hamiltonian_def_impl(ctx, coeffs, *terms):  # pragma: no cover
    raise NotImplementedError()


def _hamiltonian_lowering(jax_ctx: mlir.LoweringRuleContext, coeffs: ir.Value, *terms: tuple):
    ctx = jax_ctx.module_context.context
    ctx.allow_unregistered_dialects = True

    baseType = ir.RankedTensorType(coeffs.type).element_type
    shape = ir.RankedTensorType(coeffs.type).shape
    if not ir.F64Type.isinstance(baseType):
        baseType = ir.F64Type.get()
        resultTensorType = ir.RankedTensorType.get(shape, baseType)
        coeffs = StableHLOConvertOp(resultTensorType, coeffs).results

    result_type = ir.OpaqueType.get("quantum", "obs", ctx)

    return HamiltonianOp(result_type, coeffs, terms).results


#
# sample measurement
#
@sample_p.def_abstract_eval
def _sample_abstract_eval(obs, shots, shape):
    assert isinstance(obs, AbstractObs)

    if obs.primitive is compbasis_p:
        assert shape == (shots, obs.num_qubits)
    else:
        assert shape == (shots,)

    return core.ShapedArray(shape, jax.numpy.float64)


@sample_p.def_impl
def _sample_def_impl(ctx, obs, shots, shape):  # pragma: no cover
    raise NotImplementedError()


def _sample_lowering(jax_ctx: mlir.LoweringRuleContext, obs: ir.Value, shots: int, shape: tuple):
    ctx = jax_ctx.module_context.context
    ctx.allow_unregistered_dialects = True

    i64_type = ir.IntegerType.get_signless(64, ctx)
    shots_attr = ir.IntegerAttr.get(i64_type, shots)
    f64_type = ir.F64Type.get()
    result_type = ir.RankedTensorType.get(shape, f64_type)

    return SampleOp(result_type, obs, shots_attr).results


#
# counts measurement
#
@counts_p.def_impl
def _counts_def_impl(ctx, obs, shots, shape):  # pragma: no cover
    raise NotImplementedError()


@counts_p.def_abstract_eval
def _counts_abstract_eval(obs, shots, shape):
    assert isinstance(obs, AbstractObs)

    if obs.primitive is compbasis_p:
        assert shape == (2**obs.num_qubits,)
    else:
        assert shape == (2,)

    return core.ShapedArray(shape, jax.numpy.float64), core.ShapedArray(shape, jax.numpy.int64)


def _counts_lowering(jax_ctx: mlir.LoweringRuleContext, obs: ir.Value, shots: int, shape: tuple):
    ctx = jax_ctx.module_context.context
    ctx.allow_unregistered_dialects = True

    i64_type = ir.IntegerType.get_signless(64, ctx)
    shots_attr = ir.IntegerAttr.get(i64_type, shots)
    f64_type = ir.F64Type.get()
    eigvals_type = ir.RankedTensorType.get(shape, f64_type)
    counts_type = ir.RankedTensorType.get(shape, i64_type)

    return CountsOp(eigvals_type, counts_type, obs, shots_attr).results


#
# expval measurement
#
@expval_p.def_abstract_eval
def _expval_abstract_eval(obs, shots, shape=None):
    assert isinstance(obs, AbstractObs)
    return core.ShapedArray((), jax.numpy.float64)


@expval_p.def_impl
def _expval_def_impl(ctx, obs, shots, shape=None):  # pragma: no cover
    raise NotImplementedError()


def _expval_lowering(jax_ctx: mlir.LoweringRuleContext, obs: ir.Value, shots: int, shape=None):
    ctx = jax_ctx.module_context.context
    ctx.allow_unregistered_dialects = True

    assert ir.OpaqueType.isinstance(obs.type)
    assert ir.OpaqueType(obs.type).dialect_namespace == "quantum"
    assert ir.OpaqueType(obs.type).data == "obs"

    i64_type = ir.IntegerType.get_signless(64, ctx)
    shots_attr = ir.IntegerAttr.get(i64_type, shots) if shots is not None else None
    result_type = ir.F64Type.get()

    mres = ExpvalOp(result_type, obs, shots=shots_attr).result
    result_from_elements_op = ir.RankedTensorType.get((), result_type)
    from_elements_op = FromElementsOp(result_from_elements_op, mres)
    return from_elements_op.results


#
# var measurement
#
@var_p.def_abstract_eval
def _var_abstract_eval(obs, shots, shape=None):
    assert isinstance(obs, AbstractObs)
    return core.ShapedArray((), jax.numpy.float64)


@var_p.def_impl
def _var_def_impl(ctx, obs, shots, shape=None):  # pragma: no cover
    raise NotImplementedError()


def _var_lowering(jax_ctx: mlir.LoweringRuleContext, obs: ir.Value, shots: int, shape=None):
    ctx = jax_ctx.module_context.context
    ctx.allow_unregistered_dialects = True

    assert ir.OpaqueType.isinstance(obs.type)
    assert ir.OpaqueType(obs.type).dialect_namespace == "quantum"
    assert ir.OpaqueType(obs.type).data == "obs"

    i64_type = ir.IntegerType.get_signless(64, ctx)
    shots_attr = ir.IntegerAttr.get(i64_type, shots) if shots is not None else None
    result_type = ir.F64Type.get()

    mres = VarianceOp(result_type, obs, shots=shots_attr).result
    result_from_elements_op = ir.RankedTensorType.get((), result_type)
    from_elements_op = FromElementsOp(result_from_elements_op, mres)
    return from_elements_op.results


#
# probs measurement
#
@probs_p.def_abstract_eval
def _probs_abstract_eval(obs, shape, shots=None):
    assert isinstance(obs, AbstractObs)

    if obs.primitive is compbasis_p:
        assert shape == (2**obs.num_qubits,)
    else:
        raise TypeError("probs only supports computational basis")

    return core.ShapedArray(shape, jax.numpy.float64)


@var_p.def_impl
def _probs_def_impl(ctx, obs, shape, shots=None):  # pragma: no cover
    raise NotImplementedError()


def _probs_lowering(jax_ctx: mlir.LoweringRuleContext, obs: ir.Value, shape: tuple, shots=None):
    ctx = jax_ctx.module_context.context
    ctx.allow_unregistered_dialects = True

    result_type = ir.RankedTensorType.get(shape, ir.F64Type.get())

    return ProbsOp(result_type, obs).results


#
# state measurement
#
@state_p.def_abstract_eval
def _state_abstract_eval(obs, shape, shots=None):
    assert isinstance(obs, AbstractObs)

    if obs.primitive is compbasis_p:
        assert shape == (2**obs.num_qubits,)
    else:
        raise TypeError("state only supports computational basis")

    return core.ShapedArray(shape, jax.numpy.complex128)


@state_p.def_impl
def _state_def_impl(ctx, obs, shape, shots=None):  # pragma: no cover
    raise NotImplementedError()


def _state_lowering(jax_ctx: mlir.LoweringRuleContext, obs: ir.Value, shape: tuple, shots=None):
    ctx = jax_ctx.module_context.context
    ctx.allow_unregistered_dialects = True

    c64_type = ir.ComplexType.get(ir.F64Type.get())
    result_type = ir.RankedTensorType.get(shape, c64_type)

    return StateOp(result_type, obs).results


#
# cond
#
@cond_p.def_abstract_eval
def _cond_abstract_eval(
    *args, branch_jaxprs, nimplicit_inputs: int, nimplicit_outputs: int, **kwargs
):
    out_type = infer_output_type_jaxpr(
        [()] + branch_jaxprs[0].jaxpr.invars,
        [],
        branch_jaxprs[0].jaxpr.outvars[nimplicit_outputs:],
        expansion_strategy=cond_expansion_strategy(),
        num_implicit_inputs=nimplicit_inputs,
    )
    return out_type


@cond_p.def_impl
def _cond_def_impl(ctx, *preds_and_branch_args_plus_consts, branch_jaxprs):  # pragma: no cover
    raise NotImplementedError()


def _cond_lowering(
    jax_ctx: mlir.LoweringRuleContext,
    *preds_and_branch_args_plus_consts: tuple,
    branch_jaxprs: List[core.ClosedJaxpr],
    nimplicit_inputs: int,
    nimplicit_outputs: int,
):
    result_types = [mlir.aval_to_ir_types(a)[0] for a in jax_ctx.avals_out]
    num_preds = len(branch_jaxprs) - 1
    preds = preds_and_branch_args_plus_consts[:num_preds]
    branch_args_plus_consts = preds_and_branch_args_plus_consts[num_preds:]
    flat_args_plus_consts = mlir.flatten_lowering_ir_args(branch_args_plus_consts)

    # recursively lower if-else chains to nested IfOps
    def emit_branches(preds, branch_jaxprs, ip):
        # ip is an MLIR InsertionPoint. This allows recursive calls to emit their Operations inside
        # the 'else' blocks of preceding IfOps.
        with ip:
            pred_extracted = TensorExtractOp(ir.IntegerType.get_signless(1), preds[0], []).result
            if_op_scf = IfOp(pred_extracted, result_types, hasElse=True)
            true_jaxpr = branch_jaxprs[0]
            if_block = if_op_scf.then_block

            # if block
            source_info_util.extend_name_stack("if")
            if_ctx = jax_ctx.module_context.replace(
                name_stack=jax_ctx.module_context.name_stack.extend("if")
            )
            with ir.InsertionPoint(if_block):
                # recursively generate the mlir for the if block
                out = mlir.jaxpr_subcomp(
                    if_ctx,
                    true_jaxpr.jaxpr,
                    mlir.TokenSet(),
                    [mlir.ir_constants(c) for c in true_jaxpr.consts],
                    *([a] for a in flat_args_plus_consts),  # fn expects [a1], [a2], [a3] format
                    dim_var_values=jax_ctx.dim_var_values,
                )

                YieldOp([o[0] for o in out[0]])

            # else block
            source_info_util.extend_name_stack("else")
            else_ctx = jax_ctx.module_context.replace(
                name_stack=jax_ctx.module_context.name_stack.extend("else")
            )
            else_block = if_op_scf.else_block
            if len(preds) == 1:
                # Base case: reached the otherwise block
                otherwise_jaxpr = branch_jaxprs[-1]
                with ir.InsertionPoint(else_block):
                    out = mlir.jaxpr_subcomp(
                        else_ctx,
                        otherwise_jaxpr.jaxpr,
                        mlir.TokenSet(),
                        [mlir.ir_constants(c) for c in otherwise_jaxpr.consts],
                        *([a] for a in flat_args_plus_consts),
                        dim_var_values=jax_ctx.dim_var_values,
                    )

                    YieldOp([o[0] for o in out[0]])
            else:
                with ir.InsertionPoint(else_block) as else_ip:
                    child_if_op = emit_branches(preds[1:], branch_jaxprs[1:], else_ip)
                    YieldOp(child_if_op.results)
            return if_op_scf

    head_if_op = emit_branches(preds, branch_jaxprs, jax_ctx.module_context.ip.current)
    return head_if_op.results


#
# while loop
#
@while_p.def_abstract_eval
def _while_loop_abstract_eval(*in_type, body_jaxpr, nimplicit, preserve_dimensions, **kwargs):
    _assert_jaxpr_without_constants(body_jaxpr)
    return infer_output_type_jaxpr(
        [],
        body_jaxpr.jaxpr.invars,
        body_jaxpr.jaxpr.outvars[nimplicit:],
        expansion_strategy=while_loop_expansion_strategy(preserve_dimensions),
    )


@while_p.def_impl
def _while_loop_def_impl(
    ctx,
    *iter_args_plus_consts,
    cond_jaxpr,
    body_jaxpr,
    cond_nconsts,
    body_nconsts,
    nimplicit,
    preserve_dimensions,
):  # pragma: no cover
    raise NotImplementedError()


def _while_loop_lowering(
    jax_ctx: mlir.LoweringRuleContext,
    *iter_args_plus_consts: tuple,
    cond_jaxpr: core.ClosedJaxpr,
    body_jaxpr: core.ClosedJaxpr,
    cond_nconsts: int,
    body_nconsts: int,
    nimplicit: int,
    preserve_dimensions: bool,
):
    loop_carry_types_plus_consts = [mlir.aval_to_ir_types(a)[0] for a in jax_ctx.avals_in]
    flat_args_plus_consts = mlir.flatten_lowering_ir_args(iter_args_plus_consts)
    assert [val.type for val in flat_args_plus_consts] == loop_carry_types_plus_consts

    # split the argument list into 3 separate groups
    # 1) the constants used in the condition function
    # 2) the constants used in the body function
    # 3) the normal arguments which are not constants
    cond_consts = flat_args_plus_consts[:cond_nconsts]
    body_consts = flat_args_plus_consts[cond_nconsts : cond_nconsts + body_nconsts]
    loop_args = flat_args_plus_consts[cond_nconsts + body_nconsts :]

    # remove const types from abstract parameter types list
    loop_carry_types = loop_carry_types_plus_consts[cond_nconsts + body_nconsts :]
    assert loop_carry_types == [mlir.aval_to_ir_types(a)[0] for a in jax_ctx.avals_out]

    while_op_scf = WhileOp(loop_carry_types, loop_args)

    # cond block
    cond_block = while_op_scf.regions[0].blocks.append(*loop_carry_types)
    name_stack = jax_ctx.module_context.name_stack.extend("while")
    cond_ctx = jax_ctx.module_context.replace(name_stack=name_stack.extend("cond"))
    with ir.InsertionPoint(cond_block):
        cond_args = [cond_block.arguments[i] for i in range(len(loop_carry_types))]

        # recursively generate the mlir for the while cond
        ((pred,),), _ = mlir.jaxpr_subcomp(
            cond_ctx,
            cond_jaxpr.jaxpr,
            mlir.TokenSet(),
            [mlir.ir_constants(c) for c in cond_jaxpr.consts],
            *([a] for a in (cond_consts + cond_args)),  # fn expects [a1], [a2], [a3] format
            dim_var_values=jax_ctx.dim_var_values,
        )

        pred_extracted = TensorExtractOp(ir.IntegerType.get_signless(1), pred, []).result
        ConditionOp(pred_extracted, cond_args)

    # body block
    body_block = while_op_scf.regions[1].blocks.append(*loop_carry_types)
    body_ctx = jax_ctx.module_context.replace(name_stack=name_stack.extend("body"))
    with ir.InsertionPoint(body_block):
        body_args = [body_block.arguments[i] for i in range(len(loop_carry_types))]

        # recursively generate the mlir for the while body
        out, _ = mlir.jaxpr_subcomp(
            body_ctx,
            body_jaxpr.jaxpr,
            mlir.TokenSet(),
            [mlir.ir_constants(c) for c in cond_jaxpr.consts],
            *([a] for a in (body_consts + body_args)),  # fn expects [a1], [a2], [a3] format
            dim_var_values=jax_ctx.dim_var_values,
        )

        YieldOp([o[0] for o in out])

    return while_op_scf.results


#
# for loop
#
@for_p.def_abstract_eval
def _for_loop_abstract_eval(
    *args, body_jaxpr, nimplicit, preserve_dimensions, body_nconsts, **kwargs
):
    _assert_jaxpr_without_constants(body_jaxpr)

    return infer_output_type_jaxpr(
        body_jaxpr.jaxpr.invars[:body_nconsts],
        body_jaxpr.jaxpr.invars[body_nconsts:],
        body_jaxpr.jaxpr.outvars[nimplicit:],
        expansion_strategy=for_loop_expansion_strategy(preserve_dimensions),
        num_implicit_inputs=nimplicit,
    )


# pylint: disable=too-many-arguments
@for_p.def_impl
def _for_loop_def_impl(
    ctx,
    lower_bound,
    upper_bound,
    step,
    *iter_args_plus_consts,
    body_jaxpr,
    nimplicit=0,
    body_nconsts,
    preserve_dimensions,
):  # pragma: no cover
    raise NotImplementedError()


# pylint: disable=too-many-arguments
def _for_loop_lowering(
    jax_ctx: mlir.LoweringRuleContext,
    *iter_args_plus_consts: tuple,
    body_jaxpr: core.ClosedJaxpr,
    body_nconsts: int,
    apply_reverse_transform: bool,
    nimplicit: int,
    preserve_dimensions,
):
    body_consts = iter_args_plus_consts[:body_nconsts]
    body_implicits = iter_args_plus_consts[body_nconsts : body_nconsts + nimplicit]
    lower_bound = iter_args_plus_consts[body_nconsts + nimplicit + 0]
    upper_bound = iter_args_plus_consts[body_nconsts + nimplicit + 1]
    step = iter_args_plus_consts[body_nconsts + nimplicit + 2]
    loop_index = iter_args_plus_consts[body_nconsts + nimplicit + 3]
    loop_args = [*body_implicits, *iter_args_plus_consts[body_nconsts + nimplicit + 4 :]]

    loop_index_type = ir.RankedTensorType(loop_index.type).element_type

    all_param_types_plus_consts = [mlir.aval_to_ir_types(a)[0] for a in jax_ctx.avals_in]
    assert [lower_bound.type, upper_bound.type, step.type] == all_param_types_plus_consts[
        body_nconsts + nimplicit : body_nconsts + nimplicit + 3
    ]
    assert [val.type for val in body_consts] == all_param_types_plus_consts[:body_nconsts]

    result_types = [v.type for v in loop_args]
    assert result_types == [
        mlir.aval_to_ir_types(a)[0] for a in jax_ctx.avals_out
    ], f"\n{result_types=} doesn't match \n{jax_ctx.avals_out=}"

    def _cast_to_index(p):
        p = TensorExtractOp(
            ir.RankedTensorType(p.type).element_type, p, []
        ).result  # tensor<i64> -> i64
        p = IndexCastOp(ir.IndexType.get(), p).result  # i64 -> index
        return p

    lower_bound, upper_bound, step = map(_cast_to_index, (lower_bound, upper_bound, step))

    if apply_reverse_transform:
        zero_np = np.array(0)
        one_np = np.array(1)
        zero_attr = ir.DenseIntElementsAttr.get(zero_np)
        one_attr = ir.DenseIntElementsAttr.get(one_np)
        zero_tensor = StableHLOConstantOp(zero_attr)
        one_tensor = StableHLOConstantOp(one_attr)
        ctx = jax_ctx.module_context.context
        i64_type = ir.IntegerType.get_signless(64, ctx)
        zero_i64 = TensorExtractOp(i64_type, zero_tensor, []).result
        one_i64 = TensorExtractOp(i64_type, one_tensor, []).result
        zero = IndexCastOp(ir.IndexType.get(), zero_i64).result
        one = IndexCastOp(ir.IndexType.get(), one_i64).result

        start_val, stop_val, step_val = lower_bound, upper_bound, step

        # Iterate from 0 to the number of iterations (ceil((stop - start) / step))
        distance = SubIOp(stop_val, start_val)
        num_iterations = CeilDivSIOp(distance, step_val)
        lower_bound, upper_bound, step = zero, num_iterations, one

    for_op_scf = ForOp(lower_bound, upper_bound, step, iter_args=loop_args)

    name_stack = jax_ctx.module_context.name_stack.extend("for")
    body_block = for_op_scf.body
    body_ctx = jax_ctx.module_context.replace(name_stack=name_stack.extend("body"))

    with ir.InsertionPoint(body_block):
        body_args = list(body_block.arguments)

        # Convert the index type iteration variable expected by MLIR to tensor<i64> expected by JAX.
        if apply_reverse_transform:
            # iv = start + normalized_iv * step
            body_args[0] = AddIOp(start_val, MulIOp(body_args[0], step_val))

        body_args[0] = IndexCastOp(loop_index_type, body_args[0]).result
        result_from_elements_op = ir.RankedTensorType.get((), loop_index_type)
        from_elements_op = FromElementsOp(result_from_elements_op, body_args[0])
        body_args[0] = from_elements_op.result

        # Re-order arguments in accordance with jax dynamic API convensions
        consts = body_consts
        loop_iter = body_args[0]
        implicit_args = body_args[1 : nimplicit + 1]
        explicit_args = body_args[nimplicit + 1 :]
        loop_params = (*consts, *implicit_args, loop_iter, *explicit_args)
        body_args = [[param] for param in loop_params]

        # Recursively generate the mlir for the loop body
        out, _ = mlir.jaxpr_subcomp(
            body_ctx,
            body_jaxpr.jaxpr,
            mlir.TokenSet(),
            [mlir.ir_constants(c) for c in body_jaxpr.consts],
            *body_args,
            dim_var_values=jax_ctx.dim_var_values,
        )

        YieldOp([o[0] for o in out])

    return for_op_scf.results


#
# adjoint
#
@adjoint_p.def_impl
def _adjoint_def_impl(ctx, *args, args_tree, jaxpr):  # pragma: no cover
    raise NotImplementedError()


@adjoint_p.def_abstract_eval
def _adjoint_abstract(*args, args_tree, jaxpr):
    return jaxpr.out_avals[-1:]


def _adjoint_lowering(
    jax_ctx: mlir.LoweringRuleContext,
    *args: Iterable[ir.Value],
    args_tree: PyTreeDef,
    jaxpr: core.ClosedJaxpr,
) -> ir.Value:
    """The JAX bind handler performing the Jaxpr -> MLIR adjoint lowering by taking the `jaxpr`
    expression to be lowered and all its already lowered arguments as MLIR value references. The Jax
    requires all the arguments to be passed as a single list of positionals, thus we pass indices of
    the argument groups. The handler returns the resulting MLIR Value."""

    # [1] - MLIR Value of constans, classical and quantum arguments [2] - JAXPR types of constans,
    # classical and quantum arguments [3] - Build a body of the adjoint operator. We pass constants
    # and classical arguments as-is, but substitute the quantum arguments with the arguments of the
    # block.

    ctx = jax_ctx.module_context.context
    consts, cargs, qargs = tree_unflatten(args_tree, args)  # [1]
    _, _, aqargs = tree_unflatten(args_tree, jax_ctx.avals_in)  # [2]

    assert len(qargs) == 1, "We currently expect exactly one quantum register argument"
    output_types = util.flatten(map(mlir.aval_to_ir_types, jax_ctx.avals_out))
    assert len(output_types) == 1 and output_types[0] == ir.OpaqueType.get(
        "quantum", "reg", ctx
    ), f"Expected a single result of quantum.register type, got: {output_types}"

    # Build an adjoint operation with a single-block region.
    op = AdjointOp(output_types[0], qargs[0])
    adjoint_block = op.regions[0].blocks.append(*[mlir.aval_to_ir_types(a)[0] for a in aqargs])
    with ir.InsertionPoint(adjoint_block):
        source_info_util.extend_name_stack("adjoint")
        out, _ = mlir.jaxpr_subcomp(
            jax_ctx.module_context.replace(
                name_stack=jax_ctx.module_context.name_stack.extend("adjoint")
            ),
            jaxpr.jaxpr,
            mlir.TokenSet(),
            [mlir.ir_constants(c) for c in jaxpr.consts],
            *([a] for a in chain(consts, cargs, adjoint_block.arguments)),  # [3]
            dim_var_values=jax_ctx.dim_var_values,
        )

        QYieldOp([a[0] for a in out[-1:]])

    return op.results


#
# registration
#

mlir.register_lowering(zne_p, _zne_lowering)
mlir.register_lowering(qdevice_p, _qdevice_lowering)
mlir.register_lowering(qalloc_p, _qalloc_lowering)
mlir.register_lowering(qdealloc_p, _qdealloc_lowering)
mlir.register_lowering(qextract_p, _qextract_lowering)
mlir.register_lowering(qinsert_p, _qinsert_lowering)
mlir.register_lowering(qinst_p, _qinst_lowering)
mlir.register_lowering(gphase_p, _gphase_lowering)
mlir.register_lowering(qunitary_p, _qunitary_lowering)
mlir.register_lowering(qmeasure_p, _qmeasure_lowering)
mlir.register_lowering(compbasis_p, _compbasis_lowering)
mlir.register_lowering(namedobs_p, _named_obs_lowering)
mlir.register_lowering(hermitian_p, _hermitian_lowering)
mlir.register_lowering(tensorobs_p, _tensor__obs_lowering)
mlir.register_lowering(hamiltonian_p, _hamiltonian_lowering)
mlir.register_lowering(sample_p, _sample_lowering)
mlir.register_lowering(counts_p, _counts_lowering)
mlir.register_lowering(expval_p, _expval_lowering)
mlir.register_lowering(var_p, _var_lowering)
mlir.register_lowering(probs_p, _probs_lowering)
mlir.register_lowering(state_p, _state_lowering)
mlir.register_lowering(cond_p, _cond_lowering)
mlir.register_lowering(while_p, _while_loop_lowering)
mlir.register_lowering(for_p, _for_loop_lowering)
mlir.register_lowering(grad_p, _grad_lowering)
mlir.register_lowering(func_p, _func_lowering)
mlir.register_lowering(jvp_p, _jvp_lowering)
mlir.register_lowering(vjp_p, _vjp_lowering)
mlir.register_lowering(adjoint_p, _adjoint_lowering)
mlir.register_lowering(print_p, _print_lowering)
mlir.register_lowering(python_callback_p, _python_callback_lowering)


def _scalar_abstractify(t):
    # pylint: disable=protected-access
    if t in {int, float, complex, bool} or isinstance(t, jax._src.numpy.lax_numpy._ScalarMeta):
        return core.ShapedArray([], dtype=t, weak_type=True)
    raise TypeError(f"Argument type {t} is not a valid JAX type.")


# pylint: disable=protected-access
api_util._shaped_abstractify_handlers[type] = _scalar_abstractify
# pylint: disable=protected-access
api_util._shaped_abstractify_handlers[jax._src.numpy.lax_numpy._ScalarMeta] = _scalar_abstractify
