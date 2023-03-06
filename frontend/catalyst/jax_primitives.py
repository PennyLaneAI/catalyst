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

import numpy as np

import jax
from jax.interpreters import mlir, xla
from jax._src import util
from jax._src.lib.mlir import ir
from jaxlib.mlir.dialects._ods_common import get_op_results_or_values
from jaxlib.mlir.dialects._func_ops_gen import CallOp
from jaxlib.mlir.dialects._mhlo_ops_gen import ConstantOp, ConvertOp

from catalyst.python_bindings._arith_ops_gen import IndexCastOp
from catalyst.python_bindings._tensor_ops_gen import ExtractOp as TensorExtractOp, FromElementsOp
from catalyst.python_bindings._scf_ops_gen import IfOp, ConditionOp, ForOp, WhileOp, YieldOp
from catalyst.python_bindings import AllocOp, ExtractOp, InsertOp, DeallocOp
from catalyst.python_bindings import CustomOp, MultiRZOp, QubitUnitaryOp, MeasureOp
from catalyst.python_bindings import SampleOp, CountsOp, ExpvalOp, VarianceOp, ProbsOp, StateOp
from catalyst.python_bindings import GradOp
from catalyst.python_bindings import (
    ComputationalBasisOp,
    NamedObsOp,
    HermitianOp,
    TensorOp,
    HamiltonianOp,
)
from catalyst.utils.calculate_grad_shape import calculate_grad_shape, Signature


#########
# Types #
#########


#
# qbit
#
# pylint: disable=too-few-public-methods,abstract-method
class Qbit:
    """Qbit primitive."""

    def __init__(self):
        self.aval = AbstractQbit()


# pylint: disable=too-few-public-methods,abstract-method
class AbstractQbit(jax.core.AbstractValue):
    """Abstract Qbit"""


# pylint: disable=too-few-public-methods,abstract-method
class ConcreteQbit(AbstractQbit):
    """Concrete Qbit."""


def _qbit_lowering(aval):
    assert isinstance(aval, AbstractQbit)
    return (ir.OpaqueType.get("quantum", "bit"),)


#
# qreg
#
# pylint: disable=too-few-public-methods,abstract-method
class Qreg:
    """Quantum register primitive."""

    def __init__(self):
        self.aval = AbstractQreg()


# pylint: disable=too-few-public-methods,abstract-method
class AbstractQreg(jax.core.AbstractValue):
    """Abstract quantum register."""


# pylint: disable=too-few-public-methods,abstract-method
class ConcreteQreg(AbstractQreg):
    """Concrete quantum register."""


def _qreg_lowering(aval):
    assert isinstance(aval, AbstractQreg)
    return (ir.OpaqueType.get("quantum", "reg"),)


#
# observable
#
# pylint: disable=too-few-public-methods,abstract-method
class Obs:
    """Observable JAX type primitive."""

    def __init__(self, num_qubits, primitive):
        self.aval = AbstractObs(num_qubits, primitive)


# pylint: disable=too-few-public-methods,abstract-method
class AbstractObs(jax.core.AbstractValue):
    """Abstract observable."""

    def __init__(self, num_qubits=None, primitive=None):
        self.num_qubits = num_qubits
        self.primitive = primitive


# pylint: disable=too-few-public-methods,abstract-method
class ConcreteObs(AbstractObs):
    """Concrete observable."""


def _obs_lowering(aval):
    assert isinstance(aval, AbstractObs)
    return (ir.OpaqueType.get("quantum", "obs"),)


#
# registration
#
jax.core.pytype_aval_mappings[Qbit] = lambda x: x.aval
jax.core.raise_to_shaped_mappings[AbstractQbit] = lambda aval, _: aval
mlir.ir_type_handlers[AbstractQbit] = _qbit_lowering

jax.core.pytype_aval_mappings[Qreg] = lambda x: x.aval
jax.core.raise_to_shaped_mappings[AbstractQreg] = lambda aval, _: aval
mlir.ir_type_handlers[AbstractQreg] = _qreg_lowering

jax.core.pytype_aval_mappings[Obs] = lambda x: x.aval
jax.core.raise_to_shaped_mappings[AbstractObs] = lambda aval, _: aval
mlir.ir_type_handlers[AbstractObs] = _obs_lowering


##############
# Primitives #
##############

qalloc_p = jax.core.Primitive("qalloc")
qdealloc_p = jax.core.Primitive("qdealloc")
qdealloc_p.multiple_results = True
qextract_p = jax.core.Primitive("qextract")
qinsert_p = jax.core.Primitive("qinsert")
qinst_p = jax.core.Primitive("qinst")
qinst_p.multiple_results = True
qunitary_p = jax.core.Primitive("qunitary")
qunitary_p.multiple_results = True
qmeasure_p = jax.core.Primitive("qmeasure")
qmeasure_p.multiple_results = True
compbasis_p = jax.core.Primitive("compbasis")
namedobs_p = jax.core.Primitive("namedobs")
hermitian_p = jax.core.Primitive("hermitian")
tensorobs_p = jax.core.Primitive("tensorobs")
hamiltonian_p = jax.core.Primitive("hamiltonian")
sample_p = jax.core.Primitive("sample")
counts_p = jax.core.Primitive("counts")
counts_p.multiple_results = True
expval_p = jax.core.Primitive("expval")
var_p = jax.core.Primitive("var")
probs_p = jax.core.Primitive("probs")
state_p = jax.core.Primitive("state")
qcond_p = jax.core.AxisPrimitive("qcond")
qcond_p.multiple_results = True
qwhile_p = jax.core.AxisPrimitive("qwhile")
qwhile_p.multiple_results = True
qfor_p = jax.core.AxisPrimitive("qfor")
qfor_p.multiple_results = True
grad_p = jax.core.Primitive("grad")
grad_p.multiple_results = True
func_p = jax.core.CallPrimitive("func")

#
# func
#
mlir_fn_cache = {}


@func_p.def_impl
def _func_def_impl(ctx, *args, call_jaxpr, fn, call=True):  # pragma: no cover
    raise NotImplementedError()


def _func_symbol_lowering(ctx, fn_name, call_jaxpr):
    """Create a func::FuncOp from JAXPR."""
    if isinstance(call_jaxpr, jax.core.Jaxpr):
        call_jaxpr = jax.core.ClosedJaxpr(call_jaxpr, ())
    symbol_name = mlir.lower_jaxpr_to_fun(ctx, fn_name, call_jaxpr, tuple()).name.value
    return symbol_name


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
        symbol_name = _func_symbol_lowering(ctx.module_context, fn.__name__, call_jaxpr)
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


@grad_p.def_impl
def _grad_def_impl(ctx, *args, jaxpr, fn, method, h, argnum):  # pragma: no cover
    raise NotImplementedError()


@grad_p.def_abstract_eval
# pylint: disable=unused-argument
def _grad_abstract(*args, jaxpr, fn, method, h, argnum):
    """This function is called with abstract arguments for tracing."""
    signature = Signature(jaxpr.consts + jaxpr.in_avals, jaxpr.out_avals)
    offset = len(jaxpr.consts)
    new_argnum = [num + offset for num in argnum]
    transformed_signature = calculate_grad_shape(signature, new_argnum)
    return tuple(transformed_signature.get_results())


def _grad_lowering(ctx, *args, jaxpr, fn, method, h, argnum):
    """Lowering function to gradient.
    Args:
        ctx: the MLIR context
        args: the points in the function in which we are to calculate the derivative
        jaxpr: the jaxpr representation of the grad op
        fn: the function to be differentiated
        method: the method used for differentiation
        h: the difference for finite difference. May be None when fn is not finite difference.
        argnum: argument indices which define over which arguments to
            differentiate.
    """
    mlir_ctx = ctx.module_context.context
    finiteDiffParam = None
    if h:
        f64 = ir.F64Type.get(mlir_ctx)
        finiteDiffParam = ir.FloatAttr.get(f64, h)
    offset = len(jaxpr.consts)
    new_argnum = [num + offset for num in argnum]
    argnum_numpy = np.array(new_argnum)
    diffArgIndices = ir.DenseIntElementsAttr.get(argnum_numpy)

    _func_lowering(ctx, *args, call_jaxpr=jaxpr.eqns[0].params["call_jaxpr"], fn=fn.fn, call=False)
    symbol_name = mlir_fn_cache[fn.fn]
    output_types = list(map(mlir.aval_to_ir_types, ctx.avals_out))
    flat_output_types = util.flatten(output_types)
    constants = [ConstantOp(ir.DenseElementsAttr.get(const)).results for const in jaxpr.consts]
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
# qalloc
#
@qalloc_p.def_impl
def _qalloc_def_impl(ctx, size_value):  # pragma: no cover
    raise NotImplementedError()


def qalloc(size):
    """Bind operands to operation."""
    return qalloc_p.bind(size)


@qalloc_p.def_abstract_eval
# pylint: disable=unused-argument
def _qalloc_abstract_eval(size):
    """This function is called with abstract arguments for tracing."""
    return AbstractQreg()


def _qalloc_lowering(jax_ctx: mlir.LoweringRuleContext, size_value: ir.Value):
    ctx = jax_ctx.module_context.context
    ctx.allow_unregistered_dialects = True

    assert size_value.owner.name == "mhlo.constant"
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
def qdealloc(qreg):
    """Bind operands to operation."""
    return qdealloc_p.bind(qreg)


@qdealloc_p.def_impl
def _qdealloc_def_impl(ctx, size_value):  # pragma: no cover
    raise NotImplementedError()


@qdealloc_p.def_abstract_eval
# pylint: disable=unused-argument
def _qdealloc_abstract_eval(qreg):
    return ()


def _qdealloc_lowering(jax_ctx: mlir.LoweringRuleContext, qreg):
    ctx = jax_ctx.module_context.context
    ctx.allow_unregistered_dialects = True
    DeallocOp(qreg)
    return ()


#
# qextract
#


@qextract_p.def_impl
def _qextract_def_impl(ctx, qreg, qubit_idx):  # pragma: no cover
    raise NotImplementedError()


def qextract(qreg, qubit_idx):
    """Bind operands to operation."""
    return qextract_p.bind(qreg, qubit_idx)


@qextract_p.def_abstract_eval
# pylint: disable=unused-argument
def _qextract_abstract_eval(qreg, qubit_idx):
    """This function is called with abstract arguments for tracing."""
    assert isinstance(qreg, AbstractQreg)
    return AbstractQbit()


def _qextract_lowering(jax_ctx: mlir.LoweringRuleContext, qreg: ir.Value, qubit_idx: ir.Value):
    ctx = jax_ctx.module_context.context
    ctx.allow_unregistered_dialects = True

    assert ir.OpaqueType.isinstance(qreg.type)
    assert ir.OpaqueType(qreg.type).dialect_namespace == "quantum"
    assert ir.OpaqueType(qreg.type).data == "reg"

    if (
        ir.RankedTensorType.isinstance(qubit_idx.type)
        and ir.RankedTensorType(qubit_idx.type).shape == []
    ):
        baseType = ir.RankedTensorType(qubit_idx.type).element_type
        qubit_idx = TensorExtractOp(baseType, qubit_idx, []).result
    assert ir.IntegerType.isinstance(qubit_idx.type), "Scalar integer required for extract op!"

    qubit_type = ir.OpaqueType.get("quantum", "bit", ctx)
    return ExtractOp(qubit_type, qreg, idx=qubit_idx).results


#
# qinsert
#
@qinsert_p.def_impl
def _qinsert_def_impl(ctx, qreg_old, qubit_idx, qubit):  # pragma: no cover
    raise NotImplementedError()


def qinsert(qreg_old, qubit_idx, qubit):
    """Bind operands to operation."""
    return qinsert_p.bind(qreg_old, qubit_idx, qubit)


@qinsert_p.def_abstract_eval
# pylint: disable=unused-argument
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

    if (
        ir.RankedTensorType.isinstance(qubit_idx.type)
        and ir.RankedTensorType(qubit_idx.type).shape == []
    ):
        baseType = ir.RankedTensorType(qubit_idx.type).element_type
        qubit_idx = TensorExtractOp(baseType, qubit_idx, []).result
    assert ir.IntegerType.isinstance(qubit_idx.type), "Scalar integer required for insert op!"

    qreg_type = ir.OpaqueType.get("quantum", "reg", ctx)
    return InsertOp(qreg_type, qreg_old, qubit, idx=qubit_idx).results


#
# qinst
#
def qinst(name, qubits_len, *qubits_or_params):
    """Bind operands to operation."""
    return qinst_p.bind(*qubits_or_params, op=name, qubits_len=qubits_len)


@qinst_p.def_abstract_eval
# pylint: disable=unused-argument
def _qinst_abstract_eval(*qubits_or_params, op=None, qubits_len=-1):
    for idx in range(qubits_len):
        qubit = qubits_or_params[idx]
        assert isinstance(qubit, AbstractQbit)
    return (AbstractQbit(),) * qubits_len


@qinst_p.def_impl
def _qinst_def_impl(ctx, *qubits_or_params, op, qubits_len):  # pragma: no cover
    raise NotImplementedError()


def _qinst_lowering(
    jax_ctx: mlir.LoweringRuleContext, *qubits_or_params: tuple, op=None, qubits_len=-1
):
    ctx = jax_ctx.module_context.context
    ctx.allow_unregistered_dialects = True

    qubits = qubits_or_params[:qubits_len]
    params = qubits_or_params[qubits_len:]

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
            p = ConvertOp(resultTensorType, p).results

        p = TensorExtractOp(baseType, p, []).result

        assert ir.F64Type.isinstance(
            p.type
        ), "Only scalar double parameters are allowed for quantum gates!"

        float_params.append(p)

    name_attr = ir.StringAttr.get(op)
    name_str = str(name_attr)
    name_str = name_str.replace('"', "")

    if name_str == "MultiRZ":
        assert len(float_params) == 1, "MultiRZ takes one float parameter"
        float_param = float_params[0]
        return MultiRZOp([qubit.type for qubit in qubits], float_param, qubits).results

    return CustomOp([qubit.type for qubit in qubits], float_params, qubits, name_attr).results


#
# qubit unitary operation
#
def qunitary(matrix, *qubits):
    """Bind operands to operation."""
    return qunitary_p.bind(matrix, *qubits)


@qunitary_p.def_abstract_eval
# pylint: disable=unused-argument
def _qunitary_abstract_eval(matrix, *qubits):
    for q in qubits:
        assert isinstance(q, AbstractQbit)
    return (AbstractQbit(),) * len(qubits)


@qunitary_p.def_impl
def _qunitary_def_impl(ctx, matrix, qubits):  # pragma: no cover
    raise NotImplementedError()


def _qunitary_lowering(jax_ctx: mlir.LoweringRuleContext, matrix: ir.Value, *qubits: tuple):
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
        matrix = ConvertOp(tensor_complex_f64_type, matrix).results

    return QubitUnitaryOp([q.type for q in qubits], matrix, qubits).results


#
# qmeasure
#
def qmeasure(qubit):
    """Bind operands to operation."""
    return qmeasure_p.bind(qubit)


@qmeasure_p.def_abstract_eval
# pylint: disable=unused-argument
def _qmeasure_abstract_eval(qubit):
    assert isinstance(qubit, AbstractQbit)
    return jax.core.ShapedArray((), bool), qubit


@qmeasure_p.def_impl
def _qmeasure_def_impl(ctx, qubit):  # pragma: no cover
    raise NotImplementedError()


def _qmeasure_lowering(jax_ctx: mlir.LoweringRuleContext, qubit: ir.Value):
    ctx = jax_ctx.module_context.context
    ctx.allow_unregistered_dialects = True

    assert ir.OpaqueType.isinstance(qubit.type)
    assert ir.OpaqueType(qubit.type).dialect_namespace == "quantum"
    assert ir.OpaqueType(qubit.type).data == "bit"

    result_type = ir.IntegerType.get_signless(1)
    result, new_qubit = MeasureOp(result_type, qubit.type, qubit).results

    return (
        FromElementsOp(ir.RankedTensorType.get((), result.type), [result]).results[0],
        new_qubit,
    )


#
# compbasis observable
#
def compbasis(*qubits):
    """Bind operands to operation."""
    return compbasis_p.bind(*qubits)


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
@compbasis_p.def_impl
def _namedobs_def_impl(ctx, qubit, type):  # pragma: no cover
    raise NotImplementedError()


def namedobs(type, qubit):
    """Bind operands to operation."""
    return namedobs_p.bind(qubit, type=type)


@namedobs_p.def_abstract_eval
# pylint: disable=unused-argument
def _namedobs_abstract_eval(qubit, type):
    assert isinstance(qubit, AbstractQbit)
    return AbstractObs()


def _named_obs_lowering(jax_ctx: mlir.LoweringRuleContext, qubit: ir.Value, type: int):
    ctx = jax_ctx.module_context.context
    ctx.allow_unregistered_dialects = True

    assert ir.OpaqueType.isinstance(qubit.type)
    assert ir.OpaqueType(qubit.type).dialect_namespace == "quantum"
    assert ir.OpaqueType(qubit.type).data == "bit"

    i8_type = ir.IntegerType.get_signless(8, ctx)
    obsId = ir.IntegerAttr.get(i8_type, type)
    result_type = ir.OpaqueType.get("quantum", "obs", ctx)

    return NamedObsOp(result_type, qubit, obsId).results


#
# hermitian observable
#
def hermitian(matrix, *qubits):
    """Bind operands to operation."""
    return hermitian_p.bind(matrix, *qubits)


@hermitian_p.def_abstract_eval
# pylint: disable=unused-argument
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
def tensorobs(*terms):
    """Bind operands to operation."""
    return tensorobs_p.bind(*terms)


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
def hamiltonian(coeffs, *terms):
    """Bind operands to operation."""
    return hamiltonian_p.bind(coeffs, *terms)


@hamiltonian_p.def_abstract_eval
# pylint: disable=unused-argument
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

    result_type = ir.OpaqueType.get("quantum", "obs", ctx)

    return HamiltonianOp(result_type, coeffs, terms).results


#
# sample measurement
#
def sample(obs, shots, shape):
    """Bind operands to operation."""
    assert shots is not None, "must specify shot number for qml.sample"
    return sample_p.bind(obs, shots=shots, shape=shape)


@sample_p.def_abstract_eval
def _sample_abstract_eval(obs, shots, shape):
    assert isinstance(obs, AbstractObs)

    if obs.primitive is compbasis_p:
        assert shape == (shots, obs.num_qubits)
    else:
        assert shape == (shots,)

    return jax.core.ShapedArray(shape, jax.numpy.float64)


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
def counts(obs, shots, shape):
    """Bind operands to operation."""
    assert shots is not None, "must specify shot number for qml.counts"
    return counts_p.bind(obs, shots=shots, shape=shape)


@counts_p.def_impl
def _counts_def_impl(ctx, obs, shots, shape):  # pragma: no cover
    raise NotImplementedError()


@counts_p.def_abstract_eval
# pylint: disable=unused-argument
def _counts_abstract_eval(obs, shots, shape):
    assert isinstance(obs, AbstractObs)

    if obs.primitive is compbasis_p:
        assert shape == (2**obs.num_qubits,)
    else:
        assert shape == (2,)

    return jax.core.ShapedArray(shape, jax.numpy.float64), jax.core.ShapedArray(
        shape, jax.numpy.int64
    )


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
def expval(obs, shots):
    """Bind operands to operation."""
    return expval_p.bind(obs, shots=shots)


@expval_p.def_abstract_eval
# pylint: disable=unused-argument
def _expval_abstract_eval(obs, shots):
    assert isinstance(obs, AbstractObs)
    return jax.core.ShapedArray((), jax.numpy.float64)


@expval_p.def_impl
def _expval_def_impl(ctx, obs, shots):  # pragma: no cover
    raise NotImplementedError()


def _expval_lowering(jax_ctx: mlir.LoweringRuleContext, obs: ir.Value, shots: int):
    ctx = jax_ctx.module_context.context
    ctx.allow_unregistered_dialects = True

    assert ir.OpaqueType.isinstance(obs.type)
    assert ir.OpaqueType(obs.type).dialect_namespace == "quantum"
    assert ir.OpaqueType(obs.type).data == "obs"

    i64_type = ir.IntegerType.get_signless(64, ctx)
    shots_attr = ir.IntegerAttr.get(i64_type, shots) if shots is not None else None
    result_type = ir.F64Type.get()

    mres = ExpvalOp(result_type, obs, shots=shots_attr).result
    return FromElementsOp(ir.RankedTensorType.get((), result_type), [mres]).results


#
# var measurement
#
def var(obs, shots):
    """Bind operands to operation."""
    return var_p.bind(obs, shots=shots)


@var_p.def_abstract_eval
# pylint: disable=unused-argument
def _var_abstract_eval(obs, shots):
    assert isinstance(obs, AbstractObs)
    return jax.core.ShapedArray((), jax.numpy.float64)


@var_p.def_impl
def _var_def_impl(ctx, obs, shots):  # pragma: no cover
    raise NotImplementedError()


def _var_lowering(jax_ctx: mlir.LoweringRuleContext, obs: ir.Value, shots: int):
    ctx = jax_ctx.module_context.context
    ctx.allow_unregistered_dialects = True

    assert ir.OpaqueType.isinstance(obs.type)
    assert ir.OpaqueType(obs.type).dialect_namespace == "quantum"
    assert ir.OpaqueType(obs.type).data == "obs"

    i64_type = ir.IntegerType.get_signless(64, ctx)
    shots_attr = ir.IntegerAttr.get(i64_type, shots) if shots is not None else None
    result_type = ir.F64Type.get()

    mres = VarianceOp(result_type, obs, shots=shots_attr).result
    return FromElementsOp(ir.RankedTensorType.get((), result_type), [mres]).results


#
# probs measurement
#
def probs(obs, shape):
    """Bind operands to operation."""
    return probs_p.bind(obs, shape=shape)


@probs_p.def_abstract_eval
def _probs_abstract_eval(obs, shape):
    assert isinstance(obs, AbstractObs)

    if obs.primitive is compbasis_p:
        assert shape == (2**obs.num_qubits,)
    else:
        raise TypeError("probs only supports computational basis")

    return jax.core.ShapedArray(shape, jax.numpy.float64)


@var_p.def_impl
def _probs_def_impl(ctx, obs, shape):  # pragma: no cover
    raise NotImplementedError()


def _probs_lowering(jax_ctx: mlir.LoweringRuleContext, obs: ir.Value, shape: tuple):
    ctx = jax_ctx.module_context.context
    ctx.allow_unregistered_dialects = True

    result_type = ir.RankedTensorType.get(shape, ir.F64Type.get())

    return ProbsOp(result_type, obs).results


#
# state measurement
#
def state(obs, shape):
    """Bind operands to operation."""
    return state_p.bind(obs, shape=shape)


@state_p.def_abstract_eval
def _state_abstract_eval(obs, shape):
    assert isinstance(obs, AbstractObs)

    if obs.primitive is compbasis_p:
        assert shape == (2**obs.num_qubits,)
    else:
        raise TypeError("state only supports computational basis")

    return jax.core.ShapedArray(shape, jax.numpy.complex128)


@state_p.def_impl
def _state_def_impl(ctx, obs, shape):  # pragma: no cover
    raise NotImplementedError()


def _state_lowering(jax_ctx: mlir.LoweringRuleContext, obs: ir.Value, shape: tuple):
    ctx = jax_ctx.module_context.context
    ctx.allow_unregistered_dialects = True

    c64_type = ir.ComplexType.get(ir.F64Type.get())
    result_type = ir.RankedTensorType.get(shape, c64_type)

    return StateOp(result_type, obs).results


#
# qcond
#
def qcond(true_jaxpr, false_jaxpr, *header_and_branch_args_plus_consts):
    """Bind operands to operation."""
    return qcond_p.bind(
        *header_and_branch_args_plus_consts,
        true_jaxpr=true_jaxpr,
        false_jaxpr=false_jaxpr,
    )


@qcond_p.def_abstract_eval
# pylint: disable=unused-argument
def _qcond_abstract_eval(*args, true_jaxpr, false_jaxpr, **kwargs):
    return true_jaxpr.out_avals


@qcond_p.def_impl
def _qcond_def_impl(
    ctx, pred, *branch_args_plus_consts, true_jaxpr, false_jaxpr
):  # pragma: no cover
    raise NotImplementedError()


def _qcond_lowering(
    jax_ctx: mlir.LoweringRuleContext,
    pred: ir.Value,
    *branch_args_plus_consts: tuple,
    true_jaxpr: jax.core.ClosedJaxpr,
    false_jaxpr: jax.core.ClosedJaxpr,
):
    result_types = [mlir.aval_to_ir_types(a)[0] for a in jax_ctx.avals_out]
    flat_args_plus_consts = mlir.flatten_lowering_ir_args(branch_args_plus_consts)

    pred_extracted = TensorExtractOp(ir.IntegerType.get_signless(1), pred, []).result

    if_op_scf = IfOp(
        IfOp.build_generic(
            results=result_types,
            operands=get_op_results_or_values([pred_extracted]),
        )
    )

    # if block
    if_block = if_op_scf.regions[0].blocks.append()
    name_stack = util.extend_name_stack(jax_ctx.module_context.name_stack, "if")
    if_ctx = jax_ctx.module_context.replace(name_stack=xla.extend_name_stack(name_stack, "if"))
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
    else_block = if_op_scf.regions[1].blocks.append()
    name_stack = util.extend_name_stack(jax_ctx.module_context.name_stack, "else")
    else_ctx = jax_ctx.module_context.replace(name_stack=xla.extend_name_stack(name_stack, "else"))
    with ir.InsertionPoint(else_block):
        # recursively generate the mlir for the else block
        out = mlir.jaxpr_subcomp(
            else_ctx,
            false_jaxpr.jaxpr,
            mlir.TokenSet(),
            [mlir.ir_constants(c) for c in false_jaxpr.consts],
            *([a] for a in flat_args_plus_consts),  # fn expects [a1], [a2], [a3] format
            dim_var_values=jax_ctx.dim_var_values,
        )

        YieldOp([o[0] for o in out[0]])

    return if_op_scf.results


#
# qwhile loop
#
def qwhile(cond_jaxpr, body_jaxpr, cond_nconsts, body_nconsts, *iter_args_plus_consts):
    """Bind operands to operation."""
    return qwhile_p.bind(
        *iter_args_plus_consts,
        cond_jaxpr=cond_jaxpr,
        body_jaxpr=body_jaxpr,
        cond_nconsts=cond_nconsts,
        body_nconsts=body_nconsts,
    )


@qwhile_p.def_abstract_eval
# pylint: disable=unused-argument
def _qwhile_loop_abstract_eval(*args, cond_jaxpr, body_jaxpr, **kwargs):
    return body_jaxpr.out_avals


@qwhile_p.def_impl
def _qwhile_def_impl(
    ctx, *iter_args_plus_consts, cond_jaxpr, body_jaxpr, cond_nconsts, body_nconsts
):  # pragma: no cover
    raise NotImplementedError()


def _qwhile_lowering(
    jax_ctx: mlir.LoweringRuleContext,
    *iter_args_plus_consts: tuple,
    cond_jaxpr: jax.core.ClosedJaxpr,
    body_jaxpr: jax.core.ClosedJaxpr,
    cond_nconsts: int,
    body_nconsts: int,
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
    name_stack = util.extend_name_stack(jax_ctx.module_context.name_stack, "while")
    cond_ctx = jax_ctx.module_context.replace(name_stack=xla.extend_name_stack(name_stack, "cond"))
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
    body_ctx = jax_ctx.module_context.replace(name_stack=xla.extend_name_stack(name_stack, "body"))
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
# qfor loop
#
# pylint: disable=unused-argument
def qfor(body_jaxpr, body_nconsts, *header_and_iter_args_plus_consts):
    """Bind operands to operation."""
    return qfor_p.bind(
        *header_and_iter_args_plus_consts, body_jaxpr=body_jaxpr, body_nconsts=body_nconsts
    )


@qfor_p.def_abstract_eval
# pylint: disable=unused-argument
def _qfor_loop_abstract_eval(*args, body_jaxpr, **kwargs):
    return body_jaxpr.out_avals


@qfor_p.def_impl
def _qfor_def_impl(
    ctx, lower_bound, upper_bound, step, *iter_args_plus_consts, body_jaxpr, body_nconsts
):  # pragma: no cover
    raise NotImplementedError()


def _qfor_lowering(
    jax_ctx: mlir.LoweringRuleContext,
    lower_bound: ir.Value,
    upper_bound: ir.Value,
    step: ir.Value,
    *iter_args_plus_consts: tuple,
    body_jaxpr: jax.core.ClosedJaxpr,
    body_nconsts: int,
):
    # Separate constants from iteration arguments.
    # The MLIR value provided by JAX for the iteration index is not needed
    # (as it's identical to the lower bound value).
    body_consts = iter_args_plus_consts[:body_nconsts]
    loop_index = iter_args_plus_consts[body_nconsts]
    loop_args = iter_args_plus_consts[body_nconsts + 1 :]

    all_param_types_plus_consts = [mlir.aval_to_ir_types(a)[0] for a in jax_ctx.avals_in]

    # Remove header values: lower_bound, upper_bound, step
    assert [lower_bound.type, upper_bound.type, step.type] == all_param_types_plus_consts[:3]
    loop_carry_types_plus_consts = all_param_types_plus_consts[3:]

    # Remove loop body constants.
    assert [val.type for val in body_consts] == loop_carry_types_plus_consts[:body_nconsts]
    loop_carry_types = loop_carry_types_plus_consts[body_nconsts:]

    # Overwrite the type of the iteration index determined by JAX (= type of lower bound)
    # in favor of the 'index' type expected by MLIR.
    loop_index_type = ir.RankedTensorType(loop_index.type).element_type
    loop_carry_types[0] = ir.IndexType.get()

    # Don't include the iteration index in the result types.
    result_types = loop_carry_types[1:]
    assert [val.type for val in loop_args] == result_types
    assert result_types == [mlir.aval_to_ir_types(a)[0] for a in jax_ctx.avals_out]

    loop_operands = []
    for p in (lower_bound, upper_bound, step):
        p = TensorExtractOp(
            ir.RankedTensorType(p.type).element_type, p, []
        ).result  # tensor<i64> -> i64
        p = IndexCastOp(ir.IndexType.get(), p).result  # i64 -> index
        loop_operands.append(p)
    loop_operands.extend(loop_args)

    for_op_scf = ForOp(
        ForOp.build_generic(
            results=result_types,
            operands=loop_operands,
        )
    )

    name_stack = util.extend_name_stack(jax_ctx.module_context.name_stack, "for")
    body_block = for_op_scf.regions[0].blocks.append(*loop_carry_types)
    body_ctx = jax_ctx.module_context.replace(name_stack=xla.extend_name_stack(name_stack, "body"))

    with ir.InsertionPoint(body_block):
        body_args = list(body_block.arguments)

        # Convert the index type iteration variable expected by MLIR to tensor<i64> expected by JAX.
        body_args[0] = IndexCastOp(loop_index_type, body_args[0]).result
        body_args[0] = FromElementsOp(
            ir.RankedTensorType.get((), loop_index_type), [body_args[0]]
        ).result

        # recursively generate the mlir for the loop body
        out, _ = mlir.jaxpr_subcomp(
            body_ctx,
            body_jaxpr.jaxpr,
            mlir.TokenSet(),
            [mlir.ir_constants(c) for c in body_jaxpr.consts],
            *([a] for a in (*body_consts, *body_args)),
            dim_var_values=jax_ctx.dim_var_values,
        )

        YieldOp([o[0] for o in out])

    return for_op_scf.results


#
# registration
#
mlir.register_lowering(qdealloc_p, _qdealloc_lowering)
mlir.register_lowering(qalloc_p, _qalloc_lowering)
mlir.register_lowering(qextract_p, _qextract_lowering)
mlir.register_lowering(qinsert_p, _qinsert_lowering)
mlir.register_lowering(qinst_p, _qinst_lowering)
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
mlir.register_lowering(qcond_p, _qcond_lowering)
mlir.register_lowering(qwhile_p, _qwhile_lowering)
mlir.register_lowering(qfor_p, _qfor_lowering)
mlir.register_lowering(grad_p, _grad_lowering)
mlir.register_lowering(func_p, _func_lowering)
