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

"""This module contains JAX-compatible quantum primitives to support the lowering
of quantum operations, measurements, and observables to reference semantics JAXPR.
"""

from itertools import chain

from jax._src import source_info_util
from jax._src.lib.mlir import ir
from jax.core import AbstractValue, ShapedArray
from jax.extend.core import Primitive
from jax.interpreters import mlir
from jaxlib.mlir._mlir_libs import _mlir as _ods_cext
from jaxlib.mlir.dialects.arith import (
    ExtUIOp,
)
from jaxlib.mlir.dialects.stablehlo import ConvertOp as StableHLOConvertOp
from pennylane.capture.primitives import adjoint_transform_prim as plxpr_adjoint_transform_prim

# TODO: remove after jax v0.7.2 upgrade
# Mock _ods_cext.globals.register_traceback_file_exclusion due to API conflicts between
# Catalyst's MLIR version and the MLIR version used by JAX. The current JAX version has not
# yet updated to the latest MLIR, causing compatibility issues. This workaround will be removed
# once JAX updates to a compatible MLIR version
# pylint: disable=ungrouped-imports
from catalyst.jax_extras.patches import mock_attributes
from catalyst.jax_primitives import (
    AbstractObs,
    _named_obs_attribute,
    extract_scalar,
    safe_cast_to_f64,
)
from catalyst.utils.extra_bindings import FromElementsOp, TensorExtractOp
from catalyst.utils.patching import Patcher

with Patcher(
    (
        _ods_cext,
        "globals",
        mock_attributes(
            # pylint: disable=c-extension-no-member
            _ods_cext.globals,
            {"register_traceback_file_exclusion": lambda x: None},
        ),
    ),
):
    from mlir_quantum.dialects.qref import (
        AdjointOp,
        AllocOp,
        ComputationalBasisOp,
        CustomOp,
        DeallocOp,
        GetOp,
        GlobalPhaseOp,
        HermitianOp,
        MeasureOp,
        MultiRZOp,
        NamedObsOp,
        PauliRotOp,
        PCPhaseOp,
        QubitUnitaryOp,
        SetBasisStateOp,
        SetStateOp,
    )


#########
# Types #
#########


#
# qubit
#
class QrefQubit(AbstractValue):
    """Abstract Qubit"""

    hash_value = hash("QrefQubit")

    def __eq__(self, other):
        return isinstance(other, QrefQubit)

    def __hash__(self):
        return self.hash_value


def _qref_qubit_lowering(aval):
    assert isinstance(aval, QrefQubit)
    return ir.OpaqueType.get("qref", "bit")


#
# qreg
#
class QrefQreg(AbstractValue):
    """Abstract quantum register."""

    def __init__(self, num_qubits=None):
        self.num_qubits = num_qubits
        self.hash_value = hash("QrefQreg") + hash(num_qubits)

    def __eq__(self, other):
        return isinstance(other, QrefQreg) and self.hash_value == other.hash_value

    def __hash__(self):
        return self.hash_value


def _qref_qreg_lowering(aval):
    assert isinstance(aval, QrefQreg)
    if aval.num_qubits is None:
        tag = "?"
    else:
        tag = str(aval.num_qubits)
    return ir.OpaqueType.get("qref", "reg<" + tag + ">")


#
# registration
#
mlir.ir_type_handlers[QrefQubit] = _qref_qubit_lowering
mlir.ir_type_handlers[QrefQreg] = _qref_qreg_lowering


##############
# Primitives #
##############

qref_alloc_p = Primitive("qref_alloc")
qref_dealloc_p = Primitive("qref_dealloc")
qref_dealloc_p.multiple_results = True
qref_get_p = Primitive("qref_get")
qref_set_state_p = Primitive("qref_state_prep")
qref_set_state_p.multiple_results = True
qref_set_basis_state_p = Primitive("qref_set_basis_state")
qref_set_basis_state_p.multiple_results = True
qref_qinst_p = Primitive("qref_qinst")
qref_qinst_p.multiple_results = True
qref_gphase_p = Primitive("qref_gphase")
qref_gphase_p.multiple_results = True
qref_pauli_rot_p = Primitive("qref_pauli_rot")
qref_pauli_rot_p.multiple_results = True
qref_unitary_p = Primitive("qref_unitary")
qref_unitary_p.multiple_results = True
qref_measure_p = Primitive("qref_measure")
qref_compbasis_p = Primitive("qref_compbasis")
qref_namedobs_p = Primitive("qref_namedobs")
qref_hermitian_p = Primitive("qref_hermitian")


#
# qref_alloc_p
#
@qref_alloc_p.def_abstract_eval
def _qref_alloc_abstract_eval(*dynamic_num_qubits, static_num_qubits=None):
    assert bool(dynamic_num_qubits) ^ bool(static_num_qubits)
    if static_num_qubits:
        return QrefQreg(static_num_qubits)
    else:
        return QrefQreg()


def _qref_alloc_lowering(
    jax_ctx: mlir.LoweringRuleContext, *dynamic_num_qubits, static_num_qubits=None
):
    assert bool(dynamic_num_qubits) ^ bool(static_num_qubits)
    ctx = jax_ctx.module_context.context
    ctx.allow_unregistered_dialects = True

    if static_num_qubits:
        size_attr = ir.IntegerAttr.get(ir.IntegerType.get_signless(64, ctx), static_num_qubits)
        qreg_type = ir.OpaqueType.get("qref", "reg<" + str(static_num_qubits) + ">", ctx)
        return AllocOp(qreg_type, nqubits_attr=size_attr).results
    else:
        size_value = extract_scalar(dynamic_num_qubits[0], "qref_alloc")
        qreg_type = ir.OpaqueType.get("qref", "reg<?>", ctx)
        return AllocOp(qreg_type, nqubits=size_value).results


#
# qref_dealloc_p
#
# pylint: disable=unused-argument
@qref_dealloc_p.def_abstract_eval
def _qref_dealloc_abstract_eval(qreg):
    return ()


def _qref_dealloc_lowering(jax_ctx: mlir.LoweringRuleContext, qreg):
    ctx = jax_ctx.module_context.context
    ctx.allow_unregistered_dialects = True
    DeallocOp(qreg)
    return ()


#
# qref_get_p
#
# pylint: disable=unused-argument
@qref_get_p.def_abstract_eval
def _qref_get_abstract_eval(qreg, qubit_idx):
    assert isinstance(qreg, QrefQreg), f"Expected QrefQreg, got {qreg}"
    return QrefQubit()


def _qref_get_lowering(jax_ctx: mlir.LoweringRuleContext, qreg: ir.Value, qubit_idx: ir.Value):
    ctx = jax_ctx.module_context.context
    ctx.allow_unregistered_dialects = True

    assert ir.OpaqueType.isinstance(qreg.type), qreg.type
    assert ir.OpaqueType(qreg.type).dialect_namespace == "qref"
    assert "reg" in ir.OpaqueType(qreg.type).data

    qubit_idx = extract_scalar(qubit_idx, "wires", "index")
    if not ir.IntegerType.isinstance(qubit_idx.type):
        raise TypeError(f"Operator wires expected to be integers, got {qubit_idx.type}!")

    if ir.IntegerType(qubit_idx.type).width < 64:
        qubit_idx = ExtUIOp(ir.IntegerType.get_signless(64), qubit_idx).result
    elif not ir.IntegerType(qubit_idx.type).width == 64:
        raise TypeError(f"Operator wires expected to be 64-bit integers, got {qubit_idx.type}!")

    qubit_type = ir.OpaqueType.get("qref", "bit", ctx)
    return GetOp(qubit_type, qreg, idx=qubit_idx).results


#
# state_prep
#
@qref_set_state_p.def_abstract_eval
def _qref_set_state_abstract_eval(*qubits_or_params):
    """Abstract evaluation"""
    return ()


def _qref_set_state_lowering(jax_ctx: mlir.LoweringRuleContext, *qubits_or_params):
    """Lowering of set state"""
    qubits_or_params = list(qubits_or_params)
    param = qubits_or_params.pop()
    qubits = qubits_or_params
    SetStateOp(param, qubits)
    return ()


#
# set_basis_state
#
@qref_set_basis_state_p.def_abstract_eval
def qref_set_basis_state_abstract(*qubits_or_params):
    """Abstract evaluation"""
    return ()


def _qref_set_basis_state_lowering(jax_ctx: mlir.LoweringRuleContext, *qubits_or_params):
    """Lowering of set basis state"""
    qubits_or_params = list(qubits_or_params)
    param = qubits_or_params.pop()
    qubits = qubits_or_params
    SetBasisStateOp(param, qubits)
    return ()


#
# qref_qinst_p
#
@qref_qinst_p.def_abstract_eval
def _qref_qinst_abstract_eval(
    *qubits_or_params, op=None, qubits_len=0, params_len=0, ctrl_len=0, adjoint=False
):
    # The signature here is: (using * to denote zero or more)
    # qubits*, params*, ctrl_qubits*, ctrl_values*
    qubits = qubits_or_params[:qubits_len]
    ctrl_qubits = qubits_or_params[-2 * ctrl_len : -ctrl_len]
    all_qubits = qubits + ctrl_qubits
    assert all(isinstance(qubit, QrefQubit) for qubit in all_qubits[: qubits_len + ctrl_len])
    return ()


# pylint: disable=too-many-arguments
def _qref_qinst_lowering(
    jax_ctx: mlir.LoweringRuleContext,
    *qubits_or_params,
    op=None,
    qubits_len=0,
    params_len=0,
    ctrl_len=0,
    adjoint=False,
):
    ctx = jax_ctx.module_context.context
    ctx.allow_unregistered_dialects = True

    qubits = qubits_or_params[:qubits_len]
    params = qubits_or_params[qubits_len : qubits_len + params_len]
    ctrl_qubits = qubits_or_params[qubits_len + params_len : qubits_len + params_len + ctrl_len]
    ctrl_values = qubits_or_params[qubits_len + params_len + ctrl_len :]

    for qubit in qubits:
        assert ir.OpaqueType.isinstance(qubit.type)
        assert ir.OpaqueType(qubit.type).dialect_namespace == "qref"
        assert ir.OpaqueType(qubit.type).data == "bit"

    float_params = []
    for p in params:
        p = safe_cast_to_f64(p, op)
        p = extract_scalar(p, op)

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
        MultiRZOp(
            theta=float_param,
            qubits=qubits,
            ctrl_qubits=ctrl_qubits,
            ctrl_values=ctrl_values_i1,
            adjoint=adjoint,
        )
        return ()

    if name_str == "PCPhase":
        assert len(float_params) == 2, "PCPhase takes two float parameters"
        float_param = float_params[0]
        dim_param = float_params[1]
        PCPhaseOp(
            theta=float_param,
            dim=dim_param,
            qubits=qubits,
            ctrl_qubits=ctrl_qubits,
            ctrl_values=ctrl_values_i1,
            adjoint=adjoint,
        )
        return ()

    CustomOp(
        params=float_params,
        qubits=qubits,
        gate_name=name_attr,
        ctrl_qubits=ctrl_qubits,
        ctrl_values=ctrl_values_i1,
        adjoint=adjoint,
    )
    return ()


#
# qref_gphase_p
#
@qref_gphase_p.def_abstract_eval
def _qref_gphase_abstract_eval(*qubits_or_params, ctrl_len=0, adjoint=False):
    # The signature here is: (using * to denote zero or more)
    # param, ctrl_qubits*, ctrl_values*
    # since gphase has no target qubits.
    param = qubits_or_params[0]
    assert not isinstance(param, QrefQubit)
    ctrl_qubits = qubits_or_params[-2 * ctrl_len : -ctrl_len]
    assert all(isinstance(qubit, QrefQubit) for qubit in ctrl_qubits[:ctrl_len])
    return ()


def _qref_gphase_lowering(
    jax_ctx: mlir.LoweringRuleContext, *qubits_or_params, ctrl_len=0, adjoint=False
):
    ctx = jax_ctx.module_context.context
    ctx.allow_unregistered_dialects = True

    param = qubits_or_params[0]
    ctrl_qubits = qubits_or_params[1 : 1 + ctrl_len]
    ctrl_values = qubits_or_params[1 + ctrl_len :]

    param = safe_cast_to_f64(param, "GlobalPhase")
    param = extract_scalar(param, "GlobalPhase")

    assert ir.F64Type.isinstance(
        param.type
    ), "Only scalar double parameters are allowed for quantum gates!"

    ctrl_values_i1 = [
        TensorExtractOp(ir.IntegerType.get_signless(1), v, []).result for v in ctrl_values
    ]

    GlobalPhaseOp(
        angle=param,
        ctrl_qubits=ctrl_qubits,
        ctrl_values=ctrl_values_i1,
        adjoint=adjoint,
    )
    return ()


#
# qref_paulirot_p
#
# pylint: disable=unused-variable
@qref_pauli_rot_p.def_abstract_eval
def _pauli_rot_abstract_eval(
    *qubits_and_ctrl_qubits,
    angle=None,
    pauli_word=None,
    qubits_len=0,
    params_len=0,
    ctrl_len=0,
    adjoint=False,
):
    # The signature here is: (using * to denote zero or more)
    # qubits*, params*, ctrl_qubits*, ctrl_values*
    qubits = qubits_and_ctrl_qubits[:qubits_len]
    params = qubits_and_ctrl_qubits[qubits_len : qubits_len + params_len]
    ctrl_qubits = qubits_and_ctrl_qubits[-2 * ctrl_len : -ctrl_len]
    ctrl_values = qubits_and_ctrl_qubits[-ctrl_len:]
    all_qubits = qubits + ctrl_qubits
    assert all(isinstance(qubit, QrefQubit) for qubit in all_qubits)
    return ()


# pylint: disable=unused-argument
def _qref_pauli_rot_lowering(
    jax_ctx: mlir.LoweringRuleContext,
    *qubits_and_params: tuple,
    pauli_word=None,
    qubits_len=0,
    params_len=0,
    ctrl_len=0,
    adjoint=False,
):
    ctx = jax_ctx.module_context.context
    ctx.allow_unregistered_dialects = True

    qubits = qubits_and_params[:qubits_len]
    params = qubits_and_params[qubits_len : qubits_len + params_len]
    ctrl_qubits = qubits_and_params[qubits_len + params_len : qubits_len + params_len + ctrl_len]
    ctrl_values = qubits_and_params[qubits_len + params_len + ctrl_len :]

    for q in qubits:
        assert ir.OpaqueType.isinstance(q.type)
        assert ir.OpaqueType(q.type).dialect_namespace == "qref"
        assert ir.OpaqueType(q.type).data == "bit"

    assert params_len == 1 and params[0] is not None
    angle = params[0]
    angle = safe_cast_to_f64(angle, "PauliRot")
    angle = extract_scalar(angle, "PauliRot")
    assert ir.F64Type.isinstance(angle.type)
    assert pauli_word is not None

    pauli_word = ir.ArrayAttr.get([ir.StringAttr.get(p) for p in pauli_word])

    ctrl_values_i1 = [
        TensorExtractOp(ir.IntegerType.get_signless(1), v, []).result for v in ctrl_values
    ]

    PauliRotOp(
        angle=angle,
        pauli_product=pauli_word,
        qubits=qubits,
        ctrl_qubits=ctrl_qubits,
        ctrl_values=ctrl_values_i1,
        adjoint=adjoint,
    )

    return ()


#
# qubit unitary operation
#
@qref_unitary_p.def_abstract_eval
def _qref_unitary_abstract_eval(matrix, *qubits, qubits_len=0, ctrl_len=0, adjoint=False):
    assert all(isinstance(qubit, QrefQubit) for qubit in qubits[: qubits_len + ctrl_len])
    return ()


def _qref_unitary_lowering(
    jax_ctx: mlir.LoweringRuleContext,
    matrix: ir.Value,
    *qubits_or_controlled: tuple,
    qubits_len=0,
    ctrl_len=0,
    adjoint=False,
):
    qubits = qubits_or_controlled[:qubits_len]
    ctrl_qubits = qubits_or_controlled[qubits_len : qubits_len + ctrl_len]
    ctrl_values = qubits_or_controlled[qubits_len + ctrl_len :]

    ctx = jax_ctx.module_context.context
    ctx.allow_unregistered_dialects = True

    for q in qubits:
        assert ir.OpaqueType.isinstance(q.type)
        assert ir.OpaqueType(q.type).dialect_namespace == "qref"
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
        matrix = StableHLOConvertOp(tensor_complex_f64_type, matrix).result

    ctrl_values_i1 = [
        TensorExtractOp(ir.IntegerType.get_signless(1), v, []).result for v in ctrl_values
    ]

    QubitUnitaryOp(
        matrix=matrix,
        qubits=qubits,
        ctrl_qubits=ctrl_qubits,
        ctrl_values=ctrl_values_i1,
        adjoint=adjoint,
    )

    return ()

#
# PL adjoint primitive
#
def _pl_adjoint_lowering(
    jax_ctx,
    *plxpr_invals,
    jaxpr,
    lazy,
    n_consts,
):
    new_jaxpr = jaxpr.replace(constvars=(), invars=jaxpr.constvars + jaxpr.invars)

    op = AdjointOp()
    adjoint_block = op.regions[0].blocks.append()
    with ir.InsertionPoint(adjoint_block):
        source_info_util.extend_name_stack("adjoint")
        _, _ = mlir.jaxpr_subcomp(
            jax_ctx.module_context,
            new_jaxpr,
            jax_ctx.name_stack.extend("adjoint"),
            mlir.TokenSet(),
            [],
            *list(chain(plxpr_invals, adjoint_block.arguments)),
            dim_var_values=jax_ctx.dim_var_values,
            const_lowering=jax_ctx.const_lowering,
        )

    return ()


#
# measure
#
@qref_measure_p.def_abstract_eval
def _qref_measure_abstract_eval(qubit, postselect: int = None):
    assert isinstance(qubit, QrefQubit)
    return ShapedArray((), bool)


def _qref_measure_lowering(
    jax_ctx: mlir.LoweringRuleContext, qubit: ir.Value, postselect: int = None
):
    ctx = jax_ctx.module_context.context
    ctx.allow_unregistered_dialects = True

    assert ir.OpaqueType.isinstance(qubit.type)
    assert ir.OpaqueType(qubit.type).dialect_namespace == "qref"
    assert ir.OpaqueType(qubit.type).data == "bit"

    # Prepare postselect attribute
    if postselect is not None:
        i32_type = ir.IntegerType.get_signless(32, ctx)
        postselect = ir.IntegerAttr.get(i32_type, postselect)

    result_type = ir.IntegerType.get_signless(1)

    result = MeasureOp(result_type, qubit, postselect=postselect).results[0]

    result_from_elements_op = ir.RankedTensorType.get((), result.type)
    from_elements_op = FromElementsOp(result_from_elements_op, result)

    return (from_elements_op.results[0],)


#
# compbasis observable
#
@qref_compbasis_p.def_abstract_eval
def _qref_compbasis_abstract_eval(*qubits_or_qreg, qreg_available=False):
    if qreg_available:
        qreg = qubits_or_qreg[0]
        assert isinstance(qreg, QrefQreg)
        return AbstractObs(qreg, qref_compbasis_p)
    else:
        qubits = qubits_or_qreg
        for qubit in qubits:
            assert isinstance(qubit, QrefQubit)
        return AbstractObs(len(qubits), qref_compbasis_p)


def _qref_compbasis_lowering(
    jax_ctx: mlir.LoweringRuleContext, *qubits_or_qreg: tuple, qreg_available=False
):
    ctx = jax_ctx.module_context.context
    ctx.allow_unregistered_dialects = True

    result_type = ir.OpaqueType.get("quantum", "obs")

    if qreg_available:
        qreg = qubits_or_qreg[0]
        assert ir.OpaqueType.isinstance(qreg.type)
        assert ir.OpaqueType(qreg.type).dialect_namespace == "qref"
        assert "reg" in ir.OpaqueType(qreg.type).data
        return ComputationalBasisOp(result_type, [], qreg=qreg).results

    else:
        qubits = qubits_or_qreg
        for qubit in qubits:
            assert ir.OpaqueType.isinstance(qubit.type)
            assert ir.OpaqueType(qubit.type).dialect_namespace == "qref"
            assert ir.OpaqueType(qubit.type).data == "bit"

        return ComputationalBasisOp(result_type, qubits).results


#
# named observable
#
# pylint: disable=unused-argument
@qref_namedobs_p.def_abstract_eval
def _qref_namedobs_abstract_eval(qubit, kind):
    assert isinstance(qubit, QrefQubit)
    return AbstractObs()


def _qref_named_obs_lowering(jax_ctx: mlir.LoweringRuleContext, qubit: ir.Value, kind: str):
    ctx = jax_ctx.module_context.context
    ctx.allow_unregistered_dialects = True

    assert ir.OpaqueType.isinstance(qubit.type)
    assert ir.OpaqueType(qubit.type).dialect_namespace == "qref"
    assert ir.OpaqueType(qubit.type).data == "bit"

    obsId = _named_obs_attribute(ctx, kind)
    result_type = ir.OpaqueType.get("quantum", "obs", ctx)

    return NamedObsOp(result_type, qubit, obsId).results


#
# hermitian observable
#
# pylint: disable=unused-argument
@qref_hermitian_p.def_abstract_eval
def _hermitian_abstract_eval(matrix, *qubits):
    for q in qubits:
        assert isinstance(q, QrefQubit)
    return AbstractObs()


def _qref_hermitian_lowering(jax_ctx: mlir.LoweringRuleContext, matrix: ir.Value, *qubits: tuple):
    assert isinstance(matrix.type, ir.RankedTensorType)
    assert isinstance(matrix.type.element_type, ir.ComplexType)

    ctx = jax_ctx.module_context.context
    ctx.allow_unregistered_dialects = True

    result_type = ir.OpaqueType.get("quantum", "obs", ctx)

    return HermitianOp(result_type, matrix, qubits).results


CUSTOM_LOWERING_RULES = (
    (qref_alloc_p, _qref_alloc_lowering),
    (qref_dealloc_p, _qref_dealloc_lowering),
    (qref_get_p, _qref_get_lowering),
    (qref_set_state_p, _qref_set_state_lowering),
    (qref_set_basis_state_p, _qref_set_basis_state_lowering),
    (qref_qinst_p, _qref_qinst_lowering),
    (qref_gphase_p, _qref_gphase_lowering),
    (qref_pauli_rot_p, _qref_pauli_rot_lowering),
    (qref_unitary_p, _qref_unitary_lowering),
    (qref_measure_p, _qref_measure_lowering),
    (qref_compbasis_p, _qref_compbasis_lowering),
    (qref_namedobs_p, _qref_named_obs_lowering),
    (qref_hermitian_p, _qref_hermitian_lowering),
    (plxpr_adjoint_transform_prim, _pl_adjoint_lowering),
)
