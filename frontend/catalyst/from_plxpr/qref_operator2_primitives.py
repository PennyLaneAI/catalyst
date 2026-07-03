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
of quantum operations to reference semantics JAXPR.
"""

# pylint: disable=unused-argument
from jax._src.lib.mlir import ir
from jax.extend.core import Primitive
from jax.interpreters import mlir
from jaxlib.mlir._mlir_libs import _mlir as _ods_cext
from jaxlib.mlir.dialects.stablehlo import ConvertOp as StableHLOConvertOp
from pennylane.pytrees import unflatten

from catalyst.jax_extras.lowering import get_mlir_attribute_from_pyval

# TODO: remove after jax v0.7.2 upgrade
# Mock _ods_cext.globals.register_traceback_file_exclusion due to API conflicts between
# Catalyst's MLIR version and the MLIR version used by JAX. The current JAX version has not
# yet updated to the latest MLIR, causing compatibility issues. This workaround will be removed
# once JAX updates to a compatible MLIR version
# pylint: disable=ungrouped-imports
from catalyst.jax_extras.patches import mock_attributes
from catalyst.jax_primitives import (
    extract_scalar,
    safe_cast_to_f64,
)
from catalyst.utils.extra_bindings import TensorExtractOp
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
        CustomOp,
        GlobalPhaseOp,
        MultiRZOp,
        OperatorOp,
        PauliRotOp,
        PCPhaseOp,
        QubitUnitaryOp,
    )


_SPECIAL_LOWERINGS = {}


def _register_special_lowering(op_name):
    def decorator(f):
        _SPECIAL_LOWERINGS[op_name] = f
        return f

    return decorator


qref_operator_p = Primitive("qref_operator")
qref_operator_p.multiple_results = True


@qref_operator_p.def_abstract_eval
def _qref_operator_p_abstract_eval(*args, **kwargs):
    return []


def _is_custom_op(op_cls, avals_in):
    if op_cls.static_argnames or op_cls.hybrid_argnames or op_cls.compilable_argnames:
        return False
    if op_cls.wire_argnames != ("wires",):
        return False
    return all(p.shape == () and "float" in p.dtype.name for p in avals_in)


def _general_validation(*args, op_cls, wire_lens, **kwargs):
    num_normal_wires = sum(wire_lens)
    wires = args[len(op_cls.dynamic_argnames) : (len(op_cls.dynamic_argnames) + num_normal_wires)]
    for w in wires:
        assert ir.OpaqueType.isinstance(w.type)
        assert ir.OpaqueType(w.type).dialect_namespace == "qref"
        assert ir.OpaqueType(w.type).data == "bit"


def _qref_operator_p_lowering(
    jax_ctx: mlir.LoweringRuleContext,
    *args,
    op_cls,
    **kwargs,
):
    _general_validation(*args, op_cls=op_cls, **kwargs)
    ctx = jax_ctx.module_context.context
    ctx.allow_unregistered_dialects = True
    if op_cls.__name__ in _SPECIAL_LOWERINGS:
        return _SPECIAL_LOWERINGS[op_cls.__name__](jax_ctx, *args, op_cls=op_cls, **kwargs)
    # will be used in future improvements
    hybrid_lens = kwargs.pop("hybrid_lens")  # pylint: disable=unused-variable
    hybrid_trees = kwargs.pop("hybrid_trees")  # pylint: disable=unused-variable
    adjoint = kwargs.pop("adjoint")  # pylint: disable=unused-variable
    n_ctrls = kwargs.pop("n_ctrls")  # pylint: disable=unused-variable
    wire_lens = kwargs.pop("wire_lens")
    params = args[: len(op_cls.dynamic_argnames)]
    qubits = args[len(op_cls.dynamic_argnames) :]

    name_attr = get_mlir_attribute_from_pyval(op_cls.__name__)

    repack_static_data = {k: unflatten(*v) for k, v in kwargs.items()}
    processed_static_data = get_mlir_attribute_from_pyval(repack_static_data)

    param_map = {
        name: ir.DenseI64ArrayAttr.get([ind]) for ind, name in enumerate(op_cls.dynamic_argnames)
    }
    processed_param_map = get_mlir_attribute_from_pyval(param_map)

    qubit_map = {}
    ind = 0
    for name, size in zip(op_cls.wire_argnames, wire_lens):
        qubit_map[name] = ir.DenseI64ArrayAttr.get(list(range(ind, ind + size)))
        ind += size

    processed_qubit_map = get_mlir_attribute_from_pyval(qubit_map)

    ctrl_qubits = []
    ctrl_values = []
    adjoint = False

    if _is_custom_op(op_cls, jax_ctx.avals_in[: len(op_cls.dynamic_argnames)]):
        params = [extract_scalar(safe_cast_to_f64(p, op_cls), op_cls) for p in params]
        CustomOp(
            params=params,
            qubits=qubits,
            gate_name=name_attr,
            ctrl_qubits=ctrl_qubits,
            ctrl_values=ctrl_values,
            adjoint=adjoint,
        )
    else:
        OperatorOp(
            op_name=name_attr,
            params=params,
            qubits=qubits,
            qreg=None,
            forward_args=[],
            ctrl_qubits=[],
            ctrl_values=[],
            adjoint=False,
            UID=None,
            arr_qubit_indices=[],
            param_map=processed_param_map,
            static_data=processed_static_data,
            qubit_map=processed_qubit_map,
        )
    return []


@_register_special_lowering("MultiRZ")
def _multirz_lowering(jax_ctx: mlir.LoweringRuleContext, *args, **_):
    theta = extract_scalar(safe_cast_to_f64(args[0], "MultiRZ"), "MultiRZ")
    qubits = args[1:]
    MultiRZOp(
        theta=theta,
        qubits=qubits,
        ctrl_qubits=[],
        ctrl_values=[],
        adjoint=False,
    )
    return []


@_register_special_lowering("PCPhase")
def _pcphase_lowering(
    jax_ctx: mlir.LoweringRuleContext,
    *args,
    **_,
):
    qubits = args[2:]
    PCPhaseOp(
        theta=extract_scalar(safe_cast_to_f64(args[0], "PCPhase"), "PCPhase"),
        dim=extract_scalar(safe_cast_to_f64(args[1], "PCPhase"), "PCPhase"),
        qubits=qubits,
        ctrl_qubits=[],
        ctrl_values=[],
        adjoint=False,
    )
    return ()


@_register_special_lowering("GlobalPhase")
def _special_gphase_lowering(
    jax_ctx: mlir.LoweringRuleContext,
    *args,
    op_cls,
    **_,
):
    GlobalPhaseOp(
        angle=extract_scalar(safe_cast_to_f64(args[0], "GlobalPhase"), "GlobalPhase"),
        ctrl_qubits=[],
        ctrl_values=[],
        adjoint=False,
    )
    return ()


@_register_special_lowering("QubitUnitary")
def _special_unitary_lowering(jax_ctx: mlir.LoweringRuleContext, matrix, *qubits, **_):
    ctrl_qubits = []
    ctrl_values = []

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
        adjoint=False,
    )

    return ()


@_register_special_lowering("PauliRot")
def _special_paulirot_lowering(
    jax_ctx: mlir.LoweringRuleContext,
    angle,
    *qubits,
    pauli_word,
    **_,
):
    pauli_word = unflatten(*pauli_word)
    ctrl_qubits = []
    ctrl_values = []

    angle = safe_cast_to_f64(angle, "PauliRot")
    angle = extract_scalar(angle, "PauliRot")
    assert ir.F64Type.isinstance(angle.type)

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
        adjoint=False,
    )

    return ()
