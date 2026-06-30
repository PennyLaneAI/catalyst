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


def _is_custom_op(op_cls, params):
    if op_cls.static_argnames or op_cls.hybrid_argnames or op_cls.compilable_argnames:
        return False
    if op_cls.wire_argnames != ("wires",):
        return False

    for p in params:
        baseType = ir.RankedTensorType(p.type).element_type
        if ir.ComplexType.isinstance(baseType) or (
            ir.FloatType.isinstance(baseType) and ir.FloatType(baseType).width > 64
        ):
            return False
        shape = ir.RankedTensorType(p.type).shape
        if shape not in ([], [1]):
            return False

    return True


def _qref_operator_p_lowering(
    jax_ctx: mlir.LoweringRuleContext,
    *args,
    op_cls,
    hybrid_lens,
    hybrid_trees,
    wire_lens,
    adjoint,
    n_ctrls,
    **kwargs,
):
    ctx = jax_ctx.module_context.context
    ctx.allow_unregistered_dialects = True
    assert (
        len(hybrid_lens) == 0 and len(hybrid_trees) == 0
    ), "Hybrid arguments are not supported yet."

    if n_ctrls:
        ctrl_qubits = args[-2 * n_ctrls : -n_ctrls]
        ctrl_values = [
            TensorExtractOp(ir.IntegerType.get_signless(1), v, []).result for v in args[-n_ctrls:]
        ]
        qubits_slice = slice(len(op_cls.dynamic_argnames), -2 * n_ctrls)
    else:
        ctrl_qubits = ctrl_values = ()
        qubits_slice = slice(len(op_cls.dynamic_argnames), None)

    params = args[: len(op_cls.dynamic_argnames)]
    qubits = args[qubits_slice]

    for q in qubits:
        assert ir.OpaqueType.isinstance(q.type)
        assert ir.OpaqueType(q.type).dialect_namespace == "qref"
        assert ir.OpaqueType(q.type).data == "bit"

    if op_cls.__name__ in _SPECIAL_LOWERINGS:
        return _SPECIAL_LOWERINGS[op_cls.__name__](
            *args,
            adjoint=adjoint,
            ctrl_qubits=ctrl_qubits,
            ctrl_values=ctrl_values,
            **kwargs,
        )

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

    if _is_custom_op(op_cls, params):
        new_params = []
        for p in params:
            new_p = extract_scalar(safe_cast_to_f64(p, op_cls), op_cls)
            assert ir.F64Type.isinstance(
                new_p.type
            ), "Only scalar double parameters are allowed for quantum gates!"
            new_params.append(new_p)

        CustomOp(
            params=new_params,
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
            ctrl_qubits=ctrl_qubits,
            ctrl_values=ctrl_values,
            adjoint=adjoint,
            UID=None,
            arr_qubit_indices=[],
            param_map=processed_param_map,
            static_data=processed_static_data,
            qubit_map=processed_qubit_map,
        )
    return []


@_register_special_lowering("MultiRZ")
def _multirz_lowering(theta, *qubits, adjoint, ctrl_qubits, ctrl_values):
    theta = extract_scalar(safe_cast_to_f64(theta, "MultiRZ"), "MultiRZ")
    assert ir.F64Type.isinstance(
        theta.type
    ), "Only scalar double parameters are allowed for MultiRZ!"

    MultiRZOp(
        theta=theta,
        qubits=qubits,
        ctrl_qubits=ctrl_qubits,
        ctrl_values=ctrl_values,
        adjoint=adjoint,
    )
    return []


@_register_special_lowering("PCPhase")
def _pcphase_lowering(theta, dim, *qubits, adjoint, ctrl_qubits, ctrl_values):
    theta = extract_scalar(safe_cast_to_f64(theta, "PCPhase"), "PCPhase")
    dim = extract_scalar(safe_cast_to_f64(dim, "PCPhase"), "PCPhase")

    assert ir.F64Type.isinstance(
        theta.type
    ), "Only scalar double parameters are allowed for PCPhase!"
    assert ir.F64Type.isinstance(dim.type), "Only scalar double parameters are allowed for PCPhase!"

    PCPhaseOp(
        theta=theta,
        dim=dim,
        qubits=qubits,
        ctrl_qubits=ctrl_qubits,
        ctrl_values=ctrl_values,
        adjoint=adjoint,
    )
    return ()


@_register_special_lowering("GlobalPhase")
def _gphase_lowering(*args, adjoint, ctrl_qubits, ctrl_values):
    angle = extract_scalar(safe_cast_to_f64(args[0], "GlobalPhase"), "GlobalPhase")
    assert ir.F64Type.isinstance(
        angle.type
    ), "Only scalar double parameters are allowed for GlobalPhase!"

    GlobalPhaseOp(angle=angle, ctrl_qubits=ctrl_qubits, ctrl_values=ctrl_values, adjoint=adjoint)
    return ()


@_register_special_lowering("QubitUnitary")
def _qubit_unitary_lowering(matrix, *qubits, adjoint, ctrl_qubits, ctrl_values):
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

    QubitUnitaryOp(
        matrix=matrix,
        qubits=qubits,
        ctrl_qubits=ctrl_qubits,
        ctrl_values=ctrl_values,
        adjoint=adjoint,
    )

    return ()


@_register_special_lowering("PauliRot")
def _paulirot_lowering(angle, *qubits, adjoint, ctrl_qubits, ctrl_values, pauli_word=None):
    assert pauli_word is not None
    pauli_word = unflatten(*pauli_word)
    if not all(p in ["I", "X", "Y", "Z"] for p in pauli_word):
        raise ValueError("Only Pauli words consisting of 'I', 'X', 'Y', and 'Z' are allowed.")

    for q in qubits:
        assert ir.OpaqueType.isinstance(q.type)
        assert ir.OpaqueType(q.type).dialect_namespace == "qref"
        assert ir.OpaqueType(q.type).data == "bit"

    angle = extract_scalar(safe_cast_to_f64(angle, "PauliRot"), "PauliRot")
    assert ir.F64Type.isinstance(
        angle.type
    ), "Only scalar double parameters are allowed for PauliRot!"

    pauli_word = ir.ArrayAttr.get([ir.StringAttr.get(p) for p in pauli_word])

    PauliRotOp(
        angle=angle,
        pauli_product=pauli_word,
        qubits=qubits,
        ctrl_qubits=ctrl_qubits,
        ctrl_values=ctrl_values,
        adjoint=adjoint,
    )

    return ()
    return ()
