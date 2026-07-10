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

from .uid import generate_uid

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


def _is_qref_qubit(val) -> bool:
    val_type = val.type
    return (
        ir.OpaqueType.isinstance(val_type)
        and ir.OpaqueType(val_type).dialect_namespace == "qref"
        and ir.OpaqueType(val_type).data == "bit"
    )


def _general_validation(*args, op_cls, wire_lens, **kwargs):
    num_normal_wires = sum(wire_lens)
    wires = args[len(op_cls.dynamic_argnames) : (len(op_cls.dynamic_argnames) + num_normal_wires)]
    assert all(_is_qref_qubit(w) for w in wires)


def _process_params(
    *args, op_cls, wire_lens, hybrid_lens, forward_mask
) -> tuple[list, list, dict[str, list[int]]]:
    """Process non-qubit operands of an operator. This function returns the flattened sequence
    of qubit operands of the operator, feed-through arguments of any operator arguments, and a
    dictionary mapping argument names to the indices of their respective qubits.
    """
    params = []
    forward_params = []
    param_map = {}

    # Flat dynamic arguments
    for i, dname in enumerate(op_cls.dynamic_argnames):
        params.append(args[i])
        param_map[dname] = ir.DenseI64ArrayAttr.get([i])

    # Hybrid dynamic arguments
    args_idx = len(op_cls.dynamic_argnames) + sum(wire_lens)
    mask_idx = 0
    map_idx = len(op_cls.dynamic_argnames)

    for hname, hsize in zip(op_cls.hybrid_argnames, hybrid_lens, strict=True):
        if hname not in op_cls.wire_argnames:
            leaves = args[args_idx : args_idx + hsize]
            # Any dynamic arguments of input operators are considered feed-forward arguments for
            # decomposition rules, not parameters or qubits of the outer operator. This function
            # is used to partition feed-forward arguments from other dynamic values.
            cur_fwd_mask = forward_mask[mask_idx : mask_idx + hsize]
            cur_params = []

            for leaf, is_forward in zip(leaves, cur_fwd_mask, strict=True):
                if is_forward:
                    forward_params.append(leaf)
                elif not _is_qref_qubit(leaf):
                    cur_params.append(leaf)

            if cur_params:
                params += cur_params
                param_map[hname] = ir.DenseI64ArrayAttr.get(
                    list(range(map_idx, map_idx + len(cur_params)))
                )
                map_idx += len(cur_params)

        mask_idx += hsize
        args_idx += hsize

    param_map = get_mlir_attribute_from_pyval(param_map) if param_map else None
    return params, forward_params, param_map


def _process_qubits(*args, op_cls, wire_lens, hybrid_lens) -> tuple[list, dict[str, list[int]]]:
    """Process qubit operands of an operator. This function returns the flattened sequence
    of qubit operands of the operator, as well as a dictionary mapping argument names to
    the indices of their respective qubits.
    """
    qubits = []
    qubit_map = {}
    flat_wire_argnames = tuple(
        name for name in op_cls.wire_argnames if name not in op_cls.hybrid_argnames
    )

    # Flat wire arguments
    args_idx = len(op_cls.dynamic_argnames)
    map_idx = 0
    for wname, wsize in zip(flat_wire_argnames, wire_lens, strict=True):
        if wsize:
            # If wsize is 0, then we don't want to populate the qubit map
            qubits += args[args_idx : args_idx + wsize]
            qubit_map[wname] = ir.DenseI64ArrayAttr.get(list(range(map_idx, map_idx + wsize)))
            map_idx += wsize
            args_idx += wsize

    # Hybrid wire arguments and nested-operator wires from non-wire hybrid arguments
    for hname, hsize in zip(op_cls.hybrid_argnames, hybrid_lens, strict=True):
        leaves = args[args_idx : args_idx + hsize]
        if hname in op_cls.wire_argnames:
            cur_qubits = leaves
        else:
            cur_qubits = [l for l in leaves if _is_qref_qubit(l)]

        if cur_qubits:
            qubits += cur_qubits
            qubit_map[hname] = ir.DenseI64ArrayAttr.get(
                list(range(map_idx, map_idx + len(cur_qubits)))
            )
            map_idx += len(cur_qubits)

        args_idx += hsize

    qubit_map = get_mlir_attribute_from_pyval(qubit_map) if qubit_map else None
    return qubits, qubit_map


def _qref_operator_p_lowering(
    jax_ctx: mlir.LoweringRuleContext,
    *args,
    op_cls,
    **kwargs,
):
    ctx = jax_ctx.module_context.context
    ctx.allow_unregistered_dialects = True
    _general_validation(*args, op_cls=op_cls, **kwargs)

    hybrid_lens = kwargs.pop("hybrid_lens")
    hybrid_trees = kwargs.pop("hybrid_trees")
    forward_mask = kwargs.pop("forward_mask")
    adjoint = kwargs.pop("adjoint")
    n_ctrls = kwargs.pop("n_ctrls")
    wire_lens = kwargs.pop("wire_lens")

    if n_ctrls:
        ctrl_qubits = args[-2 * n_ctrls : -n_ctrls]
        ctrl_values = [
            TensorExtractOp(ir.IntegerType.get_signless(1), val, []).result
            for val in args[-n_ctrls:]
        ]
        args = args[: -2 * n_ctrls]
    else:
        ctrl_qubits = ctrl_values = ()

    if op_cls.__name__ in _SPECIAL_LOWERINGS:
        expected_len = len(op_cls.dynamic_argnames) + sum(wire_lens)
        assert len(args) == expected_len, f"Incorrect number of operands for {op_cls.__name__}."

        return _SPECIAL_LOWERINGS[op_cls.__name__](
            *args, ctrl_qubits=ctrl_qubits, ctrl_values=ctrl_values, adjoint=adjoint, **kwargs
        )

    name_attr = get_mlir_attribute_from_pyval(op_cls.__name__)

    if _is_custom_op(op_cls, jax_ctx.avals_in[: len(op_cls.dynamic_argnames)]):
        expected_len = len(op_cls.dynamic_argnames) + sum(wire_lens)
        assert len(args) == expected_len, f"Incorrect number of operands for {op_cls.__name__}."

        op_name = op_cls.__name__
        params = [
            extract_scalar(safe_cast_to_f64(p, op_name), op_name)
            for p in args[: len(op_cls.dynamic_argnames)]
        ]
        qubits = args[len(op_cls.dynamic_argnames) : len(op_cls.dynamic_argnames) + sum(wire_lens)]

        CustomOp(
            params=params,
            qubits=qubits,
            gate_name=name_attr,
            ctrl_qubits=ctrl_qubits,
            ctrl_values=ctrl_values,
            adjoint=adjoint,
        )
        return []

    params, forward_args, param_map = _process_params(
        *args,
        op_cls=op_cls,
        wire_lens=wire_lens,
        hybrid_lens=hybrid_lens,
        forward_mask=forward_mask,
    )
    qubits, qubit_map = _process_qubits(
        *args, op_cls=op_cls, wire_lens=wire_lens, hybrid_lens=hybrid_lens
    )
    repack_static_data = {k: unflatten(*v) for k, v in kwargs.items()}

    if op_cls.hybrid_argnames or op_cls.static_argnames:
        uid = generate_uid(
            *jax_ctx.avals_in,
            op_cls=op_cls,
            wire_lens=wire_lens,
            hybrid_lens=hybrid_lens,
            hybrid_trees=hybrid_trees,
            adjoint=adjoint,
            n_ctrls=n_ctrls,
            static_args=repack_static_data,
        )
        static_data = None
    else:
        uid = None
        static_data = get_mlir_attribute_from_pyval(repack_static_data)

    OperatorOp(
        op_name=name_attr,
        params=params,
        qubits=qubits,
        qreg=None,
        forward_args=forward_args,
        ctrl_qubits=ctrl_qubits,
        ctrl_values=ctrl_values,
        adjoint=adjoint,
        UID=uid,
        arr_qubit_indices=[],
        param_map=param_map,
        static_data=static_data,
        qubit_map=qubit_map,
    )

    return []


@_register_special_lowering("MultiRZ")
def _multirz_lowering(theta, *qubits, ctrl_qubits, ctrl_values, adjoint):
    MultiRZOp(
        theta=extract_scalar(safe_cast_to_f64(theta, "MultiRZ"), "MultiRZ"),
        qubits=qubits,
        ctrl_qubits=ctrl_qubits,
        ctrl_values=ctrl_values,
        adjoint=adjoint,
    )
    return []


@_register_special_lowering("PCPhase")
def _pcphase_lowering(theta, dim, *qubits, ctrl_qubits, ctrl_values, adjoint):
    PCPhaseOp(
        theta=extract_scalar(safe_cast_to_f64(theta, "PCPhase"), "PCPhase"),
        dim=extract_scalar(safe_cast_to_f64(dim, "PCPhase"), "PCPhase"),
        qubits=qubits,
        ctrl_qubits=ctrl_qubits,
        ctrl_values=ctrl_values,
        adjoint=adjoint,
    )
    return ()


@_register_special_lowering("GlobalPhase")
def _special_gphase_lowering(angle, *_, ctrl_qubits, ctrl_values, adjoint):
    GlobalPhaseOp(
        angle=extract_scalar(safe_cast_to_f64(angle, "GlobalPhase"), "GlobalPhase"),
        ctrl_qubits=ctrl_qubits,
        ctrl_values=ctrl_values,
        adjoint=adjoint,
    )
    return ()


@_register_special_lowering("QubitUnitary")
def _special_unitary_lowering(matrix, *qubits, ctrl_qubits, ctrl_values, adjoint):
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
def _special_paulirot_lowering(angle, *qubits, ctrl_qubits, ctrl_values, adjoint, pauli_word):
    pauli_word = unflatten(*pauli_word)
    pauli_word = ir.ArrayAttr.get([ir.StringAttr.get(p) for p in pauli_word])

    PauliRotOp(
        angle=extract_scalar(safe_cast_to_f64(angle, "PauliRot"), "PauliRot"),
        pauli_product=pauli_word,
        qubits=qubits,
        ctrl_qubits=ctrl_qubits,
        ctrl_values=ctrl_values,
        adjoint=adjoint,
    )

    return ()
