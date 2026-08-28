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
import pennylane as qp
from jax._src.lib.mlir import ir
from jax.core import ShapedArray
from jax.extend.core import Primitive
from jax.interpreters import mlir
from jaxlib.mlir._mlir_libs import _mlir as _ods_cext
from jaxlib.mlir.dialects.stablehlo import ConvertOp as StableHLOConvertOp
from pennylane.core.operator.utils import abstractify
from pennylane.pytrees import unflatten
from pennylane.typing import AbstractArray
from pennylane.wires import AbstractQubit

# TODO: remove after jax v0.7.2 upgrade
# Mock _ods_cext.globals.register_traceback_file_exclusion due to API conflicts between
# Catalyst's MLIR version and the MLIR version used by JAX. The current JAX version has not
# yet updated to the latest MLIR, causing compatibility issues. This workaround will be removed
# once JAX updates to a compatible MLIR version
from catalyst.decomposition.decomposition_rules import (
    fetch_all_reachable_decomposition_rules_from_op,
    inject_new_rules_into_module,
)
from catalyst.decomposition.graph_op_id import _SPECIAL_LOWERINGS
from catalyst.decomposition.type_utils import (
    convert_types_to_mlir_strings,
    format_dynamic_params_for_id,
)
from catalyst.jax_extras.lowering import get_mlir_attribute_from_pyval
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
        SetBasisStateOp,
    )


def _register_special_lowering(op_cls):
    def decorator(f):
        _SPECIAL_LOWERINGS[op_cls] = f
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
    if list(op_cls._sig.parameters.keys())[-1] != "wires":
        return False
    # Complex dtypes cannot be safely cast to float64
    return all(p.shape == () and p.dtype.kind in "ifu" for p in avals_in)


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
    """Process the dynamic arguments of an operator. This function returns the flattened sequence
    of non-qubit operands of the operator, feed-through arguments of any operator arguments,
    and a dictionary mapping argument names to the indices of their respective values.
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
            # decomposition rules, not parameters of the outer operator. This function is used
            # to partition feed-forward arguments from other dynamic values. Note that _wires_ of
            # operator arguments _ are not_ considered feed-forward arguments. Rather, it is assumed
            # that an operator of operators acts on all the wires of its operator arguments.
            cur_fwd_mask = forward_mask[mask_idx : mask_idx + hsize]
            cur_params = []

            for leaf, is_forward in zip(leaves, cur_fwd_mask, strict=True):
                if is_forward:
                    forward_params.append(leaf)
                elif not _is_qref_qubit(leaf):
                    # Qubits are handled in _process_qubits
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
    for param in params:
        assert isinstance(param.type, ir.RankedTensorType)
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
            # If wsize is 0, then we don't need to populate the qubit map. It will be empty anyway
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


@abstractify.register(ShapedArray)
def _abstractify_jax_array(val):
    return AbstractArray(val.shape, val.dtype)


# pylint: disable=too-many-arguments,too-many-branches
def compile_decomp_rules(
    module,
    op_cls,
    is_custom_op=False,
    params=None,
    param_map=None,
    wire_lens=None,
    qubit_map=None,
    hybrid_lens=None,
    hybrid_trees=None,
    repack_static_data=None,
    uid=None,
    avals_in=None,
):
    """
    Generate all the decomposition rules registered on the current gate, recursively generating all
    the rules that are registered on the resource gates of these rules as well.
    """
    if is_custom_op:
        dynamic_shape = {str(i): ["f64"] for i in range(len(op_cls.dynamic_argnames))}

        op_id = (
            op_cls.__name__
            + format_dynamic_params_for_id(dynamic_shape)
            + "{"
            + f"wires:{wire_lens[0]}"
            + "}{}"
        )

        decomp_rules = fetch_all_reachable_decomposition_rules_from_op(
            op_name=op_cls.__name__,
            op_id=op_id,
            dynamic_shape=dynamic_shape,
            wire_lens={"wires": wire_lens[0]},
            static_data={},
            is_custom_op=True,
        )

    elif op_cls is qp.MultiRZ:
        dynamic_shape = {qp.MultiRZ.dynamic_argnames[0]: ["f64"]}
        wire_argname = qp.MultiRZ.wire_argnames[0]
        op_id = (
            "MultiRZ"
            + format_dynamic_params_for_id(dynamic_shape)
            + "{"
            + f"{wire_argname}:{wire_lens[0]}"
            + "}{}"
        )

        decomp_rules = fetch_all_reachable_decomposition_rules_from_op(
            op_name="MultiRZ",
            op_id=op_id,
            dynamic_shape=dynamic_shape,
            wire_lens={f"{wire_argname}": wire_lens[0]},
            static_data={},
        )

    elif op_cls is qp.PauliRot:
        dynamic_shape = {qp.PauliRot.dynamic_argnames[0]: ["f64"]}
        wire_argname = qp.PauliRot.wire_argnames[0]
        pauliword_argname = qp.PauliRot.compilable_argnames[0]
        op_id = (
            "PauliRot"
            + format_dynamic_params_for_id(dynamic_shape)
            + "{"
            + f"{wire_argname}:{wire_lens[0]}"
            + "}"
            + "{"
            + f"{pauliword_argname}:{repack_static_data[pauliword_argname]}"
            + "}"
        )

        decomp_rules = fetch_all_reachable_decomposition_rules_from_op(
            op_name="PauliRot",
            op_id=op_id,
            dynamic_shape=dynamic_shape,
            wire_lens={f"{wire_argname}": wire_lens[0]},
            static_data=repack_static_data,
        )

    elif op_cls is qp.PCPhase:
        dynamic_shape = {qp.PCPhase.dynamic_argnames[0]: ["f64"]}
        wire_argname = qp.PCPhase.wire_argnames[0]
        op_id = (
            "PCPhase"
            + format_dynamic_params_for_id(dynamic_shape)
            + "{"
            + f"{wire_argname}:{wire_lens[0]}"
            + "}{"
            + f"dim:{repack_static_data["dim"]}"
            + "}"
        )

        decomp_rules = fetch_all_reachable_decomposition_rules_from_op(
            op_name="PCPhase",
            op_id=op_id,
            dynamic_shape=dynamic_shape,
            wire_lens={f"{wire_argname}": wire_lens[0]},
            static_data=repack_static_data,
        )

    elif op_cls is qp.GlobalPhase:
        dynamic_shape = {qp.GlobalPhase.dynamic_argnames[0]: ["f64"]}
        op_id = "GlobalPhase" + format_dynamic_params_for_id(dynamic_shape) + "{}{}"

        decomp_rules = fetch_all_reachable_decomposition_rules_from_op(
            op_name="GlobalPhase",
            op_id=op_id,
            dynamic_shape=dynamic_shape,
            wire_lens={},
            static_data={},
        )

    elif op_cls is qp.BasisState:
        # TODO: qp.BasisState decomp rule calls allclose, but the current infra cannot support
        # rules that call other funcops
        # When the above is implemented, uncomment the BasisState decomp rule collection impl below
        op_id = ""
        decomp_rules = []

        # # qp.BasisState has the same number of booleans as the number of wires
        # num_wires = wire_lens[0]
        # dynamic_shape = {qp.BasisState.dynamic_argnames[0]: ["i64"] * num_wires}
        # wire_argname = qp.BasisState.wire_argnames[0]
        # op_id = (
        #     "BasisState"
        #     + format_dynamic_params_for_id(dynamic_shape)
        #     + "{"
        #     + f"{wire_argname}:{num_wires}"
        #     + "}{}"
        # )

        # decomp_rules = fetch_all_reachable_decomposition_rules_from_op(
        #     op_name="BasisState",
        #     op_id=op_id,
        #     dynamic_shape=dynamic_shape,
        #     wire_lens={f"{wire_argname}": num_wires},
        #     static_data={},
        # )

    elif op_cls is qp.QubitUnitary:
        # TODO: qp.QubitUnitary decomp rule calls det, but the current infra cannot support
        # rules that call other funcops
        # When the above is implemented, uncomment the Unitary decomp rule collection impl below

        op_id = ""
        decomp_rules = []

        # num_wires = wire_lens[0]
        # matrix_size = 2**num_wires
        # dynamic_shape = {
        #     qp.QubitUnitary.dynamic_argnames[0]: [["complex<f64>"] * matrix_size] * matrix_size
        # }
        # wire_argname = qp.QubitUnitary.wire_argnames[0]
        # op_id = (
        #     "QubitUnitary"
        #     + "[" + format_dynamic_params_for_id(dynamic_shape) + "]"
        #     + "{"
        #     + f"{wire_argname}:{wire_lens[0]}"
        #     + "}{}"
        # )

        # decomp_rules = fetch_all_reachable_decomposition_rules_from_op(
        #     op_name="QubitUnitary",
        #     op_id=op_id,
        #     dynamic_shape=dynamic_shape,
        #     wire_lens={f"{wire_argname}": wire_lens[0]},
        #     static_data={},
        # )

    else:
        # Operator Op
        non_hybrid_dynamic_shape = {}

        indices_to_remove = set()
        if param_map is not None:
            for named_attr in param_map:
                if named_attr.name in op_cls.hybrid_argnames:
                    for idx in named_attr.attr:
                        indices_to_remove.add(int(idx))
        non_hybrid_params = [p for i, p in enumerate(params) if i not in indices_to_remove]

        for dynamic_argname, param in zip(op_cls.dynamic_argnames, non_hybrid_params, strict=True):
            non_hybrid_dynamic_shape[dynamic_argname] = param.type
        non_hybrid_dynamic_shape = convert_types_to_mlir_strings(non_hybrid_dynamic_shape)

        non_hybrid_wire_argnames = []
        for wire_argname in op_cls.wire_argnames:
            if wire_argname not in op_cls.hybrid_argnames:
                non_hybrid_wire_argnames.append(wire_argname)
        non_hybrid_wire_lens = {
            a: b for a, b in zip(non_hybrid_wire_argnames, wire_lens, strict=True)
        }

        extra_data = {}
        non_hybrid_wire_len = 0
        if qubit_map is not None:
            for w in non_hybrid_wire_argnames:
                if w in qubit_map:
                    non_hybrid_wire_len += len(qubit_map[w])
        hybrid_arg_start_idx = len(non_hybrid_params) + non_hybrid_wire_len
        for hybrid_argname, hybrid_len, hybrid_tree in zip(
            op_cls.hybrid_argnames, hybrid_lens, hybrid_trees
        ):
            replaced_leaves = []
            for leaf in avals_in[hybrid_arg_start_idx : hybrid_arg_start_idx + hybrid_len]:
                if isinstance(leaf, AbstractQubit):
                    replaced_leaves.append(ShapedArray((), dtype=int))
                else:
                    replaced_leaves.append(leaf)

            with Patcher(
                (AbstractArray, "__hash__", lambda x: id(x)),
            ):
                replaced_leaves = abstractify(replaced_leaves)
                unflattened = unflatten(replaced_leaves, hybrid_tree)
                unflattened = abstractify(unflattened)
            extra_data[hybrid_argname] = unflattened
            hybrid_arg_start_idx += hybrid_len

        with_hybrid_dynamic_shape = {}
        if param_map is not None:
            for named_attr in param_map:
                with_hybrid_dynamic_shape[named_attr.name] = [
                    params[idx].type for idx in named_attr.attr
                ]
            with_hybrid_dynamic_shape = convert_types_to_mlir_strings(with_hybrid_dynamic_shape)

        with_hybrid_wire_lens = {}
        if qubit_map is not None:
            for wire_attr in qubit_map:
                with_hybrid_wire_lens[wire_attr.name] = len(wire_attr.attr)

        op_id = (
            op_cls.__name__
            + format_dynamic_params_for_id(dict(sorted(with_hybrid_dynamic_shape.items())))
            + "{"
            + ",".join(f"{name}:{shape}" for name, shape in sorted(with_hybrid_wire_lens.items()))
            + "}"
        )
        if not (op_cls.hybrid_argnames or op_cls.static_argnames):
            op_id += "{" + ",".join(f"{k}:{v}" for k, v in sorted(repack_static_data.items())) + "}"
        else:
            op_id += "{}"
        if uid is not None:
            op_id += f"[{str(uid)}]"

        decomp_rules = fetch_all_reachable_decomposition_rules_from_op(
            op_name=op_cls.__name__,
            op_id=op_id,
            dynamic_shape=non_hybrid_dynamic_shape,
            wire_lens=non_hybrid_wire_lens,
            static_data=repack_static_data,
            extra_data=extra_data,
        )

    inject_new_rules_into_module(module, decomp_rules)


def _qref_operator_p_lowering(jax_ctx: mlir.LoweringRuleContext, *args, op_cls, **kwargs):
    ctx = jax_ctx.module_context.context
    ctx.allow_unregistered_dialects = True
    _general_validation(*args, op_cls=op_cls, **kwargs)

    hybrid_lens = kwargs.pop("hybrid_lens")
    hybrid_trees = kwargs.pop("hybrid_trees")
    forward_mask = kwargs.pop("forward_mask")
    adjoint = kwargs.pop("adjoint")
    n_ctrls = kwargs.pop("n_ctrls")
    wire_lens = kwargs.pop("wire_lens")
    collect_decomp_rules = kwargs.pop("collect_decomp_rules")

    repack_static_data = {k: unflatten(*v) for k, v in kwargs.items()}

    if n_ctrls:
        ctrl_qubits = args[-2 * n_ctrls : -n_ctrls]
        ctrl_values = [
            TensorExtractOp(ir.IntegerType.get_signless(1), val, []).result
            for val in args[-n_ctrls:]
        ]
        args = args[: -2 * n_ctrls]
    else:
        ctrl_qubits = ctrl_values = ()

    # Custom lowerings (qref.multirz, qref.pcphase, etc.)
    if op_cls in _SPECIAL_LOWERINGS:
        expected_len = len(op_cls.dynamic_argnames) + sum(wire_lens)
        assert len(args) == expected_len, f"Incorrect number of operands for {op_cls.__name__}."

        if collect_decomp_rules:
            compile_decomp_rules(
                module=jax_ctx.module_context.module,
                op_cls=op_cls,
                wire_lens=wire_lens,
                repack_static_data=repack_static_data,
            )

        return _SPECIAL_LOWERINGS[op_cls](
            *args,
            ctrl_qubits=ctrl_qubits,
            ctrl_values=ctrl_values,
            adjoint=adjoint,
            **kwargs,
        )

    name_attr = get_mlir_attribute_from_pyval(op_cls.__name__)

    # Lowering to qref.custom
    # Custom op only has float dynamic args, followed by a single wire argname "wires" at the end
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

        if collect_decomp_rules:
            compile_decomp_rules(
                module=jax_ctx.module_context.module,
                op_cls=op_cls,
                is_custom_op=True,
                wire_lens=wire_lens,
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

    # Collect decomp rules reachable from the current op
    if collect_decomp_rules:
        compile_decomp_rules(
            module=jax_ctx.module_context.module,
            op_cls=op_cls,
            is_custom_op=False,
            params=params,
            param_map=param_map,
            wire_lens=wire_lens,
            qubit_map=qubit_map,
            hybrid_lens=hybrid_lens,
            hybrid_trees=hybrid_trees,
            repack_static_data=repack_static_data,
            uid=uid,
            avals_in=jax_ctx.avals_in,
        )

    return []


@_register_special_lowering(qp.MultiRZ)
def _multirz_lowering(theta, *qubits, ctrl_qubits, ctrl_values, adjoint):
    MultiRZOp(
        theta=extract_scalar(safe_cast_to_f64(theta, "MultiRZ"), "MultiRZ"),
        qubits=qubits,
        ctrl_qubits=ctrl_qubits,
        ctrl_values=ctrl_values,
        adjoint=adjoint,
    )
    return []


@_register_special_lowering(qp.PCPhase)
def _pcphase_lowering(theta, *qubits, ctrl_qubits, ctrl_values, adjoint, dim):
    dim = unflatten(*dim)
    PCPhaseOp(
        theta=extract_scalar(safe_cast_to_f64(theta, "PCPhase"), "PCPhase"),
        dim=get_mlir_attribute_from_pyval(dim),
        qubits=qubits,
        ctrl_qubits=ctrl_qubits,
        ctrl_values=ctrl_values,
        adjoint=adjoint,
    )
    return ()


@_register_special_lowering(qp.GlobalPhase)
def _special_gphase_lowering(angle, *_, ctrl_qubits, ctrl_values, adjoint):
    GlobalPhaseOp(
        angle=extract_scalar(safe_cast_to_f64(angle, "GlobalPhase"), "GlobalPhase"),
        ctrl_qubits=ctrl_qubits,
        ctrl_values=ctrl_values,
        adjoint=adjoint,
    )
    return ()


@_register_special_lowering(qp.BasisState)
def _special_basis_state_lowering(state, *qubits, ctrl_qubits, ctrl_values, adjoint):
    assert not ctrl_qubits and not ctrl_values, "ctrl(BasisState) is not supported."
    assert not adjoint, "adjoint(BasisState) is not supported."
    SetBasisStateOp(state, qubits)
    return ()


@_register_special_lowering(qp.QubitUnitary)
def _special_unitary_lowering(matrix, *qubits, ctrl_qubits, ctrl_values, adjoint, unitary_check):
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


@_register_special_lowering(qp.PauliRot)
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
