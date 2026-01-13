# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This file contains utility functions for parsing PennyLane objects from xDSL."""

from __future__ import annotations

import contextlib
import inspect
from collections.abc import Callable
from copy import deepcopy
from itertools import compress
from typing import TYPE_CHECKING

from pennylane import capture, ops
from pennylane.ftqc.operations import RotXZX
from pennylane.measurements import counts, expval, probs, sample, state, var
from pennylane.operation import Operator
from pennylane.ops import MidMeasure
from pennylane.ops import __all__ as ops_all
from pennylane.ops import measure
from xdsl.dialects.builtin import DenseIntOrFPElementsAttr, IntegerAttr, IntegerType
from xdsl.dialects.scf import ForOp
from xdsl.dialects.tensor import ExtractOp as TensorExtractOp
from xdsl.ir import Block, SSAValue

from catalyst.jit import QJIT, qjit
from catalyst.python_interface.dialects.qec import (
    PPMeasurementOp,
    PPRotationArbitraryOp,
    PPRotationOp,
)

from ..dialects.quantum import (
    CustomOp,
    ExtractOp,
    GlobalPhaseOp,
    MeasureOp,
    MultiRZOp,
    NamedObsOp,
    PauliRotOp,
    QubitUnitaryOp,
    SetBasisStateOp,
    SetStateOp,
)

if TYPE_CHECKING:
    from jaxlib.mlir._mlir_libs._mlir.ir import Module
    from pennylane.measurements import MeasurementProcess
    from pennylane.workflow.qnode import QNode

has_jax = True
try:
    import jax
except ImportError:
    has_jax = False


def conditional_pause(pause):
    """Will only pause capture if it is enabled."""
    if capture.enabled():
        return pause()

    @contextlib.contextmanager
    def dont_do_anything():
        yield

    return dont_do_anything()


def get_mlir_module(qnode: QNode | QJIT, args, kwargs) -> Module:
    """Ensure the QNode is compiled and return its MLIR module."""
    if hasattr(qnode, "mlir_module") and qnode.mlir_module is not None:
        return qnode.mlir_module

    if isinstance(qnode, QJIT):
        # Deep copy as to not mutate compile_options
        compile_options = deepcopy(qnode.compile_options)
        compile_options.autograph = False  # Autograph has already been applied for `user_function`

        jitted_qnode = QJIT(qnode.user_function, compile_options)
    else:
        jitted_qnode = qjit(qnode)

    jitted_qnode.jit_compile(args, **kwargs)
    return jitted_qnode.mlir_module


from_str_to_PL_gate = {
    name: getattr(ops, name)
    for name in ops_all
    if inspect.isclass(getattr(ops, name, None)) and issubclass(getattr(ops, name), Operator)
}

from_str_to_PL_gate["RotXZX"] = RotXZX  # Include FTQC gates, not in primary module

from_str_to_PL_measurement = {
    "quantum.counts": counts,
    "quantum.state": state,
    "quantum.probs": probs,
    "quantum.sample": sample,
    "quantum.expval": expval,
    "quantum.var": var,
    "quantum.measure": measure,
}


######################################################
### Gate/Measurement resolution
######################################################


def _resolve(name: str, mapping: dict, kind: str):
    try:
        return mapping[name]
    except KeyError as exc:
        raise NotImplementedError(f"Unsupported {kind}: {name}") from exc


def resolve_gate(name: str) -> Operator:
    """Resolve the gate from the name."""
    return _resolve(name, from_str_to_PL_gate, "gate")


def resolve_measurement(name: str) -> MeasurementProcess:
    """Resolve the measurement from the name."""
    return _resolve(name, from_str_to_PL_measurement, "measurement")


######################################################
### Helpers
######################################################


def _tensor_shape_from_ssa(ssa: SSAValue) -> list[int]:
    """Extract the concrete shape from an SSA tensor value."""
    ssa_type = ssa.type

    if hasattr(ssa_type, "shape") and hasattr(ssa_type.shape, "data"):
        return [dim.data for dim in ssa_type.shape.data]

    raise AttributeError(
        f"Could not extract tensor shape from {ssa.type}. "
        f"SSA type {ssa.type} does not have expected shape data."
    )


def _extract(op, attr: str, resolver: Callable, single: bool = False):
    """Helper to extract and resolve attributes."""
    values = getattr(op, attr, None)
    if not values:
        return [] if not single else None
    return resolver(values) if single else [resolver(v) for v in values if v is not None]


def _extract_dense_constant_value(op) -> float | int:
    """Extract the first value from a stablehlo.constant op."""
    attr = op.properties.get("value")
    if isinstance(attr, DenseIntOrFPElementsAttr):
        # TODO: handle multi-value cases if needed
        return attr.get_values()[0]
    raise NotImplementedError(f"Unexpected attr type in constant: {type(attr)}")


def _apply_adjoint_and_ctrls(qml_op: Operator, xdsl_op) -> Operator:
    """Apply adjoint and control modifiers to a gate if needed."""
    if xdsl_op.properties.get("adjoint"):
        qml_op = ops.op_math.adjoint(qml_op)
    ctrls = ssa_to_qml_wires(xdsl_op, control=True)
    if ctrls:
        cvals = ssa_to_qml_params(xdsl_op, control=True)
        qml_op = ops.op_math.ctrl(qml_op, control=ctrls, control_values=cvals)
    return qml_op


# pylint: disable=too-many-return-statements
def resolve_constant_params(ssa: SSAValue) -> float | int | str:
    """Resolve a constant parameter SSA value to a Python float or int."""
    op = ssa.owner

    if isinstance(op, Block):
        arg_name = next(compress(op.args, map(lambda arg: arg is ssa, op.args)))
        return arg_name.name_hint

    if isinstance(op, TensorExtractOp):
        return resolve_constant_params(op.tensor)

    if not hasattr(op, "name"):
        raise NotImplementedError(f"Cannot resolve parameters for operation: {op}")

    match op.name:
        case "func.call":
            if op.callee.string_value() == "remainder":
                x = resolve_constant_params(op.operands[0])
                y = resolve_constant_params(op.operands[1])
                return f"({x} % {y})"
            raise NotImplementedError(f"Function call to {op.callee} not supported")

        case "tensor.from_elements":
            return resolve_constant_params(op.operands[0])

        case "arith.index_cast":
            return resolve_constant_params(op.operands[0])

        case "arith.addf":
            return sum(resolve_constant_params(o) for o in op.operands)

        case "arith.constant":
            return op.value.value.data  # Catalyst

        case "arith.index_cast":
            return resolve_constant_params(op.input)

        case "stablehlo.add":
            x, y = (
                resolve_constant_params(op.operands[0]),
                resolve_constant_params(op.operands[1]),
            )
            return f"({x} + {y})"

        case "stablehlo.subtract":
            x, y = (
                resolve_constant_params(op.operands[0]),
                resolve_constant_params(op.operands[1]),
            )
            return f"({x} - {y})"

        case "stablehlo.constant":
            return _extract_dense_constant_value(op)

        case "stablehlo.convert" | "stablehlo.broadcast_in_dim":
            return resolve_constant_params(op.operands[0])

        case "stablehlo.concatenate":
            return [resolve_constant_params(operand) for operand in op.operands]

        case "stablehlo.reshape":
            res_type = op.result_types[0]
            shape = res_type.get_shape()
            type_ = res_type.get_element_type()
            return jax.numpy.array(shape, dtype=int if isinstance(type_, IntegerType) else float)

        case _:
            raise NotImplementedError(f"Cannot resolve parameters for operation: {op}")


def count_static_loop_iterations(for_op: ForOp) -> int:
    """
    Calculates static loop iterations for a given ForOp.

    Requires that the loop bounds and step are constant values.
    """

    lower_bound = resolve_constant_params(for_op.lb)
    upper_bound = resolve_constant_params(for_op.ub)
    step = resolve_constant_params(for_op.step)
    if not all(isinstance(x, int) for x in [lower_bound, upper_bound, step]):
        raise NotImplementedError("Dynamic loop iterations (strings) are not supported.")

    if upper_bound <= lower_bound:
        return 0

    num_elements = upper_bound - lower_bound
    return (num_elements + step - 1) // step


def dispatch_wires_extract(op: ExtractOp):
    """Dispatch the wire resolution for the given extract operation."""
    if op.idx_attr is not None:  # used by Catalyst
        return resolve_constant_wire(op.idx_attr)
    return resolve_constant_wire(op.idx)  # used by xDSL


def resolve_constant_wire(ssa: SSAValue) -> float | int | str:
    """Resolve the wire for the given SSA qubit."""
    if isinstance(ssa, IntegerAttr):  # Catalyst
        return ssa.value.data

    op = ssa.owner

    if isinstance(op, Block):
        arg_name = next(compress(op.args, map(lambda arg: arg is ssa, op.args)))
        return arg_name.name_hint

    match op:
        case _ if op.name == "func.call":
            if op.callee.string_value() == "remainder":
                x = resolve_constant_params(op.operands[0])
                y = resolve_constant_params(op.operands[1])
                return f"({x} % {y})"
            raise NotImplementedError(f"Function call to {op.callee} not supported")

        case _ if op.name == "stablehlo.reshape":
            return resolve_constant_wire(op.operands[0])

        case _ if op.name == "stablehlo.add":
            x, y = (resolve_constant_wire(op.operands[0]), resolve_constant_wire(op.operands[1]))
            return f"({x} + {y})"

        case _ if op.name == "stablehlo.subtract":
            x, y = (resolve_constant_wire(op.operands[0]), resolve_constant_wire(op.operands[1]))
            return f"({x} - {y})"

        case _ if op.name == "tensor.from_elements":
            return resolve_constant_wire(op.operands[0])

        case _ if op.name == "arith.index_cast":
            return resolve_constant_params(op.operands[0])

        case _ if op.name == "arith.constant":
            return op.value.value.data  # Catalyst

        case TensorExtractOp(tensor=tensor):
            return resolve_constant_wire(tensor)

        case _ if op.name == "stablehlo.convert":
            return resolve_constant_wire(op.operands[0])

        case _ if op.name == "stablehlo.constant":
            return _extract_dense_constant_value(op)

        case (
            CustomOp()
            | GlobalPhaseOp()
            | QubitUnitaryOp()
            | SetStateOp()
            | MultiRZOp()
            | SetBasisStateOp()
            | PPRotationOp()
            | PPRotationArbitraryOp()
            | PauliRotOp()
        ):
            all_qubits = list(getattr(op, "in_qubits", [])) + list(
                getattr(op, "in_ctrl_qubits", [])
            )
            return resolve_constant_wire(all_qubits[ssa.index])

        case ExtractOp():
            return dispatch_wires_extract(op)

        case MeasureOp(in_qubit=in_qubit):
            return resolve_constant_wire(in_qubit)

        case PPMeasurementOp():
            # NOTE: This branch is needed to cover two PPMs in a row
            # subtract one as the first ssa index is the result,
            # %res, %q0, ... = qec.ppm [PAULI_WORD] %q0, ...
            return resolve_constant_wire(op.operands[ssa.index - 1])
        case _:
            raise NotImplementedError(f"Cannot resolve wire for op: {op}")


######################################################
### Parameters/Wires Conversion
######################################################


def ssa_to_qml_params(
    op, control: bool = False, single: bool = False
) -> list[float | int] | float | int | None:
    """Get the parameters from the operation."""
    return _extract(op, "in_ctrl_values" if control else "params", resolve_constant_params, single)


def ssa_to_qml_wires(op: CustomOp, control: bool = False) -> list[int]:
    """Get the wires from the operation."""
    return _extract(op, "in_ctrl_qubits" if control else "in_qubits", resolve_constant_wire)


def ssa_to_qml_wires_named(op: NamedObsOp) -> int:
    """Get the wire from the named observable operation."""
    if not op.qubit:
        raise ValueError("No qubit found for named observable operation.")
    return resolve_constant_wire(op.qubit)


############################################################
### xDSL ---> PennyLane Operators/Measurements conversion
############################################################


def xdsl_to_qml_op(op) -> Operator:
    """Convert an xDSL operation into a PennyLane Operator.

    Args:
        op: The xDSL operation to convert.

    Returns:
        A PennyLane Operator.
    """
    # Pause capture *only if active* so we can allow strings (dynamic wires) as allowed wires
    with conditional_pause(capture.pause):
        match op.name:
            case "quantum.paulirot":
                pw = []
                for str_attr in op.pauli_product.data:
                    pw.append(str(str_attr).replace('"', ""))
                pw = "".join(pw)
                gate = ops.PauliRot(
                    theta=_extract(op, "angle", resolve_constant_params, single=True),
                    pauli_word=pw,
                    wires=ssa_to_qml_wires(op),
                )
            case "quantum.gphase":
                gate = ops.GlobalPhase(
                    ssa_to_qml_params(op, single=True), wires=ssa_to_qml_wires(op)
                )

            case "quantum.unitary":
                gate = ops.qubit.matrix_ops.QubitUnitary(
                    U=jax.numpy.zeros(_tensor_shape_from_ssa(op.matrix)),
                    wires=ssa_to_qml_wires(op),
                )

            case "quantum.set_state":
                gate = ops.qubit.state_preparation.StatePrep(
                    state=jax.numpy.zeros(_tensor_shape_from_ssa(op.in_state)),
                    wires=ssa_to_qml_wires(op),
                )

            case "quantum.multirz":
                gate = ops.qubit.parametric_ops_multi_qubit.MultiRZ(
                    theta=_extract(op, "theta", resolve_constant_params, single=True),
                    wires=ssa_to_qml_wires(op),
                )

            case "quantum.set_basis_state":
                gate = ops.qubit.state_preparation.BasisState(
                    state=jax.numpy.zeros(_tensor_shape_from_ssa(op.basis_state)),
                    wires=ssa_to_qml_wires(op),
                )

            case "quantum.custom":
                gate_cls = resolve_gate(op.properties.get("gate_name").data)
                gate = gate_cls(*ssa_to_qml_params(op), wires=ssa_to_qml_wires(op))

            case _:
                raise NotImplementedError(f"Unsupported gate: {op.name}")

    return _apply_adjoint_and_ctrls(gate, op)


def xdsl_to_qml_op_name(op, adjoint_mode: bool) -> str:
    """Convert an xDSL operation into a string representing a PennyLane Operator.

    Args:
        op: The xDSL operation to convert.
        adjoint_mode: If True, treat all non-adjoint gates as adjoint, and vice versa.

    Returns:
        A string representing the PennyLane operator.
    """

    name_map = {
        "quantum.gphase": "GlobalPhase",
        "quantum.multirz": "MultiRZ",
        "quantum.set_basis_state": "BasisState",
        "quantum.set_state": "StatePrep",
        "quantum.unitary": "QubitUnitary",
        "quantum.paulirot": "PauliRot",
    }

    if op.name == "quantum.custom":
        gate_name = op.properties.get("gate_name").data
    elif op.name in name_map:
        gate_name = name_map[op.name]
    else:
        raise NotImplementedError(f"Unsupported gate: {op.name}")

    is_adjoint = op.properties.get("adjoint") is not None
    if adjoint_mode:
        # Adjoint-mode means all non-adjoint gates are treated as adjoint, and vice versa
        is_adjoint = not is_adjoint

    if is_adjoint and not gate_name in ops.qubit.attributes.self_inverses:
        gate_name = f"Adjoint({gate_name})"

    if hasattr(op, "in_ctrl_qubits"):
        n_ctrls = len(op.in_ctrl_qubits)
        if n_ctrls == 1:
            gate_name = f"C({gate_name})"
        elif n_ctrls > 1:
            gate_name = f"{n_ctrls}C({gate_name})"
    return gate_name


def xdsl_to_qml_measurement(op, *args, **kwargs) -> MeasurementProcess | Operator:
    """Convert any xDSL measurement/observable operation to a PennyLane object.

    Args:
        op: The xDSL measurement/observable operation to convert.

    Returns:
        A PennyLane MeasurementProcess or Operator.
    """

    with conditional_pause(capture.pause):
        match op.name:
            case "quantum.measure":
                postselect = op.postselect.value.data if op.postselect is not None else None
                return MidMeasure([resolve_constant_wire(op.in_qubit)], postselect=postselect)

            case "quantum.namedobs":
                return resolve_gate(op.type.data.value)(wires=ssa_to_qml_wires_named(op))

            case "quantum.tensor":
                return ops.op_math.prod(
                    *(xdsl_to_qml_measurement(operand.owner) for operand in op.operands)
                )

            case "quantum.hamiltonian":
                coeffs = _extract(op, "coeffs", resolve_constant_params, single=True)
                ops_list = [xdsl_to_qml_measurement(term.owner) for term in op.terms]
                return ops.LinearCombination(coeffs, ops_list)
            case "quantum.compbasis":
                return _extract(op, "qubits", resolve_constant_wire)

            case (
                "quantum.state"
                | "quantum.probs"
                | "quantum.sample"
                | "quantum.expval"
                | "quantum.var"
            ):
                return resolve_measurement(op.name)(*args, **kwargs)

            case _:
                raise NotImplementedError(f"Unsupported measurement/observable: {op.name}")


def xdsl_to_qml_measurement_name(op, obs_op=None) -> str:
    """Convert any xDSL measurement/observable operation into a string representing a PennyLane
    measurement.

    Args:
        op: The xDSL measurement/observable operation to convert.
        obs_op: An optional string representing the observable operation.

    Returns:
        A string representing the PennyLane measurement.
    """

    if op.name == "quantum.measure":
        gate_name = "MidMeasure"

    elif op.name == "quantum.compbasis":
        # Defines a pseudo-observable to represent measurements in the computational basis
        # Used within e.g. `probs()`
        if len(op.qubits) == 0:
            # No specified qubits means use all qubits
            gate_name = "all wires"
        else:
            gate_name = f"{len(op.qubits)} wires"

    elif op.name == "quantum.hamiltonian":
        gate_name = f"Hamiltonian(num_terms={len(op.terms)})"

    elif op.name == "quantum.tensor":
        gate_name = f"Prod(num_terms={len(op.operands)})"

    elif op.name == "quantum.namedobs":
        gate_name = op.type.data.value

    elif op.name in from_str_to_PL_measurement:
        gate_name = op.name.split(".")[-1]

    else:
        raise NotImplementedError(f"Unsupported measurement/observable: {op.name}")

    if obs_op != None:
        gate_name = f"{gate_name}({obs_op})"

    return gate_name
