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

"""Utilities for AOT compiling PennyLane's decomposition rules to MLIR Bytecode."""

from pathlib import Path

import pennylane as qp
from jax._src.lib.mlir import ir
from pennylane.operation import Operator, Operator2

from catalyst.compiler import _quantum_opt
from catalyst.device.python_decompositions import get_graph_op_id, python_decomposition
from catalyst.utils.runtime_environment import BYTECODE_FILE_PATH

# TODO: Uncomment dynamic size wires ops once they are supported
COMPILER_OPS_FOR_DECOMPOSITION = {
    qp.CNOT,
    qp.ControlledPhaseShift,
    qp.CRot,
    qp.CRX,
    qp.CRY,
    qp.CRZ,
    qp.CSWAP,
    qp.CY,
    qp.CZ,
    qp.H,
    # qp.I,
    qp.IsingXX,
    qp.IsingXY,
    qp.IsingYY,
    qp.IsingZZ,
    qp.SingleExcitation,
    qp.SingleExcitationPlus,
    qp.SingleExcitationMinus,
    qp.DoubleExcitation,
    qp.DoubleExcitationPlus,
    qp.DoubleExcitationMinus,
    qp.ISWAP,
    qp.PauliX,
    qp.PauliY,
    qp.PauliZ,
    # qp.PauliRot,
    # qp.PauliMeasure,
    qp.PhaseShift,
    qp.PSWAP,
    qp.Rot,
    qp.RX,
    qp.RY,
    qp.RZ,
    qp.S,
    qp.SWAP,
    qp.T,
    # qp.Toffoli, // adjoint not supported
    qp.U1,
    qp.U2,
    qp.U3,
    # qp.MultiRZ,
    # qp.GlobalPhase,
}


def get_abstract_args(op_class: type[Operator]) -> list[type]:
    """
    Create jax-compatible abstract args for catalyst DecompositionRules that apply to op_class.

    Args:
        op_class: operator to create args for.

    Returns:
        list: abstract args for DecompositionRules.
    """
    # decomposition rule signatures are of the form
    #   (*op_params, wires, **op_resource_params, **hyperparams)
    # see https://github.com/PennyLaneAI/catalyst/pull/2531#discussion_r2949351413
    if isinstance(op_class.ndim_params, tuple) and any(dim > 0 for dim in op_class.ndim_params):
        raise ValueError(
            f"Cannot generate arguments for {op_class.__name__} with multi-dimensional parameters."
        )
    return [float for _ in range(op_class.num_params)]


def get_rules_from_module(module) -> str:
    """
    Parse and modify decomposition rules from a ModuleOp.

    Args:
        module: an MLIR module object containing a FuncOp named `rule_wrapper` to be extracted

    Returns:
        str: The string representation of any decomposition rules from `module`, pre-pending the
             `__builtin_` prefix to their names.
    """
    funcOps = []

    def find_condition(op):
        if op.name == "func.func":
            if "target_gate" in op.attributes:
                op.attributes["sym_name"] = ir.StringAttr.get(
                    "__builtin_" + op.attributes["sym_name"].value.strip('"')
                )
                funcOps.append(op)
                return ir.WalkResult.SKIP
        return ir.WalkResult.ADVANCE

    module.operation.walk(find_condition)

    return "\n".join(str(funcOp) for funcOp in funcOps) if funcOps else ""


def parse_operator_data(op):
    """Parse operator data from an Operator/Operator2 instance."""
    if isinstance(op, Operator2):
        # TODO: use real getters here
        dynamic_shape = op.getDynamicShape()
        wire_lens = op.getWireLens()
        static_data = op.getStaticData()
        return dynamic_shape, wire_lens, static_data
    if issubclass(op, Operator):
        # NOTE: handling this the old-fashioned way, remove once Operator2 migration is complete
        dynamic_shape = get_abstract_args(op)
        num_wires = op.num_wires if op.num_wires else 0
        return dynamic_shape, [num_wires], {}
    else:
        raise ValueError(
            "Only AbstractOperator and CompressedResourceOp types are supported for generating a "
            f"graph ID, got {op} of type {type(op)}"
        )


def precompile_decomp_rules(decomp_file_path: str = BYTECODE_FILE_PATH):
    """
    Compile PennyLane built-in decomposition rules to MLIR Bytecode.

    Intended for use with `make decomp-rules` in catalyst/mlir.

    Args:
        decomp_file_path (Path): path to compile rules to.
    """
    Path(decomp_file_path).parent.mkdir(parents=True, exist_ok=True)

    bytecode_lib = ""

    with ir.Context():
        # TODO: update this for Operator2, PL will implement a precompilation registry
        for op in COMPILER_OPS_FOR_DECOMPOSITION:
            dynamic_data, wire_lens, static_data = parse_operator_data(op)
            if static_data:
                # we cannot precompile if the rule takes static data
                continue

            mlir_rules = python_decomposition(
                op.__name__, get_graph_op_id(op), dynamic_data, wire_lens, {}
            )

            bytecode_lib += get_rules_from_module(mlir_rules) + "\n"

    bytecode = _quantum_opt(
        "--emit-bytecode",
        "--canonicalize",
        "--convert-to-value-semantics",
        "--canonicalize",
        "--register-decomp-rule-resource",
        stdin=bytecode_lib.encode("utf-8"),
        text=None,
    )

    with open(decomp_file_path, "wb") as bytecode_file:
        bytecode_file.write(bytecode)


if __name__ == "__main__":  # pragma: no cover
    precompile_decomp_rules()
