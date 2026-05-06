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

import warnings
from pathlib import Path

import jax
import pennylane as qp
from jax._src.lib.mlir import ir
from pennylane.operation import Operator

from catalyst.compiler import _quantum_opt
from catalyst.jax_primitives import decomposition_rule
from catalyst.utils.exceptions import CompileError
from catalyst.utils.runtime_environment import BYTECODE_FILE_PATH

# TODO: Uncomment dynamic size wires ops once they are supported
# FIXME: Use the Gate class instead of this list of compiler ops
#              https://github.com/PennyLaneAI/pennylane/pull/8767
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


def get_func_from_circuit(module) -> str | None:
    """
    Get the string representation of `rule_wrapper` from module, if it exists.

    Args:
        module: an MLIR module object containing a FuncOp named `rule_wrapper` to be extracted

    Returns:
        str: string representation of FuncOp named `rule_wrapper` from module
        None: if no such FuncOp can be found
    """
    decomp_func_op = None

    def find_condition(op):
        nonlocal decomp_func_op
        if op.name == "func.func":
            if ir.StringAttr(op.attributes["sym_name"]).value == "rule_wrapper":
                decomp_func_op = op
                return ir.WalkResult.INTERRUPT
        return ir.WalkResult.ADVANCE

    module.operation.walk(find_condition)

    return str(decomp_func_op) + "\n" if decomp_func_op else None


def compile_rule(
    op_class,
    abstract_args,
    op_num_wires,
    rule,
    dev,
) -> str | None:
    """
    Get the string representation of a compiled rule from a python decomposition rule, if possible.

    NOTE: rules with string params are not currently supported.

    Args:
        op_class: A PennyLane class subclassing Operation
        op_num_wires: the number of wires used by op_class
        rule (DecompositionRule): the decomposition rule to be compiled
        dev (Device): a device for qjit

    Returns:
        str: string representation of the mlir of the decomposition rule.
    """
    qp.decomposition.enable_graph()

    # WARNING: do not rename this function, we use it to extract the rule from the compiled
    # circuit
    @decomposition_rule(is_qreg=True, op_type=op_class.__name__)
    def rule_wrapper(*args, wires, **_):
        return rule(*args, wires=wires, **_)

    @qp.qjit(capture=True, target="mlir")
    @qp.qnode(dev)
    def circuit():
        rule_wrapper(*abstract_args, wires=jax.core.ShapedArray((op_num_wires,), int))
        return qp.probs()

    return get_func_from_circuit(circuit.mlir_module)


def compile_op_decomp_rules(
    op_class: type[Operator],
) -> dict[str, str | None]:
    """
    Compile all decomposition rules for op_class.

    Note: the modules include the full circuit IR.

    Args:
        op_class (type[Operator]): the op class to compile decomposition rules for.

    Returns:
        dict[str, str | None]: decomposition rule names to compiled mlir modules.
    """
    op_decomp_rules = qp.decomposition.decomposition_graph.list_decomps(op_class)

    mlir_modules: dict[str, str | None] = {}

    if not hasattr(op_class, "num_wires") or not op_class.num_wires:
        warnings.warn(
            f"Cannot compile decomposition rules for op {op_class.__name__} with an unknown number "
            + "of wires."
        )
        return mlir_modules

    dev = qp.device("null.qubit", wires=op_class.num_wires)

    abstract_args = get_abstract_args(op_class)  # pylint: disable=protected-access

    for rule in op_decomp_rules:
        try:
            rule_name = rule._impl.__name__  # pylint: disable=protected-access
            mlir_modules[rule_name] = compile_rule(
                op_class, abstract_args, op_class.num_wires, rule, dev
            )
        except CompileError as e:
            warnings.warn(f"Failed to compile {rule_name}: {e}")
        except Exception as e:  # pylint: disable=broad-exception-caught
            warnings.warn(f"Unexpected error while trying to compile {rule_name}: {e}")
        finally:
            qp.decomposition.disable_graph()

    return mlir_modules


def precompile_decomp_rules(decomp_file_path: Path = BYTECODE_FILE_PATH):
    """
    Compile PennyLane built-in decomposition rules to MLIR Bytecode.

    Intended for use with `make decomp-rules` in catalyst/mlir.

    Args:
        decomp_file_path (Path): path to compile rules to.
    """
    decomp_file_path.parent.mkdir(parents=True, exist_ok=True)

    mlir_rules = "".join(
        str(mlir).replace("@rule_wrapper", f"@__builtin_{name}")
        for func in COMPILER_OPS_FOR_DECOMPOSITION
        for name, mlir in compile_op_decomp_rules(func).items()
    )

    bytecode = _quantum_opt(
        "--emit-bytecode",
        "--register-decomp-rule-resource",
        stdin=mlir_rules.encode("utf-8"),
        text=None,
    )

    with open(decomp_file_path, "wb") as bytecode_file:
        bytecode_file.write(bytecode)


if __name__ == "__main__":  # pragma: no cover
    precompile_decomp_rules()  # pragma: no cover
