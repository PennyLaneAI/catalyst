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

"""
This module provides the utilities necessary to AOT compile PennyLane's decomposition rules to MLIR
Bytecode.
"""

import inspect
import subprocess
import warnings
from functools import lru_cache
from pathlib import Path
from types import UnionType
from typing import Callable, Union, get_args, get_origin

import jax
import pennylane as qp
from jax._src.lib.mlir import ir
from pennylane.operation import Operator
from pennylane.typing import TensorLike
from pennylane.wires import Wires

from catalyst.jax_primitives import decomposition_rule
from catalyst.utils.exceptions import CompileError
from catalyst.utils.runtime_environment import DEFAULT_BIN_PATHS

DEFAULT_RULE_DIR = Path(__file__).parent.parent / Path("resources")
BYTECODE_FILE_NAME = Path("decomposition_rules.mlirbc")

COMPILER_OPS_FOR_DECOMPOSITION = {
    "CNOT",
    "ControlledPhaseShift",
    "CRot",
    "CRX",
    "CRY",
    "CRZ",
    "CSWAP",
    "CY",
    "CZ",
    "Hadamard",
    "Identity",
    "IsingXX",
    "IsingXY",
    "IsingYY",
    "IsingZZ",
    "SingleExcitation",
    "DoubleExcitation",
    "ISWAP",
    "PauliX",
    "PauliY",
    "PauliZ",
    "PauliRot",
    "PauliMeasure",
    "PhaseShift",
    "PSWAP",
    "Rot",
    "RX",
    "RY",
    "RZ",
    "S",
    "SWAP",
    "T",
    "Toffoli",
    "U1",
    "U2",
    "U3",
    "MultiRZ",
    "GlobalPhase",
}


@lru_cache
def get_compiler_ops() -> tuple[set[type[Operator]], int]:
    """
    Extracts all ops from pennylane that have decompositions in catalyst

    Returns:
        set[Operation]: the set of PennyLane ops that are compiler-compatible
                        (as defined by COMPILER_OPS_FOR_DECOMPOSITION)
        int: the number of compiler-compatible ops that could not be found in PennyLane
    """
    num_failures = 0

    pl_op_classes = set(
        value
        for _, value in inspect.getmembers(
            qp, lambda obj: inspect.isclass(obj) and issubclass(obj, Operator)
        )
    )

    # FIXME: manual override for PauliMeasure
    pl_op_classes.add(qp.ops.PauliMeasure)

    supported_compiler_op_names = set(COMPILER_OPS_FOR_DECOMPOSITION)

    compiler_op_classes = set(
        op_class for op_class in pl_op_classes if op_class.__name__ in supported_compiler_op_names
    )

    compiler_op_class_names = set(op_class.__name__ for op_class in compiler_op_classes)

    for class_name in supported_compiler_op_names.difference(compiler_op_class_names):
        warnings.warn(f"failed to collect pennylane op with name {class_name}")
        num_failures += 1

    return compiler_op_classes, num_failures


def get_dummy_args(func: Callable) -> list[float | int]:
    """
    Create dummy args for a callable.

    Args:
        func: callable to create args for

    Returns:
        list: dummy args matching the (positional) signature of func.
    """
    # pylint: disable=too-many-branches

    func_sig = inspect.signature(func)
    dummy_args: list = []
    for param in func_sig.parameters.values():
        type_annotation = param.annotation
        if get_origin(type_annotation) is Union or get_origin(type_annotation) is UnionType:
            type_annotation = get_args(type_annotation)
        else:
            type_annotation = (type_annotation,)

        if param.default is not inspect.Parameter.empty:  # use defaults whenever possible
            continue
        if param.name == "wires" or Wires in type_annotation:  # wires are handled separately
            continue
        if param.kind == inspect.Parameter.VAR_POSITIONAL:
            continue
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            continue

        elif float in type_annotation or TensorLike in type_annotation:
            dummy_args.append(0.0)
        elif int in type_annotation:
            dummy_args.append(0)
        elif param.name in ["theta", "phi", "delta", "omega"]:
            dummy_args.append(0.0)
        else:
            raise ValueError(
                f"Cannot resolve the {param.name} parameter of the {func} decomposition rule."
            )

    return dummy_args


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
    op_num_wires,
    rule,
    dev,
) -> str | None:
    """
    Get the string representation of a compiled rule from a python decomposition rule, if possible.

    NOTE: rules with string params are not currently supported.

    Args:
        op_class: A PennyLane class subclassing Operation
        op_args: a valid (positional) arguments to op_class
        op_num_wires: the number of wires used by op_class
        rule (DecompositionRule): the decomposition rule to be compiled
        dev (Device): a device for qjit

    Returns:
        str: string representation of the mlir of the decomposition rule.
    """
    abstract_args = [
        type(arg) for arg in get_dummy_args(rule._impl)  # pylint: disable=protected-access
    ]

    if str in abstract_args:
        raise ValueError("Cannot compile decomposition rules with string arguments.")

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
    op_class: Operator,
) -> dict[str, str | None]:
    """
    Compile all decomposition rules for op_class.

    Note: the modules include the full circuit IR.

    Returns:
        dict[str, str | None]: decomposition rule names to compiled mlir modules.
        int: the number of rules that successfully compiled
        int: the number of rules that failed to compile
    """
    op_decomp_rules = qp.decomposition.decomposition_graph.list_decomps(op_class)

    mlir_modules = {}

    op_num_wires = op_class.num_wires if op_class.num_wires else 2
    dev = qp.device("null.qubit", wires=op_num_wires)

    for rule in op_decomp_rules:
        try:
            rule_name = rule._impl.__name__  # pylint: disable=protected-access
            mlir_modules[rule_name] = compile_rule(op_class, op_num_wires, rule, dev)
        except TypeError as e:
            warnings.warn(f"dummy args failed to compile {rule_name}: {e}")
        except CompileError as e:
            warnings.warn(f"failed to compile {rule_name}: {e}")
        except Exception as e:  # pylint: disable=broad-exception-caught
            warnings.warn(f"Unexpected error while trying to compile {rule_name}: {e}")
        finally:
            qp.decomposition.disable_graph()

    return mlir_modules


def precompile_decomp_rules(decomp_dir_path: Path = DEFAULT_RULE_DIR):
    """
    filters compiler-compatible decomposition ops from PennyLane, grabs their associated
    decomposition rules, and compiles them via a wrapper function with qjit to mlir.
    Intended for use with `make decomp-rules` in catalyst/mlir.
    """

    decomp_dir_path.mkdir(parents=True, exist_ok=True)

    target_ops, num_ops_missed = get_compiler_ops()
    if num_ops_missed:
        warnings.warn(f"failed to collect {num_ops_missed} op(s) from PennyLane")

    mlir_rules = ""
    for func in target_ops:
        results = compile_op_decomp_rules(func)
        if results:
            for name, circuit_mlir in results.items():
                if circuit_mlir:
                    mlir_rules += str(circuit_mlir).replace("@rule_wrapper", "@" + name)

    # FIXME use catalyst.compiler._quantum_opt once the catalyst dangling options are fixed
    bytecode = subprocess.run(
        (
            f"{DEFAULT_BIN_PATHS['cli']}/quantum-opt",
            "--emit-bytecode",
            "--register-decomp-rule-resource",
        ),
        input=mlir_rules.encode("utf-8"),
        capture_output=True,
        check=True,
    ).stdout

    with open(decomp_dir_path / BYTECODE_FILE_NAME, "wb") as bytecode_file:
        bytecode_file.write(bytecode)


if __name__ == "__main__":
    precompile_decomp_rules()
