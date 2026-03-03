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
import warnings
from textwrap import indent
from typing import Callable, get_args

import jax
import pennylane as qp
from jax._src.lib.mlir import ir
from pennylane.operation import Operation
from pennylane.typing import TensorLike
from pennylane.wires import WiresLike

from catalyst.from_plxpr.decompose import COMPILER_OPS_FOR_DECOMPOSITION
from catalyst.jax_primitives import decomposition_rule
from catalyst.utils.exceptions import CompileError

# TODO document this directory + functionality

# NOTE: paths are relative to catalyst root, not mlir directory
DECOMP_RULE_DIR = "./decomposition-rules/"
DECOMPS_FILE = DECOMP_RULE_DIR + "decompositions.mlir"
MLIRBC_DECOMPS_FILE = DECOMP_RULE_DIR + "decompositions.mlirbc"


def get_compiler_ops() -> tuple[list[Operation], int]:
    """
    Extracts all ops from pennylane that have decompositions in catalyst
    """
    num_failures = 0

    pl_op_classes = [
        obj
        for _, obj in inspect.getmembers(qp)
        if inspect.isclass(obj) and issubclass(obj, Operation)
    ]

    compiler_op_classes = [
        op_class
        for op_class in pl_op_classes
        if op_class.__name__ in COMPILER_OPS_FOR_DECOMPOSITION
    ]

    compiler_op_class_names = [op_class.__name__ for op_class in compiler_op_classes]

    for class_name in COMPILER_OPS_FOR_DECOMPOSITION:
        if class_name not in compiler_op_class_names:
            warnings.warn(f"failed to collect pennylane op with name {class_name}")
            num_failures += 1

    return compiler_op_classes, num_failures


def get_dummy_args(func: Callable) -> list:
    """
    Return a list of dummy args to allow compilation of op_class
    """
    func_sig = inspect.signature(func)
    dummy_args = []
    for param in func_sig.parameters.values():
        type_annotation = param.annotation
        annotation_args = get_args(type_annotation)
        if param.default is not inspect.Parameter.empty:  # use defaults whenever possible
            continue
        elif param.name == "wires" or annotation_args is WiresLike:  # wires are handled separately
            continue
        elif param.kind == inspect.Parameter.VAR_POSITIONAL:
            continue
        elif param.kind == inspect.Parameter.VAR_KEYWORD:
            continue
        elif type_annotation is inspect.Parameter.empty:
            print(f"no annotation for parameter {param} from function {func} with sig {func_sig}")
            dummy_args.append(0)  # use 0, works for the most types
        elif str in annotation_args:
            dummy_args.append("XX")  # we default to two wires
        elif float in annotation_args or TensorLike in annotation_args:
            dummy_args.append(0.1)  # float for angles
        elif int in annotation_args:
            dummy_args.append(0)
    print(f"got args {dummy_args} from signature {func_sig}")
    return dummy_args


def get_func_from_circuit(module) -> str | None:
    """
    Return the string representation of FuncOp named `rule_wrapper` from mlir_circuit.
    """

    decompFuncOp = None

    def find_condition(op):
        nonlocal decompFuncOp
        if op.name == "func.func":
            if ir.StringAttr(op.attributes["sym_name"]).value == "rule_wrapper":
                decompFuncOp = op
                return ir.WalkResult.INTERRUPT
        return ir.WalkResult.ADVANCE

    module.operation.walk(find_condition)

    return "builtin.module {\n" + indent(str(decompFuncOp), "  ") + "\n}\n" if decompFuncOp else ""


def compile_decomps_via_dummy_circuit(
    op_class: Operation,
) -> tuple[dict[str, str | None], int, int]:
    """
    Compile all decomposition rules for op_class. Returns a dictionary of decomposition rule names
    to compiled mlir modules.

    Note: the modules include the full circuit IR.
    """
    num_failures = 0
    num_successes = 0

    op_decomp_rules = qp.decomposition.decomposition_graph.list_decomps(op_class)

    mlir_modules = {}

    try:
        op_num_wires = (
            op_class.num_wires if op_class.num_wires and isinstance(op_class.num_wires, int) else 2
        )
        op_wires = list(range(op_num_wires))
        dev = qp.device("lightning.qubit", wires=op_num_wires)
    except Exception:
        print(f"failed to compile {len(op_decomp_rules)} rules")
        return {}, 0, len(op_decomp_rules)

    op_args = get_dummy_args(op_class)

    for rule in op_decomp_rules:
        rule_name = rule._impl.__name__  # pylint: disable=protected-access
        abstract_args = [
            type(arg) for arg in get_dummy_args(rule._impl)
        ]  # pylint: disable=protected-access

        try:
            qp.capture.enable()
            qp.decomposition.enable_graph()

            # WARNING: do not rename this function, we use it to extract the rule from the compiled
            # circuit
            @decomposition_rule(is_qreg=True, op_type=op_class.__name__)
            def rule_wrapper(*args, wires, **_):
                return rule(*args, wires=wires, **_)

            @qp.qjit(target="mlir")
            @qp.transform(pass_name="decompose-lowering")
            @qp.qnode(dev)
            def circuit():
                rule_wrapper(*abstract_args, wires=jax.core.ShapedArray((op_num_wires,), int))
                op_class(*op_args, wires=op_wires)
                return qp.probs()

            circuit()
            mlir_modules[rule_name] = get_func_from_circuit(circuit.mlir_module)
            num_successes += 1
        except TypeError as e:
            warnings.warn(str(e))
        except CompileError as e:
            warnings.warn(f"failed to compile {rule_name}: {e}")
            num_failures += 1
        except Exception as e:
            warnings.warn(f"Unexpected error while trying to compile {rule_name}: {e}")
            num_failures += 1
        finally:
            qp.capture.disable()

    return (mlir_modules, num_successes, num_failures)


if __name__ == "__main__":
    target_ops, num_ops_missed = get_compiler_ops()
    if num_ops_missed:
        print(f"failed to collect {num_ops_missed} op(s) from PennyLane")

    num_successes = 0
    num_failures = 0
    with open(DECOMPS_FILE, "w") as mlir_file:
        for func in target_ops:
            results, num_new_successes, num_new_failures = compile_decomps_via_dummy_circuit(func)
            num_successes += num_new_successes
            num_failures += num_new_failures
            if results:
                for name, circuit_mlir in results.items():
                    if circuit_mlir:
                        mlir_file.write(circuit_mlir.replace("rule_wrapper", name))
    if num_failures:
        print(f"compiled {num_successes} / {num_failures + num_successes} decomposition rules")
    else:
        print("successfully compiled decomposition rules")
