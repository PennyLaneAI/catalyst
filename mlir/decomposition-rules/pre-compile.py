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

import jax
import pennylane as qp
from pennylane.operation import Operation

from catalyst.from_plxpr.decompose import COMPILER_OPS_FOR_DECOMPOSITION
from catalyst.jax_primitives import decomposition_rule
from catalyst.utils.exceptions import CompileError

# TODO document this directory + functionality

# NOTE: paths are relative to catalyst root, not mlir directory
DECOMP_RULE_DIR = "./decomposition-rules/"
DECOMPS_FILE = DECOMP_RULE_DIR + "decompositions.mlir"
MLIRBC_DECOMPS_FILE = DECOMP_RULE_DIR + "decompositions.mlirbc"


def get_compiler_ops() -> list[Operation]:
    """
    Extracts all ops from pennylane that have decompositions in catalyst
    """
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
            warnings.warn(f"could not find pennylane op with name {class_name}")

    return compiler_op_classes


def get_dummy_args(op_class: Operation) -> list:
    """
    Return a list of dummy args to allow compilation of op_class
    """
    op_class_sig = inspect.signature(op_class)
    # print(f"{op_class} has signature {op_class_sig}")
    dummy_args = []
    for param in op_class_sig.parameters.values():
        if param.name == "wires":  # wires are handled separately
            continue
        elif param.name == "pauli_word":
            dummy_args.append("XX")  # we default to two wires
        elif param.default == inspect.Parameter.empty:  # assume float/int
            dummy_args.append(0)

    return dummy_args


def compile_decomps_via_dummy_circuit(op_class: Operation) -> dict[str, str] | None:
    """
    Compile all decomposition rules for op_class. Returns a dictionary of decomposition rule names
    to compiled mlir modules.

    Note: the modules include the full circuit IR.

    TODO: remove circuit IR, return only the func.func decomposition rules.
    """
    op_decomp_rules = qp.decomposition.decomposition_graph.list_decomps(op_class)

    mlir_modules = {}

    try:
        op_num_wires = op_class.num_wires if op_class.num_wires else 2
        op_wires = list(range(op_num_wires))
        dev = qp.device("lightning.qubit", wires=op_num_wires)
    except Exception:
        return None

    # naively assume all inputs are floats
    args = get_dummy_args(op_class)
    abstract_args = [type(arg) for arg in args]

    for rule in op_decomp_rules:
        rule_name = rule._impl.__name__  # pylint: disable=protected-access

        try:
            qp.capture.enable()
            qp.decomposition.enable_graph()

            # WARNING: do not rename this function, we use it to extract the rule from the compiled
            # circuit
            @decomposition_rule(is_qreg=True, op_type=op_class.__name__)
            def rule_wrapper(*args, wires, **_):
                return rule(*args, wires, **_)

            @qp.qjit(target="mlir")
            @qp.transform(pass_name="decompose-lowering")
            @qp.qnode(dev)
            def circuit():
                rule_wrapper(*abstract_args, wires=jax.core.ShapedArray((op_num_wires,), int))
                op_class(*args, wires=op_wires)
                return qp.probs()

        except CompileError, TypeError:  # type error to include failed dummy-args
            warnings.warn(f"failed to qjit {rule_name}")
            return None
        except Exception as e:
            warnings.warn(f"Unexpected error while trying to qjit {rule_name}:")
            raise e

        try:
            circuit()
            mlir_modules[rule_name] = circuit.mlir

        except CompileError:
            warnings.warn(f"failed to compile {rule_name}")
            return None
        except Exception as e:
            warnings.warn(f"Unexpected error while trying to compile {rule_name}:")
            raise e
        finally:
            qp.capture.disable()

    return mlir_modules


if __name__ == "__main__":
    target_ops = get_compiler_ops()

    with open(DECOMPS_FILE, "w") as mlir_file:
        for op_class in target_ops:
            results = compile_decomps_via_dummy_circuit(op_class)
            if results:
                for name, circuit_mlir in results.items():
                    mlir_file.write(circuit_mlir.replace("rule_wrapper", name))
                    mlir_file.write("// -----\n")  # for splitting input file to quantum-opt
