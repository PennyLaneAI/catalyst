import inspect
import pathlib
import warnings

import jax
import pennylane as qp

from catalyst.from_plxpr.decompose import COMPILER_OPS_FOR_DECOMPOSITION
from catalyst.jax_primitives import decomposition_rule

# TODO merge into one file
# TODO update ADR to reflect that we chose one file

DECOMP_RULE_DIR = "./mlir/decomposition-rules/compiled-decomp-rules/"


def get_compiler_ops():
    """
    Extracts all ops from pennylane that have decompositions in catalyst
    """
    pl_op_classes = [
        obj
        for _, obj in inspect.getmembers(qp)
        if inspect.isclass(obj) and issubclass(obj, qp.operation.Operation)
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


def get_dummy_args(op_class):
    op_class_sig = inspect.signature(op_class)
    # print(f"{op_class} has signature {op_class_sig}")
    dummy_args = []
    for param in op_class_sig.parameters.values():
        if param.name == "wires":
            continue
        elif param.name == "pauli_word":
            dummy_args.append("XX")  # we default to two wires
        elif param.default == inspect.Parameter.empty:  # assume float
            dummy_args.append(0)

    return dummy_args


def compile_decomps_via_dummy_circuit(op_class):
    op_decomp_rules = qp.decomposition.decomposition_graph.list_decomps(op_class)

    mlir_modules = {}

    try:
        op_num_wires = op_class.num_wires if op_class.num_wires else 2
        op_wires = [i for i in range(op_num_wires)]
        dev = qp.device("lightning.qubit", wires=op_num_wires)
    except Exception as e:
        # print(f"error getting num_wires from {op_class}: {e}")
        return

    # naively assume all inputs are floats
    args = get_dummy_args(op_class)
    abstract_args = [type(arg) for arg in args]

    for rule in op_decomp_rules:
        rule_name = rule._impl.__name__

        try:
            qp.capture.enable()
            qp.decomposition.enable_graph()

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

        except Exception as e:
            print(f"failed to qjit {op_class}")
            # print(f"failed to qjit: {e}")
            return

        try:
            circuit()
            mlir_modules[rule_name] = circuit.mlir

        except Exception as e:
            print(f"failed to compile {op_class}")
            # print(f"Compilation Failed: {e}")
            pass
        finally:
            qp.capture.disable()

    return mlir_modules


if __name__ == "__main__":
    target_ops = get_compiler_ops()

    pathlib.Path(DECOMP_RULE_DIR).mkdir(exist_ok=True)

    for op_class in target_ops:
        results = compile_decomps_via_dummy_circuit(op_class)
        if results:
            for name, mlir in results.items():
                with open(DECOMP_RULE_DIR + f"{name}.mlir", "w") as mlir_file:
                    mlir_file.write(mlir)
