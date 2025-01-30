import pennylane as qml
import pytest

import catalyst
from catalyst import qjit
from catalyst.debug import get_compilation_stage, replace_ir
from catalyst.third_party.oqd import OQDDevice

dev = OQDDevice(backend="default", shots=1000, wires=2)


@qjit  # (keep_intermediate=True)
@qml.qnode(dev)
def f():
    # qml.Hadamard(wires=0)
    return qml.counts()


# breakpoint()
with open("5_MLIRToLLVMDialect.mlir", "r") as file:
    ir = file.read()
replace_ir(f, "MLIRToLLVMDialect", ir)

"""
with open("6_llvm_ir_manual.ll", "r") as file:
    ir = file.read()
replace_ir(f, "llvm_ir", ir)
"""

f()
