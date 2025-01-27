import pennylane as qml
import pytest

import catalyst
from catalyst import qjit
from catalyst.third_party.oqd import OQDDevice
from catalyst.debug import get_compilation_stage, replace_ir

dev = OQDDevice(backend="default", shots=1000, wires=8)

@qjit#(keep_intermediate=True)
@qml.qnode(dev)
def f():
    qml.Hadamard(wires=0)
    return qml.probs()


with open("5_MLIRToLLVMDialect.mlir", "r") as file:
    ir = file.read()
replace_ir(f, "MLIRToLLVMDialect", ir)

'''
with open("6_llvm_ir_manual.ll", "r") as file:
    ir = file.read()
replace_ir(f, "llvm_ir", ir)
'''

f()
