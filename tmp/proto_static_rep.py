from inspect import signature

import pennylane as qml
import numpy as np

from pennylane.decomposition import resource_rep


def static_resource_rep(op_type, *args, **kwargs):
    bound = signature(op_type.__init__).bind(*args, **kwargs)
    signature_key = tuple(bound.arguments.values())
    return resource_rep(op_type, signature_key=signature_key)

from pennylane.templates import AbstractArray

class HigherOp(qml.operation.Operation):

    def __init__(self, angle, wires, meta=None):
        self.hyperparameters["meta"] = meta
        super().__init__(angle, wires=wires)

class MyOp(qml.operation.Operation):

    resource_keys = {"signature_key"}

    def __init__(self, angle, wires, meta=None):
        self.hyperparameters["meta"] = meta
        super().__init__(angle, wires=wires)

abstract_params = AbstractArray((1, ), float)
abstract_wires = AbstractArray(()) # a single wire
meta = 3.14159
rep = static_resource_rep(MyOp, abstract_params, abstract_wires, meta)

@qml.decomposition.register_resources({rep: 1})
def higher_op_decomposition(angle, wires, meta=np.pi):
    MyOp(angle, wires, meta * 2)

qml.add_decomps(HigherOp, higher_op_decomposition)

qml.decomposition.enable_graph()

@qml.qnode(qml.device('reference.qubit', wires=1))
def c():
    HigherOp(1.0, wires=0, meta=3.14159)
    return qml.state()

print(qml.draw(qml.decompose(c, max_expansion=1))())
