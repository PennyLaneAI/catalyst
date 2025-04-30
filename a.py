import pennylane as qml
from pennylane.tape import QuantumScript, QuantumScriptBatch
from pennylane.transforms import cancel_inverses as pl_cancel_inverses
from pennylane.typing import PostprocessingFn
from catalyst.tracing.contexts import EvaluationContext
from catalyst.passes.xdsl_plugin import getXDSLPluginAbsolutePath

import catalyst
import functools

from dataclasses import dataclass
from xdsl import context, passes
from xdsl.utils import parse_pipeline
from xdsl.dialects import builtin

def xdsl_transform(_klass):

    def identity_transform(tape):
        return tape, lambda args: args[0]

    identity_transform.__name__ = "xdsl_transform" + _klass.__name__
    transform = qml.transform(identity_transform)
    catalyst.from_plxpr.register_transform(transform, _klass.name, False)

    return transform


@xdsl_transform
#@dataclass(frozen=True)
# All passes inherit from passes.ModulePass
class PrintModule(passes.ModulePass):
    # All passes require a name field
    name = "remove-chained-self-inverse"

    # All passes require an apply method with this signature.
    def apply(self, ctx: context.MLContext, module: builtin.ModuleOp) -> None:
        print("Hello from inside the pass\n", module)


qml.capture.enable()

@catalyst.qjit(pass_plugins=[getXDSLPluginAbsolutePath()])
@PrintModule
@qml.qnode(qml.device("lightning.qubit", wires=1))
def captured_circuit(x: float):
    qml.RX(x, wires=0)
    qml.Hadamard(wires=0)
    qml.Hadamard(wires=0)
    return qml.state()


qml.capture.disable()

print(captured_circuit.mlir)

captured_circuit(0.0)
