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
from xdsl.dialects import builtin, func

def xdsl_transform(_klass):

    def identity_transform(tape):
        return tape, lambda args: args[0]

    identity_transform.__name__ = "xdsl_transform" + _klass.__name__
    transform = qml.transform(identity_transform)
    catalyst.from_plxpr.register_transform(transform, _klass.name, False)
    from catalyst.python_compiler import register_pass
    register_pass(_klass.name, lambda : _klass())

    return transform


from xdsl.rewriter import InsertPoint
from xdsl import pattern_rewriter

from catalyst.python_compiler import quantum


self_inverses = ("PauliZ", "PauliX", "PauliY", "Hadamard", "Identity")

class DeepCancelInversesSingleQubitPattern(pattern_rewriter.RewritePattern):
    @pattern_rewriter.op_type_rewrite_pattern
    def match_and_rewrite(self, funcOp: func.FuncOp, rewriter: pattern_rewriter.PatternRewriter):
        """Deep Cancel for Self Inverses"""
        for op in funcOp.body.walk():

            while isinstance(op, quantum.CustomOp) and op.gate_name.data in self_inverses:

                next_user = None
                for use in op.results[0].uses:
                    user = use.operation
                    if isinstance(user, quantum.CustomOp) and user.gate_name.data == op.gate_name.data:
                        next_user = user
                        break

                if next_user is None:
                    break

                rewriter.replace_all_uses_with(next_user.results[0], op.in_qubits[0])
                rewriter.erase_op(next_user)
                rewriter.erase_op(op)

                op = op.in_qubits[0].owner


@xdsl_transform
class DeepCancelInversesSingleQubitPass(passes.ModulePass):
    name = "deep-cancel-inverses-single-qubit"

    def apply(self, ctx: context.MLContext, module: builtin.ModuleOp) -> None:
        pattern_rewriter.PatternRewriteWalker(
            pattern_rewriter.GreedyRewritePatternApplier([DeepCancelInversesSingleQubitPattern()])
        ).rewrite_module(module)

qml.capture.enable()

@catalyst.qjit(pass_plugins=[getXDSLPluginAbsolutePath()])
@DeepCancelInversesSingleQubitPass
@qml.qnode(qml.device("lightning.qubit", wires=1))
def captured_circuit(x: float):
    qml.RX(x, wires=0)
    qml.Hadamard(wires=0)
    qml.Hadamard(wires=0)
    return qml.state()


qml.capture.disable()


print(captured_circuit(1.))
