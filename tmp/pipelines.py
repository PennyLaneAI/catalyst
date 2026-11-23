import pennylane as qml
from pennylane import qjit
from pennylane.decomposition import enable_graph
from pennylane.queuing import AnnotatedQueue
from pennylane.transforms import inspect_decomp_graph

# the compile pipeline we want to replace our QFTs *before* they get to Catalyst

pipeline = qml.CompilePipeline(
    qml.decompose(gate_set=qml.decomposition.gate_sets.CLIFFORD_T_PLUS_RZ)
)

# shared setup

enable_graph()
dev = qml.device("null.qubit", wires=(0, 1, 2))

# This example works as expected


@qml.qnode(dev)
def circuit():
    qml.QFT((0, 1, 2))
    return qml.expval(qml.Z(1))


decomposed = pipeline(circuit)
qjited = qjit(decomposed, capture=True)
qjited.jit_compile(tuple())

# This example fails, as the QFT is still in the circuit by the time we get to
# the DecompRuleInterpreter in Catalyst, which cannot handle it and errors out


# @inspect_decomp_graph(gate_set=qml.gate_sets.CLIFFORD_T_PLUS_RZ)
@qml.qnode(dev)
def circuit():
    qml.adjoint(qml.QFT)((0, 1, 2))
    return qml.expval(qml.Z(1))


# inspector = circuit()
# decomps = inspector.inspect_decomps(
#     qml.adjoint(qml.QFT)((0, 1, 2))
# )
# print(decomps)

decomposed = pipeline(circuit)
qjited = qjit(decomposed, capture=True)
qjited.jit_compile(tuple())
