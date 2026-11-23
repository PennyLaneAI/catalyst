import jax.numpy as jnp
import pennylane as qml

qml.capture.enable()

dev = qml.device("null.qubit", wires=3)


def qfunc():
    qml.Hadamard(0)
    qml.Hadamard(2)
    qml.RZ(0.1, wires=1)
    qml.PauliRot(jnp.pi / 4, pauli_word="XYZ", wires=[0, 1, 2])
    ppm = qml.pauli_measure(pauli_word="XY", wires=[0, 2])
    qml.cond(ppm, qml.X)(wires=1)


@qml.qjit(target="mlir")
@qml.transforms.ppm_compilation
@qml.transforms.gridsynth
@qml.transforms.clifford_t_decomposition
@qml.qnode(dev)
def circuit():
    qfunc()
    return qml.expval(qml.Z(0))


print(qml.specs(circuit, level="all")())
# AttributeError: DynamicJaxprTracer has no attribute measurements
