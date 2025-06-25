import catalyst
from catalyst import qjit
import pennylane as qml

with catalyst.profiler("python", 5000000):
    @qjit
    def workflow(wires):
        @qml.qnode(qml.device("lightning.qubit", wires=5, shots=20))
        def circuit():
            qml.Hadamard(wires=0)
            for i in range(1000):
                qml.Hadamard(wires=0)
                qml.Hadamard(wires=1)
                qml.Hadamard(wires=0)
                qml.Hadamard(wires=1)
            return qml.sample()
        return circuit()


    res = workflow(5)
    print(res)
