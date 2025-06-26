import catalyst
from catalyst import qjit
import pennylane as qml

#with catalyst.profiler("python"):
#with catalyst.profiler():
#with catalyst.profiler("passes"):
with catalyst.profiler("memory"):
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

# Currently only works with PLxPR due to the use
# of subroutines
with catalyst.profiler("memory"):
    qml.capture.enable()
    @qml.qjit
    @qml.qnode(qml.device("null.qubit", wires=1))
    def foo():
        return qml.probs()

    foo()
    qml.capture.disable()
