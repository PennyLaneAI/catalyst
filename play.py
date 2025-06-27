import jax
import pennylane as qml

from catalyst.jax_primitives import subroutine

# with catalyst.profiler("python"):
# with catalyst.profiler():
with catalyst.profiler():
#with catalyst.profiler("user memory"):  # breaks for non plxpr
#with catalyst.profiler("fake mode"):  # breaks
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


# # Currently only works with PLxPR due to the use
# # # of subroutines
# #with catalyst.profiler():
with catalyst.profiler("user memory"):

    qml.capture.enable()

    @subroutine
    def H_0():
        qml.Hadamard(wires=[0])

    @subroutine
    def identity_plus(y):
        return jax.numpy.array([[1, 0], [0, 1]], dtype=complex) + y

    @qml.qjit(keep_intermediate=True, autograph=False)
    @qml.qnode(qml.device("null.qubit", wires=1), autograph=False)
    def foo():
        H_0()
        H_0()
        identity_plus(0)
        return qml.probs()

    print(foo())

    qml.capture.disable()
