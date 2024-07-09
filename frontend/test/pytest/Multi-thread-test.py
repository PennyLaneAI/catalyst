import jax
import pennylane as qml

def repeat(n):
    agg = []
    for x in range(n):
        @jax.jit
        def identity(x):
            return x
        agg += [identity]
    return agg

@qml.qjit(keep_intermediate=True, multi_threaded_compilation=True)
def foo(x : float):
    old = x
    for func in repeat(100):
        new = func(old)
        old = new
    return new


foo.mlir