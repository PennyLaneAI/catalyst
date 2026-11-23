from functools import partial
import pennylane as qml

# pass-by-pass specs

# import pennylane as qml
# from pennylane.devices import device
# import time
#
# dev = device(wires=[0, 1], name="null.qubit")
#
# @qml.qjit
# @qml.transforms.merge_rotations
# @qml.transforms.cancel_inverses
# @qml.qnode(dev)
# def circuit(x):
#   qml.RX(x, wires=0)
#   qml.RX(x, wires=0)
#   qml.X(0)
#   qml.X(0)
#   qml.CNOT([0, 1])
#   return qml.probs()
#
# start = time.time()
# with_level = qml.specs(circuit, level="all-mlir")(1.23)
# end = time.time()
# print(end - start)
#
# start = time.time()
# without_level = qml.specs(circuit)(1.23)
# end = time.time()
# print(end - start)
#
# print(qml.specs(circuit, level="all-mlir")(1.23))

# bug reproduction

# import pennylane as qml
# from pennylane.devices import device
#
# dev = device(wires=[0, 1], name="null.qubit")
#
# @qml.qjit
# @qml.transforms.merge_rotations
# @qml.transforms.cancel_inverses
# @qml.qnode(dev)
# def circuit(x):
#   qml.RX(x, wires=0)
#   qml.RX(x, wires=0)
#   qml.X(0)
#   qml.X(0)
#   qml.CNOT([0, 1])
#   return qml.probs()

# will default to level=gradient, and error: `ValueError: Marker name 'gradient' not found in the transform program.`
# details = qml.specs(circuit)(1.23)
# print(details)

# partial

# import pennylane as qml
# import numpy as np
# import time
#
# start = time.time()
#
# @partial(qml.transforms.decompose, gate_set={qml.RX, qml.CNOT})
# @qml.qnode(qml.device('default.qubit', wires=2))
# def circuit():
#     qml.Hadamard(wires=0)
#     qml.CZ(wires=[0, 1])
#     return qml.expval(qml.Z(0))
#
# res = circuit()
#
# end = time.time()
# print(end - start)
#
# start = time.time()
#
# @qml.transforms.decompose(gate_set={qml.RX, qml.CNOT})
# @qml.qnode(qml.device('default.qubit', wires=2))
# def circuit():
#     qml.Hadamard(wires=0)
#     qml.CZ(wires=[0, 1])
#     return qml.expval(qml.Z(0))
#
# decompose_res = circuit()
#
# end = time.time()
# print(end - start)
#
# assert np.allclose(res, decompose_res)

# pennylane-catalyst unification

# import pennylane as qml
# import time
#
# start = time.time()
#
# @qml.while_loop(lambda n: n < 4)
# def w(n):
#     return n + 1
#
# res = qml.qjit(w)(0)
#
# end = time.time()
# print(end - start)
#
# assert qml.math.allclose(res, 4)
#
# start = time.time()
#
# @qml.for_loop(5)
# def f(i, x):
#     return x + i
#
# res = qml.qjit(f)(0)
#
# end = time.time()
# print(end - start)
#
# assert qml.math.allclose(res, 10)
#
# start = time.time()
#
# @qml.qjit
# @qml.qnode(qml.device("lightning.qubit", wires=3))
# def c():
#     qml.X(0)
#     m = qml.measure(0, reset=True, postselect=True)
#
#     def f():
#         qml.X(1)
#
#     qml.cond(m, f)()
#
#     return qml.expval(qml.Z(0)), qml.expval(qml.Z(1))
#
# z0, z1 = c()
#
# end = time.time()
# print(end - start)
#
# assert qml.math.allclose(z0, 1)
# assert qml.math.allclose(z1, -1)
#
# start = time.time()
#
# @qml.qnode(qml.device("default.qubit", wires=3))
# def c():
#     qml.X(0)
#     m = qml.measure(0, reset=True, postselect=True)
#
#     def f():
#         qml.X(1)
#
#     qml.cond(m, f)()
#
#     return qml.expval(qml.Z(0)), qml.expval(qml.Z(1))
#
# z0, z1 = c()
#
# end = time.time()
# print(end - start)
#
# assert qml.math.allclose(z0, 1)
# assert qml.math.allclose(z1, -1)

# qmc
#
# import pennylane as qml
# import numpy as np
#
# from pennylane.templates.state_preparations.mottonen import (
#     _apply_uniform_rotation_dagger as r_unitary,
# )
#
# from scipy.stats import norm
#
# m = 5
# M = 2 ** m
#
# xmax = np.pi  # bound to region [-pi, pi]
# xs = np.linspace(-xmax, xmax, M)
#
# probs = np.array([norm().pdf(x) for x in xs])
# probs /= np.sum(probs)
#
# func = lambda i: np.sin(xs[i]) ** 2
# r_rotations = np.array([2 * np.arcsin(np.sqrt(func(i))) for i in range(M)])
#
# n = 6
# N = 2 ** n
#
# a_wires = range(m)
# wires = range(m + 1)
# target_wire = m
# estimation_wires = range(m + 1, n + m + 1)
#
# dev = qml.device("default.qubit", wires=(n + m + 1))
#
# def fn():
#     qml.templates.MottonenStatePreparation(np.sqrt(probs), wires=a_wires)
#     r_unitary(qml.RY, r_rotations, control_wires=a_wires[::-1], target_wire=target_wire)
#
# @qml.qnode(dev)
# def qmc():
#     qml.quantum_monte_carlo(fn, wires, target_wire, estimation_wires)()
#     return qml.probs(estimation_wires)
#
# phase_estimated = np.argmax(qmc()[:int(N / 2)]) / N
#
# specs = qml.specs(qmc, level="device")()
# from pprint import pprint
# pprint(specs)

# demos

# import pennylane as qml
# import numpy as np
#
# dev = qml.device("default.qubit")
# @qml.qnode(dev)
# def original_circuit(x, y):
#     qml.RX(x, 0)
#     qml.CNOT([0, 1])
#     qml.RY(y, 0)
#     return qml.probs()
#
# x, y = np.pi / 2, np.pi / 4
# unrolled_circuit = qml.transforms.clifford_t_decomposition(original_circuit)
#
# with qml.Tracker(dev) as tracker:
#     unrolled_circuit(x, y)
#
# resources_lst = tracker.history["resources"]
# print(resources_lst[0])
#
# shots = 100
# dev = qml.device("default.qubit")

# import pennylane as qml
# import numpy as np
#
# def init_state(x):
#     # Rotate the first three qubits
#     for w in range(3):
#         qml.RX(x[w], w)
#     # Entangle the first three qubits
#     qml.CNOT([0, 1])
#     qml.CNOT([1, 2])
#     qml.CNOT([2, 0])
#
# shots = 100
# dev = qml.device("default.qubit", seed=0)
#
# @qml.set_shots(shots)
# @qml.qnode(dev)
# def create_half_filled_state(x):
#     init_state(x)
#     for w in range(3):
#         # Measure one qubit at a time and flip another, fresh qubit if measured 0
#         mcm = qml.measure(w)
#         qml.cond(~mcm, qml.X)(w + 3)
#
#     return qml.counts(wires=range(6))
#
# np.random.seed(652)
# x = np.random.random(3) * np.pi
#
# print(qml.draw(create_half_filled_state)(x))
#
# counts = create_half_filled_state(x)
# print(f"Sampled bit strings:\n{list(counts.keys())}")
#
#
# print("The probabilities for the bit strings are:")
# for key, val in counts.items():
#     print(f"    {key}: {val/shots*100:4.1f} %")

import jax
dev = qml.device("default.qubit")

@qml.qnode(dev, diff_method="hadamard", gradient_kwargs={"mode": "standard"})
def circuit(params):
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=0)
    qml.RX(params[2], wires=0)
    return qml.expval(qml.Z(0)), qml.probs(wires=0)
params = jax.numpy.array([0.1, 0.2, 0.3])
jax.jacobian(circuit)(params)