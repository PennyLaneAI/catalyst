# for x in range(10):
#     print(1 << x)
#     print(2**x)
# from collections import defaultdict
# from functools import partial

# import jax
# from catalyst.passes import to_ppr

# from pennylane.tape import QuantumScript
# from pennylane.templates.core import Subroutine
# from pennylane.wires import Wires

# from pennylane.ops import Controlled, PauliZ, change_op_basis

# from pennylane.decomposition import resource_rep, register_resources, add_decomps

# from pennylane.operation import Operation
# from pennylane.queuing import AnnotatedQueue

# from pennylane.templates import SelectOnlyQRAM, QFT

# bits = [True, False, True]
# print("".join([str(int(bit)) for bit in bits]))

# import pennylane.estimator as qre
# iqp = qre.IQP(num_wires=4, pattern=[[[0]], [[1]], [[2]], [[3]]])
# print(qre.estimate(iqp))

# import numpy as np
# import pennylane.math as math
#
# data = np.array([[0, 1, 1, 0], [1, 0, 0, 1]])
#
# one = np.array([int("".join([str(b) for b in bits]), 2) for bits in data])
#
# powers_of_two = 2**np.arange(data.shape[1])[::-1]
# two = math.sum(powers_of_two * data, axis=1)
#
# assert np.allclose(two, one)

# import jax
# import jax.numpy as jnp
#
# @jax.jit
# def f(x, y):
#     return x[y: y + 1]
#
# f(jnp.ones((3,)), 1)

# import numpy as np
#
# data = np.array([[0, 1, 0, 1], [0, 1, 0, 1]])
#
# print(len(data))
# print(data.shape[0])
#
# print(len(data[0]))
# print(data.shape[1])

import numpy as np
import pennylane
import pennylane as qml
import pennylane.math as math
from pennylane import (
    BBQRAM,
    BasisEmbedding,
    ControlledSequence,
    HybridQRAM,
    PauliX,
    PhaseAdder,
    adjoint,
    device,
    inspect_decomps,
    list_decomps,
    ops,
    probs,
    qjit,
    qnode,
    queuing,
    state,
)
from pennylane.transforms import cancel_inverses
from pennylane.workflow import construct_tape
from tests.templates.subroutines.test_iqp import local_gates

# @qml.qnode(qml.device('lightning.qubit'))
# def circuit():
#     qml.BBQRAM(
#         bitstrings=("010", "111", "110", "000"),
#         control_wires=(0, 1),
#         target_wires=(2, 3, 4),
#         work_wires=tuple([5] + [6, 7, 8] + [12, 13, 14] + [9, 10, 11]),
#     )
#     return qml.state()
#
# print(qml.specs(circuit, level="device")())

# iters = 10
#
# print([-j for j in range(1, iters // 2 + 1)][: iters // 2])
#
# lst = []
# for iter in range(iters // 2):
#     lst.append(-(iter + 1))
# print(lst[: iters // 2])
#
# lst = []
# for iter in range(1, iters // 2 + 1):
#     lst.append(iters // 2 + 1 - iter)
# print(lst[: iters // 2])


dev = qml.device("default.qubit")
#
#
# @qml.qnode(dev)
# def iqp_circuit(weights, pattern, spin_sym):
#     qml.IQP(weights=weights, num_wires=2, pattern=pattern, spin_sym=spin_sym)
#     return [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))]
#
# print(qml.draw(iqp_circuit, level="device")([0.89, 0.54], [[[0]], [[1]]], False))

# dev = device("default.qubit")
#
#
# @qnode(dev)
# def bb_quantum(data, control_wires, target_wires, work_wires, address):
#     BasisEmbedding(address, wires=control_wires)
#
#     BBQRAM(
#         data,
#         control_wires=control_wires,
#         target_wires=target_wires,
#         work_wires=work_wires,
#     )
#     return probs(wires=target_wires)
#
# qml.draw_mpl(bb_quantum, level="device", style="pennylane_sketch")(
#     [
#         "01",
#         "11",
#     ],
#     [0],
#     [2, 3],
#     [5, 6, 7, 8],
#     1,
# )

# @cancel_inverses(recursive=True)
# @qnode(dev)
# def select_only_quantum(
#     data, control_wires, target_wires, select_wires, select_value, address
# ):  # pylint: disable=too-many-arguments
#     BasisEmbedding(address, wires=control_wires)
#
#     SelectOnlyQRAM(
#         data,
#         control_wires=control_wires,
#         target_wires=target_wires,
#         select_wires=select_wires,
#         select_value=select_value,
#     )
#     return probs(wires=target_wires)
#
# qml.draw_mpl(select_only_quantum, level="device", style="pennylane_sketch")(
#     [
#         "001",
#         "010",
#         "011",
#         "100",
#     ],
#     [0, 1],
#     [2, 3, 4],
#     None,
#     None,
#     0
# )

# print(math.sum([[0, 1, 1], [1, 1, 1]]))

# @qml.decomposition.register_resources({
#     qml.H: 2,
#     qml.CZ: 1
# })
# def _custom_cnot_decomposition(wires, **_):
#     qml.Hadamard(wires=wires[1])
#     qml.CZ(wires=[wires[0], wires[1]])
#     qml.Hadamard(wires=wires[1])
#
# qml.decomposition.add_decomps(qml.CNOT, _custom_cnot_decomposition)
#
# qml.decomposition.enable_graph()
#
# def circuit():
#     qml.CNOT(wires=[0, 1])
#     return qml.expval(qml.X(1))
#
# dev = qml.device('default.qubit', wires=2)
# qnode = qml.QNode(circuit, dev)
# decomp_qnode = qml.transforms.decompose(circuit, gate_set={qml.CZ, qml.H})
# print(qml.draw(decomp_qnode, level="device")())

# @cancel_inverses(recursive=True)
# @qnode(dev)
# def hybrid_quantum():  # pylint: disable=too-many-arguments
#     BasisEmbedding(1, wires=[1])
#
#     HybridQRAM(
#         [
#             "010",
#             "111",
#             "110",
#             "000",
#         ],
#         [0, 1],
#         [2, 3, 4],
#         [5, 6, 7, 8, 9],
#         1,
#     )
#     return probs(wires=[2, 3, 4])
#
# qml.draw_mpl(hybrid_quantum, level="device", style="pennylane_sketch")()

# print(math.sum([[0, 1, 1], [1, 1, 1]]))

# print(1 << 5)

# block_encoding = qml.Hadamard(wires=0)  # note H is a block encoding of 1/sqrt(2)
# phase_shifts = [qml.RZ(-2 * theta, wires=0) for theta in (1.23, -0.5, 4)]  # -2*theta to match convention
#
# q_script = qml.tape.QuantumScript(ops=[qml.QSVT(block_encoding, phase_shifts)])
# q_scripts, func = qml.transforms.decompose(q_script, gate_set={"H", "Adjoint(H)", "RZ"})
# q_script = func(q_scripts)
# print(q_script.draw(decimals=2))
# 6, wires=[0]), CNOT(wires=[1, 0]), PhaseShift(1.5707963267948966, wires=[0]), X(1)]

# import pennylane as qml
#
# dev = qml.device('default.qubit', wires=2)
#
# @qml.qnode(dev)
# def circuit(theta):
#     qml.X(0)
#     return qml.expval(qml.PauliZ(0))
#
# result = circuit(0.2)

# dictionary = {
#     0: [0, 1, 2],
#     1: [0, 1, 2],
# }
#
# cpy = defaultdict(list, {k: v[:] for k, v in dictionary.items()})
#
# print(cpy)

# @qml.transforms.decompose
# @qml.qnode(dev)
# def circuit():
#     SelectOnlyQRAM(
#         data=[
#             "000",
#             "101",
#             "010",
#             "111",
#             "000",
#             "101",
#             "010",
#             "111",
#             "000",
#             "101",
#             "010",
#             "111",
#             "000",
#             "101",
#             "010",
#             "111",
#         ],
#         control_wires=(0, 1),
#         target_wires=(2, 3, 4),
#         select_wires=(5, 6),
#         select_value=0,
#     )
#
# print(qml.specs(circuit)())
# print(qml.draw(circuit)())

# @qml.transforms.decompose
# @qml.qnode(dev)
# def circuit():
#     SelectOnlyQRAM(
#         data=[
#             "0",
#             "1",
#             "0",
#             "1",
#         ],
#         control_wires=(0,),
#         target_wires=(1),
#         select_wires=(2,),
#         select_value=0,
#     )
#
# print(qml.specs(circuit)())
# print(qml.draw(circuit)())

# @qml.qnode(dev)
# def circuit():
#     qml.QROM(
#         [
#             "001",
#             "010",
#             "011",
#             "100",
#         ],
#         [0, 1],
#         [2, 3, 4],
#         [],
#         clean=True
#     )
#
# qml.draw_mpl(circuit, level="device", style="pennylane_sketch")()

import pennylane as qp

import catalyst

# dev = qml.device("default.qubit", wires=2)
#
# tape = qml.tape.QuantumScript(
#     [qml.Rot(0.3, 0.1, 0.2, wires=0)], [qml.expval(qml.PauliZ(1))]
# )
#
# print([tape.wires.index(w) for w in dev.wires])

# @qjit
# @qml.qnode(dev)
# def circuit(state):
#     qml.MottonenStatePreparation(state_vector=state, wires=range(3))
#     return qml.state()
#
# state = np.array([1, 2j, 3, 4j, 5, 6j, 7, 8j])
# state = state / np.linalg.norm(state)
#
# circuit(state)

# dev = qml.device("lightning.qubit", wires=10)
# @qml.qjit(capture=True)
# @qml.qnode(dev)
# def c():
#     poly = np.array([0, -1, 0, 0.5, 0, 0.5])
#     angles = qml.poly_to_angles(poly, "QSVT")
#     input_matrix = np.array([[0.2, 0.1], [0.1, -0.1]])
#
#     wires = [0, 1]
#     block_encode = qml.BlockEncode(input_matrix, wires=wires)
#     projectors = [
#         qml.PCPhase(angles[i], dim=len(input_matrix), wires=wires)
#         for i in range(len(angles))
#     ]
#     qml.QSVT(block_encode, projectors)
#     return qml.state()
#
# c()
#
# dev = qml.device("lightning.qubit", wires=10)
#
# @qjit
# @qml.qnode(dev)
# def circuit():
#     qml.IQP(
#         weights=math.random.uniform(0, 2 * np.pi, 4),
#         num_wires=4,
#         pattern=local_gates(4, 1),
#         spin_sym=False,
#     )
#     return qml.state()
#
# circuit()
# from scipy.stats import ortho_group
# O = ortho_group.rvs(4, random_state=51)
#
# @qml.decompose(gate_set=qml.gate_sets.ALL_OPS)
# def circ():
#     qml.BasisRotation(wires=range(4), unitary_matrix=O)
#
# print(qml.draw(circ)())

# @qml.transforms.cancel_inverses
# @qml.qnode(dev)
# def circ():
#     qml.QFT(wires=[0, 1, 2])
#     qml.adjoint(qml.QFT)(wires=[0, 1, 2])
#
# print(qml.draw(circ)())
#
# import pennylane as qml
#
# qml.capture.enable()

# @qml.transforms.cancel_inverses
# @qml.qnode(qml.device("null.qubit", wires=3))
# def circ():
#     qml.QFT(wires=[0, 1, 2])
#     qml.adjoint(qml.QFT)(wires=[0, 1, 2])
#
# print(qml.draw(circ)())

# print(qml.matrix(qml.QFT))


# from pennylane.devices.default_qubit import ALL_DQ_GATE_SET
# print(*ALL_DQ_GATE_SET, sep="\n")

# class Op1(qml.operation.Operator):
#     resource_keys = set()
#
#     def decomposition(self):
#         return [qml.X(self.wires)]
#
# class Op2(qml.operation.Operator):
#
#     resource_keys = set()
#
#     @property
#     def resources(self):
#         return {}
#
# @qml.decomposition.register_resources({Op1: 1})
# def f(wires):
#     Op1(wires)
#
# qml.add_decomps(Op2, f)
#
# tape = qml.tape.QuantumScript([Op2(0)])
# print(qml.decompose(tape)[0][0].circuit)

# from jax import numpy as jnp
#
# device = qml.device("lightning.qubit", wires=3)
#
# theta = 0.4
#
# # Capture enabled
#
# qml.capture.enable()
#
# @qml.qnode(device)
# def circuit(theta):
#     qml.ctrl(qml.PCPhase, control=[1], control_values=[False])(theta, 2, wires=[0])
#     return qml.state()
#
# capture_result = qjit(circuit)(theta)
#
# # Capture disabled
#
# qml.capture.disable()
#
# assert jnp.allclose(capture_result, circuit(theta))


# manual = list(map(lambda op: qml.adjoint(op), qml.QFT.operator(wires=[0, 1]).decomposition()))[::-1]
#
# @qml.qnode(dev)
# def circ():
#     qml.adjoint(qml.QFT)(wires=[0, 1])
#     return qml.expval(qml.PauliZ(0))
#
# with AnnotatedQueue() as q:
#     circ()
#
# auto = q.queue
#
# print(manual)
# print("==========================================")
# print(auto)

# from pennylane.templates import AbstractArray, adjoint_subroutine_resource_rep
#
#
# def S0_resources(params, wires, rotation):
#     return {qml.resource_rep(rotation): params.shape[0]}
#
#
# @partial(qml.templates.core.Subroutine, static_argnames="rotation", compute_resources=S0_resources)
# def S0(params, wires, rotation):
#     for x in params:
#         rotation(x, wires)
#
# class MyOp(qml.operation.Operation):
#     pass
#
# abstract_params = AbstractArray((4, ), float)
# abstract_wires = AbstractArray(()) # a single wire
# S0_resources = adjoint_subroutine_resource_rep(S0, abstract_params, abstract_wires, qml.RX)
#
# @qml.decomposition.register_resources({S0_resources: 1})
# def MyOpDecomposition(wires):
#     # data of shape (4, ) and dtype float
#     params = np.array([1.0, 2.0, 3.0, 4.0])
#     adjoint(S0)(params, wires, qml.RX)
#
# qml.add_decomps(MyOp, MyOpDecomposition)
#
# qml.decomposition.enable_graph()
# @qml.qnode(qml.device('reference.qubit', wires=1))
# def c():
#     MyOp(wires=0)
#     return qml.state()
#
# print(qml.draw(qml.decompose(c, max_expansion=1))())

# state = np.array([[1.0, 0.0]])
# base = qml.StatePrep(state, wires=list(range(1)))
# op = Controlled(PauliZ(0), Wires("wire1"))

# op.decomposition()

# change_op_basis(QFT.operator((0, 1)), PauliX(0))

# import pennylane as qml
#
# qml.decomposition.enable_graph()
#
# dev = qml.device("default.qubit")
# @qml.qnode(dev)
# def circuit():
#     qml.H(0)
#     qml.CNOT([1,2])
#     qml.ctrl(
#         qml.change_op_basis(qml.SWAP([1,2]), qml.PhaseAdder(1, x_wires=[1,2])),
#         control=0
#     )
#     return qml.state()
#
# circuit2 = qml.decompose(circuit, max_expansion=1)
#
# print(qml.draw(circuit2)())

# # works
# qml.ControlledPhaseShift(0.1, wires=[0, 1])
#
# # works
# qml.ctrl(
#     qml.PhaseShift(0.1, wires=[1]),
#     0
# )
#
# # does not work?
# Controlled(
#     qml.PhaseShift(0.1, wires=[1]),
#     control_wires=0,
#     control_values=[0],
#     work_wires=[],
#     work_wire_type="borrowed",
# )

# Controlled(base_class=<class 'pennylane.ops.qubit.parametric_ops_single_qubit.PhaseShift'>, base_params={}, num_control_wires=1, num_work_wires=0, num_zero_control_values=1, signature_key=(AbstractArray(shape=(1,), dtype=<class 'int'>), AbstractArray(shape=(1,), dtype=<class 'int'>), AbstractArray(shape=(1,), dtype=<class 'int'>), AbstractArray(shape=(), dtype=<class 'int'>), (PyTreeStructure(), (AbstractArray(shape=(), dtype=<class 'NoneType'>),)), 'borrowed'), work_wire_type=borrowed)
# Controlled(base_class=<class 'pennylane.ops.qubit.parametric_ops_single_qubit.PhaseShift'>, base_params={'signature_key': (AbstractArray(shape=(1,), dtype=<class 'int'>), AbstractArray(shape=(1,), dtype=<class 'int'>))}, num_control_wires=1, num_work_wires=0, num_zero_control_values=1, work_wire_type=borrowed)


# with AnnotatedQueue() as q:
#     rot = np.array(
#         [
#             [-0.618452, -0.68369054 - 0.38740723j],
#             [-0.78582258, 0.53807284 + 0.30489424j],
#         ]
#     )
#     wires = range(len(rot))
#     qml.adjoint(qml.BasisRotation)(wires, rot)
#
# print(q.queue)

# @partial(a=0.1, reg1=Wires([0]), reg2=Wires([1]))
# def f(a, reg1, reg2):
#     qml.BasisState(np.zeros(len(reg2)), reg2)
#     qml.QFT(reg1)
#     qml.RX(a, reg1[0])
#
#
# def g(wires):
#     qml.PauliX(wires[0])
#
# @qnode(dev)
# def circuit():
#     qml.change_op_basis(f, partial(g, Wires([0])))
#     return qml.state()
#
# circuit3 = qml.decompose(circuit, max_expansion=1)
#
# print(qml.draw(circuit3)())
#
# @qjit
# @qnode(dev)
# def power():
#     qml.RX(0.1, 0)
#
#
# power()

# assert issubclass(qml.ops.Pow, qml.ops.Pow)

# ops.change_op_basis(
#     partial(ops.adjoint(QFT), wires=[0, 1]),
#     ops.ctrl(ops.X(2), control=3, control_values=1),
#     partial(QFT, wires=[0, 1]),
# )

# import pennylane as qp
#
# wires = 3
#
# qp.capture.enable()
#
# dev = qp.device('lightning.qubit', wires=wires)
#
# @qjit
# @qp.qnode(dev)
# def circuit_qft():
#     qp.QFT(wires=range(wires))
#     return qp.state()
#
# circuit_qft()
#
# @Subroutine
# def MyTemplate(x, y, wires):
#   qml.RX(x, wires[0])
#   qml.RY(y, wires[0])
#
# @qml.qnode(qml.device('lightning.qubit'))
# def c():
#   MyTemplate(0.1, 0.2, 0)
#   return qml.state()

# qml.decomposition.enable_graph()
#
# op = qml.ctrl(qml.pow(qml.QubitUnitary([[0, 1], [1, 0]], wires=0), 1), control=1)
# tape = qml.tape.QuantumScript([op])
# [new_tape], _ = qml.decompose(tape, gate_set={qml.ControlledQubitUnitary})
# assert new_tape.operations == [qml.ControlledQubitUnitary([[0, 1], [1, 0]], wires=[1, 0])]

# print(qp.matrix(qp.QFT.operator([0,1,2])))

# import numpy as np
#
#
# def _make_first_flipped_bits(c, i=0):
#     """Compute the pattern [c-1, c-2, c-1, c-3, c-1, c-2, c-1, c-4...] recursively.
#
#     For example, for ``c=4``, we get a first call (with ``i=0``) that produces
#     ``output =_make_first_flipped_bit(4, 0) = sub_0 + [0] + sub_0``, where
#     ``sub_0 = _make_first_flipped_bit(3, 1) = sub_1 + [1] + sub_1``, where
#     ``sub_1 = _make_first_flipped_bit(2, 2) = sub_2 + [2] + sub_2``, where
#     ``sub_2 = _make_first_flipped_bit(1, 3) = [3]``.
#
#     Overall this gives
#     ``sub_1 = [3, 2, 3]``
#     ``sub_0 = [3, 2, 3, 1, 3, 2, 3]``
#     ``output = [3, 2, 3, 1, 3, 2, 3, 0, 3, 2, 3, 1, 3, 2, 3]``.
#     """
#     if c == 1:
#         return [i]
#     sub = _make_first_flipped_bits(c - 1, i=i + 1)
#     return sub + [i] + sub
#
#
# def old(c, K):
#     bits = _make_first_flipped_bits(c)
#     first_flipped_bits = np.array(bits[: K - 1], dtype=int)
#     num_elbows = c - 1 + np.sum(c - 1 - np.clip(first_flipped_bits, a_min=1, a_max=None))
#     return num_elbows
#
#
# def new(c, K):
#     ell = (K - 1).bit_count() + int(np.floor((K - 1) / 2 ** (c - 1)))
#     num_elbows = c + K - ell - 2
#     return num_elbows
#
#
# cs = range(1, 15)
# for c in cs:
#     for K in range(1, 2 ** c + 1):
#         assert new(c, K) == old(c, K)


# dev = qml.device("lightning.qubit", wires=1)
#
# @qml.qjit(capture=True)
# @qml.qnode(dev)
# def circuit():
#     # mat = qml.matrix(qml.X)(0)
#     mat = qml.matrix(qml.X)
#     return qml.state()
#
# circuit()

# import pennylane as qml
#
# device = qml.device("default.qubit", wires=2)
#
# @qml.set_shots(1000)
# @qml.qnode(device)
# def circuit():
#     qml.Hadamard(wires=0)
#     qml.CNOT(wires=[0, 1])
#     qml.GlobalPhase(1/4)
#     return qml.expval(qml.PauliZ(wires=0)), qml.expval(qml.PauliZ(wires=1))
#
# loaded_circuit = qml.to_openqasm(circuit)()
# print(loaded_circuit)
#
# qasm = """
# OPENQASM 2.0;
# qreg q[2];
# creg c[2];
# h q[0];
# cx q[0],q[1];
# gphase(0.25);
# measure q -> c;
# """
#
# with queuing.AnnotatedQueue() as q:
#     qml.from_qasm3(qasm)()
#
# print(q.queue)
#
# dev = qp.device("default.qubit")
#
# circ_full = qp.tape.qscript.QuantumScript([qp.Identity(q) for q in Wires(["a", "b"])], [qp.state()])
# result = qp.execute([circ_full], dev, diff_method=None)
# print(result)

# import cirq
#
# state = np.array([1., 0., 1., 0.])
# dm = np.outer(state, np.conj(state))
# dm_reshaped = dm.reshape([2] * 2 * 2)
#
# print(qml.math.partial_trace(dm, indices=[index for index in range(len(state.shape)) if index not in list(range(1))]))
#
# print(cirq.partial_trace(dm, keep_indices=list(range(1))))
#
# print(list(range(5)))

# from pennylane.templates.subroutines.ffft import FFFT
# from pennylane import device, draw_mpl, qnode, state
#
# dev = device("default.qubit", wires=(0, 1, 2, 3))
#
# wires = (0, 1, 2, 3)
#
# @qnode(dev)
# def circuit(wires):
#     FFFT(wires)
#     return state()
#
# # print(draw(circuit, level="device")(wires))
# draw_mpl(circuit, level="device", style="pennylane_sketch")(wires)

# dev = qml.device("lightning.qubit", wires=10)
# @qml.qjit(capture=True)
# @qml.qnode(dev)
# def c():
#     poly = np.array([0, -1, 0, 0.5, 0, 0.5])
#     angles = qml.poly_to_angles(poly, "QSVT")
#     input_matrix = np.array([[0.2, 0.1], [0.1, -0.1]])
#
#     wires = [0, 1]
#     block_encode = qml.BlockEncode(input_matrix, wires=wires)
#     projectors_cos = [
#         qml.PCPhase(angles[i], dim=len(input_matrix), wires=wires)
#         for i in range(len(angles))
#     ]
#     qp.ctrl(qp.QSVT, control=2, control_values=[1])(block_encode, projectors_cos)
#     return qml.state()
#
# c()

# rng = np.random.default_rng(42)
# real_parts = rng.random(4)
# imag_parts = rng.random(4)
# state_vector = real_parts + 1j * imag_parts
# state_vector = state_vector / np.linalg.norm(state_vector)
#
# # Set up device
# dev = qp.device("lightning.qubit", wires=(0, 1))
#
# @qp.qnode(dev)
# def mottonen_qnode():
#     qp.MottonenStatePreparation(state_vector, wires=range(2))
#     return qp.probs(wires=0)
#
# qjited = qjit(mottonen_qnode, capture=True)
# qjited.jit_compile(tuple())

# mod = 4
# qft_output_wires = [0, 1]
# x_wires = [2, 3]
# y_wires = [5, 6]
# work_wire = 7
#
# @qnode(dev)
# def circuit():
#     ControlledSequence(
#         ControlledSequence(PhaseAdder(1, qft_output_wires, mod, work_wire), control=x_wires),
#         control=y_wires,
#     )
#     # qp.PhaseAdder(k, x_wires=x_wires, mod=mod, work_wire=work_wire)
#
# qml.draw_mpl(circuit, level="device", style="pennylane_sketch")()

# from pennylane.templates import OutMultiplier
#
# decomps = list_decomps(OutMultiplier)
#
# @qnode(dev)
# def circ():
#     decomps['_out_multiplier_with_adder'](
#         x_wires=[0, 1],
#         y_wires=[2, 3],
#         output_wires=[4, 5],
#         mod=4,
#         work_wires=[6, 7],
#         output_wires_zeroed=False
#     )
#
# print(qml.draw(circ, level="device")())


# from pennylane.templates.subroutines.ffft import FFFT
# from pennylane import device, draw, qnode, state, draw_mpl
#
# dev = device("default.qubit", wires=(0, 1, 2, 3, 4, 5, 6, 7))
#
# wires = (0, 1, 2, 3, 4, 5, 6, 7)
#
# @qnode(dev)
# def circuit(wires):
#     FFFT(wires, True)
#     return state()
#
# # print(draw(circuit, level="device")(wires))
# draw_mpl(circuit, level="device", style="pennylane_sketch")(wires)

# import pennylane as qp
# from pennylane import Incrementer, state
# from pennylane.ops import Controlled
#
# dev = qp.device("default.qubit")
#
# @qp.qnode(dev)
# def inc(wires, work_wires):
#     Controlled(
#         Incrementer(wires, work_wires),
#         11,
#         [1]
#     )
#     return state()
#
# wires, work_wires = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), (6, 7, 8, 9, 10)
#
# print(qp.draw(inc, level="device")(wires, work_wires))

# import pennylane as qml
# from pennylane import qjit
# from pennylane.decomposition import enable_graph

# # the compile pipeline we want to replace our QFTs *before* they get to Catalyst
# pipeline = qml.CompilePipeline(
#     qml.decompose(gate_set=qml.decomposition.gate_sets.CLIFFORD_T_PLUS_RZ)
# )

# # shared setup
# enable_graph()
# dev = qml.device("null.qubit", wires=(0, 1, 2))

# # This example works as expected
# @qml.qnode(dev)
# def circuit():
#     qml.QFT((0, 1, 2))
#     return qml.expval(qml.Z(1))

# decomposed = pipeline(circuit)
# qjited = qjit(decomposed, capture=True)
# qjited.jit_compile(tuple())

# This example fails, as the QFT is still in the circuit by the time we get to
# the DecompRuleInterpreter in Catalyst, which cannot handle it and errors out
# @qml.qnode(dev)
# def circuit():
#     qml.adjoint(qml.QFT((0, 1, 2)))
#     return qml.expval(qml.Z(1))

# print(inspect_decomps(qml.QFT((0, 1, 2))))
# print(inspect_decomps(qml.adjoint(qml.QFT)((0, 1, 2))))

# qscript = qml.decompose(QuantumScript([qml.adjoint(qml.QFT)((0, 1, 2))]), gate_set=qml.decomposition.gate_sets.CLIFFORD_T_PLUS_RZ)

# tape = construct_tape(circuit, level="user")()
# print(tape.operations)

# decomposed = qml.decompose(circuit, gate_set=qml.decomposition.gate_sets.CLIFFORD_T_PLUS_RZ) # pipeline(circuit)

# tape = construct_tape(decomposed, level="user")()
# print(tape.operations)

# qjited = qjit(decomposed, capture=True)
# qjited.jit_compile(tuple())

# import numpy as np
# import pennylane as qml
#
#
# def random_su2(rng):
#     X = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
#     Q, R = np.linalg.qr(X)
#     d = np.diag(R)
#     Q = Q * (d / np.abs(d))
#     Q /= np.linalg.det(Q) ** 0.5
#     return Q
#
#
# def compose_ops(ops):
#     U = np.eye(4, dtype=complex)
#     for op in ops:
#         U = qml.matrix(op, wire_order=(0, 1)) @ U
#     return U
#
#
# def phase_align_error(U, V):
#     overlap = np.vdot(V.reshape(-1), U.reshape(-1))
#     phase = overlap / abs(overlap)
#     return np.linalg.norm(U - phase * V)
#
#
# rng = np.random.default_rng(123)
# A, B, C, D = [random_su2(rng) for _ in range(4)]
#
# ops = [
#     qml.QubitUnitary(A, 0),
#     qml.QubitUnitary(B, 1),
#     qml.CNOT((1, 0)),
#     qml.RZ(8.471910266820082e-04, 0),
#     qml.RX(-1.3556395803158893e-03, 1),
#     qml.CNOT((1, 0)),
#     qml.QubitUnitary(C, 0),
#     qml.QubitUnitary(D, 1),
# ]
#
# U = compose_ops(ops)
# decomp = qml.ops.two_qubit_decomposition(U, wires=(0, 1))
# U2 = compose_ops(decomp)
#
# print("returned ops:", [op.name for op in decomp])
# print("num CNOTs:", sum(op.name == "CNOT" for op in decomp))
# print("phase-aligned error:", phase_align_error(U, U2))

# import pennylane as qml
#
# dev = qml.device("default.clifford", wires=3)
#
# @qml.qnode(dev)
# def circuit():
#     qml.Hadamard(0)
#     qml.CNOT(wires=[0, 1])
#     qml.CNOT(wires=[1, 2])
#     return qml.mutual_info(wires0=[0], wires1=[1])
#
# result = circuit()
# print(f"Got: {result}")
# print(f"Expected: 0.6931")

# qp.apply


qp.decomposition.enable_graph()

gate_set = {"RZ", "S", "Adjoint(S)", "Hadamard", "CNOT", "GlobalPhase"}
fixed = {qp.RY: qp.list_decomps(qp.RY)["_ry_to_rz_cliff"]}  # a rule that already ships with PL


@qp.qjit(capture=True, target="mlir")
@qp.transforms.decompose(gate_set=gate_set, fixed_decomps=fixed)
@qp.qnode(qp.device("null.qubit", wires=1))
def circuit():
    qp.RY(0.3, wires=0)
    return qp.probs()


print(qp.specs(circuit, level="all")())
