import pennylane as qp

from catalyst import cond, for_loop
from catalyst.passes import graph_decomposition, phase_folding

# @qp.qjit(autograph=True, keep_intermediate=2, capture=False)
# @qp.transform(pass_name="convert-to-value-semantics")
# @phase_folding(report_stats=True, trace_abstraction=True)
# @qp.transform(pass_name="convert-to-reference-semantics")
# # Clifford + Rz
# # @qp.decompose(gates=[qp.X, qp.Y, qp.Z, qp.RZ, qp.CNOT])
# @qp.qnode(device=qp.device("lightning.qubit", wires=2))
# def circuit_base():
#     qp.CNOT([0, 1])
#     qp.T(1)
#     qp.CNOT([0, 1])
#     qp.CNOT([1, 0])
#     qp.T(0)
#     qp.CNOT([1, 0])
#     return qp.probs()


# @qp.qjit(keep_intermediate=2, capture=False)
# @qp.transform(pass_name="phase-folding")
# @qp.qnode(device=qp.device("lightning.qubit", wires=2))
# def circuit_ex424():
#     qp.T(0)
#     qp.T(1)
#     qp.CNOT([0, 1])
#     qp.adjoint(qp.T(1))
#     qp.CNOT([0, 1])

#     qp.H(1)
#     qp.T(0)
#     qp.T(1)
#     qp.CNOT([1, 0])
#     qp.adjoint(qp.T(0))
#     qp.CNOT([1, 0])


@qp.qjit(keep_intermediate=2, capture=True)
# @qp.transform(pass_name="phase-folding")
# @phase_folding(report_stats=True, trace_abstraction=True)
@qp.transform(pass_name="convert-to-reference-semantics")
@qp.qnode(device=qp.device("lightning.qubit", wires=2))
def circuit_ex425():
    # qp.StatePrep([0, 1], wires=1)
    qp.BasisState(0, wires=1)  # becomes dynamic, so not yet
    # qp.T(1)
    # qp.CNOT([0, 1])
    # qp.T(0)
    # qp.T(1)


# @qp.qjit(keep_intermediate=2, capture=False)
# @qp.transform(pass_name="phase-folding")
# @qp.qnode(device=qp.device("lightning.qubit", wires=1))
# def circuit_ex426():
#     qp.T(0)
#     qp.PauliX(0)
#     qp.adjoint(qp.T(0))
#     qp.PauliX(0)
#     return qp.probs()


# @qp.qjit(autograph=True, keep_intermediate=2, capture=False)
# @qp.transform(pass_name="convert-to-value-semantics")
# @phase_folding(report_stats=True, trace_abstraction=True)
# @qp.transform(pass_name="convert-to-reference-semantics")
# @qp.qnode(device=qp.device("lightning.qubit", wires=2))
# def circuit_if_aff_val(x: float):
#     qp.T(0)

# if x > 1.4:
#     qp.X(0)
#     qp.Hadamard(wires=1)
# else:
#     qp.Y(0)
#     qp.RZ(x, wires=1)

#     qp.T(0)

#     return qp.probs()

# @qp.qjit(autograph=True, keep_intermediate=2, capture=False)
# @qp.transform(pass_name="convert-to-value-semantics")
# @phase_folding(report_stats=True, trace_abstraction=True)
# @qp.transform(pass_name="convert-to-reference-semantics")
# @qp.qnode(device=qp.device("lightning.qubit", wires=2))
# def circuit_if_simple(x: float):
#     qp.T(0)

#     if x > 1.4:
#         qp.CNOT([0, 1])

#     qp.T(0)

#     return qp.probs()

# @qp.qjit(autograph=True, keep_intermediate=2, capture=False)
# @qp.transform(pass_name="convert-to-value-semantics")
# @phase_folding(report_stats=True, trace_abstraction=True)
# @qp.transform(pass_name="convert-to-reference-semantics")
# @qp.qnode(device=qp.device("lightning.qubit", wires=2))
# def circuit_loop_block(x: float):
#     qp.T(0)
#     qp.Hadamard(0)

#     for i in range(10):
#         qp.T(0)

#     qp.Hadamard(0)
#     qp.T(0)

#     return qp.probs()

# @qp.qjit(autograph=True, keep_intermediate=2, capture=False)
# @qp.transform(pass_name="convert-to-value-semantics")
# @phase_folding(report_stats=True, trace_abstraction=True)
# @qp.transform(pass_name="convert-to-reference-semantics")
# @qp.qnode(device=qp.device("lightning.qubit", wires=3))
# def circuit_loop_cycle(x: float):
#     qp.T(1)
#     qp.Hadamard(0)

#     for i in range(10):
#         qp.SWAP([0, 1])
#         qp.SWAP([1, 2])

#     qp.T(1)

#     return qp.probs()

# @qp.qjit(autograph=True, keep_intermediate=2, capture=False)
# @qp.transform(pass_name="convert-to-value-semantics")
# @phase_folding(report_stats=True, trace_abstraction=True)
# @qp.transform(pass_name="convert-to-reference-semantics")
# @qp.qnode(device=qp.device("lightning.qubit", wires=2))
# def circuit_loop_h(x: float):
#     qp.T(1)

#     for i in range(10):
#         qp.Hadamard(0)

#     qp.T(1)

#     return qp.probs()


# from catalyst.jax_primitives import decomposition_rule
# import jax

# @decomposition_rule(op_type=qp.Toffoli)
# def decompose_toffoli(wires):
#     qp.Z(0)

# @qp.qjit(autograph=True, keep_intermediate=2, capture=False)
# # @qp.transform(pass_name="convert-to-value-semantics")
# # @phase_folding(report_stats=True, trace_abstraction=True)
# # @qp.transform(pass_name="convert-to-reference-semantics")
# @graph_decomposition(gate_set=[qp.H, qp.X, qp.Z, qp.Y, qp.S, qp.CNOT, qp.SWAP, qp.T, qp.RZ])
# @qp.qnode(device=qp.device("lightning.qubit", wires=3))
# def circuit_decomp(x: float):
#     qp.CNOT([0, 1])
#     qp.Toffoli([0, 1, 2])
#     qp.T(1)

#     decompose_toffoli(jax.core.ShapedArray((3,), int))

#     # if x > 1.4:
#     #     qp.Toffoli([1, 2, 0])
#     # else:
#     #     qp.Y(0)
#     #     qp.RZ(x, wires=1)

#     return qp.probs()

# circuit_base()
# circuit_ex424()
# circuit_ex425()
# circuit_ex426()
# circuit_if_aff_val(1.5)
# circuit_if_simple(1.5)
# circuit_loop_block(1.5)
# circuit_loop_cycle(1.5)
# circuit_loop_h(1.5)
# circuit_decomp(1.5)


# print("=== Generated MLIR ===")
# print(circuit.mlir)

# @qjit(autograph=True, keep_intermediate=2, capture=True)
# @phase_folding(report_stats=True, trace_abstraction=True)
# # @qp.transforms.decompose(gate_set=clifford_rz_gates)
# @qp.qnode(device=qp.device("lightning.qubit", wires=3))
# def circ1(x: float):
#     # Toffoli(0, 1, 2)

#     qp.RZ(1.2, 0)
#     if x > 1.4:
#         qp.X(0)
#         qp.H(1)
#     else:
#         qp.Y(0)
#         qp.RZ(0.3, 1)
#     qp.RZ(2.3, 0)

#     # Toffoli(2, 1, 0)
#     return qp.probs()