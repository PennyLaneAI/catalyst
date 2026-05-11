from qiskit import QuantumCircuit

# Bell state |Φ-⟩ = (|00⟩ - |11⟩)/√2
qc = QuantumCircuit(2, 2)
qc.h(0)
qc.z(0)
qc.cx(0, 1)
qc.measure([0, 1], [0, 1])
