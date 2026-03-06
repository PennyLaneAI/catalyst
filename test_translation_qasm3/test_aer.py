from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import qiskit.qasm3

qc1 = QuantumCircuit.from_qasm_file("qasm3_circuits/cascade_measure.qasm")
sim = AerSimulator()
qc_t1 = transpile(qc1, sim)
c1 = sim.run(qc_t1, shots=10).result().get_counts()

qasm = """OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q0;
h q0[0];
cx q0[0], q0[1];
bit m_2;
m_2 = measure q0[0];
bit m_3;
m_3 = measure q0[1];
"""

qc2 = qiskit.qasm3.loads(qasm)
qc_t2 = transpile(qc2, sim)
c2 = sim.run(qc_t2, shots=10).result().get_counts()

print("Qiskit format:", c1)
print("QASM3 format:", c2)
