OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];

// Bell state |Ψ-⟩ = (|01⟩ - |10⟩)/√2
h q[0];
z q[0];
x q[1];
cx q[0], q[1];

measure q -> c;
