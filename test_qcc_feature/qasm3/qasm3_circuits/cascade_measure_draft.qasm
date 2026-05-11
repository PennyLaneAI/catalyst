OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
creg c[3];

h q[0];
measure q[0] -> c[0];
if(c==1) x q[1];
measure q[1] -> c[1];
if(c==2) h q[2]; // c is treated as integer? Qiskit parser nuances. 
// c==1 means c[0]=1 (if c[1,2] are 0). c==2 means c[1]=1.
// Let's stick to single bit conditions if possible or ensure qiskit importer handles register conditions (it does not yet, only bit)
// So we use standard qiskit if(creg, val).
// Our importer only handles single bit conditions currently: if_else(clbits=[bit]).
// So we should verify what Qiskit standard 'if' produces.
// Qiskit 'if' on register checks the integer value of the register.
// Our importer checks 'clbits[0]'. If qiskit produces 'if_else' with clbits=entire_register, our importer will fail or check only first bit.
// Let's use single bit measurements to single bit registers to match our importer logic "if(c1==1)".
// But here we defined `creg c[3]`.
// Let's redefine to use separate registers to be safe with our basic importer.
