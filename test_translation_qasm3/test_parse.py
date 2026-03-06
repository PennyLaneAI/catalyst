import openqasm3

qasm_str = """OPENQASM 3.0;
include "stdgates.inc";

qubit q0[2];
h q0[0];
bit m_2 = measure q0[0];
"""

try:
    openqasm3.parse(qasm_str)
    print("Parsed successfully!")
except Exception as e:
    import traceback
    traceback.print_exc()

