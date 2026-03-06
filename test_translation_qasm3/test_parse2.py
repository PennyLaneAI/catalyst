import openqasm3

qasm_str = "OPENQASM 3.0;\nqubit[2] q0;\nh q0[0];\nbit m_2;\nm_2 = measure q0[0];"

try:
    print(openqasm3.parse(qasm_str))
    print("Parsed!")
except Exception as e:
    import traceback
    traceback.print_exc()

