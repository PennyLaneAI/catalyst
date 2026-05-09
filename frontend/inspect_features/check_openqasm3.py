import pennylane as qml

from catalyst import qjit

# Use the local simulator instead of the AWS-hosted one
dev = qml.device("braket.local.qubit", wires=2)


@qjit(keep_intermediate=True)
@qml.qnode(dev)
def circuit(x):
    qml.Hadamard(wires=0)
    qml.RX(x, wires=0)
    return qml.expval(qml.PauliZ(0))


# To see the OpenQASM 3 without running, you can inspect the MLIR
# which Catalyst uses as the source for the OQ3 generator.
# with open("circuit.mlir", "w") as f:
#     f.write(circuit.__str__())
# print(dir(circuit))
x = 0.54
result = circuit(x)
print(f"Result: {result}")


# # Define a device that uses OpenQASM 3 for execution
# dev = qml.device("braket.aws.qubit", device_arn="arn:aws:braket:::device/quantum-simulator/amazon/sv1", wires=2)

# @qjit
# @qml.qnode(dev)
# def circuit(x):
#     qml.Hadamard(wires=0)
#     qml.CNOT(wires=[0, 1])
#     qml.RX(x, wires=0)
#     return qml.expval(qml.PauliZ(0))

# # Catalyst compiles this and the runtime generates OpenQASM 3
# # to send to the Braket service during execution.
# result = circuit(0.54)
# print(result)
