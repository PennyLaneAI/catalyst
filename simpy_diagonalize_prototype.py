import pennylane as qp
from catalyst.python_interface.transforms import diagonalize_final_measurements_pass

dev = qp.device("null.qubit", wires=4)

def diagonalize_measurements_setup_inputs(to_eigvals: bool = False, supported_base_obs: list[str] = "PauliZ"):
    "Docstring for my_transform."
    return (), {"to_eigvals": to_eigvals, "supported_base_obs": supported_base_obs}

diagonalize_measurements = qp.transform(pass_name="diagonalize-final-measurements", setup_inputs=diagonalize_measurements_setup_inputs)

@qp.qjit(target="mlir", keep_intermediate=True)
@diagonalize_measurements(to_eigvals=True)
@qp.qnode(dev, shots=1000)
def circuit():
    qp.CRX(0.1, wires=[0, 1])
    return qp.expval(qp.Z(0))

circuit()