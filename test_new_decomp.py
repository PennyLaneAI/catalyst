from functools import partial

import jax
import numpy as np
import pennylane as qml
from pennylane.ftqc import RotXZX
from pennylane.typing import TensorLike
from pennylane.wires import WiresLike

from catalyst import qjit
from catalyst.from_plxpr import from_plxpr
from catalyst.jax_primitives import decomposition_rule

qml.capture.enable()
qml.decomposition.enable_graph()


######################################
# Custom decomposition rules
######################################


@decomposition_rule
def Rule_ry_to_rz_rx(phi, wires: WiresLike, **__):
    qml.RZ(-np.pi / 2, wires=wires)
    qml.RX(phi, wires=wires)
    qml.RZ(np.pi / 2, wires=wires)


@decomposition_rule
def Rule_rot_to_rz_ry_rz(phi, theta, omega, wires: WiresLike, **__):
    qml.RZ(phi, wires=wires)
    qml.RY(theta, wires=wires)
    qml.RZ(omega, wires=wires)


@decomposition_rule
def Rule_u2_phaseshift_rot(phi, delta, wires, **__):
    pi_half = qml.math.ones_like(delta) * (np.pi / 2)
    qml.Rot(delta, pi_half, -delta, wires=wires)
    qml.PhaseShift(delta, wires=wires)
    qml.PhaseShift(phi, wires=wires)


@decomposition_rule
def Rule_xzx_decompose(phi, theta, omega, wires, **__):
    qml.RX(phi, wires=wires)
    qml.RZ(theta, wires=wires)
    qml.RX(omega, wires=wires)


@qml.qjit()
@partial(qml.transforms.decompose, gate_set={"RX", "RZ", "PhaseShift"})
@qml.qnode(qml.device("lightning.qubit", wires=3))
def circuit():

    qml.RY(0.5, wires=0)
    qml.Rot(0.1, 0.2, 0.3, wires=1)
    qml.U2(0.4, 0.5, wires=2)
    RotXZX(0.6, 0.7, 0.8, wires=0)

    Rule_ry_to_rz_rx(0, 0)
    Rule_rot_to_rz_ry_rz(0, 0, 0, 1)
    Rule_u2_phaseshift_rot(0, 0, 2)
    Rule_xzx_decompose(0, 0, 0, 0)

    return qml.expval(qml.Z(0))


print(circuit.mlir)


###################################################
# MBQC Example with custom decomposition to RotXZX
###################################################

qml.decomposition.enable_graph()
qml.capture.enable()


@qml.register_resources({qml.ftqc.RotXZX: 1})
@decomposition_rule
def Rule_rot_to_xzx(phi, theta, omega, wires, **__):
    mat = qml.Rot.compute_matrix(phi, theta, omega)
    lam, theta, phi = qml.math.decomposition.xzx_rotation_angles(mat)
    qml.ftqc.RotXZX(lam, theta, phi, wires)


@qml.qjit()
@partial(
    qml.transforms.decompose,
    gate_set={"X", "Y", "Z", "S", "H", "CNOT", "RZ", "RotXZX", "GlobalPhase"},
    fixed_decomps={qml.Rot: Rule_rot_to_xzx},
)
@qml.qnode(qml.device("null.qubit", wires=3))
def mbqc_circ(x: float, y: float):
    qml.RX(x, 0)
    qml.RY(y, 1)

    Rule_rot_to_xzx(
        float, float, float, int
    )  # this needs to be here to include the custom decomposition in the graph
    Rule_ry_to_rz_rx(float, int)
    Rule_xzx_decompose(float, float, float, int)

    return qml.expval(qml.Z(0)), qml.expval(qml.Z(1))


print(mbqc_circ.mlir)
