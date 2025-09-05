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


@decomposition_rule
def _ry_to_rz_rx(phi, wires: WiresLike, **__):
    qml.RZ(-np.pi / 2, wires=wires)
    qml.RX(phi, wires=wires)
    qml.RZ(np.pi / 2, wires=wires)


@decomposition_rule
def _rot_to_rz_ry_rz(phi, theta, omega, wires: WiresLike, **__):
    qml.RZ(phi, wires=wires)
    qml.RY(theta, wires=wires)
    qml.RZ(omega, wires=wires)


@decomposition_rule
def _u2_phaseshift_rot(phi, delta, wires, **__):
    pi_half = qml.math.ones_like(delta) * (np.pi / 2)
    qml.Rot(delta, pi_half, -delta, wires=wires)
    qml.PhaseShift(delta, wires=wires)
    qml.PhaseShift(phi, wires=wires)


@decomposition_rule
def _xzx_decompose(phi, theta, omega, wires, **__):
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

    _ry_to_rz_rx(0, 0)
    _rot_to_rz_ry_rz(0, 0, 0, 1)
    _u2_phaseshift_rot(0, 0, 2)
    _xzx_decompose(0, 0, 0, 0)

    return qml.expval(qml.Z(0))


print(circuit.mlir)
