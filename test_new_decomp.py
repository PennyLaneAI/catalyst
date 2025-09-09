"""
This file contains a few tests for the end-to-end custom decomposition rules

TODO: remove the file after testing
"""

from functools import partial

import jax
import numpy as np
import pennylane as qml
from pennylane.ftqc import RotXZX
from pennylane.wires import WiresLike

from catalyst.jax_primitives import decomposition_rule

qml.capture.enable()
qml.decomposition.enable_graph()


######################################
# Custom decomposition rules
######################################


@decomposition_rule
def _ry_to_rz_rx(phi, wires: WiresLike, **__):
    """Decomposition of RY gate using RZ and RX gates."""
    qml.RZ(-np.pi / 2, wires=wires)
    qml.RX(phi, wires=wires)
    qml.RZ(np.pi / 2, wires=wires)


@decomposition_rule
def _rot_to_rz_ry_rz(phi, theta, omega, wires: WiresLike, **__):
    """Decomposition of Rot gate using RZ and RY gates."""
    qml.RZ(phi, wires=wires)
    qml.RY(theta, wires=wires)
    qml.RZ(omega, wires=wires)


@decomposition_rule
def _u2_phaseshift_rot(phi, delta, wires, **__):
    """Decomposition of U2 gate using Rot and PhaseShift gates."""
    pi_half = qml.math.ones_like(delta) * (np.pi / 2)
    qml.Rot(delta, pi_half, -delta, wires=wires)
    qml.PhaseShift(delta, wires=wires)
    qml.PhaseShift(phi, wires=wires)


@decomposition_rule
def _xzx_decompose(phi, theta, omega, wires, **__):
    """Decomposition of Rot gate using RX and RZ gates in XZX format."""
    qml.RX(phi, wires=wires)
    qml.RZ(theta, wires=wires)
    qml.RX(omega, wires=wires)


@qml.qjit()
@partial(qml.transforms.decompose, gate_set={"RX", "RZ", "PhaseShift"})
@qml.qnode(qml.device("lightning.qubit", wires=3))
def circuit():
    """Circuit to test custom decomposition rules."""
    qml.RY(0.5, wires=0)
    # qml.Rot(0.1, 0.2, 0.3, wires=1)
    # qml.U2(0.4, 0.5, wires=2)
    # RotXZX(0.6, 0.7, 0.8, wires=0)

    _ry_to_rz_rx(float, jax.core.ShapedArray((1,), jax.numpy.dtype("int64")))
    # _rot_to_rz_ry_rz(0, 0, 0, 1)
    # _u2_phaseshift_rot(0, 0, 2)
    # _xzx_decompose(0, 0, 0, 0)

    return qml.expval(qml.Z(0))


print(circuit.mlir)


###################################################
# MBQC Example with custom decomposition to RotXZX
###################################################

qml.decomposition.enable_graph()
qml.capture.enable()


@qml.register_resources({qml.ftqc.RotXZX: 1})
@decomposition_rule
def _rot_to_xzx(phi, theta, omega, wires, **__):
    """Decomposition of Rot gate using RotXZX gate."""
    mat = qml.Rot.compute_matrix(phi, theta, omega)
    lam, theta, phi = qml.math.decomposition.xzx_rotation_angles(mat)
    qml.ftqc.RotXZX(lam, theta, phi, wires)


@qml.qjit()
@qml.transforms.cancel_inverses
@qml.transforms.merge_rotations
@partial(
    qml.transforms.decompose,
    gate_set={"X", "Y", "Z", "S", "H", "CNOT", "RZ", "RotXZX", "GlobalPhase"},
    fixed_decomps={qml.Rot: _rot_to_xzx},
)
@qml.qnode(qml.device("null.qubit", wires=3))
def mbqc_circ(x: float, y: float):
    """MBQC example to test custom decomposition to RotXZX."""
    qml.RX(x, 0)
    qml.RY(y, 1)

    _rot_to_xzx(float, float, float, int)
    _ry_to_rz_rx(float, int)
    _xzx_decompose(float, float, float, int)

    return qml.expval(qml.Z(0)), qml.expval(qml.Z(1))


print(mbqc_circ.mlir)
