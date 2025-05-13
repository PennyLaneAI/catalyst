# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# RUN: %PYTHON %s | FileCheck %s

# pylint: disable=line-too-long

"""Lit tests for capturing and compiling circuits with parametric arbitrary-basis mid-circuit
measurements from PennyLane's ftqc module in Catalyst.
"""

from functools import partial

import pennylane as qml
import pennylane.ftqc as plft

from catalyst import qjit


def test_measure_x():
    """Test the compilation of the qml.ftqc.measure_x function, which performs a mid-circuit
    measurement in the Pauli X basis.

    `measure_x` is implemented as a measurement in the XY plane with angle=0.
    """
    dev = qml.device("null.qubit", wires=1)

    qml.capture.enable()

    # CHECK-LABEL: @workload_measure_x
    @qjit(target="mlir")
    @qml.qnode(dev)
    def workload_measure_x():
        # CHECK: [[angle:%.+]] = arith.constant 0.000000e+00 : f64
        # CHECK: [[qreg:%.+]] = quantum.alloc( 1) : !quantum.reg
        # CHECK: [[q0:%.+]] = quantum.extract [[qreg]][ 0] : !quantum.reg -> !quantum.bit
        # CHECK: [[mres:%.+]], [[out_qubit:%.+]] = mbqc.measure_in_basis[ XY, [[angle]]] [[q0]] : i1, !quantum.bit
        _ = plft.measure_x(0)
        return qml.expval(qml.Z(0))

    qml.capture.disable()

    print(workload_measure_x.mlir)


test_measure_x()


def test_measure_y():
    """Test the compilation of the qml.ftqc.measure_y function, which performs a mid-circuit
    measurement in the Pauli Y basis.

    `measure_y` is implemented as a measurement in the XY plane with angle=pi/2.
    """
    dev = qml.device("null.qubit", wires=1)

    qml.capture.enable()

    # CHECK-LABEL: @workload_measure_y
    @qjit(target="mlir")
    @qml.qnode(dev)
    def workload_measure_y():
        # CHECK: [[angle:%.+]] = arith.constant 1.5707963267948966 : f64
        # CHECK: [[qreg:%.+]] = quantum.alloc( 1) : !quantum.reg
        # CHECK: [[q0:%.+]] = quantum.extract [[qreg]][ 0] : !quantum.reg -> !quantum.bit
        # CHECK: [[mres:%.+]], [[out_qubit:%.+]] = mbqc.measure_in_basis[ XY, [[angle]]] [[q0]] : i1, !quantum.bit
        _ = plft.measure_y(0)
        return qml.expval(qml.Z(0))

    qml.capture.disable()

    print(workload_measure_y.mlir)


test_measure_y()


def test_measure_z():
    """Test the compilation of the qml.ftqc.measure_z function, which performs a mid-circuit
    measurement in the Pauli Z (computational) basis.

    `measure_z` is a wrapper around the standard computation-basis measurement op (qml.measure).

    NOTE: At the time of writing, Catalyst does not support qml.measure() with program capture
    enabled. This test is expected to fail until support is added. The expected output has been
    commented out using FileCheck comment directives, `COM:`.
    """
    dev = qml.device("null.qubit", wires=1)

    qml.capture.enable()

    # COM: CHECK-LABEL: @workload_measure_z
    @qjit(target="mlir")
    @qml.qnode(dev)
    def workload_measure_z():
        # COM: CHECK: [[qreg:%.+]] = quantum.alloc( 1) : !quantum.reg
        # COM: CHECK: [[q0:%.+]] = quantum.extract [[qreg]][ 0] : !quantum.reg -> !quantum.bit
        # COM: CHECK: [[mres:%.+]], [[out_qubit:%.+]] = quantum.measure [[q0]] : i1, !quantum.bit
        _ = plft.measure_z(0)
        return qml.expval(qml.Z(0))

    qml.capture.disable()

    print(workload_measure_z.mlir)


try:
    test_measure_z()

except NotImplementedError:
    ...


def test_measure_arbitrary_basis(angle, plane):
    """Test the compilation of the qml.ftqc.measure_arbitrary_basis function, which performs a
    mid-circuit measurement in an arbitrary basis defined by a plane and rotation angle about that
    plane on the supplied qubit.

    In this case, the rotation angle is static (known at compile time).
    """
    dev = qml.device("null.qubit", wires=1)

    qml.capture.enable()

    @qjit(target="mlir")
    @qml.qnode(dev)
    def workload_measure_arbitrary_basis():
        _ = plft.measure_arbitrary_basis(wires=0, angle=angle, plane=plane)
        return qml.expval(qml.Z(0))

    qml.capture.disable()

    print(workload_measure_arbitrary_basis.mlir)


# CHECK-LABEL: public @workload_measure_arbitrary_basis()
# CHECK: [[angle:%.+]] = arith.constant 1.000000e-01 : f64
# CHECK: [[qreg:%.+]] = quantum.alloc( 1) : !quantum.reg
# CHECK: [[q0:%.+]] = quantum.extract [[qreg]][ 0] : !quantum.reg -> !quantum.bit
# CHECK: [[mres:%.+]], [[out_qubit:%.+]] = mbqc.measure_in_basis[ XY, [[angle]]] [[q0]] : i1, !quantum.bit
test_measure_arbitrary_basis(0.1, "XY")


# CHECK-LABEL: public @workload_measure_arbitrary_basis()
# CHECK: [[angle:%.+]] = arith.constant 2.000000e-01 : f64
# CHECK: [[qreg:%.+]] = quantum.alloc( 1) : !quantum.reg
# CHECK: [[q0:%.+]] = quantum.extract [[qreg]][ 0] : !quantum.reg -> !quantum.bit
# CHECK: [[mres:%.+]], [[out_qubit:%.+]] = mbqc.measure_in_basis[ YZ, [[angle]]] [[q0]] : i1, !quantum.bit
test_measure_arbitrary_basis(0.2, "YZ")


# CHECK-LABEL: public @workload_measure_arbitrary_basis()
# CHECK: [[angle:%.+]] = arith.constant 3.000000e-01 : f64
# CHECK: [[qreg:%.+]] = quantum.alloc( 1) : !quantum.reg
# CHECK: [[q0:%.+]] = quantum.extract [[qreg]][ 0] : !quantum.reg -> !quantum.bit
# CHECK: [[mres:%.+]], [[out_qubit:%.+]] = mbqc.measure_in_basis[ ZX, [[angle]]] [[q0]] : i1, !quantum.bit
test_measure_arbitrary_basis(0.3, "ZX")


try:
    # The plane 'XZ' is invalid; check that this raise a ValueError
    test_measure_arbitrary_basis(0.4, "XZ")

except ValueError as e:
    assert "Measurement plane must be one of ['XY', 'YZ', 'ZX']" in str(e)


def test_measure_arbitrary_basis_dyn_angle(plane):
    """Test the compilation of the qml.ftqc.measure_arbitrary_basis function, which performs a
    mid-circuit measurement in an arbitrary basis defined by a plane and rotation angle about that
    plane on the supplied qubit.

    In this case, the rotation angle is dynamic (not known until runtime).
    """
    dev = qml.device("null.qubit", wires=1)

    qml.capture.enable()

    @qjit(target="mlir")
    @qml.qnode(dev)
    def workload_measure_arbitrary_basis_dyn_angle(_angle: float):
        _ = plft.measure_arbitrary_basis(wires=0, angle=_angle, plane=plane)
        return qml.expval(qml.Z(0))

    qml.capture.disable()

    print(workload_measure_arbitrary_basis_dyn_angle.mlir)


# CHECK-LABEL: public @workload_measure_arbitrary_basis_dyn_angle(%arg0: tensor<f64>)
# CHECK-DAG: [[qreg:%.+]] = quantum.alloc( 1) : !quantum.reg
# CHECK-DAG: [[q0:%.+]] = quantum.extract [[qreg]][ 0] : !quantum.reg -> !quantum.bit
# CHECK-DAG: [[angle_tensor:%.+]] = stablehlo.convert %arg0 : tensor<f64>
# CHECK-DAG: [[angle:%.+]] = tensor.extract [[angle_tensor]][] : tensor<f64>
# CHECK: %mres, %out_qubit = mbqc.measure_in_basis[ XY, [[angle]]] %2 : i1, !quantum.bit
test_measure_arbitrary_basis_dyn_angle("XY")


def test_pseudo_mbqc_workload():
    """Test the compilation of a pseudo-MBQC workload.

    This workload implements a simplified mock circuit in an explicit MBQC-like representation that
    contains all of the elements of a typical gate represented in the MBQC formalism, but with
    reduced complexity for the sake of testing.
    """
    dev = qml.device("null.qubit", wires=3)

    qml.capture.enable()

    # CHECK-LABEL: public @workload_pseudo_mbqc(
    @qjit(target="mlir")
    @qml.qnode(dev)
    def workload_pseudo_mbqc(rotation_angle: float):
        # CHECK: [[cst_zero:%.+]] = arith.constant 0.000000e+00 : f64

        # Create graph state
        # CHECK: quantum.custom "Hadamard"() {{.+}} : !quantum.bit
        # CHECK: quantum.custom "Hadamard"() {{.+}} : !quantum.bit
        # CHECK: quantum.custom "Hadamard"() {{.+}} : !quantum.bit
        qml.Hadamard(0)
        qml.Hadamard(1)
        qml.Hadamard(2)

        # CHECK: quantum.custom "CZ"() {{.+}}, {{.+}} : !quantum.bit, !quantum.bit
        # CHECK: quantum.custom "CZ"() {{.+}}, {{.+}} : !quantum.bit, !quantum.bit
        qml.CZ([0, 1])
        qml.CZ([1, 2])

        # Perform measurement pattern
        # CHECK: [[m0:%.+]], [[out0:%.+]] = mbqc.measure_in_basis[ XY, [[cst_zero]]] {{.+}} : i1, !quantum.bit
        m0 = plft.measure_x(0)

        # CHECK: [[neg_angle:%.+]] = stablehlo.negate %arg0 : tensor<f64>
        # CHECK: scf.if [[m0]] -> (tensor<i1>, !quantum.reg) {
        # CHECK:   stablehlo.convert %arg0 : tensor<f64>
        # CHECK:   [[angle_ext:%.+]] = tensor.extract {{.+}} : tensor<f64>
        # CHECK:   mbqc.measure_in_basis[ XY, [[angle_ext]]] {{.+}} : i1, !quantum.bit
        # CHECK: } else {
        # CHECK:   stablehlo.convert [[neg_angle]] : tensor<f64>
        # CHECK:   [[neg_angle_ext:%.+]] = tensor.extract {{.+}} : tensor<f64>
        # CHECK:   mbqc.measure_in_basis[ XY, [[neg_angle_ext]]] {{.+}} : i1, !quantum.bit
        # CHECK: }
        m1 = qml.ftqc.cond_measure(
            m0,
            partial(qml.ftqc.measure_arbitrary_basis, angle=rotation_angle),
            partial(qml.ftqc.measure_arbitrary_basis, angle=-rotation_angle),
        )(plane="XY", wires=1)

        # Apply by-product correction
        # CHECK: stablehlo.xor {{.+}}, {{.+}} : tensor<i1>
        # CHECK: [[xor_res:%.+]] = tensor.extract {{.+}}[] : tensor<i1>
        # CHECK: scf.if [[xor_res]] -> (!quantum.reg) {
        # CHECK:   quantum.custom "PauliX"() {{.+}} : !quantum.bit
        # CHECK: } else {
        # CHECK:   quantum.custom "Identity"() {{.+}} : !quantum.bit
        # CHECK: }
        qml.cond(m0 ^ m1, qml.X, qml.I)(2)

        # CHECK: [[obs:%.+]] = quantum.namedobs {{.+}}[ PauliZ] : !quantum.obs
        # CHECK: quantum.expval [[obs]] : f64
        return qml.expval(qml.Z(2))

    qml.capture.disable()

    print(workload_pseudo_mbqc.mlir)


test_pseudo_mbqc_workload()
