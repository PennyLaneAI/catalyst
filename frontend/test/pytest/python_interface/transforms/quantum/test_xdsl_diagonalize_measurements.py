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
"""Unit test module for the xDSL implementation of the diagonalize_final_measurements pass"""

import re

import numpy as np
import pennylane as qp
import pytest
from pennylane.exceptions import CompileError

from catalyst.passes.builtin_passes import diagonalize_measurements
from catalyst.python_interface.transforms import (
    DiagonalizeFinalMeasurementsPass,
    diagonalize_final_measurements_pass,
)

pytestmark = pytest.mark.xdsl


_non_commuting_err_msg = (
    "Observables are not qubit-wise commuting. Please apply the `split-non-commuting` pass first"
)


class TestDiagonalizeFinalMeasurementsPass:
    """Unit tests for the diagonalize-final-measurements pass."""

    @pytest.mark.parametrize("to_eigvals", [1, 0, True, "False"])
    def test_with_to_eigvals_raise_errors(self, to_eigvals):
        """Test a ValueError is raised if to_eigvals is not set as False."""
        expected_msg = "Only to_eigvals = False is supported."
        with pytest.raises(ValueError, match=expected_msg):
            _ = DiagonalizeFinalMeasurementsPass(to_eigvals=to_eigvals)

    @pytest.mark.parametrize(
        "supported_base_obs",
        [
            "PauliX",
            "pauliz",
            ("paulix",),
            ("paulix", "pauliy"),
        ],
    )
    def test_with_supported_base_obs_raise_errors(self, supported_base_obs):
        """Test a ValueError is raised if supported_base_obs is not a subset of [PauliX,
        PauliY, PauliZ, Hadamard, and Identity]."""
        expected_msg = (
            "Supported base observables must be a subset of (PauliX, PauliY, PauliZ, Hadamard, "
            f"and Identity) passed as a tuple[str], but received {supported_base_obs}"
        )
        with pytest.raises(ValueError, match=re.escape(expected_msg)):
            _ = DiagonalizeFinalMeasurementsPass(supported_base_obs=supported_base_obs)

    def test_with_pauli_z(self, run_filecheck):
        """Test that a PauliZ observable is not affected by diagonalization"""

        program = """
            func.func @test_func() attributes {quantum.node} {
                %0 = "test.op"() : () -> !quantum.bit

                // CHECK: quantum.namedobs %0[PauliZ] : !quantum.obs
                %1 = quantum.namedobs %0[PauliZ] : !quantum.obs

                // CHECK: quantum.expval %1 : f64
                %2 = quantum.expval %1 : f64
                return
            }
            """

        pipeline = (DiagonalizeFinalMeasurementsPass(),)
        run_filecheck(program, pipeline)

    @pytest.mark.parametrize(
        "supported_base_obs",
        [("PauliX",), ("PauliX", "PauliY")],
    )
    def test_with_supported_base_obs(self, supported_base_obs, run_filecheck):
        """Check observables in the supported_base_obs would not be diagonalized."""
        program = """
            func.func @test_func() attributes {quantum.node} {
                // CHECK: [[q0:%.*]] = "test.op"() : () -> !quantum.bit
                %0 = "test.op"() : () -> !quantum.bit

                // CHECK-NEXT: quantum.namedobs %0[PauliX] : !quantum.obs
                %1 = quantum.namedobs %0[PauliX] : !quantum.obs

                // CHECK: quantum.var %1 : f64
                %2 = quantum.var %1 : f64
                return
            }
            """
        pipeline = (DiagonalizeFinalMeasurementsPass(supported_base_obs=supported_base_obs),)
        run_filecheck(program, pipeline)

    def test_with_identity(self, run_filecheck):
        """Test that an Identity observable is not affected by diagonalization."""

        program = """
            func.func @test_func() attributes {quantum.node} {
                // CHECK: [[q0:%.*]] = "test.op"() : () -> !quantum.bit
                %0 = "test.op"() : () -> !quantum.bit

                // CHECK: quantum.namedobs %0[Identity] : !quantum.obs
                %1 = quantum.namedobs %0[Identity] : !quantum.obs

                // CHECK: quantum.var %1 : f64
                %2 = quantum.var %1 : f64
                return
            }
            """
        pipeline = (DiagonalizeFinalMeasurementsPass(),)
        run_filecheck(program, pipeline)

    def test_with_pauli_x_base_obs(self, run_filecheck):
        """Test that when a PauliX observable diagonalization is not expected."""

        program = """
            func.func @test_func() attributes {quantum.node} {
                // CHECK: [[q0:%.*]] = "test.op"() : () -> !quantum.bit
                %0 = "test.op"() : () -> !quantum.bit

                // CHECK-NEXT: [[obs:%.*]] = quantum.namedobs [[q0]][PauliX] : !quantum.obs
                %1 = quantum.namedobs %0[PauliX] : !quantum.obs

                // CHECK: quantum.expval [[obs]]
                %2 = quantum.expval %1 : f64
            }
            """

        pipeline = (DiagonalizeFinalMeasurementsPass(supported_base_obs=("PauliX",)),)
        run_filecheck(program, pipeline)

    def test_with_pauli_x(self, run_filecheck):
        """Test that when diagonalizing a PauliX observable, the expected diagonalizing
        gates are inserted and the observable becomes PauliZ."""

        program = """
            func.func @test_func() attributes {quantum.node} {
                // CHECK: [[q0:%.*]] = "test.op"() : () -> !quantum.bit
                %0 = "test.op"() : () -> !quantum.bit

                // CHECK: [[q0_1:%.*]] = quantum.custom "Hadamard"() [[q0]]
                // CHECK-NEXT: [[q0_2:%.*]] =  quantum.namedobs [[q0_1]][PauliZ]
                // CHECK-NOT: quantum.namedobs [[q:%.+]][PauliX]
                %1 = quantum.namedobs %0[PauliX] : !quantum.obs

                // CHECK: quantum.expval [[q0_2]]
                %2 = quantum.expval %1 : f64
                return
            }
            """

        pipeline = (DiagonalizeFinalMeasurementsPass(),)
        run_filecheck(program, pipeline)

    def test_with_pauli_y_base_obs(self, run_filecheck):
        """Test that when a PauliY observable diagonalization is not expected."""

        program = """
            func.func @test_func() attributes {quantum.node} {
                // CHECK: [[q0:%.*]] = "test.op"() : () -> !quantum.bit
                %0 = "test.op"() : () -> !quantum.bit

                // CHECK-NEXT: [[obs:%.*]] = quantum.namedobs [[q0]][PauliY] : !quantum.obs
                %1 = quantum.namedobs %0[PauliY] : !quantum.obs

                // CHECK: quantum.expval [[obs]]
                %2 = quantum.expval %1 : f64
            }
            """

        pipeline = (DiagonalizeFinalMeasurementsPass(supported_base_obs=("PauliY",)),)
        run_filecheck(program, pipeline)

    def test_with_pauli_y(self, run_filecheck):
        """Test that when diagonalizing a PauliY observable, the expected diagonalizing
        gates are inserted and the observable becomes PauliZ."""

        program = """
            func.func @test_func() attributes {quantum.node} {
                // CHECK: [[q0:%.*]] = "test.op"() : () -> !quantum.bit
                %0 = "test.op"() : () -> !quantum.bit

                // CHECK: [[q0_1:%.*]] = quantum.custom "PauliZ"() [[q0]]
                // CHECK-NEXT: [[q0_2:%.*]] = quantum.custom "S"() [[q0_1]]
                // CHECK-NEXT: [[q0_3:%.*]] = quantum.custom "Hadamard"() [[q0_2]]
                // CHECK-NEXT: [[q0_4:%.*]] =  quantum.namedobs [[q0_3]][PauliZ]
                // CHECK-NOT: quantum.namedobs [[q:%.+]][PauliY]
                %1 = quantum.namedobs %0[PauliY] : !quantum.obs

                // CHECK: quantum.expval [[q0_4]]
                %2 = quantum.expval %1 : f64
                return
            }
            """

        pipeline = (DiagonalizeFinalMeasurementsPass(),)
        run_filecheck(program, pipeline)

    def test_with_hadamard_base_obs(self, run_filecheck):
        """Test that when a Hadamard observable diagonalization is not expected."""

        program = """
            func.func @test_func() attributes {quantum.node} {
                // CHECK: [[q0:%.*]] = "test.op"() : () -> !quantum.bit
                %0 = "test.op"() : () -> !quantum.bit

                // CHECK-NEXT: [[obs:%.*]] = quantum.namedobs [[q0]][Hadamard] : !quantum.obs
                %1 = quantum.namedobs %0[Hadamard] : !quantum.obs

                // CHECK: quantum.expval [[obs]]
                %2 = quantum.expval %1 : f64
            }
            """

        pipeline = (DiagonalizeFinalMeasurementsPass(supported_base_obs=("Hadamard",)),)
        run_filecheck(program, pipeline)

    def test_with_hadamard(self, run_filecheck):
        """Test that when diagonalizing a Hadamard observable, the expected diagonalizing
        gates are inserted and the observable becomes PauliZ."""

        program = """
            func.func @test_func() attributes {quantum.node} {
                // CHECK: [[q0:%.*]] = "test.op"() : () -> !quantum.bit
                %0 = "test.op"() : () -> !quantum.bit

                // CHECK: [[quarter_pi:%.*]] = arith.constant -0.78539816339744828 : f64
                // CHECK-NEXT: [[q0_1:%.*]] = quantum.custom "RY"([[quarter_pi]]) [[q0]]
                // CHECK-NEXT: [[q0_2:%.*]] =  quantum.namedobs [[q0_1]][PauliZ]
                // CHECK-NOT: quantum.namedobs [[q:%.+]][Hadamard]
                %1 = quantum.namedobs %0[Hadamard] : !quantum.obs

                // CHECK: quantum.expval [[q0_2]]
                %2 = quantum.expval %1 : f64
            }
            """

        pipeline = (DiagonalizeFinalMeasurementsPass(),)
        run_filecheck(program, pipeline)

    def test_with_composite_observable_with_supported_base(self, run_filecheck):
        """Test transform on a measurement process with a tuple of base observable. In this
        case, the simplified program is based on the MLIR generated by the circuit

        @qp.qjit(target="mlir")
        @qp.qnode(qp.device("lightning.qubit", wires=3))
        def circuit():
            return qp.expval(qp.Y(0)@qp.X(1) + qp.Z(2))
        """

        program = """
            func.func @test_func() attributes {quantum.node} {
                // CHECK: [[q0:%.*]] = "test.op"() : () -> !quantum.bit
                // CHECK: [[q1:%.*]] = "test.op"() : () -> !quantum.bit
                // CHECK: [[q2:%.*]] = "test.op"() : () -> !quantum.bit
                %0 = "test.op"() : () -> !quantum.bit
                %1 = "test.op"() : () -> !quantum.bit
                %2 = "test.op"() : () -> !quantum.bit

                // CHECK: [[q_y:%.*]] = quantum.namedobs [[q0]][PauliY]
                %3 = quantum.namedobs %0[PauliY] : !quantum.obs

                // CHECK: [[q_x:%.*]] = quantum.namedobs [[q1]][PauliX]
                %4 = quantum.namedobs %1[PauliX] : !quantum.obs

                // CHECK: [[tensor0:%.*]] = quantum.tensor [[q_y]], [[q_x]] : !quantum.obs
                %5 = quantum.tensor %3, %4 : !quantum.obs

                // CHECK: [[q_z:%.*]] = quantum.namedobs [[q2]][PauliZ] : !quantum.obs
                %6 = quantum.namedobs %2[PauliZ] : !quantum.obs

                // CHECK: [[size:%.*]] = "test.op"() : () -> tensor<2xf64>
                %size_info = "test.op"() : () -> tensor<2xf64>

                // CHECK: quantum.hamiltonian([[size]] : tensor<2xf64>) [[tensor0]], [[q_z]] : !quantum.obs
                %7 = quantum.hamiltonian(%size_info : tensor<2xf64>) %5, %6 : !quantum.obs

                // CHECK: quantum.expval
                %8 = quantum.expval %7 : f64
                return
            }
            """

        pipeline = (DiagonalizeFinalMeasurementsPass(supported_base_obs=("PauliX", "PauliY")),)
        run_filecheck(program, pipeline)

    def test_with_composite_observable(self, run_filecheck):
        """Test transform on a measurement process with a composite observable. In this
        case, the simplified program is based on the MLIR generated by the circuit

        @qp.qjit(target="mlir")
        @qp.qnode(qp.device("lightning.qubit", wires=3))
        def circuit():
            return qp.expval(qp.Y(0)@qp.X(1) + qp.Z(2))
        """

        program = """
            func.func @test_func() attributes {quantum.node} {
                // CHECK: [[q0:%.*]] = "test.op"() : () -> !quantum.bit
                // CHECK: [[q1:%.*]] = "test.op"() : () -> !quantum.bit
                // CHECK: [[q2:%.*]] = "test.op"() : () -> !quantum.bit
                %0 = "test.op"() : () -> !quantum.bit
                %1 = "test.op"() : () -> !quantum.bit
                %2 = "test.op"() : () -> !quantum.bit

                // CHECK: [[q0_1:%.*]] = quantum.custom "PauliZ"() [[q0]]
                // CHECK: [[q0_2:%.*]] = quantum.custom "S"() [[q0_1]]
                // CHECK: [[q0_3:%.*]] = quantum.custom "Hadamard"() [[q0_2]]
                // CHECK: [[q_y:%.*]] =  quantum.namedobs [[q0_3]][PauliZ]
                // CHECK-NOT: quantum.namedobs [[q:%.+]][PauliY]
                %3 = quantum.namedobs %0[PauliY] : !quantum.obs

                // CHECK: [[q1_1:%.*]] = quantum.custom "Hadamard"() [[q1]]
                // CHECK: [[q_x:%.*]] = quantum.namedobs [[q1_1]][PauliZ]
                // CHECK-NOT: quantum.namedobs [[q:%.+]][PauliX]
                %4 = quantum.namedobs %1[PauliX] : !quantum.obs

                // CHECK: [[tensor0:%.*]] = quantum.tensor [[q_y]], [[q_x]] : !quantum.obs
                %5 = quantum.tensor %3, %4 : !quantum.obs

                // CHECK: [[q_z:%.*]] = quantum.namedobs [[q2]][PauliZ] : !quantum.obs
                %6 = quantum.namedobs %2[PauliZ] : !quantum.obs

                // CHECK: [[size:%.*]] = "test.op"() : () -> tensor<2xf64>
                %size_info = "test.op"() : () -> tensor<2xf64>

                // CHECK: quantum.hamiltonian([[size]] : tensor<2xf64>) [[tensor0]], [[q_z]] : !quantum.obs
                %7 = quantum.hamiltonian(%size_info : tensor<2xf64>) %5, %6 : !quantum.obs

                // CHECK: quantum.expval
                %8 = quantum.expval %7 : f64
                return
            }
            """

        pipeline = (DiagonalizeFinalMeasurementsPass(),)
        run_filecheck(program, pipeline)

    def test_with_multiple_measurements_supported_base_obs(self, run_filecheck):
        """Test a circuit with multiple measurements and all obs are supported. The
        simplified program for this test is based on the circuit

        @qp.qjit(target="mlir")
        @qp.qnode(qp.device("lightning.qubit", wires=3))
        def circuit():
            return qp.var(qp.Y(0)), qp.var(qp.X(1))
        """

        program = """
            func.func @test_func() attributes {quantum.node} {
                // CHECK: [[q0:%.*]] = "test.op"() : () -> !quantum.bit
                // CHECK-NEXT: [[q1:%.*]] = "test.op"() : () -> !quantum.bit
                %0 = "test.op"() : () -> !quantum.bit
                %1 = "test.op"() : () -> !quantum.bit


                // CHECK-NEXT: quantum.namedobs [[q0:%.+]][PauliY]
                %2 = quantum.namedobs %0[PauliY] : !quantum.obs
                // CHECK: quantum.var
                %3 = quantum.var %2 : f64


                // CHECK-NEXT: quantum.namedobs [[q1:%.+]][PauliX]
                %4 = quantum.namedobs %1[PauliX] : !quantum.obs

                // CHECK: quantum.expval
                %5 = quantum.expval %4 : f64
                return
            }
            """

        pipeline = (DiagonalizeFinalMeasurementsPass(supported_base_obs=("PauliX", "PauliY")),)
        run_filecheck(program, pipeline)

    def test_with_multiple_measurements(self, run_filecheck):
        """Test diagonalizing a circuit with multiple measurements. The simplified program
        for this test is based on the circuit

        @qp.qjit(target="mlir")
        @qp.qnode(qp.device("lightning.qubit", wires=3))
        def circuit():
            return qp.var(qp.Y(0)), qp.var(qp.X(1))
        """

        program = """
            func.func @test_func() attributes {quantum.node} {
                %0 = "test.op"() : () -> !quantum.bit
                %1 = "test.op"() : () -> !quantum.bit

                // CHECK: quantum.custom "PauliZ"()
                // CHECK-NEXT: quantum.custom "S"()
                // CHECK-NEXT: quantum.custom "Hadamard"()
                // CHECK-NEXT: quantum.namedobs [[q:%.+]][PauliZ]
                %2 = quantum.namedobs %0[PauliY] : !quantum.obs
                // CHECK: quantum.var
                %3 = quantum.var %2 : f64


                // CHECK: quantum.custom "Hadamard"()
                // CHECK-NEXT: quantum.namedobs [[q:%.+]][PauliZ]
                %4 = quantum.namedobs %1[PauliX] : !quantum.obs

                // CHECK: quantum.expval
                %5 = quantum.expval %4 : f64
                return
            }
            """

        pipeline = (DiagonalizeFinalMeasurementsPass(),)
        run_filecheck(program, pipeline)

    def test_overlapping_commuting_observables(self, run_filecheck):
        """Test the case where multiple overlapping (commuting) observables exist in
        the same circuit (diagonalization is only performed once).

        @qp.qjit(target="mlir")
        @qp.qnode(qp.device("lightning.qubit", wires=1))
        def circuit():
            return qp.var(qp.X(0)), qp.var(qp.X(0))
        """

        program = """
            func.func @test_func() attributes {quantum.node} {
                %0 = "test.op"() : () -> !quantum.bit
                // CHECK: quantum.custom "Hadamard"()
                // CHECK-NEXT: quantum.namedobs [[q:%.+]][PauliZ]
                %1 = quantum.namedobs %0[PauliX] : !quantum.obs
                %2 = quantum.var %1 : f64
                // CHECK-NOT: quantum.custom "Hadamard"()
                // CHECK: quantum.namedobs [[q:%.+]][PauliZ]
                %3 = quantum.namedobs %0[PauliX] : !quantum.obs
                %4 = quantum.var %3 : f64
            }
            """

        pipeline = (DiagonalizeFinalMeasurementsPass(),)

        run_filecheck(program, pipeline)

    def test_additional_qubit_uses_are_updated(self, run_filecheck):
        """Test that when diagonalizing the circuit, if the MLIR contains
        later manipulations of the qubit going into the observable, these are
        updated as well. While quantum.custom operations can't be applied to
        the same SSA value that is passed to the observable, it can still
        be inserted into a register or deallocated.

        The simplified program for this test is based on the circuit

        @qp.qjit(target="mlir")
        @qp.qnode(qp.device("lightning.qubit", wires=3))
        def circuit():
            return qp.expval(qp.X(1))
        """

        # we expect that instead of the SSA value that comes out of quantum.extract being passed to
        # both quantum.namedobs and the quantum.insert, it will be passed to the Hadamard, and the
        # SSA value that is output by the *Hadmard* operation will be passed to namedobs and insert.
        program = """
            func.func @test_func() attributes {quantum.node} {
                %0 = quantum.alloc(3) : !quantum.reg
                %1 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
                %2 = tensor.extract %1[] : tensor<i64>
                // CHECK: [[q0:%.*]] = quantum.extract
                %3 = quantum.extract %0[%2] : !quantum.reg -> !quantum.bit

                // CHECK: [[q0_1:%.*]] = quantum.custom "Hadamard"() [[q0]]
                // CHECK-NEXT: quantum.namedobs [[q0_1]][PauliZ]
                %4 = quantum.namedobs %3[PauliX] : !quantum.obs
                %5 = quantum.expval %4 : f64

                // CHECK: quantum.insert [[q:%.+]][[[q:%.+]]], [[q0_1]]
                %6 = tensor.extract %1[] : tensor<i64>
                %7 = quantum.insert %0[%6], %3 : !quantum.reg, !quantum.bit
                quantum.dealloc %7 : !quantum.reg
            }
        """

        pipeline = (DiagonalizeFinalMeasurementsPass(),)
        run_filecheck(program, pipeline)


@pytest.mark.usefixtures("use_both_frontend")
class TestDiagonalizeFinalMeasurementsProgramCaptureExecution:
    """Integration tests going through plxpr (program capture enabled)"""

    # pylint: disable=unnecessary-lambda
    @pytest.mark.parametrize(
        "mp, obs, expected_res",
        [
            (qp.expval, qp.Identity, lambda x: 1),
            (qp.var, qp.Identity, lambda x: 0),
            (qp.expval, qp.X, lambda x: 0),
            (qp.var, qp.X, lambda x: 1),
            (qp.expval, qp.Y, lambda x: -np.sin(x)),
            (qp.var, qp.Y, lambda x: 1 - np.sin(x) ** 2),
            (qp.expval, qp.Z, lambda x: np.cos(x)),
            (qp.var, qp.Z, lambda x: 1 - np.cos(x) ** 2),
            (qp.expval, qp.Hadamard, lambda x: np.cos(x) / np.sqrt(2)),
            (qp.var, qp.Hadamard, lambda x: (2 - np.cos(x) ** 2) / 2),
        ],
    )
    def test_with_single_obs(self, mp, obs, expected_res):
        """Test the diagonalization transform for a circuit with a single measurement
        of a single supported observable"""

        dev = qp.device("lightning.qubit", wires=1)

        @qp.qnode(dev)
        def circuit_ref(phi):
            qp.RX(phi, 0)
            return mp(obs(0))

        angle = 0.7692

        assert np.allclose(
            expected_res(angle), circuit_ref(angle)
        ), "Sanity check failed, is expected_res correct?"
        circuit_compiled = qp.qjit(
            diagonalize_final_measurements_pass(circuit_ref),
        )

        assert np.allclose(expected_res(angle), circuit_compiled(angle))

    def test_with_composite_observables(self):
        """Test the transform works for an observable built using operator arithmetic
        (sprod, prod, sum)"""

        dev = qp.device("lightning.qubit", wires=3)

        @qp.qnode(dev)
        def circuit_ref(x, y):
            qp.RX(x, 0)
            qp.RY(y, 1)
            qp.RY(y / 2, 2)
            return qp.expval(qp.Y(0) @ qp.X(1) + 3 * qp.X(2))

        def expected_res(x, y):
            y0_res = -np.sin(x)
            x1_res = np.sin(y)
            x2_res = np.sin(y / 2)
            return y0_res * x1_res + 3 * x2_res

        phi = 0.3867
        theta = 1.394

        assert np.allclose(
            expected_res(phi, theta), circuit_ref(phi, theta)
        ), "Sanity check failed, is expected_res correct?"
        circuit_compiled = qp.qjit(
            diagonalize_final_measurements_pass(circuit_ref),
        )

        assert np.allclose(expected_res(phi, theta), circuit_compiled(phi, theta))

    @pytest.mark.usefixtures("use_capture")
    def test_with_multiple_measurements(self):
        """Test that the transform runs and returns the expected results for
        a circuit with multiple measurements"""

        dev = qp.device("lightning.qubit", wires=2)

        @qp.qnode(dev)
        def circuit_ref(x, y):
            qp.RX(x, 0)
            qp.RY(y, 1)
            return qp.expval(qp.Y(0)), qp.var(qp.X(1))

        def expected_res(x, y):
            return -np.sin(x), 1 - np.sin(y) ** 2

        phi = 0.3867
        theta = 1.394

        assert np.allclose(
            expected_res(phi, theta), circuit_ref(phi, theta)
        ), "Sanity check failed, is expected_res correct?"

        circuit_compiled = qp.qjit(
            diagonalize_final_measurements_pass(circuit_ref),
        )

        assert np.allclose(expected_res(phi, theta), circuit_compiled(phi, theta))

    @pytest.mark.parametrize("phi", [0.123, 1.23])
    def test_overlapping_commute_observables_multiple_measurements(self, phi):
        """Test the case where multiple overlapping (commuting) observables exist in
        the same circuit."""

        dev = qp.device("lightning.qubit", wires=2)

        @qp.qjit
        @diagonalize_final_measurements_pass
        @qp.qnode(dev)
        def circuit(x):
            qp.RX(x, 0)
            return qp.expval(qp.Y(0)), qp.var(qp.Y(0))

        expected_expval = -np.sin(phi)
        expected_var = 1 - np.sin(phi) ** 2

        expval, var = circuit(phi)
        assert np.allclose(expected_expval, expval)
        assert np.allclose(expected_var, var)

    def test_non_commuting_observables_raise_error(self):
        """Check that an error is raised if we try to diagonalize a circuit that contains
        non-commuting observables."""
        dev = qp.device("lightning.qubit", wires=1)

        @qp.qjit
        @diagonalize_final_measurements_pass
        @qp.qnode(dev)
        def circuit(x):
            qp.RX(x, 0)
            return qp.expval(qp.Y(0)), qp.expval(qp.X(0))

        with pytest.raises(CompileError, match=f"{_non_commuting_err_msg}"):
            _ = circuit(0.7)


@pytest.mark.usefixtures("use_both_frontend")
class TestDiagonalizeFinalMeasurementsCatalystFrontend:
    """Integration tests going through the catalyst frontend (program capture disabled)"""

    # pylint: disable=unnecessary-lambda
    @pytest.mark.parametrize(
        "mp, obs, expected_res",
        [
            (qp.expval, qp.Identity, lambda x: 1),
            (qp.var, qp.Identity, lambda x: 0),
            (qp.expval, qp.X, lambda x: 0),
            (qp.var, qp.X, lambda x: 1),
            (qp.expval, qp.Y, lambda x: -np.sin(x)),
            (qp.var, qp.Y, lambda x: 1 - np.sin(x) ** 2),
            (qp.expval, qp.Z, lambda x: np.cos(x)),
            (qp.var, qp.Z, lambda x: 1 - np.cos(x) ** 2),
            (qp.expval, qp.Hadamard, lambda x: np.cos(x) / np.sqrt(2)),
            (qp.var, qp.Hadamard, lambda x: (2 - np.cos(x) ** 2) / 2),
        ],
    )
    def test_with_single_obs(self, mp, obs, expected_res):
        """Test the diagonalization transform for a circuit with a single measurement
        of a single supported observable"""

        dev = qp.device("lightning.qubit", wires=1)

        @qp.qnode(dev)
        def circuit_ref(phi):
            qp.RX(phi, 0)
            return mp(obs(0))

        angle = 0.7692

        assert np.allclose(
            expected_res(angle), circuit_ref(angle)
        ), "Sanity check failed, is expected_res correct?"

        circuit_compiled = qp.qjit(
            diagonalize_final_measurements_pass(circuit_ref),
        )

        np.allclose(expected_res(angle), circuit_compiled(angle))

    def test_with_composite_observables(self):
        """Test the transform works for an observable built using operator arithmetic
        (sprod, prod, sum)"""

        dev = qp.device("lightning.qubit", wires=3)

        @qp.qnode(dev)
        def circuit_ref(x, y):
            qp.RX(x, 0)
            qp.RY(y, 1)
            qp.RY(y / 2, 2)
            return qp.expval(qp.Y(0) @ qp.X(1) + 3 * qp.X(2))

        def expected_res(x, y):
            y0_res = -np.sin(x)
            x1_res = np.sin(y)
            x2_res = np.sin(y / 2)
            return y0_res * x1_res + 3 * x2_res

        phi = 0.3867
        theta = 1.394

        assert np.allclose(
            expected_res(phi, theta), circuit_ref(phi, theta)
        ), "Sanity check failed, is expected_res correct?"

        circuit_compiled = qp.qjit(
            diagonalize_final_measurements_pass(circuit_ref),
        )

        assert np.allclose(expected_res(phi, theta), circuit_compiled(phi, theta))

    def test_diagonalize_measurements_bultin_pass(self, run_filecheck_qjit):
        """Unit test for the diagonalize_measurements builtin pass."""

        dev = qp.device("lightning.qubit", wires=10)
        obs = qp.X(0)

        @qp.qjit
        @diagonalize_measurements(supported_base_obs=("PauliX",))
        @qp.qnode(dev)
        def circuit():
            qp.CNOT(wires=[0, 1])
            # CHECK: quantum.namedobs [[q:%.+]][PauliX]
            return qp.expval(qp.X(0))

        run_filecheck_qjit(circuit)

        res = circuit()

        @qp.qjit
        @qp.qnode(dev)
        def circuit_ref():
            qp.CNOT(wires=[0, 1])
            return qp.expval(obs)

        res_ref = circuit_ref()
        assert np.allclose(res, res_ref)

    def test_with_split_non_commuting_multiple_measurements(self, run_filecheck_qjit):
        """Test the executable file can be generated and run with lightning.qubit when applying
        both the diagonalize-final-measurements and the split-non-commuting passes"""

        dev = qp.device("lightning.qubit", wires=10)

        obs = qp.Hamiltonian([1.0, 2.0, 3.0], [qp.X(0), qp.Z(0), qp.I(2)])

        @qp.for_loop(0, 10, 1)
        def for_fn(i):
            qp.H(i)
            qp.S(i)
            qp.RZ(phi=0.1, wires=[i])

        @qp.while_loop(lambda i: i < 10)
        def while_fn(i):
            qp.H(i)
            qp.S(i)
            qp.RZ(phi=0.1, wires=[i])
            i = i + 1
            return i

        @qp.qjit
        @diagonalize_measurements
        @qp.transform(pass_name="split-non-commuting")
        @qp.qnode(dev)
        def circuit():
            for_fn()  # pylint: disable=no-value-for-parameter
            while_fn(0)
            qp.CNOT(wires=[0, 1])
            # CHECK: quantum.namedobs [[q:%.+]][PauliZ]
            return qp.expval(obs)

        run_filecheck_qjit(circuit)

        res = circuit()

        @qp.qjit
        @qp.qnode(dev)
        def circuit_ref():
            for_fn()  # pylint: disable=no-value-for-parameter
            while_fn(0)
            qp.CNOT(wires=[0, 1])
            return qp.expval(obs)

        res_ref = circuit_ref()
        assert np.allclose(res, res_ref)

    def test_with_multiple_measurements(self):
        """Test that the transform runs and returns the expected results for
        a circuit with multiple measurements"""

        dev = qp.device("lightning.qubit", wires=2)

        @qp.qnode(dev)
        def circuit_ref(x, y):
            qp.RX(x, 0)
            qp.RY(y, 1)
            return qp.expval(qp.Y(0)), qp.var(qp.X(1))

        def expected_res(x, y):
            return -np.sin(x), 1 - np.sin(y) ** 2

        phi = 0.3867
        theta = 1.394

        assert np.allclose(
            expected_res(phi, theta), circuit_ref(phi, theta)
        ), "Sanity check failed, is expected_res correct?"

        circuit_compiled = qp.qjit(
            diagonalize_final_measurements_pass(circuit_ref),
        )

        assert np.allclose(expected_res(phi, theta), circuit_compiled(phi, theta))


@pytest.mark.usefixtures("use_both_frontend")
class TestDiagonalizeFinalMeasurementsNonCommuteValidate:
    """Integrate test for NonCommutingObservableValidator"""

    A = np.array([[complex(1.0, 0.0), complex(2.0, 0.0)], [complex(2.0, 0.0), complex(-1.0, 0.0)]])
    # non-commuting Hamiltonians (Single Measurement)
    NON_COMMUTE_SINGLE_OBS_LIST = [
        qp.Hamiltonian([1.0, 1.0], [qp.Z(0), qp.X(0)]),  # qwc check
        qp.Hamiltonian([1.0, 1.0], [qp.Z(2), qp.Hadamard(2)]),  # non-overlap check via Hadamard
        qp.Hamiltonian(
            [1.0, 1.0], [qp.Z(1), qp.X(0) @ qp.Hermitian(A, wires=1)]
        ),  # non-overlap check via Hermitian
    ]

    # commuting Hamiltonians (Single Measurement)
    COMMUTE_SINGLE_OBS_LIST = [
        qp.Hamiltonian([1.0, 1.0], [qp.Z(0), qp.Z(0)]),  # qwc check
        qp.Hamiltonian([1.0, 1.0], [qp.Z(0), qp.I(0)]),  # qwc check
        qp.Hamiltonian([1.0, 1.0], [qp.Hadamard(0), qp.I(0)]),  # qwc check
        qp.Hamiltonian([1.0, 1.0], [qp.X(0), qp.I(0)]),  # qwc check
        qp.Hamiltonian(
            [1.0, 1.0], [qp.Z(1), qp.Hermitian(A, wires=0)]
        ),  # non-overlap check via Hermitian
    ]

    # non-commuting pairs (Multiple Measurements)
    NON_COMMUTE_MULTI_OBS_LIST = [
        (qp.X(0), qp.Y(0)),  # NamedObs
        (qp.X(0) @ qp.Y(1), qp.Z(1)),  # TensorObs
        (qp.Hamiltonian([1.0], [qp.Z(0)]), qp.X(0)),  # Hamiltonians vs NamedObs
        (qp.Hermitian(A, wires=1), qp.Z(1)),  # HermitianOps
    ]

    # commuting pairs (Multiple Measurements)
    COMMUTE_MULTI_OBS_LIST = [
        (qp.X(0), qp.I(0)),  # NamedObs
        (qp.X(0) @ qp.Z(1), qp.Z(1)),  # TensorObs
        (qp.Hamiltonian([1.0], [qp.Z(0)]), qp.I(0)),  # Hamiltonians vs NamedObs
        (qp.Hermitian(A, wires=1), qp.X(2)),  # HermitianOps
    ]

    @pytest.fixture
    def device(self):
        """Create a lightning device fixture."""
        return qp.device("lightning.qubit", wires=4)

    @pytest.mark.parametrize("add_compbasis_meas", ["false", "wires", "qreg"])
    @pytest.mark.parametrize("obs", NON_COMMUTE_SINGLE_OBS_LIST)
    @pytest.mark.parametrize("measurements", [qp.expval, qp.var])
    def test_non_commuting_single_measurement(self, add_compbasis_meas, device, obs, measurements):
        """An CompileError is raised for single measurement non-commuting Hamiltonians."""

        # pylint: disable=inconsistent-return-statements
        @qp.qjit()
        @diagonalize_final_measurements_pass
        @qp.set_shots(10)
        @qp.qnode(device)
        def circuit(x):
            qp.RX(x, 0)
            if add_compbasis_meas == "false":
                return measurements(obs)
            if add_compbasis_meas == "wires":
                return qp.sample(wires=obs.wires), measurements(obs)
            if add_compbasis_meas == "qreg":
                return qp.sample(), measurements(obs)

        with pytest.raises(CompileError, match=_non_commuting_err_msg):
            circuit(0.7)

    def test_non_commuting_multiple_measurements_only(self, device):
        """A CompileError is raised for non-commuting measurements without gates
        applied."""

        @diagonalize_final_measurements_pass
        @qp.set_shots(10)
        @qp.qnode(device)
        def circuit():
            return qp.expval(qp.X(0)), qp.var(qp.Z(0))

        with pytest.raises(CompileError, match=_non_commuting_err_msg):
            qp.qjit(circuit)

    @pytest.mark.parametrize("obs", NON_COMMUTE_MULTI_OBS_LIST)
    @pytest.mark.parametrize("m", [(qp.expval, qp.var)])
    def test_non_commuting_multiple_measurements(self, device, obs, m):
        """An CompileError is raised for multiple non-commuting measurements."""

        @qp.qjit()
        @diagonalize_final_measurements_pass
        @qp.set_shots(10)
        @qp.qnode(device)
        def circuit(x):
            qp.RX(x, 0)
            return m[0](obs[0]), m[1](obs[1])

        with pytest.raises(CompileError, match=_non_commuting_err_msg):
            circuit(0.7)

    @pytest.mark.parametrize("obs", COMMUTE_SINGLE_OBS_LIST)
    @pytest.mark.parametrize("measurements", [qp.expval, qp.var])
    def test_commuting_single_measurement(self, device, obs, measurements):
        """No error is raised for single measurement commuting Hamiltonians."""

        @qp.qjit()
        @diagonalize_final_measurements_pass
        @qp.set_shots(10)
        @qp.qnode(device)
        def circuit(x):
            qp.RX(x, 0)
            return measurements(obs)

        circuit(0.7)

    @pytest.mark.parametrize("obs", COMMUTE_MULTI_OBS_LIST)
    @pytest.mark.parametrize("m", [(qp.expval, qp.var)])
    def test_commuting_multiple_measurements(self, device, obs, m):
        """No error is raised for multiple commuting measurements."""

        @qp.qjit()
        @diagonalize_final_measurements_pass
        @qp.set_shots(10)
        @qp.qnode(device)
        def circuit(x):
            qp.RX(x, 0)
            return m[0](obs[0]), m[1](obs[1])

        circuit(0.7)
