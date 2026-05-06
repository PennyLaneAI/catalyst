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

"""Unit and integration tests for the unified compiler `measurements_from_samples` transform."""

# pylint: disable=line-too-long

import pytest
from pennylane.exceptions import CompileError

from catalyst.python_interface.transforms import (
    MeasurementsFromSamplesPass,
)

pytestmark = pytest.mark.xdsl


class TestMeasurementsFromSamplesPass:
    """Unit tests for the measurements-from-samples pass."""

    def test_no_shots_raises_error(self, run_filecheck):
        """Test that when no shots are provided, the pass raises an error"""
        program = """
        builtin.module @module_circuit {
            // CHECK-LABEL: circuit
            func.func public @circuit() -> (tensor<f64>) attributes {quantum.node} {
                %0 = arith.constant 0 : i64
                quantum.device shots(%0) ["", "", ""]
            }
        }
        """

        pipeline = (MeasurementsFromSamplesPass(),)
        with pytest.raises(
            ValueError, match="measurements_from_samples pass requires non-zero shots"
        ):
            run_filecheck(program, pipeline)

    def test_1_wire_expval(self, run_filecheck):
        """Test the measurements-from-samples pass on a 1-wire circuit terminating with an expval(Z)
        measurement.
        """
        program = """
        builtin.module @module_circuit {
            // CHECK-LABEL: circuit
            // CHECK: [[raw_samples:%.+]] = func.call @circuit.from_samples() : () -> tensor<5x1xf64>
            // CHECK: [[res:%.+]] = func.call @expval_from_samples.tensor.5x1xf64([[raw_samples]]) :
            // CHECK-SAME: (tensor<5x1xf64>) -> tensor<f64>
            // CHECK: func.return [[res]] : tensor<f64>
            // CHECK: func.func public @circuit.from_samples{{.*}} attributes {quantum.node}
            func.func public @circuit() -> (tensor<f64>) attributes {quantum.node}  {
                %0 = "stablehlo.constant"() <{value = dense<5> : tensor<i64>}> : () -> tensor<i64>
                %1 = tensor.extract %0[] : tensor<i64>
                quantum.device shots(%1) ["", "", ""]

                // CHECK: [[q0:%.+]] = "test.op"() : () -> !quantum.bit
                %2 = "test.op"() : () -> !quantum.bit

                // CHECK-NOT: quantum.namedobs
                %3 = quantum.namedobs %2[PauliZ] : !quantum.obs

                // CHECK: [[obs:%.+]] = quantum.compbasis qubits [[q0]] : !quantum.obs
                // CHECK: [[samples:%.+]] = quantum.sample [[obs]] : tensor<5x1xf64>
                // CHECK-NOT: quantum.expval
                %4 = quantum.expval %3 : f64
                %5 = "tensor.from_elements"(%4) : (f64) -> tensor<f64>

                // CHECK: func.return [[samples]] : tensor<5x1xf64>
                func.return %5 : tensor<f64>
            }
            // CHECK-LABEL: func.func public @expval_from_samples.tensor.5x1xf64
        }
        """

        pipeline = (MeasurementsFromSamplesPass(),)
        run_filecheck(program, pipeline)

    def test_1_wire_var(self, run_filecheck):
        """Test the measurements-from-samples pass on a 1-wire circuit terminating with a var(Z)
        measurement.
        """
        program = """
        builtin.module @module_circuit {
            // CHECK-LABEL: circuit
            // CHECK: [[samples:%.+]] = func.call @circuit.from_samples() : () -> tensor<3x1xf64>
            // CHECK: [[res:%.+]] = func.call @var_from_samples.tensor.3x1xf64([[samples]]) :
            // CHECK-SAME: (tensor<3x1xf64>) -> tensor<f64>
            // CHECK: func.return [[res]] : tensor<f64>
            // CHECK: func.func public @circuit.from_samples{{.*}} attributes {quantum.node}
            func.func public @circuit() -> (tensor<f64>) attributes {quantum.node}  {
                %0 = "stablehlo.constant"() <{value = dense<3> : tensor<i64>}> : () -> tensor<i64>
                %1 = tensor.extract %0[] : tensor<i64>
                quantum.device shots(%1) ["", "", ""]

                // CHECK: [[q0:%.+]] = "test.op"() : () -> !quantum.bit
                %2 = "test.op"() : () -> !quantum.bit

                // CHECK-NOT: quantum.namedobs
                %3 = quantum.namedobs %2[PauliZ] : !quantum.obs

                // CHECK: [[obs:%.+]] = quantum.compbasis qubits [[q0]] : !quantum.obs
                // CHECK: [[samples:%.+]] = quantum.sample [[obs]] : tensor<3x1xf64>
                // CHECK-NOT: quantum.var
                %4 = quantum.var %3 : f64
                %5 = "tensor.from_elements"(%4) : (f64) -> tensor<f64>

                // CHECK: func.return [[samples]] : tensor<3x1xf64>
                func.return %5 : tensor<f64>
            }
            // CHECK-LABEL: func.func public @var_from_samples.tensor.3x1xf64
        }
        """

        pipeline = (MeasurementsFromSamplesPass(),)
        run_filecheck(program, pipeline)

    def test_1_wire_probs(self, run_filecheck):
        """Test the measurements-from-samples pass on a 1-wire circuit terminating with a probs
        measurement.
        """
        program = """
        builtin.module @module_circuit {
            // CHECK-LABEL: circuit
            // CHECK: [[samples:%.+]] = func.call @circuit.from_samples() : () -> tensor<3x1xf64>
            // CHECK: [[res:%.+]] = func.call @probs_from_samples.tensor.3x1xf64([[samples]]) :
            // CHECK-SAME: (tensor<3x1xf64>) -> tensor<2xf64>
            // CHECK: func.return [[res]] : tensor<2xf64>
            // CHECK: func.func public @circuit.from_samples{{.*}} attributes {quantum.node}
            func.func public @circuit() -> (tensor<f64>) attributes {quantum.node}  {
                %0 = "stablehlo.constant"() <{value = dense<3> : tensor<i64>}> : () -> tensor<i64>
                %1 = tensor.extract %0[] : tensor<i64>
                quantum.device shots(%1) ["", "", ""]

                // CHECK: [[q0:%.+]] = "test.op"() <{nqubits_attr = 1 : i64}> : () -> !quantum.reg
                %2 = "test.op"() <{nqubits_attr = 1 : i64}> : () -> !quantum.reg

                // CHECK: [[compbasis:%.+]] = quantum.compbasis qreg [[q0]] : !quantum.obs
                %3 = quantum.compbasis qreg %2 : !quantum.obs

                // CHECK: [[samples:%.+]] = quantum.sample [[compbasis]] : tensor<3x1xf64>
                // CHECK-NOT: quantum.probs
                %4 = quantum.probs %3 : tensor<2xf64>

                // CHECK: func.return [[samples]] : tensor<3x1xf64>
                func.return %4 : tensor<2xf64>
            }
            // CHECK-LABEL: func.func public @probs_from_samples.tensor.3x1xf64
        }
        """

        pipeline = (MeasurementsFromSamplesPass(),)
        run_filecheck(program, pipeline)

    def test_1_wire_sample(self, run_filecheck):
        """Test the measurements-from-samples pass on a 1-wire circuit terminating with a sample
        measurement.

        This pass should be a no-op.
        """
        program = """
        builtin.module @module_circuit {
            // CHECK-LABEL: circuit
            // CHECK: [[samples:%.+]] = func.call @circuit.from_samples() : () -> tensor<7x1xf64>
            // CHECK: func.return [[samples]] : tensor<7x1xf64>
            // CHECK: func.func public @circuit.from_samples{{.*}} attributes {quantum.node}
            func.func public @circuit() -> (tensor<7x1xf64>) attributes {quantum.node}  {
                %0 = "stablehlo.constant"() <{value = dense<7> : tensor<i64>}> : () -> tensor<i64>
                %1 = tensor.extract %0[] : tensor<i64>
                quantum.device shots(%1) ["", "", ""]

                // CHECK: [[q0:%.+]] = "test.op"() : () -> !quantum.reg
                %2 = "test.op"() : () -> !quantum.reg

                // CHECK: [[compbasis:%.+]] = quantum.compbasis qreg [[q0]] : !quantum.obs
                %3 = quantum.compbasis qreg %2 : !quantum.obs

                // CHECK: [[samples:%.+]] = quantum.sample [[compbasis]] : tensor<7x1xf64>
                %4 = quantum.sample %3 : tensor<7x1xf64>

                // CHECK: func.return [[samples]] : tensor<7x1xf64>
                func.return %4 : tensor<7x1xf64>
            }
        }
        """

        pipeline = (MeasurementsFromSamplesPass(),)
        run_filecheck(program, pipeline)

    @pytest.mark.xfail(reason="Counts not supported", strict=True, raises=NotImplementedError)
    def test_1_wire_counts(self, run_filecheck):
        """Test the measurements-from-samples pass on a 1-wire circuit terminating with a counts
        measurement.
        """
        program = """
        builtin.module @module_circuit {
            // CHECK-LABEL: circuit
            // CHECK: func.func public @circuit.from_samples{{.*}} attributes {quantum.node}
            func.func public @circuit() -> (tensor<f64>) attributes {quantum.node}  {
                %0 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
                %1 = tensor.extract %0[] : tensor<i64>
                quantum.device shots(%1) ["", "", ""]

                // CHECK: [[q0:%.+]] = "test.op"() : () -> !quantum.reg
                %2 = "test.op"() : () -> !quantum.reg

                // CHECK: [[compbasis:%.+]] = quantum.compbasis qreg [[q0]] : !quantum.obs
                %3 = quantum.compbasis qreg %2 : !quantum.obs

                // CHECK: [[samples:%.+]] = quantum.sample %3 : tensor<1x1xf64>
                // CHECK: [[eigvals:%.+]], [[counts:%.+]] = func.call @counts_from_samples.tensor.1x1xf64([[samples]]) :
                // CHECK-SAME: (tensor<1x1xf64>) -> tensor<2xf64>, tensor<2xi64>
                // CHECK-NOT: quantum.counts
                %eigvals, %counts = quantum.counts %3 : tensor<2xf64>, tensor<2xi64>

                // CHECK: [[eigvals_converted:%.+]] = {{.*}}stablehlo.convert{{.+}}[[eigvals]] :
                // CHECK-SAME: (tensor<2xf64>) -> tensor<2xi64>
                %4 = "stablehlo.convert"(%eigvals) : (tensor<2xf64>) -> tensor<2xi64>

                // CHECK: func.return [[eigvals_converted]], [[counts]] : tensor<1x1xf64>
                func.return %4, %counts : tensor<2xi64>, tensor<2xi64>
            }
            // CHECK-LABEL: func.func public @counts_from_samples.tensor.1x1xf64
        }
        """

        pipeline = (MeasurementsFromSamplesPass(),)
        run_filecheck(program, pipeline)

    def test_1_wire_state(self, run_filecheck):
        """Test the measurements-from-samples pass on a 1-wire circuit terminating with a state
        measurement. Note that this is not a valid IR, because state and shots don't work together.
        We would expect to either encounter a "no shots" error from this pass (for a circuit with
        no shots) or a "state and shots are incompatible" error with shots when creating an IR.
        """

        program = """
        builtin.module @module_circuit {
            // CHECK-LABEL: circuit
            // CHECK: func.func public @circuit.from_samples{{.*}} attributes {quantum.node}
            func.func public @circuit() -> (tensor<4xcomplex<f64>>) attributes {quantum.node}  {
                %0 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
                %1 = tensor.extract %0[] : tensor<i64>
                quantum.device shots(%1) ["", "", ""]

                // CHECK: [[q0:%.+]] = "test.op"() : () -> !quantum.reg
                %2 = "test.op"() : () -> !quantum.reg

                // CHECK: [[compbasis:%.+]] = quantum.compbasis qreg [[q0]] : !quantum.obs
                %3 = quantum.compbasis qreg %2 : !quantum.obs
                %4 = quantum.state %3 : tensor<4xcomplex<f64>>

                func.return %4 : tensor<4xcomplex<f64>>
            }
        }
        """

        pipeline = (MeasurementsFromSamplesPass(),)
        with pytest.raises(CompileError, match="operations are not compatible with"):
            run_filecheck(program, pipeline)

    def test_2_wire_expval(self, run_filecheck):
        """Test the measurements-from-samples pass on a 2-wire circuit terminating with an expval(Z)
        measurement on each wire. Includes a non-diagonalized operation (pass inserts diagonalizing gate).
        """

        program = """
        builtin.module @module_circuit {
            // CHECK-LABEL: circuit
            // CHECK: [[samples0:%.+]], [[samples1:%.+]] = func.call @circuit.from_samples() : () -> (tensor<5x1xf64>, tensor<5x1xf64>)
            // CHECK: [[res0:%.+]] = func.call @expval_from_samples.tensor.5x1xf64([[samples0]]) :
            // CHECK-SAME: (tensor<5x1xf64>) -> tensor<f64>
            // CHECK: [[res1:%.+]] = func.call @expval_from_samples.tensor.5x1xf64([[samples1]]) :
            // CHECK-SAME: (tensor<5x1xf64>) -> tensor<f64>
            // CHECK: func.return [[res0]], [[res1]] : tensor<f64>, tensor<f64>
            // CHECK: func.func public @circuit.from_samples{{.*}} attributes {quantum.node}
            func.func public @circuit() -> (tensor<f64>, tensor<f64>) attributes {quantum.node}  {
                %0 = "stablehlo.constant"() <{value = dense<5> : tensor<i64>}> : () -> tensor<i64>
                %1 = tensor.extract %0[] : tensor<i64>
                quantum.device shots(%1) ["", "", ""]

                // CHECK: [[q0:%.+]] = "test.op"() : () -> !quantum.bit
                // CHECK: [[q1:%.+]] = "test.op"() : () -> !quantum.bit
                %2 = "test.op"() : () -> !quantum.bit
                %3 = "test.op"() : () -> !quantum.bit

                // CHECK-NOT: quantum.namedobs
                // CHECK: [[q0_1:%.+]] = quantum.custom "Hadamard"() [[q0]]
                %4 = quantum.namedobs %2[PauliX] : !quantum.obs
                %5 = quantum.namedobs %3[PauliZ] : !quantum.obs

                // CHECK: [[obs0:%.+]] = quantum.compbasis qubits [[q0_1]] : !quantum.obs
                // CHECK: [[samples0:%.+]] = quantum.sample [[obs0]] : tensor<5x1xf64>
                // CHECK: [[obs1:%.+]] = quantum.compbasis qubits [[q1]] : !quantum.obs
                // CHECK: [[samples1:%.+]] = quantum.sample [[obs1]] : tensor<5x1xf64>
                // CHECK-NOT: quantum.expval
                %6 = quantum.expval %4 : f64
                %7 = "tensor.from_elements"(%6) : (f64) -> tensor<f64>
                %8 = quantum.expval %5 : f64
                %9 = "tensor.from_elements"(%8) : (f64) -> tensor<f64>

                // CHECK: func.return [[samples0]], [[samples1]] : tensor<5x1xf64>, tensor<5x1xf64>
                func.return %7, %9 : tensor<f64>, tensor<f64>
            }
            // CHECK-LABEL: func.func public @expval_from_samples.tensor.5x1xf64
        }
        """

        pipeline = (MeasurementsFromSamplesPass(),)
        run_filecheck(program, pipeline)

    def test_2_wire_var(self, run_filecheck):
        """Test the measurements-from-samples pass on a 2-wire circuit terminating with a var(Z)
        measurement on each wire.
        """
        program = """
        builtin.module @module_circuit {
            // CHECK-LABEL: circuit
            // CHECK: [[samples0:%.+]], [[samples1:%.+]] = func.call @circuit.from_samples() : () -> (tensor<5x1xf64>, tensor<5x1xf64>)
            // CHECK: [[res0:%.+]] = func.call @var_from_samples.tensor.5x1xf64([[samples0]]) :
            // CHECK-SAME: (tensor<5x1xf64>) -> tensor<f64>
            // CHECK: [[res1:%.+]] = func.call @var_from_samples.tensor.5x1xf64([[samples1]]) :
            // CHECK-SAME: (tensor<5x1xf64>) -> tensor<f64>
            // CHECK: func.return [[res0]], [[res1]] : tensor<f64>, tensor<f64>
            // CHECK: func.func public @circuit.from_samples{{.*}} attributes {quantum.node}
            func.func public @circuit() -> (tensor<f64>, tensor<f64>) attributes {quantum.node}  {
                %0 = "stablehlo.constant"() <{value = dense<5> : tensor<i64>}> : () -> tensor<i64>
                %1 = tensor.extract %0[] : tensor<i64>
                quantum.device shots(%1) ["", "", ""]

                // CHECK: [[q0:%.+]] = "test.op"() : () -> !quantum.bit
                %2 = "test.op"() : () -> !quantum.bit

                // CHECK: [[q1:%.+]] = "test.op"() : () -> !quantum.bit
                %3 = "test.op"() : () -> !quantum.bit

                // CHECK-NOT: quantum.namedobs
                // CHECK: [[q0_1:%.+]] = quantum.custom "Hadamard"() [[q0]]
                %4 = quantum.namedobs %2[PauliX] : !quantum.obs
                %5 = quantum.namedobs %3[PauliZ] : !quantum.obs

                // CHECK: [[obs0:%.+]] = quantum.compbasis qubits [[q0_1]] : !quantum.obs
                // CHECK: [[samples0:%.+]] = quantum.sample [[obs0]] : tensor<5x1xf64>
                // CHECK: [[obs1:%.+]] = quantum.compbasis qubits [[q1]] : !quantum.obs
                // CHECK: [[samples1:%.+]] = quantum.sample [[obs1]] : tensor<5x1xf64>
                // CHECK-NOT: quantum.var
                %6 = quantum.var %4 : f64
                %7 = "tensor.from_elements"(%6) : (f64) -> tensor<f64>
                %8 = quantum.var %5 : f64
                %9 = "tensor.from_elements"(%8) : (f64) -> tensor<f64>

                // CHECK: func.return [[samples0]], [[samples1]] : tensor<5x1xf64>, tensor<5x1xf64>
                func.return %7, %9 : tensor<f64>, tensor<f64>
            }
            // CHECK-LABEL: func.func public @var_from_samples.tensor.5x1xf64
        }
        """

        pipeline = (MeasurementsFromSamplesPass(),)
        run_filecheck(program, pipeline)

    def test_2_wire_probs_global(self, run_filecheck):
        """Test the measurements-from-samples pass on a 2-wire circuit terminating with a "global"
        probs measurement (one that implicitly acts on all wires).
        """
        program = """
        builtin.module @module_circuit {
            // CHECK-LABEL: circuit
            // CHECK: [[samples:%.+]] = func.call @circuit.from_samples() : () -> tensor<3x2xf64>
            // CHECK: [[res:%.+]] = func.call @probs_from_samples.tensor.3x2xf64([[samples]]) :
            // CHECK-SAME: (tensor<3x2xf64>) -> tensor<4xf64>
            // CHECK: func.return [[res]] : tensor<4xf64>
            // CHECK: func.func public @circuit.from_samples{{.*}} attributes {quantum.node}
            func.func public @circuit() -> (tensor<f64>) attributes {quantum.node}  {
                %0 = "stablehlo.constant"() <{value = dense<3> : tensor<i64>}> : () -> tensor<i64>
                %1 = tensor.extract %0[] : tensor<i64>
                quantum.device shots(%1) ["", "", ""]

                // CHECK: [[qreg:%.+]] = quantum.alloc
                %2 = quantum.alloc(2) : !quantum.reg

                // CHECK: [[compbasis:%.+]] = quantum.compbasis qreg [[qreg]] : !quantum.obs
                %3 = quantum.compbasis qreg %2 : !quantum.obs

                // CHECK: [[samples:%.+]] = quantum.sample [[compbasis]] : tensor<3x2xf64>
                // CHECK-NOT: quantum.probs
                %4 = quantum.probs %3 : tensor<4xf64>

                // CHECK: func.return [[samples]] : tensor<3x2xf64>
                func.return %4 : tensor<4xf64>
            }
            // CHECK-LABEL: func.func public @probs_from_samples.tensor.3x2xf64
        }
        """

        pipeline = (MeasurementsFromSamplesPass(),)
        run_filecheck(program, pipeline)

    def test_2_wire_probs_per_wire(self, run_filecheck):
        """Test the measurements-from-samples pass on a 2-wire circuit terminating with separate
        probs measurements per wire.
        """
        program = """
        builtin.module @module_circuit {
            // CHECK-LABEL: circuit
            // CHECK: [[samples0:%.+]], [[samples1:%.+]] = func.call @circuit.from_samples() : () -> (tensor<3x1xf64>, tensor<3x1xf64>)
            // CHECK: [[res0:%.+]] = func.call @probs_from_samples.tensor.3x1xf64([[samples0]]) :
            // CHECK-SAME: (tensor<3x1xf64>) -> tensor<2xf64>
            // CHECK: [[res1:%.+]] = func.call @probs_from_samples.tensor.3x1xf64([[samples1]]) :
            // CHECK-SAME: (tensor<3x1xf64>) -> tensor<2xf64>
            // CHECK: func.return [[res0]], [[res1]] : tensor<2xf64>, tensor<2xf64>
            // CHECK: func.func public @circuit.from_samples{{.*}} attributes {quantum.node}
            func.func public @circuit() -> (tensor<f64>, tensor<f64>) attributes {quantum.node}  {
                %0 = "stablehlo.constant"() <{value = dense<3> : tensor<i64>}> : () -> tensor<i64>
                %1 = tensor.extract %0[] : tensor<i64>
                quantum.device shots(%1) ["", "", ""]

                // CHECK: [[qreg:%.+]] = quantum.alloc
                %2 = quantum.alloc(2) : !quantum.reg

                // CHECK: [[q0:%.+]] = quantum.extract [[qreg]][0]
                %3 = quantum.extract %2[0] : !quantum.reg -> !quantum.bit

                // CHECK: [[compbasis0:%.+]] = quantum.compbasis qubits [[q0]] : !quantum.obs
                %4 = quantum.compbasis qubits %3 : !quantum.obs

                // CHECK: [[samples0:%.+]] = quantum.sample [[compbasis0]] : tensor<3x1xf64>
                // CHECK-NOT: quantum.probs
                %5 = quantum.probs %4 : tensor<2xf64>

                // CHECK: [[q1:%.+]] = quantum.extract [[qreg]][1]
                %6 = quantum.extract %2[1] : !quantum.reg -> !quantum.bit

                // CHECK: [[compbasis1:%.+]] = quantum.compbasis qubits [[q1]] : !quantum.obs
                %7 = quantum.compbasis qubits %6 : !quantum.obs

                // CHECK: [[samples1:%.+]] = quantum.sample [[compbasis1]] : tensor<3x1xf64>
                // CHECK-NOT: quantum.probs
                %8 = quantum.probs %7 : tensor<2xf64>

                // CHECK: func.return [[samples0]], [[samples1]] : tensor<3x1xf64>, tensor<3x1xf64>
                func.return %5, %8 : tensor<2xf64>, tensor<2xf64>
            }
            // CHECK-LABEL: func.func public @probs_from_samples.tensor.3x1xf64
        }
        """

        pipeline = (MeasurementsFromSamplesPass(),)
        run_filecheck(program, pipeline)

    def test_expval_from_sample_diagonalizes(self, run_filecheck):
        """Test that diagonalization is performed as expected when the circuit contains non-Z observables"""

        program = """
        builtin.module @module_circuit {
            // CHECK-LABEL: circuit
            // CHECK: [[samples:%.+]] = func.call @circuit.from_samples() : () -> tensor<1x1xf64>
            // CHECK: [[res:%.+]] = func.call @expval_from_samples.tensor.1x1xf64([[samples]]) :
            // CHECK-SAME: (tensor<1x1xf64>) -> tensor<f64>
            // CHECK: func.return [[res]] : tensor<f64>
            // CHECK: func.func public @circuit.from_samples{{.*}} attributes {quantum.node}            
            func.func public @circuit() -> (tensor<f64>) attributes {quantum.node} {
                %0 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
                %1 = tensor.extract %0[] : tensor<i64>
                quantum.device shots(%1) ["", "", ""]

                // CHECK: [[qreg:%.+]] = quantum.alloc
                %2 = quantum.alloc(1) : !quantum.reg

                // CHECK: [[q0:%.+]] = quantum.extract [[qreg]][0]
                %3 = quantum.extract %2[0] : !quantum.reg -> !quantum.bit

                // CHECK: [[q1:%.+]] = quantum.custom "Hadamard"() [[q0]]
                // CHECK: [[obs:%.+]] = quantum.compbasis qubits [[q1]]
                %4 = quantum.namedobs %3[PauliX] : !quantum.obs

                // CHECK: [[samples:%.+]] = quantum.sample [[obs]] : tensor<1x1xf64>
                // CHECK-NOT: quantum.expval
                %5 = quantum.expval %4 : f64
                %6 = tensor.from_elements %5 : tensor<f64>

                // CHECK: return [[samples]] : tensor<1x1xf64>
                func.return %6 : tensor<f64>
            }
            // CHECK-LABEL: func.func public @expval_from_samples.tensor.1x1xf64
        }
        """

        pipeline = (MeasurementsFromSamplesPass(),)
        run_filecheck(program, pipeline)

    def test_expval_tensor_obs(self, run_filecheck):
        """Test the measurements-from-samples pass on a circuit terminating with the variance
        of a 3-wire tensor, including non-Z observables.
        """

        program = """
        builtin.module @module_circuit {
            // CHECK-LABEL: circuit
            // CHECK: [[samples:%.+]] = func.call @circuit.from_samples() : () -> tensor<5x3xf64>
            // CHECK: [[res:%.+]] = func.call @expval_from_samples.tensor.5x3xf64([[samples]]) :
            // CHECK-SAME: (tensor<5x3xf64>) -> tensor<f64>
            // CHECK: func.return [[res]] : tensor<f64>
            func.func public @circuit() -> tensor<f64> attributes {quantum.node}  {
                %0 = "stablehlo.constant"() <{value = dense<5> : tensor<i64>}> : () -> tensor<i64>
                %1 = tensor.extract %0[] : tensor<i64>
                quantum.device shots(%1) ["", "", ""]

                // CHECK: [[q0:%.+]] = "test.op"() : () -> !quantum.bit
                // CHECK: [[q1:%.+]] = "test.op"() : () -> !quantum.bit
                // CHECK: [[q2:%.+]] = "test.op"() : () -> !quantum.bit
                %2 = "test.op"() : () -> !quantum.bit
                %3 = "test.op"() : () -> !quantum.bit
                %4 = "test.op"() : () -> !quantum.bit

                // CHECK-NOT: quantum.namedobs
                // CHECK-NOT: quantum.tensor
                // CHECK: [[q0_1:%.+]] = quantum.custom "Hadamard"() [[q0]]
                // CHECK: [[q1_1:%.+]] = quantum.custom "PauliZ"() [[q1]]
                // CHECK: [[q1_2:%.+]] = quantum.custom "S"() [[q1_1]]
                // CHECK: [[q1_3:%.+]] = quantum.custom "Hadamard"() [[q1_2]]
                // CHECK: [[obs0:%.+]] = quantum.compbasis qubits [[q0_1]], [[q1_3]], [[q2]] : !quantum.obs
                %5 = quantum.namedobs %2[PauliX] : !quantum.obs
                %6 = quantum.namedobs %3[PauliY] : !quantum.obs
                %7 = quantum.namedobs %4[PauliZ] : !quantum.obs
                %8 = quantum.tensor %5, %6, %7 : !quantum.obs

                // CHECK: [[samples:%.+]] = quantum.sample [[obs0]] : tensor<5x3xf64>
                // CHECK-NOT: quantum.expval
                %9 = quantum.expval %8 : f64
                %10 = "tensor.from_elements"(%9) : (f64) -> tensor<f64>

                // CHECK: func.return [[samples]] : tensor<5x3xf64>
                func.return %10 : tensor<f64>
            }
            // CHECK-LABEL: func.func public @expval_from_samples.tensor.5x3xf64
        }
        """

        pipeline = (MeasurementsFromSamplesPass(),)
        run_filecheck(program, pipeline)

    def test_var_tensor_obs(self, run_filecheck):
        """Test the measurements-from-samples pass on a circuit terminating with the variance
        of a 3-wire tensor, including non-Z observables.
        """

        program = """
        builtin.module @module_circuit {
            // CHECK-LABEL: circuit
            // CHECK: [[samples:%.+]] = func.call @circuit.from_samples() : () -> tensor<5x3xf64>
            // CHECK: [[res:%.+]] = func.call @var_from_samples.tensor.5x3xf64([[samples]]) :
            // CHECK-SAME: (tensor<5x3xf64>) -> tensor<f64>
            // CHECK: func.return [[res]] : tensor<f64>
            func.func public @circuit() -> tensor<f64> attributes {quantum.node}  {
                %0 = "stablehlo.constant"() <{value = dense<5> : tensor<i64>}> : () -> tensor<i64>
                %1 = tensor.extract %0[] : tensor<i64>
                quantum.device shots(%1) ["", "", ""]

                // CHECK: [[q0:%.+]] = "test.op"() : () -> !quantum.bit
                // CHECK: [[q1:%.+]] = "test.op"() : () -> !quantum.bit
                // CHECK: [[q2:%.+]] = "test.op"() : () -> !quantum.bit
                %2 = "test.op"() : () -> !quantum.bit
                %3 = "test.op"() : () -> !quantum.bit
                %4 = "test.op"() : () -> !quantum.bit

                // CHECK-NOT: quantum.namedobs
                // CHECK-NOT: quantum.tensor
                // CHECK: [[q0_1:%.+]] = quantum.custom "Hadamard"() [[q0]]
                // CHECK: [[q1_1:%.+]] = quantum.custom "PauliZ"() [[q1]]
                // CHECK: [[q1_2:%.+]] = quantum.custom "S"() [[q1_1]]
                // CHECK: [[q1_3:%.+]] = quantum.custom "Hadamard"() [[q1_2]]
                // CHECK: [[obs0:%.+]] = quantum.compbasis qubits [[q0_1]], [[q1_3]], [[q2]] : !quantum.obs
                %5 = quantum.namedobs %2[PauliX] : !quantum.obs
                %6 = quantum.namedobs %3[PauliY] : !quantum.obs
                %7 = quantum.namedobs %4[PauliZ] : !quantum.obs
                %8 = quantum.tensor %5, %6, %7 : !quantum.obs

                // CHECK: [[samples:%.+]] = quantum.sample [[obs0]] : tensor<5x3xf64>
                // CHECK-NOT: quantum.var
                %9 = quantum.var %8 : f64
                %10 = "tensor.from_elements"(%9) : (f64) -> tensor<f64>

                // CHECK: func.return [[samples]] : tensor<5x3xf64>
                func.return %10 : tensor<f64>
            }
            // CHECK-LABEL: func.func public @var_from_samples.tensor.5x3xf64
        }
        """

        pipeline = (MeasurementsFromSamplesPass(),)
        run_filecheck(program, pipeline)


if __name__ == "__main__":
    pytest.main(["-x", __file__])
