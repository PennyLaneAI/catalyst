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

from functools import partial

import numpy as np
import pennylane as qml
import pytest

from catalyst.python_interface.transforms import (
    MeasurementsFromSamplesPass,
    measurements_from_samples_pass,
)

pytestmark = pytest.mark.xdsl


class TestMeasurementsFromSamplesPass:
    """Unit tests for the measurements-from-samples pass."""

    def test_no_shots_raises_error(self, run_filecheck):
        """Test that when no shots are provided, the pass raises an error"""
        program = """
        builtin.module @module_circuit {
            // CHECK-LABEL: circuit
            func.func public @circuit() -> (tensor<f64>) {
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
            func.func public @circuit() -> (tensor<f64>) {
                %0 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
                %1 = tensor.extract %0[] : tensor<i64>
                quantum.device shots(%1) ["", "", ""]

                // CHECK: [[q0:%.+]] = "test.op"() : () -> !quantum.bit
                %2 = "test.op"() : () -> !quantum.bit

                // CHECK-NOT: quantum.namedobs
                %3 = quantum.namedobs %2[PauliZ] : !quantum.obs

                // CHECK: [[obs:%.+]] = quantum.compbasis qubits [[q0]] : !quantum.obs
                // CHECK: [[samples:%.+]] = quantum.sample [[obs]] : tensor<1x1xf64>
                // CHECK: [[c0:%.+]] = arith.constant dense<0> : tensor<i64>
                // CHECK: [[res:%.+]] = func.call @expval_from_samples.tensor.1x1xf64([[samples]], [[c0]]) :
                // CHECK-SAME: (tensor<1x1xf64>, tensor<i64>) -> tensor<f64>
                // CHECK-NOT: quantum.expval
                %4 = quantum.expval %3 : f64
                %5 = "tensor.from_elements"(%4) : (f64) -> tensor<f64>

                // CHECK: func.return [[res]] : tensor<f64>
                func.return %5 : tensor<f64>
            }
            // CHECK-LABEL: func.func public @expval_from_samples.tensor.1x1xf64
        }
        """

        pipeline = (MeasurementsFromSamplesPass(),)
        run_filecheck(program, pipeline)

    def test_1_wire_expval_shots_from_arith_constantop(self, run_filecheck):
        """Test the measurements-from-samples pass on a 1-wire circuit with shots from an arith.constant op and an expval(Z) measurement."""
        program = """
        builtin.module @module_circuit {
            // CHECK-LABEL: circuit
            func.func public @circuit() -> (tensor<f64>) {
                %0 = arith.constant 1 : i64
                quantum.device shots(%0) ["", "", ""]

                // CHECK: [[q0:%.+]] = "test.op"() : () -> !quantum.bit
                %1 = "test.op"() : () -> !quantum.bit

                // CHECK-NOT: quantum.namedobs
                %2 = quantum.namedobs %1[PauliZ] : !quantum.obs

                // CHECK: [[obs:%.+]] = quantum.compbasis qubits [[q0]] : !quantum.obs
                // CHECK: [[samples:%.+]] = quantum.sample [[obs]] : tensor<1x1xf64>
                // CHECK: [[c0:%.+]] = arith.constant dense<0> : tensor<i64>
                // CHECK: [[res:%.+]] = func.call @expval_from_samples.tensor.1x1xf64([[samples]], [[c0]]) :
                // CHECK-SAME: (tensor<1x1xf64>, tensor<i64>) -> tensor<f64>
                // CHECK-NOT: quantum.expval
                %3 = quantum.expval %2 : f64
                %4 = "tensor.from_elements"(%3) : (f64) -> tensor<f64>

                // CHECK: func.return [[res]] : tensor<f64>
                func.return %4 : tensor<f64>
            }
            // CHECK-LABEL: func.func public @expval_from_samples.tensor.1x1xf64
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
            func.func public @circuit() -> (tensor<f64>) {
                %0 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
                %1 = tensor.extract %0[] : tensor<i64>
                quantum.device shots(%1) ["", "", ""]

                // CHECK: [[q0:%.+]] = "test.op"() : () -> !quantum.bit
                %2 = "test.op"() : () -> !quantum.bit

                // CHECK-NOT: quantum.namedobs
                %3 = quantum.namedobs %2[PauliZ] : !quantum.obs

                // CHECK: [[obs:%.+]] = quantum.compbasis qubits [[q0]] : !quantum.obs
                // CHECK: [[samples:%.+]] = quantum.sample [[obs]] : tensor<1x1xf64>
                // CHECK: [[c0:%.+]] = arith.constant dense<0> : tensor<i64>
                // CHECK: [[res:%.+]] = func.call @var_from_samples.tensor.1x1xf64([[samples]], [[c0]]) :
                // CHECK-SAME: (tensor<1x1xf64>, tensor<i64>) -> tensor<f64>
                // CHECK-NOT: quantum.var
                %4 = quantum.var %3 : f64
                %5 = "tensor.from_elements"(%4) : (f64) -> tensor<f64>

                // CHECK: func.return [[res]] : tensor<f64>
                func.return %5 : tensor<f64>
            }
            // CHECK-LABEL: func.func public @var_from_samples.tensor.1x1xf64
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
            func.func public @circuit() -> (tensor<f64>) {
                %0 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
                %1 = tensor.extract %0[] : tensor<i64>
                quantum.device shots(%1) ["", "", ""]

                // CHECK: [[q0:%.+]] = "test.op"() <{nqubits_attr = 1 : i64}> : () -> !quantum.reg
                %2 = "test.op"() <{nqubits_attr = 1 : i64}> : () -> !quantum.reg

                // CHECK: [[compbasis:%.+]] = quantum.compbasis qreg [[q0]] : !quantum.obs
                %3 = quantum.compbasis qreg %2 : !quantum.obs

                // CHECK: [[samples:%.+]] = quantum.sample [[compbasis]] : tensor<1x1xf64>
                // CHECK: [[res:%.+]] = func.call @probs_from_samples.tensor.1x1xf64([[samples]]) :
                // CHECK-SAME: (tensor<1x1xf64>) -> tensor<2xf64>
                // CHECK-NOT: quantum.probs
                %4 = quantum.probs %3 : tensor<2xf64>

                // CHECK: func.return [[res]] : tensor<2xf64>
                func.return %4 : tensor<2xf64>
            }
            // CHECK-LABEL: func.func public @probs_from_samples.tensor.1x1xf64
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
            func.func public @circuit() -> (tensor<f64>) {
                %0 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
                %1 = tensor.extract %0[] : tensor<i64>
                quantum.device shots(%1) ["", "", ""]

                // CHECK: [[q0:%.+]] = "test.op"() : () -> !quantum.reg
                %2 = "test.op"() : () -> !quantum.reg

                // CHECK: [[compbasis:%.+]] = quantum.compbasis qreg [[q0]] : !quantum.obs
                %3 = quantum.compbasis qreg %2 : !quantum.obs

                // CHECK: [[samples:%.+]] = quantum.sample [[compbasis]] : tensor<1x1xf64>
                %4 = quantum.sample %3 : tensor<1x1xf64>

                // CHECK: func.return [[samples]] : tensor<1x1xf64>
                func.return %4 : tensor<1x1xf64>
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
            func.func public @circuit() -> (tensor<f64>) {
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

    def test_2_wire_expval(self, run_filecheck):
        """Test the measurements-from-samples pass on a 2-wire circuit terminating with an expval(Z)
        measurement on each wire.
        """
        program = """
        builtin.module @module_circuit {
            // CHECK-LABEL: circuit
            func.func public @circuit() -> (tensor<f64>) {
                %0 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
                %1 = tensor.extract %0[] : tensor<i64>
                quantum.device shots(%1) ["", "", ""]

                // CHECK: [[q0:%.+]] = "test.op"() : () -> !quantum.bit
                %2 = "test.op"() : () -> !quantum.bit

                // CHECK: [[q1:%.+]] = "test.op"() : () -> !quantum.bit
                %3 = "test.op"() : () -> !quantum.bit

                // CHECK-NOT: quantum.namedobs
                %4 = quantum.namedobs %2[PauliZ] : !quantum.obs
                %5 = quantum.namedobs %3[PauliZ] : !quantum.obs

                // CHECK: [[obs0:%.+]] = quantum.compbasis qubits [[q0]] : !quantum.obs
                // CHECK: [[samples0:%.+]] = quantum.sample [[obs0]] : tensor<1x1xf64>
                // CHECK: [[c0:%.+]] = arith.constant dense<0> : tensor<i64>
                // CHECK: [[obs1:%.+]] = quantum.compbasis qubits [[q1]] : !quantum.obs
                // CHECK: [[samples1:%.+]] = quantum.sample [[obs1]] : tensor<1x1xf64>
                // CHECK: [[c1:%.+]] = arith.constant dense<0> : tensor<i64>
                // CHECK: [[res0:%.+]] = func.call @expval_from_samples.tensor.1x1xf64([[samples0]], [[c0]]) :
                // CHECK-SAME: (tensor<1x1xf64>, tensor<i64>) -> tensor<f64>
                // CHECK: [[res1:%.+]] = func.call @expval_from_samples.tensor.1x1xf64([[samples1]], [[c1]]) :
                // CHECK-SAME: (tensor<1x1xf64>, tensor<i64>) -> tensor<f64>
                // CHECK-NOT: quantum.expval
                %6 = quantum.expval %4 : f64
                %7 = "tensor.from_elements"(%6) : (f64) -> tensor<f64>
                %8 = quantum.expval %5 : f64
                %9 = "tensor.from_elements"(%8) : (f64) -> tensor<f64>

                // CHECK: func.return [[res0]], [[res1]] : tensor<f64>, tensor<f64>
                func.return %7, %9 : tensor<f64>, tensor<f64>
            }
            // CHECK-LABEL: func.func public @expval_from_samples.tensor.1x1xf64
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
            func.func public @circuit() -> (tensor<f64>) {
                %0 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
                %1 = tensor.extract %0[] : tensor<i64>
                quantum.device shots(%1) ["", "", ""]

                // CHECK: [[q0:%.+]] = "test.op"() : () -> !quantum.bit
                %2 = "test.op"() : () -> !quantum.bit

                // CHECK: [[q1:%.+]] = "test.op"() : () -> !quantum.bit
                %3 = "test.op"() : () -> !quantum.bit

                // CHECK-NOT: quantum.namedobs
                %4 = quantum.namedobs %2[PauliZ] : !quantum.obs
                %5 = quantum.namedobs %3[PauliZ] : !quantum.obs

                // CHECK: [[obs0:%.+]] = quantum.compbasis qubits [[q0]] : !quantum.obs
                // CHECK: [[samples0:%.+]] = quantum.sample [[obs0]] : tensor<1x1xf64>
                // CHECK: [[c0:%.+]] = arith.constant dense<0> : tensor<i64>
                // CHECK: [[obs1:%.+]] = quantum.compbasis qubits [[q1]] : !quantum.obs
                // CHECK: [[samples1:%.+]] = quantum.sample [[obs1]] : tensor<1x1xf64>
                // CHECK: [[c1:%.+]] = arith.constant dense<0> : tensor<i64>
                // CHECK: [[res0:%.+]] = func.call @var_from_samples.tensor.1x1xf64([[samples0]], [[c0]]) :
                // CHECK-SAME: (tensor<1x1xf64>, tensor<i64>) -> tensor<f64>
                // CHECK: [[res1:%.+]] = func.call @var_from_samples.tensor.1x1xf64([[samples1]], [[c1]]) :
                // CHECK-SAME: (tensor<1x1xf64>, tensor<i64>) -> tensor<f64>
                // CHECK-NOT: quantum.var
                %6 = quantum.var %4 : f64
                %7 = "tensor.from_elements"(%6) : (f64) -> tensor<f64>
                %8 = quantum.var %5 : f64
                %9 = "tensor.from_elements"(%8) : (f64) -> tensor<f64>

                // CHECK: func.return [[res0]], [[res1]] : tensor<f64>, tensor<f64>
                func.return %7, %9 : tensor<f64>, tensor<f64>
            }
            // CHECK-LABEL: func.func public @var_from_samples.tensor.1x1xf64
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
            func.func public @circuit() -> (tensor<f64>) {
                %0 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
                %1 = tensor.extract %0[] : tensor<i64>
                quantum.device shots(%1) ["", "", ""]

                // CHECK: [[qreg:%.+]] = quantum.alloc
                %2 = quantum.alloc(2) : !quantum.reg

                // CHECK: [[compbasis:%.+]] = quantum.compbasis qreg [[qreg]] : !quantum.obs
                %3 = quantum.compbasis qreg %2 : !quantum.obs

                // CHECK: [[samples:%.+]] = quantum.sample [[compbasis]] : tensor<1x2xf64>
                // CHECK: [[res:%.+]] = func.call @probs_from_samples.tensor.1x2xf64([[samples]]) :
                // CHECK-SAME: (tensor<1x2xf64>) -> tensor<4xf64>
                // CHECK-NOT: quantum.probs
                %4 = quantum.probs %3 : tensor<4xf64>

                // CHECK: func.return [[res]] : tensor<4xf64>
                func.return %4 : tensor<4xf64>
            }
            // CHECK-LABEL: func.func public @probs_from_samples.tensor.1x2xf64
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
            func.func public @circuit() -> (tensor<f64>) {
                %0 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
                %1 = tensor.extract %0[] : tensor<i64>
                quantum.device shots(%1) ["", "", ""]

                // CHECK: [[qreg:%.+]] = quantum.alloc
                %2 = quantum.alloc(2) : !quantum.reg

                // CHECK: [[q0:%.+]] = quantum.extract [[qreg]][0]
                %3 = quantum.extract %2[0] : !quantum.reg -> !quantum.bit

                // CHECK: [[compbasis0:%.+]] = quantum.compbasis qubits [[q0]] : !quantum.obs
                %4 = quantum.compbasis qubits %3 : !quantum.obs

                // CHECK: [[samples0:%.+]] = quantum.sample [[compbasis0]] : tensor<1x1xf64>
                // CHECK: [[res0:%.+]] = func.call @probs_from_samples.tensor.1x1xf64([[samples0]]) :
                // CHECK-SAME: (tensor<1x1xf64>) -> tensor<2xf64>
                // CHECK-NOT: quantum.probs
                %5 = quantum.probs %4 : tensor<2xf64>

                // CHECK: [[q1:%.+]] = quantum.extract [[qreg]][1]
                %6 = quantum.extract %2[1] : !quantum.reg -> !quantum.bit

                // CHECK: [[compbasis1:%.+]] = quantum.compbasis qubits [[q1]] : !quantum.obs
                %7 = quantum.compbasis qubits %6 : !quantum.obs

                // CHECK: [[samples1:%.+]] = quantum.sample [[compbasis1]] : tensor<1x1xf64>
                // CHECK: [[res1:%.+]] = func.call @probs_from_samples.tensor.1x1xf64([[samples1]]) :
                // CHECK-SAME: (tensor<1x1xf64>) -> tensor<2xf64>
                // CHECK-NOT: quantum.probs
                %8 = quantum.probs %7 : tensor<2xf64>

                // CHECK: func.return [[res0]], [[res1]] : tensor<2xf64>, tensor<2xf64>
                func.return %5, %8 : tensor<2xf64>, tensor<2xf64>
            }
            // CHECK-LABEL: func.func public @probs_from_samples.tensor.1x1xf64
        }
        """

        pipeline = (MeasurementsFromSamplesPass(),)
        run_filecheck(program, pipeline)


@pytest.mark.usefixtures("use_capture")
class TestItegrationUsefulErrors:
    """Test that useful error messages are raised in the frontend for unsupported behaviour"""

    def test_no_shots_raises_error(self):
        """Test that when no shots are provided, the pass raises an error"""

        @qml.qjit
        @measurements_from_samples_pass
        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def circuit(x):
            qml.RX(x, 0)
            return qml.expval(qml.Z(0))

        with pytest.raises(
            ValueError, match="measurements_from_samples pass requires non-zero shots"
        ):
            circuit(1.2)

    def test_dynamic_shots_raises_useful_error(self):
        """Test that when dynamic shots are provided, the pass raises an error"""

        @qml.qjit
        @measurements_from_samples_pass
        def workflow(a, shots):
            
            @qml.set_shots(shots)
            @qml.qnode(qml.device("lightning.qubit", wires=1))
            def circuit(x):
                qml.RX(x, 0)
                return qml.expval(qml.Z(0))
            
            circuit(a)

        with pytest.raises(
            ValueError, match="measurements_from_samples pass requires non-zero shots"
        ):
            workflow(1.2, 100)

    def test_counts_raises_not_implemented(self):
        """Test that a circuit with counts causes measurements_from_samples_pass 
        to raise a NotImplementedError"""

        dev = qml.device("lightning.qubit", wires=4)

        with pytest.raises(
            NotImplementedError, match="not implemented with measurements_from_samples"
        ):
            @qml.qjit
            @measurements_from_samples_pass
            @qml.set_shots(1000)
            @qml.qnode(dev)
            def circuit(theta: float):
                qml.RX(theta, 0)
                return qml.counts()

    @pytest.mark.xfail(reason="Dynamic shots not supported")
    def test_exec_expval_dynamic_shots(self):
        """Test the measurements_from_samples transform where the number of shots is dynamic.

        This use case is not currently supported.
        """

        @qml.qjit
        def workload(shots):
            dev = qml.device("lightning.qubit", wires=1)

            @measurements_from_samples_pass
            @qml.qnode(dev, shots=shots)
            def circuit():
                return qml.expval(qml.Z(wires=0))

            return circuit()

        result = workload(2)
        assert result == 1.0

    # TODO: not sure what this raises yet, depends on behaviour of diagonalize and whether 
    # this is already caught there. Placeholder for now.
    @pytest.mark.parametrize("mp", (qml.expval, qml.var))
    def test_overlapping_tensor(self, mp):
                
        dev = qml.device("lightning.qubit", wires=2)

        @qml.qnode(dev, shots=shots)
        def circuit_ref():
            initial_ops[0](wires=0)
            initial_ops[1](wires=1)
            return mp(qml.Z(0) @ qml.X(0))


@pytest.mark.usefixtures("use_capture")
class TestIntegrationWithOtherPasses:
    """Tests the integration of the xDSL-based MeasurementsFromSamplesPass transform with 
    other key passes.
    """

    def test_integrate_with_decompose(self):
        """Test that the measurements_from_samples pass works correctly when used in combination
        with the decompose pass."""
        dev = qml.device("null.qubit", wires=4)

        @qml.qjit(target="mlir")
        @measurements_from_samples_pass
        @partial(
            qml.transforms.decompose,
            gate_set={"X", "Y", "Z", "S", "H", "CNOT", "RZ", "RY", "GlobalPhase"},
        )
        @qml.qnode(dev, shots=1000)
        def circuit():
            qml.CRX(0.1, wires=[0, 1])
            return qml.expval(qml.Z(0))

        res = circuit()
        assert res == 1.0

@pytest.mark.usefixtures("use_capture")
class TestIntegrationWithExecution:
    """Tests of the execution of simple workloads with the xDSL-based MeasurementsFromSamplesPass
    transform and compare to expected results.
    """

    @pytest.mark.parametrize("transform", [measurements_from_samples_pass, qml.transform(pass_name="measurements-from-samples")])
    def test_qjit_filecheck(self, transform, run_filecheck_qjit):
        """Test that the measurements_from_samples_pass works correctly with qjit when 
        applied directly and as a qml.transform."""

        dev = qml.device("lightning.qubit", wires=2)
    
        @qml.qjit(target="mlir")
        @transform
        @qml.qnode(dev, shots=25)
        def circuit():
            # CHECK-NOT: quantum.namedobs
            # CHECK: [[obs:%.+]] = quantum.compbasis
            # CHECK: [[samples:%.+]] = quantum.sample [[obs]] : tensor<25x1xf64>
            # CHECK: [[c0:%.+]] = arith.constant dense<0> : tensor<i64>
            # CHECK: [[res:%.+]] = func.call @expval_from_samples.tensor.25x1xf64([[samples]], [[c0]]) :
            # CHECK-SAME: (tensor<25x1xf64>, tensor<i64>) -> tensor<f64>
            # CHECK-NOT: quantum.expval
            return qml.expval(qml.Z(wires=0))

        run_filecheck_qjit(circuit)

    # ---- Test circuits returning expectation values -------------------------------------------- #
    @pytest.mark.parametrize("shots", [1, 2])
    @pytest.mark.parametrize(
        "initial_op, obs, expected_res",
        [
            # PauliZ observables
            (qml.I, qml.Z, 1.0),
            (qml.X, qml.Z, -1.0),
            # PauliX observables
            pytest.param(
                partial(qml.RY, phi=np.pi / 2),
                qml.X,
                1.0,
            ),
            pytest.param(
                partial(qml.RY, phi=-np.pi / 2),
                qml.X,
                -1.0,
            ),
            # PauliY observables
            pytest.param(
                partial(qml.RX, phi=-np.pi / 2),
                qml.Y,
                1.0,
            ),
            pytest.param(
                partial(qml.RX, phi=np.pi / 2),
                qml.Y,
                -1.0,
            ),
        ],
    )
    def test_expval_single_wire(self, shots, initial_op, obs, expected_res):
        """Test the measurements_from_samples transform on a circuit that measures an 
        expval of an observable on the single wire.
        """

        dev = qml.device("lightning.qubit", wires=1)

        @qml.qnode(dev, shots=shots)
        def circuit_ref():
            initial_op(wires=0)
            return qml.expval(obs(wires=0))

        assert expected_res == circuit_ref(), "Sanity check failed, is expected_res correct?"
        circuit_compiled = qml.qjit(
            measurements_from_samples_pass(circuit_ref),
        )

        assert expected_res == circuit_compiled()

    @pytest.mark.parametrize("shots", [1, 2])
    @pytest.mark.parametrize(
        "initial_ops, obs, expected_res",
        [
            ((qml.I, qml.I), qml.Z, (1.0, 1.0)),
            ((qml.I, qml.X), qml.Z, (1.0, -1.0)),
            ((qml.X, qml.I), qml.Z, (-1.0, 1.0)),
            ((qml.X, qml.X), qml.Z, (-1.0, -1.0)),
        ],
    )
    def test_expval_2_mps(self, shots, initial_ops, obs, expected_res):
        """Test the measurements_from_samples transform on a device with two wires and terminal
        measurements that require an observable (i.e. expval and var).

        In this test, the terminal measurements are performed separately per wire.
        """

        dev = qml.device("lightning.qubit", wires=2)

        @qml.qnode(dev, shots=shots)
        def circuit_ref():
            initial_ops[0](wires=0)
            initial_ops[1](wires=1)
            return qml.expval(obs(wires=0)), qml.expval(obs(wires=1))

        assert expected_res == circuit_ref(), "Sanity check failed, is expected_res correct?"
        circuit_compiled = qml.qjit(
            measurements_from_samples_pass(circuit_ref),
        )

        assert expected_res == circuit_compiled()

    @pytest.mark.parametrize("shots", [1, 2])
    @pytest.mark.parametrize(
        "initial_ops, expected_res",
        [
            ((qml.I, qml.I), 1.0),
            ((qml.I, qml.X), -1.0),
            ((qml.X, qml.I), -1.0),
            ((qml.X, qml.X), 1.0),
        ],
    )
    def test_expval_tensor(self, shots, initial_ops, expected_res):
        """Test the measurements_from_samples transform on a device with two wires and terminal
        measurements that require an observable (i.e. expval and var).

        In this test, the terminal measurements are performed on the combination of both wires.
        """

        dev = qml.device("lightning.qubit", wires=2)

        @qml.qnode(dev, shots=shots)
        def circuit_ref():
            initial_ops[0](wires=0)
            initial_ops[1](wires=1)
            return qml.expval(qml.Z(wires=0) @ qml.Z(wires=1))

        assert expected_res == circuit_ref(), "Sanity check failed, is expected_res correct?"

        circuit_compiled = qml.qjit(measurements_from_samples_pass(circuit_ref))

        assert expected_res == circuit_compiled()

    @pytest.mark.parametrize("shots", [1, 2])
    @pytest.mark.parametrize(
        "initial_ops, expected_res",
        [
            ((qml.I, qml.I), 1.0),
            ((qml.I, qml.X), -1.0),
            ((qml.X, qml.I), -1.0),
            ((qml.X, qml.X), 1.0),
        ],
    )
    def test_expval_sprod(self, shots, initial_ops, expected_res):
        """Test the measurements_from_samples transform on a device with two wires and terminal
        measurements that require an observable (i.e. expval and var).

        In this test, the terminal measurements are performed on the combination of both wires.
        """

        dev = qml.device("lightning.qubit", wires=2)

        @qml.qjit
        @qml.transform(pass_name="measurements-from-samples")
        @qml.qnode(dev, shots=shots)
        def circuit_ref():
            initial_ops[0](wires=0)
            initial_ops[1](wires=1)
            return qml.expval(2* qml.Z(wires=0))

        assert expected_res == circuit_compiled()

    @pytest.mark.parametrize("shots", [1000, 2000])
    @pytest.mark.parametrize(
        "initial_ops, expected_res",
        [
            ((qml.I, qml.I), 2.0),
            # ((qml.I, qml.X), -1.0),
            # ((qml.X, qml.I), -1.0),
            # ((qml.X, qml.X), 1.0),
        ],
    )
    def test_expval_sum(self, shots, initial_ops, expected_res):
        """Test the measurements_from_samples transform on a device with two wires and terminal
        measurements that require an observable (i.e. expval and var).

        In this test, the terminal measurements are performed on the combination of both wires.
        """

        dev = qml.device("lightning.qubit", wires=2)

        @qml.qjit
        @qml.transform(pass_name="measurements-from-samples")
        @qml.qnode(dev, shots=shots)
        def circuit():
            initial_ops[0](wires=0)
            initial_ops[1](wires=1)
            return qml.expval(2*qml.Z(wires=0) + qml.X(1))

        assert np.isclose(circuit(), expected_res, atol=0.05)

    @pytest.mark.parametrize("shots", [1, 2])
    @pytest.mark.parametrize(
        "initial_ops, expected_res",
        [
            ((qml.I, qml.I), 1.0),
            ((qml.I, qml.X), -1.0),
            ((qml.X, qml.I), -1.0),
            ((qml.X, qml.X), 1.0),
        ],
    )
    def test_expval_mcm(self, shots, initial_ops, expected_res):
        """Test the measurements_from_samples transform on a device with two wires and terminal
        measurements that require an observable (i.e. expval and var).

        In this test, the terminal measurements provide statistics on an MCM.
        """

        dev = qml.device("lightning.qubit", wires=2)

        @qml.qjit
        @qml.transform(pass_name="measurements-from-samples")
        @qml.qnode(dev, shots=shots)
        def circuit():
            initial_ops[0](wires=0)
            initial_ops[1](wires=1)
            m = qml.measure(0)
            return qml.expval(m)

        assert np.allclose(expected_res, circuit(), atol=0.05)


    # ---- Test circuits returning variance ------------------------------------------------------ #
    @pytest.mark.parametrize("shots", [1, 2])
    @pytest.mark.parametrize(
        "initial_op, obs, expected_res",
        [
            # PauliZ observables
            (qml.I, qml.Z, 0.0),
            (qml.X, qml.Z, 0.0),
            # PauliX observables
            pytest.param(
                partial(qml.RY, phi=np.pi / 2),
                qml.X,
                0.0,
            ),
            pytest.param(
                partial(qml.RY, phi=-np.pi / 2),
                qml.X,
                0.0,
            ),
            # PauliY observables
            pytest.param(
                partial(qml.RX, phi=-np.pi / 2),
                qml.Y,
                0.0,
            ),
            pytest.param(
                partial(qml.RX, phi=np.pi / 2),
                qml.Y,
                0.0,
            ),
        ],
    )
    def test_single_wire_var(self, shots, initial_op, obs, expected_res):
        """Test the measurements_from_samples transform on a circuit that measures an 
        variance of an observable on the single wire.
        """

        dev = qml.device("lightning.qubit", wires=1)

        @qml.qnode(dev, shots=shots)
        def circuit_ref():
            initial_op(wires=0)
            return qml.var(obs(wires=0))

        assert expected_res == circuit_ref(), "Sanity check failed, is expected_res correct?"
        circuit_compiled = qml.qjit(
            measurements_from_samples_pass(circuit_ref),
        )

        assert expected_res == circuit_compiled()


    @pytest.mark.parametrize("shots", [1, 2])
    @pytest.mark.parametrize(
        "initial_ops, obs, expected_res",
        [
            ((qml.I, qml.I), qml.Z, (0.0, 0.0)),
            ((qml.I, qml.X), qml.Z, (0.0, 0.0)),
            ((qml.X, qml.I), qml.Z, (0.0, 0.0)),
            ((qml.X, qml.X), qml.Z, (0.0, 0.0)),
        ],
    )
    def test_2_variance_mps(self, shots, initial_ops, obs, expected_res):
        """Test the measurements_from_samples transform on a device with two wires and terminal
        measurements that require an observable (i.e. expval and var).

        In this test, the terminal measurements are performed separately per wire.
        """

        dev = qml.device("lightning.qubit", wires=2)

        @qml.qnode(dev, shots=shots)
        def circuit_ref():
            initial_ops[0](wires=0)
            initial_ops[1](wires=1)
            return qml.var(obs(wires=0)), qml.var(obs(wires=1))

        assert expected_res == circuit_ref(), "Sanity check failed, is expected_res correct?"
        circuit_compiled = qml.qjit(
            measurements_from_samples_pass(circuit_ref),
        )

        assert expected_res == circuit_compiled()


    @pytest.mark.parametrize("shots", [1, 2])
    @pytest.mark.parametrize(
        "initial_ops, expected_res",
        [
            ((qml.I, qml.I), 0.0),
            ((qml.I, qml.X), 0.0),
            ((qml.X, qml.I), 0.0),
            ((qml.X, qml.X), 0.0),
        ],
    )
    def test_var_tensor(self, shots, initial_ops, expected_res):
        """Test the measurements_from_samples transform on a device with two wires and terminal
        measurements that require an observable (i.e. expval and var).

        In this test, the terminal measurements are performed on the combination of both wires.
        """

        dev = qml.device("lightning.qubit", wires=2)

        @qml.qnode(dev, shots=shots)
        def circuit_ref():
            initial_ops[0](wires=0)
            initial_ops[1](wires=1)
            return qml.var(qml.Z(wires=0) @ qml.Z(wires=1))

        assert expected_res == circuit_ref(), "Sanity check failed, is expected_res correct?"

        circuit_compiled = qml.qjit(measurements_from_samples_pass(circuit_ref))

        assert expected_res == circuit_compiled()


    @pytest.mark.parametrize("shots", [1, 2])
    @pytest.mark.parametrize(
        "initial_ops, expected_res",
        [
            ((qml.I, qml.I), 0.0),
            ((qml.I, qml.X), 0.0),
            ((qml.X, qml.I), 0.0),
            ((qml.X, qml.X), 0.0),
        ],
    )
    def test_var_sprod(self, shots, initial_ops, expected_res):
        """Test the measurements_from_samples transform on a device with two wires and terminal
        measurements that require an observable (i.e. expval and var).

        In this test, the terminal measurements are performed on the combination of both wires.
        """

        dev = qml.device("lightning.qubit", wires=2)

        @qml.qnode(dev, shots=shots)
        def circuit_ref():
            initial_ops[0](wires=0)
            initial_ops[1](wires=1)
            return qml.var(qml.Z(wires=0) @ qml.Z(wires=1))

        assert expected_res == circuit_ref(), "Sanity check failed, is expected_res correct?"

        circuit_compiled = qml.qjit(measurements_from_samples_pass(circuit_ref))

        assert expected_res == circuit_compiled()

    # ---- Test circuits returning probability --------------------------------------------------- #
    @pytest.mark.parametrize("shots", [1, 2])
    @pytest.mark.parametrize(
        "initial_op, expected_res",
        [
            (qml.I, [1.0, 0.0]),
            (qml.X, [0.0, 1.0]),
        ],
    )
    def test_exec_1_wire_probs(self, shots, initial_op, expected_res):
        """Test the measurements_from_samples transform on a device with a single wire and terminal
        probs measurements.
        """

        dev = qml.device("lightning.qubit", wires=1)

        @qml.qnode(dev, shots=shots)
        def circuit_ref():
            initial_op(wires=0)
            return qml.probs(wires=0)

        assert np.array_equal(
            expected_res, circuit_ref()
        ), "Sanity check failed, is expected_res correct?"
        circuit_compiled = qml.qjit(
            measurements_from_samples_pass(circuit_ref),
        )

        assert np.array_equal(expected_res, circuit_compiled())

    @pytest.mark.parametrize("shots", [1, 2])
    @pytest.mark.parametrize(
        "initial_ops, expected_res",
        [
            ((qml.I, qml.I), [1.0, 0.0, 0.0, 0.0]),
            ((qml.I, qml.X), [0.0, 1.0, 0.0, 0.0]),
            ((qml.X, qml.I), [0.0, 0.0, 1.0, 0.0]),
            ((qml.X, qml.X), [0.0, 0.0, 0.0, 1.0]),
        ],
    )
    def test_exec_2_wire_probs_global(self, shots, initial_ops, expected_res):
        """Test the measurements_from_samples transform on a device with two wires and a terminal,
        "global" probs measurements (one that implicitly acts on all wires).
        """
        dev = qml.device("lightning.qubit", wires=2)

        @qml.qnode(dev, shots=shots)
        def circuit_ref():
            initial_ops[0](wires=0)
            initial_ops[1](wires=1)
            return qml.probs()

        assert np.array_equal(
            expected_res, circuit_ref()
        ), "Sanity check failed, is expected_res correct?"
        circuit_compiled = qml.qjit(measurements_from_samples_pass(circuit_ref))

        assert np.array_equal(expected_res, circuit_compiled())

    @pytest.mark.parametrize("shots", [1, 2])
    @pytest.mark.parametrize(
        "initial_ops, expected_res",
        [
            ((qml.I, qml.I), ([1.0, 0.0], [1.0, 0.0])),
            ((qml.I, qml.X), ([1.0, 0.0], [0.0, 1.0])),
            ((qml.X, qml.I), ([0.0, 1.0], [1.0, 0.0])),
            ((qml.X, qml.X), ([0.0, 1.0], [0.0, 1.0])),
        ],
    )
    def test_exec_2_wire_probs_per_wire(self, shots, initial_ops, expected_res):
        """Test the measurements_from_samples transform on a device with two wires and a terminal,
        "global" probs measurements (one that implicitly acts on all wires).
        """
        dev = qml.device("lightning.qubit", wires=2)

        @qml.qnode(dev, shots=shots)
        def circuit_ref():
            initial_ops[0](wires=0)
            initial_ops[1](wires=1)
            return qml.probs(wires=0), qml.probs(wires=1)

        assert np.array_equal(
            expected_res, circuit_ref()
        ), "Sanity check failed, is expected_res correct?"
        circuit_compiled = qml.qjit(measurements_from_samples_pass(circuit_ref))

        assert np.array_equal(expected_res, circuit_compiled())

    # ---- Test circuits returning samples ------------------------------------------------------- #
    @pytest.mark.parametrize("shots", [1, 2])
    @pytest.mark.parametrize(
        "initial_op, expected_res_base",
        [
            (qml.I, 0),
            (qml.X, 1),
        ],
    )
    def test_exec_1_wire_sample(self, shots, initial_op, expected_res_base):
        """Test the measurements_from_samples transform on a device with a single wire and terminal
        sample measurements.

        In this case, the measurements_from_samples pass should effectively be a no-op.
        """
        dev = qml.device("lightning.qubit", wires=1)

        @qml.qnode(dev, shots=shots)
        def circuit_ref():
            initial_op(wires=0)
            return qml.sample(wires=0)

        circuit_compiled = qml.qjit(
            measurements_from_samples_pass(circuit_ref),
        )

        expected_res = expected_res_base * np.ones(shape=(shots, 1), dtype=int)

        assert np.array_equal(expected_res, circuit_compiled())

    # -------------------------------------------------------------------------------------------- #

    def test_measurements_from_samples_multiple_measurements(self):
        """Test the transform measurements_from_samples with multiple measurement types
        as part of the Catalyst pipeline."""

        dev = qml.device("lightning.qubit", wires=4)

        @qml.set_shots(5000)
        @qml.qnode(dev)
        def basic_circuit(theta: float):
            qml.RY(theta, 0)
            qml.RY(theta / 2, 1)
            qml.RY(2 * theta, 2)
            qml.RY(theta, 3)
            return (
                qml.expval(qml.PauliX(wires=0) @ qml.PauliX(wires=1)),
                qml.var(qml.PauliX(wires=2)),
                qml.sample(qml.PauliX(wires=0) @ qml.PauliX(wires=1) @ qml.PauliX(wires=2)),
                qml.probs(wires=[3]),
            )

        transformed_circuit = qml.qjit(measurements_from_samples_pass(basic_circuit), seed=37)

        # mlir = transformed_circuit.mlir
        # assert "expval" not in mlir
        # assert "quantum.var" not in mlir
        # assert "sample" in mlir

        theta = 1.9

        expval_res, var_res, sample_res, probs_res = qjit(transformed_circuit, seed=37)(theta)

        expval_expected = np.sin(theta) * np.sin(theta / 2)
        var_expected = 1 - np.sin(2 * theta) ** 2
        sample_expected = basic_circuit(theta)[2]
        probs_expected = [np.cos(theta / 2) ** 2, np.sin(theta / 2) ** 2]

        assert np.isclose(expval_res, expval_expected, atol=0.05)
        assert np.isclose(var_res, var_expected, atol=0.05)
        assert np.allclose(probs_res, probs_expected, atol=0.05)

        # sample comparison
        assert np.isclose(np.mean(sample_res), np.mean(sample_expected), atol=0.05)
        assert len(sample_res) == len(sample_expected)
        assert set(np.array(sample_res)) == set(sample_expected)


def _counts_catalyst_to_pl(basis_states, counts):
    """Helper function to convert counts in the Catalyst format to the PennyLane format.

    Example:

    >>> basis_states, counts = ([0, 1], [6, 4])
    >>> _counts_catalyst_to_pl(basis_states, counts)
    {'0': 6, '1': 4}
    """
    return {format(int(state), "01b"): count for state, count in zip(basis_states, counts)}


if __name__ == "__main__":
    pytest.main(["-x", __file__])
