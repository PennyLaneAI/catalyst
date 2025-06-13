# Copyright 2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Test for the device preprocessing."""
import platform
from dataclasses import replace

import numpy as np
import pennylane as qml
import pytest
from conftest import CONFIG_CUSTOM_DEVICE
from pennylane.devices import Device, NullQubit
from pennylane.devices.capabilities import DeviceCapabilities, OperatorProperties
from pennylane.tape import QuantumScript

from catalyst import CompileError, ctrl, qjit
from catalyst.api_extensions.control_flow import (
    Cond,
    ForLoop,
    WhileLoop,
    cond,
    for_loop,
    while_loop,
)
from catalyst.api_extensions.quantum_operators import HybridAdjoint, adjoint
from catalyst.compiler import get_lib_path
from catalyst.device.decomposition import catalyst_decompose, decompose_ops_to_unitary
from catalyst.jax_tracer import HybridOpRegion
from catalyst.tracing.contexts import EvaluationContext, EvaluationMode

# pylint: disable=unused-argument


class OtherHadamard(qml.Hadamard):
    """A version of the Hadamard operator that won't be recognized by the QJit device, and will
    need to be decomposed"""

    @property
    def name(self):
        """name"""
        return "OtherHadamard"


class OtherIsingXX(qml.IsingXX):
    """A version of the IsingXX operator that won't be recognized by the QJit device, and will
    need to be decomposed"""

    @property
    def name(self):
        """name"""
        return "OtherIsingXX"


class OtherRX(qml.RX):
    """A version of the RX operator that won't be recognized by the QJit device, and will need to
    be decomposed"""

    @property
    def name(self):
        """Name of the operator is UnknownOp"""
        return "OtherRX"

    def decomposition(self):
        """decomposes to normal RX"""
        return [qml.RX(*self.parameters, self.wires)]


class CustomDevice(Device):
    """A dummy device from the device API."""

    config_filepath = CONFIG_CUSTOM_DEVICE

    _to_matrix_ops = {
        "DiagonalQubitUnitary": OperatorProperties(),
        "BlockEncode": OperatorProperties(),
    }

    def __init__(self, wires, shots=1024):
        super().__init__(wires=wires, shots=shots)
        self.qjit_capabilities = self.capabilities

    @staticmethod
    def get_c_interface():
        """Returns a tuple consisting of the device name, and
        the location to the shared object with the C/C++ device implementation.
        """
        system_extension = ".dylib" if platform.system() == "Darwin" else ".so"
        lib_path = (
            get_lib_path("runtime", "RUNTIME_LIB_DIR") + "/librtd_null_qubit" + system_extension
        )
        return "dummy.remote", lib_path

    def execute(self, circuits, execution_config):
        """Execution."""
        raise NotImplementedError


CustomDevice.capabilities.operations.pop("BlockEncode")


class TestDecomposition:
    """Test the preprocessing transforms implemented in Catalyst."""

    def test_decompose_integration(self):
        """Test the decompose transform as part of the Catalyst pipeline."""
        dev = NullQubit(wires=4, shots=None)

        @qjit
        @qml.qnode(dev)
        def circuit(theta: float):
            qml.SingleExcitationPlus(theta, wires=[0, 1])
            return qml.state()

        mlir = qjit(circuit, target="mlir").mlir
        assert "PauliX" in mlir
        assert "CNOT" in mlir
        assert "ControlledPhaseShift" in mlir
        assert "SingleExcitationPlus" not in mlir

    def test_decompose_ops_to_unitary(self):
        """Test the decompose ops to unitary transform."""
        operations = [qml.CNOT(wires=[0, 1]), qml.RX(0.1, wires=0)]
        tape = qml.tape.QuantumScript(ops=operations)
        ops_to_decompose = ["CNOT"]

        tapes, _ = decompose_ops_to_unitary(tape, ops_to_decompose)
        decomposed_ops = tapes[0].operations
        assert isinstance(decomposed_ops[0], qml.QubitUnitary)
        assert isinstance(decomposed_ops[1], qml.RX)

    def test_decompose_ops_to_unitary_integration(self):
        """Test the decompose ops to unitary transform as part of the Catalyst pipeline."""
        dev = CustomDevice(wires=4, shots=None)

        @qjit
        @qml.qnode(dev)
        def circuit():
            qml.BlockEncode(np.array([[1, 1, 1], [0, 1, 0]]), wires=[0, 1, 2])
            return qml.state()

        mlir = qjit(circuit, target="mlir").mlir
        assert "quantum.unitary" in mlir
        assert "BlockEncode" not in mlir

    def test_no_matrix(self):
        """Test that controlling an operation without a matrix method raises an error."""
        dev = NullQubit(wires=4)

        class OpWithNoMatrix(qml.operation.Operation):
            """Op without matrix."""

            def matrix(self, wire_order=None):
                """Matrix is overriden."""
                raise NotImplementedError()

        @qml.qnode(dev)
        def f():
            ctrl(OpWithNoMatrix(wires=[0, 1]), control=[2, 3])
            return qml.probs()

        with pytest.raises(CompileError, match="could not be decomposed, it might be unsupported."):
            qjit(f, target="jaxpr")


# tapes and regions for generating HybridOps
tape1 = QuantumScript([qml.X(0), qml.Hadamard(1)])
tape2 = QuantumScript([qml.RY(1.23, 1), qml.Y(0), qml.Hadamard(2)])
region1 = HybridOpRegion([], tape1, [], [])
region2 = HybridOpRegion([], tape2, [], [])

# catalyst.pennylane_extensions.Adjoint:
#   Adjoint([], [], regions=[HybridOpRegion([], quantum_tape, [], [])])
adj_op = HybridAdjoint([], [], [region1])

# catalyst.pennylane_extensions.ForLoop:
#     ForLoop([], [], regions=[HybridOpRegion([], quantum_tape, [], [])])
forloop_op = ForLoop([], [], [region1])

# catalyst.pennylane_extensions.WhileLoop:
#     cond_region = HybridOpRegion([], None, [], [])
#     body_region = HybridOpRegion([], quantum_tape, [], [])
#     WhileLoop([], [], regions=[cond_region, body_region])
cond_region = HybridOpRegion([], None, [], [])
whileloop_op = WhileLoop([], [], regions=[cond_region, region1])

# catalyst.pennylane_extensions.Cond:
# Cond([], [], regions=[one Hybrid region per branch of the if-else tree])
cond_op = Cond([], [], regions=[region1, region2])

# each entry contains (initialized_op, op_class, num_regions)
HYBRID_OPS = [
    (adj_op, HybridAdjoint, 1),
    (forloop_op, ForLoop, 1),
    (whileloop_op, WhileLoop, 2),
    (cond_op, Cond, 2),
]

TEST_DEVICE_CONFIG_TEXT = """
schema = 3
[operators.gates]
PauliX = { }
PauliZ = { }
RX = { }
RY = { }
RZ = { }
CNOT = { }
HybridAdjoint = { }
ForLoop = { }
WhileLoop = { }
Cond = { }
QubitUnitary = { }
"""


class TestPreprocessHybridOp:
    """Test that the operators on the tapes nested inside HybridOps are also decomposed"""

    @pytest.mark.usefixtures("create_temporary_toml_file")
    @pytest.mark.parametrize("create_temporary_toml_file", [TEST_DEVICE_CONFIG_TEXT], indirect=True)
    @pytest.mark.parametrize("op, op_class, num_regions", HYBRID_OPS)
    def test_hybrid_op_decomposition(self, op, op_class, num_regions, request):
        """Tests that for a tape containing a HybridOp that contains unsupported
        Operators, the unsupported Operators are decomposed"""

        # hack for unit test (since it doesn't create a full context)
        for region in op.regions:
            region.trace = None

        capabilities = DeviceCapabilities.from_toml_file(request.node.toml_file)
        setattr(capabilities, "to_matrix_ops", {"S": OperatorProperties()})

        # create and decompose the tape
        tape = QuantumScript([op, qml.X(0), qml.Hadamard(3)])
        with EvaluationContext(EvaluationMode.QUANTUM_COMPILATION) as ctx:
            (new_tape,), _ = catalyst_decompose(tape, ctx, capabilities)

        old_op = tape[0]
        new_op = new_tape[0]

        # the pre- and post decomposition HybridOps have the same type and number of regions
        assert isinstance(old_op, op_class)
        assert isinstance(new_op, op_class)
        assert len(new_op.regions) == len(old_op.regions) == num_regions

        # the HybridOp on the original tape is unmodified, i.e. continues to contain ops
        # not in `expected_ops`. The post-decomposition HybridOp tape does not
        expected_ops = capabilities.operations
        for i in range(num_regions):
            if old_op.regions[i].quantum_tape:
                assert not np.all(
                    [op.name in expected_ops for op in old_op.regions[i].quantum_tape.operations]
                )
                assert np.all(
                    [op.name in expected_ops for op in new_op.regions[i].quantum_tape.operations]
                )

    @pytest.mark.parametrize("x, y", [(1.23, -0.4), (0.7, 0.25), (-1.51, 0.6)])
    def test_decomposition_of_adjoint_circuit(self, x, y):
        """Test that unsupported operators nested in Adjoint are decompsed
        and the resulting circuit has the expected result, obtained analytically"""

        dev = qml.device("lightning.qubit", wires=1)

        @qjit
        @qml.qnode(dev)
        def circuit(x: float, y: float):
            qml.RY(y, 0)
            adjoint(lambda: OtherRX(x, 0))()
            return qml.expval(qml.PauliZ(0))

        mlir = qjit(circuit, target="mlir").mlir

        assert "quantum.adjoint" in mlir
        assert "RX" in mlir
        assert "RY" in mlir
        assert "OtherRX" not in mlir

        assert np.isclose(circuit(x, y), np.cos(-x) * np.cos(y))

    def test_decomposition_of_cond_circuit(self):
        """Test that unsupported operators nested in Cond are decompsed, and the
        resulting circuit has the expected result, obtained analytically"""

        dev = qml.device("lightning.qubit", wires=[0, 1])

        @qjit
        @qml.qnode(dev)
        def circuit(phi: float):
            OtherHadamard(wires=0)

            # define a conditional ansatz
            @cond(phi > 1.4)
            def ansatz():
                OtherIsingXX(phi, wires=(0, 1))

            @ansatz.otherwise
            def ansatz():
                OtherIsingXX(2 * phi, wires=(0, 1))

            # apply the conditional ansatz
            ansatz()

            return qml.state()

        # mlir contains expected gate names, and not the unsupported gate names
        mlir = qjit(circuit, target="mlir").mlir
        assert "RX" in mlir
        assert "CNOT" in mlir
        assert "PhaseShift" in mlir
        assert "OtherHadamard" not in mlir
        assert "OtherIsingXX" not in mlir

        # results are correct for cond is True (IsingXX angle is phi)
        phi = 1.6
        x1 = np.cos(phi / 2) / np.sqrt(2)
        x2 = -1j * np.sin(phi / 2) / np.sqrt(2)
        expected_res = np.array([x1, x2, x1, x2])
        assert np.allclose(expected_res, circuit(phi))

        # results are correct for cond is False (IsingXX angle is 2*phi)
        phi = 1.2
        x1 = np.cos(phi) / np.sqrt(2)
        x2 = -1j * np.sin(phi) / np.sqrt(2)
        expected_res = np.array([x1, x2, x1, x2])
        assert np.allclose(expected_res, circuit(phi))

    @pytest.mark.parametrize("reps, angle", [(3, 1.72), (5, 1.6), (10, 0.4)])
    def test_decomposition_of_forloop_circuit(self, reps, angle):
        """Test that unsupported operators nested in ForLoop are decompsed, and
        the resulting circuit has the expected result, obtained analytically"""

        dev = qml.device("lightning.qubit", wires=2)

        @qjit
        @qml.qnode(dev)
        def circuit(n: int, x: float):
            OtherHadamard(wires=0)

            def loop_rx(i, phi):
                OtherIsingXX(phi, wires=(0, 1))
                # update the value of phi for the next iteration
                return phi / 2

            # apply the for loop
            for_loop(0, n, 1)(loop_rx)(x)

            return qml.state()

        def expected_res(n, x):
            """Analytic result for a loop with n reps and initial angle x"""
            phi = x * sum(1 / 2**i for i in range(0, n))

            x1 = np.cos(phi / 2) / np.sqrt(2)
            x2 = -1j * np.sin(phi / 2) / np.sqrt(2)

            return np.array([x1, x2, x1, x2])

        assert np.allclose(circuit(reps, angle), expected_res(reps, angle))

    @pytest.mark.parametrize("phi", [1.1, 1.6, 2.1])
    def test_decomposition_of_whileloop_circuit(self, phi):
        """Test that unsupported operators nested in WhileLoop are decompsed, and
        the resulting circuit has the expected result, obtained analytically"""

        dev = qml.device("lightning.qubit", wires=1)

        @qjit
        @qml.qnode(dev)
        def circuit(x: float):
            @while_loop(lambda x: x < 2.0)
            def loop_rx(x):
                # perform some work and update (some of) the arguments
                OtherRX(x, wires=0)
                return x**2

            # apply the while loop
            final_x = loop_rx(x)

            return qml.expval(qml.PauliY(0)), final_x

        res, final_phi = circuit(phi)

        total_angle = 0
        while phi < 2:
            total_angle += phi
            phi = phi**2

        expected_res = -np.sin(total_angle)

        assert np.isclose(res, expected_res)
        assert final_phi > 2.0

    @pytest.mark.usefixtures("create_temporary_toml_file")
    @pytest.mark.parametrize("create_temporary_toml_file", [TEST_DEVICE_CONFIG_TEXT], indirect=True)
    def test_decomposition_of_nested_HybridOp(self, request):
        """Tests that HybridOps with HybridOps nested inside them are still decomposed correctly"""

        # make a weird nested op
        adjoint_op = HybridAdjoint([], [], [region1])
        ops = [qml.RY(1.23, 1), adjoint_op, qml.Hadamard(2)]  # Hadamard will decompose
        adj_region = HybridOpRegion([], QuantumScript(ops), [], [])

        conditional_op = Cond([], [], regions=[adj_region, region2])
        ops = [conditional_op, qml.Y(1)]  # PauliY will decompose
        conditional_region = HybridOpRegion([], qml.tape.QuantumScript(ops), [], [])

        for_loop_op = ForLoop([], [], [conditional_region])
        ops = [for_loop_op, qml.X(0), qml.Hadamard(3)]  # Hadamard will decompose
        tape = qml.tape.QuantumScript(ops)

        # hack to avoid needing a full trace in unit test
        adjoint_op.regions[0].trace = None
        for region in conditional_op.regions:
            region.trace = None
        for_loop_op.regions[0].trace = None

        capabilities = DeviceCapabilities.from_toml_file(request.node.toml_file)
        setattr(capabilities, "to_matrix_ops", {"S": OperatorProperties()})

        # do the decomposition and get the new tape
        with EvaluationContext(EvaluationMode.QUANTUM_COMPILATION) as ctx:
            (new_tape,), _ = catalyst_decompose(tape, ctx, capabilities)

        # unsupported ops on the top-level tape have been decomposed (no more Hadamard)
        assert "Hadamard" not in [op.name for op in new_tape.operations]
        assert "RZ" in [op.name for op in new_tape.operations]

        # the first element on the top-level tape is a for-loop
        assert isinstance(new_tape[0], ForLoop)
        # unsupported op on it has been decomposed (no more PauliY)
        forloop_subtape = new_tape[0].regions[0].quantum_tape
        assert "PauliY" not in [op.name for op in forloop_subtape.operations]
        assert "RY" in [op.name for op in forloop_subtape.operations]

        # first op on the for-loop tape is a cond op
        assert isinstance(forloop_subtape[0], Cond)
        cond_subtapes = (
            forloop_subtape[0].regions[0].quantum_tape,
            forloop_subtape[0].regions[1].quantum_tape,
        )
        # unsupported ops in the subtape decomposed (original tapes contained Hadamard)
        for subtape in cond_subtapes:
            assert np.all([op.name in capabilities.operations for op in subtape.operations])
            assert "Hadamard" not in [op.name for op in subtape.operations]
            assert "RZ" in [op.name for op in subtape.operations]

        # the seconds element on the first subtape of the cond op is an adjoint
        assert isinstance(cond_subtapes[0][1], HybridAdjoint)
        # unsupported op on it has been decomposed (no more Hadamard)
        adj_subtape = cond_subtapes[0][1].regions[0].quantum_tape
        assert "Hadamard" not in [op.name for op in adj_subtape.operations]
        assert "RZ" in [op.name for op in adj_subtape.operations]

    @pytest.mark.usefixtures("create_temporary_toml_file")
    @pytest.mark.parametrize("create_temporary_toml_file", [TEST_DEVICE_CONFIG_TEXT], indirect=True)
    def test_controlled_decomposes_to_unitary_listed(self, request):
        """Test that a PennyLane toml-listed operation is decomposed to a QubitUnitary"""

        tape = qml.tape.QuantumScript([qml.PauliX(0), qml.S(0)])

        capabilities = DeviceCapabilities.from_toml_file(request.node.toml_file)
        setattr(capabilities, "to_matrix_ops", {"S": OperatorProperties()})

        with EvaluationContext(EvaluationMode.QUANTUM_COMPILATION) as ctx:
            (new_tape,), _ = catalyst_decompose(tape, ctx, capabilities)

        assert len(new_tape.operations) == 2
        assert isinstance(new_tape.operations[0], qml.PauliX)
        assert isinstance(new_tape.operations[1], qml.QubitUnitary)

    @pytest.mark.usefixtures("create_temporary_toml_file")
    @pytest.mark.parametrize("create_temporary_toml_file", [TEST_DEVICE_CONFIG_TEXT], indirect=True)
    def test_controlled_decomposes_to_unitary_controlled(self, request):
        """Test that a PennyLane controlled operation is decomposed to a QubitUnitary"""

        tape = qml.tape.QuantumScript([qml.ctrl(qml.RX(1.23, 0), 1)])

        capabilities = DeviceCapabilities.from_toml_file(request.node.toml_file)
        setattr(capabilities, "to_matrix_ops", {"S": OperatorProperties()})

        with EvaluationContext(EvaluationMode.QUANTUM_COMPILATION) as ctx:
            (new_tape,), _ = catalyst_decompose(tape, ctx, capabilities)

        assert len(new_tape.operations) == 1
        new_op = new_tape.operations[0]

        assert isinstance(new_op, qml.QubitUnitary)
        assert np.allclose(new_op.matrix(), tape.operations[0].matrix())

    @pytest.mark.usefixtures("create_temporary_toml_file")
    @pytest.mark.parametrize("create_temporary_toml_file", [TEST_DEVICE_CONFIG_TEXT], indirect=True)
    def test_error_for_pennylane_midmeasure_decompose(self, request):
        """Test that an error is raised in decompose if a PennyLane mid-circuit measurement
        is encountered"""

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(1.23, wires=0)
            qml.measure(0)

        ops, measurements = qml.queuing.process_queue(q)
        tape = qml.tape.QuantumScript(ops, measurements)

        capabilities = DeviceCapabilities.from_toml_file(request.node.toml_file)
        setattr(capabilities, "to_matrix_ops", {"S": OperatorProperties()})

        with pytest.raises(
            CompileError, match="Must use 'measure' from Catalyst instead of PennyLane."
        ):
            with EvaluationContext(EvaluationMode.QUANTUM_COMPILATION) as ctx:
                _ = catalyst_decompose(tape, ctx, capabilities)

    @pytest.mark.usefixtures("create_temporary_toml_file")
    @pytest.mark.parametrize("create_temporary_toml_file", [TEST_DEVICE_CONFIG_TEXT], indirect=True)
    def test_error_for_pennylane_midmeasure_decompose_nested(self, request):
        """Test that an error is raised in decompose if a PennyLane mid-circuit measurement
        is encountered"""

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(1.23, wires=0)
            qml.measure(0)

        ops, measurements = qml.queuing.process_queue(q)
        subtape = qml.tape.QuantumScript(ops, measurements)

        region = HybridOpRegion([], subtape, [], [])
        region.trace = None
        adjoint_op = HybridAdjoint([], [], [region])

        tape = qml.tape.QuantumScript([adjoint_op, qml.Y(1)], [])

        capabilities = DeviceCapabilities.from_toml_file(request.node.toml_file)
        setattr(capabilities, "to_matrix_ops", {"S": OperatorProperties()})

        with pytest.raises(
            CompileError, match="Must use 'measure' from Catalyst instead of PennyLane."
        ):
            with EvaluationContext(EvaluationMode.QUANTUM_COMPILATION) as ctx:
                _ = catalyst_decompose(tape, ctx, capabilities)

    @pytest.mark.usefixtures("create_temporary_toml_file")
    @pytest.mark.parametrize("create_temporary_toml_file", [TEST_DEVICE_CONFIG_TEXT], indirect=True)
    def test_unsupported_op_with_no_decomposition_raises_error(self, request):
        """Test that an unsupported operator that doesn't provide a decomposition
        raises a CompileError"""

        tape = qml.tape.QuantumScript([qml.Y(0)])

        capabilities = DeviceCapabilities.from_toml_file(request.node.toml_file)
        setattr(capabilities, "to_matrix_ops", {"S": OperatorProperties()})

        with pytest.raises(
            CompileError,
            match="not supported with catalyst on this device and does not provide a decomposition",
        ):
            with EvaluationContext(EvaluationMode.QUANTUM_COMPILATION) as ctx:
                _ = catalyst_decompose(tape, ctx, replace(capabilities, operations={}))

    @pytest.mark.usefixtures("create_temporary_toml_file")
    @pytest.mark.parametrize("create_temporary_toml_file", [TEST_DEVICE_CONFIG_TEXT], indirect=True)
    def test_decompose_to_matrix_raises_error(self, request):
        """Test that _decompose_to_matrix raises a CompileError if the operator has no matrix"""

        class NoMatrixMultiControlledX(qml.MultiControlledX):
            """A version of MulitControlledX with no matrix defined"""

            def matrix(self):
                """raise an error"""
                raise qml.operation.MatrixUndefinedError

        tape = qml.tape.QuantumScript([NoMatrixMultiControlledX(wires=[0, 1, 2, 3])])

        capabilities = DeviceCapabilities.from_toml_file(request.node.toml_file)
        setattr(capabilities, "to_matrix_ops", {"S": OperatorProperties()})

        with pytest.raises(CompileError, match="could not be decomposed, it might be unsupported"):
            with EvaluationContext(EvaluationMode.QUANTUM_COMPILATION) as ctx:
                _ = catalyst_decompose(
                    tape,
                    ctx,
                    replace(capabilities, operations={"QubitUnitary": OperatorProperties()}),
                )


if __name__ == "__main__":
    pytest.main(["-x", __file__])
