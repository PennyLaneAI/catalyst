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
"""
This module tests the decompose transformation.
"""

from functools import partial

import numpy as np
import pennylane as qml
import pytest
from pennylane.exceptions import DecompositionWarning
from pennylane.typing import TensorLike
from pennylane.wires import WiresLike


class TestGraphDecomposition:
    """Test the new graph-based decomposition integration with from_plxpr."""

    @pytest.mark.usefixtures("use_capture_dgraph")
    def test_with_multiple_decomps_transforms(self):
        """Test that a circuit with multiple decompositions and transforms can be converted."""

        @qml.qjit(target="mlir")
        @partial(
            qml.transforms.decompose,
            gate_set={"RX", "RY"},
        )
        @partial(
            qml.transforms.decompose,
            gate_set={"NOT", "GlobalPhase"},
        )
        @qml.qnode(qml.device("lightning.qubit", wires=0))
        def circuit(x):
            qml.GlobalPhase(x)
            return qml.expval(qml.PauliX(0))

        with pytest.raises(
            NotImplementedError, match="Multiple decomposition transforms are not yet supported."
        ):
            circuit(0.2)

    @pytest.mark.usefixtures("use_capture_dgraph")
    def test_fallback_warnings(self):
        """Test the fallback to legacy decomposition system with warnings."""

        @qml.qjit
        @partial(qml.transforms.decompose, gate_set={qml.GlobalPhase})
        @qml.qnode(qml.device("lightning.qubit", wires=2))
        def circuit(x):
            qml.Hadamard(x)
            return qml.state()

        # TODO: RZ/RX warnings  should not be raised, remove (PL issue #8885)
        with pytest.warns(UserWarning, match="Falling back to the legacy decomposition system"):
            with pytest.warns(
                DecompositionWarning, match="unable to find a decomposition for {'Hadamard'}"
            ):
                with pytest.warns(UserWarning, match="Operator RX does not define"):
                    with pytest.warns(UserWarning, match="Operator RZ does not define"):
                        circuit(0)

    @pytest.mark.usefixtures("use_capture_dgraph")
    def test_decompose_lowering_on_empty_circuit(self):
        """Test that the decompose lowering pass works on an empty circuit."""

        @partial(
            qml.transforms.decompose,
            gate_set={"RX", "RY"},
        )
        @qml.qnode(qml.device("lightning.qubit", wires=2))
        def circuit():
            return qml.expval(qml.X(0))

        without_qjit = circuit()
        with_qjit = qml.qjit(circuit)

        assert qml.math.allclose(without_qjit, with_qjit())

        expected_resources = qml.specs(circuit, level="device")()["resources"].gate_types
        resources = qml.specs(with_qjit, level="device")()["resources"].gate_types
        assert resources == expected_resources

    @pytest.mark.usefixtures("use_capture_dgraph")
    def test_alt_decomps(self):
        """Test the conversion of a circuit with a custom decomposition."""

        @qml.register_resources({qml.H: 2, qml.CZ: 1})
        def my_cnot(wires):
            qml.H(wires=wires[1])
            qml.CZ(wires=wires)
            qml.H(wires=wires[1])

        @partial(
            qml.transforms.decompose,
            gate_set={"H", "CZ", "GlobalPhase"},
            alt_decomps={qml.CNOT: [my_cnot]},
        )
        @qml.qnode(qml.device("lightning.qubit", wires=2))
        def circuit():
            qml.H(0)
            qml.CNOT(wires=[0, 1])
            return qml.state()

        qjited_circuit = qml.qjit(circuit)

        expected = np.array([1, 0, 0, 1]) / np.sqrt(2)
        assert qml.math.allclose(qjited_circuit(), expected)

        expected_resources = qml.specs(circuit, level="device")()["resources"].gate_types
        resources = qml.specs(qjited_circuit, level="device")()["resources"].gate_types
        assert resources == expected_resources

    @pytest.mark.usefixtures("use_capture_dgraph")
    def test_fixed_rules(self):
        """Test the decompose lowering pass with custom decomposition rules."""

        @qml.register_resources({qml.RZ: 2, qml.RX: 1})
        def rz_rx(phi, wires: WiresLike, **__):
            """Decomposition of RY gate using RZ and RX gates."""
            qml.RZ(-np.pi / 2, wires=wires)
            qml.RX(phi, wires=wires)
            qml.RZ(np.pi / 2, wires=wires)

        @qml.register_resources({qml.RZ: 2, qml.RY: 1})
        def rz_ry_rz(phi, theta, omega, wires: WiresLike, **__):
            """Decomposition of Rot gate using RZ and RY gates."""
            qml.RZ(phi, wires=wires)
            qml.RY(theta, wires=wires)
            qml.RZ(omega, wires=wires)

        @qml.register_resources({qml.RY: 1, qml.GlobalPhase: 1})
        def ry_gp(wires: WiresLike, **__):
            """Decomposition of PauliY gate using RY and GlobalPhase gates."""
            qml.RY(np.pi, wires=wires)
            qml.GlobalPhase(-np.pi / 2, wires=wires)

        qml.decomposition.enable_graph()

        @partial(
            qml.transforms.decompose,
            gate_set={"RX", "RZ", "GlobalPhase"},
            fixed_decomps={
                qml.RY: rz_rx,
                qml.Rot: rz_ry_rz,
                qml.PauliY: ry_gp,
            },
        )
        @qml.qnode(qml.device("lightning.qubit", wires=3))
        def circuit():
            qml.RY(0.5, wires=0)
            qml.Rot(0.2, 0.3, 0.4, wires=1)
            qml.PauliY(wires=2)
            qml.Rot(0.2, 0.3, 0.4, wires=2)
            qml.RY(0.5, wires=1)
            qml.PauliX(wires=0)
            return qml.expval(qml.Z(0))

        without_qjit = circuit()
        with_qjit = qml.qjit(circuit)

        assert qml.math.allclose(without_qjit, with_qjit())

        expected_resources = qml.specs(circuit, level="device")()["resources"].gate_types
        resources = qml.specs(with_qjit, level="device")()["resources"].gate_types
        assert resources == expected_resources

    @pytest.mark.usefixtures("use_capture_dgraph")
    def test_tensorlike(self):
        """Test that TensorLike parameters are handled correctly in rules."""

        @qml.register_resources({qml.RZ: 1, qml.CNOT: 4})
        def custom_multirz(params: TensorLike, wires: WiresLike, **__):
            qml.CNOT(wires=(wires[2], wires[1]))
            qml.CNOT(wires=(wires[1], wires[0]))
            qml.RZ(params, wires=wires[0])
            qml.CNOT(wires=(wires[1], wires[0]))
            qml.CNOT(wires=(wires[2], wires[1]))

        @partial(
            qml.transforms.decompose,
            gate_set={"RY", "RX", qml.CNOT},
            fixed_decomps={qml.MultiRZ: custom_multirz},
        )
        @qml.qnode(qml.device("lightning.qubit", wires=3), shots=1000)
        def circuit(x, y):
            qml.MultiRZ(x + y, wires=[0, 1, 2])
            return qml.expval(qml.Z(0))

        x = 0.5
        y = 0.3

        without_qjit = circuit(x, y)
        with_qjit = qml.qjit(circuit)

        assert qml.math.allclose(without_qjit, with_qjit(x, y))
        expected_resources = qml.specs(circuit, level="device")(x, y)["resources"].gate_types
        resources = qml.specs(with_qjit, level="device")(x, y)["resources"].gate_types
        assert resources == expected_resources

    @pytest.mark.usefixtures("use_capture_dgraph")
    def test_inordered_params(self):
        """Test that unordered parameters in rules are handled correctly."""

        @partial(qml.transforms.decompose, gate_set=[qml.RX, qml.RY, qml.RZ])
        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def circuit(x, y, z):
            qml.Rot(x, y, z, wires=0)
            return qml.expval(qml.Z(0))

        x = 0.5
        y = 0.3
        z = 0.2

        without_qjit = circuit(x, y, z)
        with_qjit = qml.qjit(circuit)

        assert qml.math.allclose(without_qjit, with_qjit(x, y, z))

        expected_resources = qml.specs(circuit, level="device")(x, y, z)["resources"].gate_types
        resources = qml.specs(with_qjit, level="device")(x, y, z)["resources"].gate_types
        assert resources == expected_resources

    @pytest.mark.skip(
        reason="inconsistent type and error msg across gcc/clang on arm/x86 for undefined symbols"
    )
    @pytest.mark.usefixtures("use_capture_dgraph")
    def test_gateset_with_rotxzx(self):
        """Test the runtime raises an error if RotXZX is not decomposed."""

        @partial(
            qml.transforms.decompose,
            gate_set={qml.ftqc.RotXZX},
        )
        @qml.qnode(qml.device("lightning.qubit", wires=2))
        def circuit():
            qml.ftqc.RotXZX(0.5, 0.3, 0.7, wires=0)
            return qml.expval(qml.X(0))

        with pytest.raises(
            OSError,
            match="undefined symbol",  # ___catalyst__qis__RotXZX
        ):
            qml.qjit(circuit)()

    @pytest.mark.usefixtures("use_capture_dgraph")
    def test_ftqc_rotxzx(self):
        """Test that FTQC RotXZX decomposition works with from_plxpr."""

        @partial(
            qml.transforms.decompose,
            gate_set={"CNOT", "GlobalPhase", "RX", "RZ", "PauliRot"},
        )
        @qml.qnode(qml.device("lightning.qubit", wires=2))
        def circuit():
            qml.ftqc.RotXZX(0.5, 0.3, 0.7, wires=0)
            qml.ctrl(qml.ftqc.RotXZX(0.4, 0.2, 0.6, wires=1), control=0)
            return qml.expval(qml.X(0))

        without_qjit = circuit()
        with_qjit = qml.qjit(circuit)

        assert qml.math.allclose(without_qjit, with_qjit())

        expected_resources = qml.specs(circuit, level="device")()["resources"].gate_types
        resources = qml.specs(with_qjit, level="device")()["resources"].gate_types
        assert resources == expected_resources

    @pytest.mark.xfail(reason="unstable global phase numbers", strict=False)
    @pytest.mark.usefixtures("use_capture_dgraph")
    def test_multirz(self):
        """Test that multirz decomposition works with from_plxpr."""

        @partial(
            qml.transforms.decompose,
            gate_set={"X", "Y", "Z", "S", "H", "CNOT", "RZ", "Rot", "GlobalPhase"},
        )
        @qml.qnode(qml.device("lightning.qubit", wires=4))
        def circuit():
            qml.Hadamard(0)
            qml.ctrl(qml.MultiRZ(0.345, wires=[1, 2]), control=0)
            qml.adjoint(qml.MultiRZ(0.25, wires=[1, 2]))
            qml.MultiRZ(0.5, wires=[0, 1])
            qml.MultiRZ(0.5, wires=[0])
            qml.MultiRZ(0.5, wires=[0, 1, 3])
            return qml.expval(qml.X(0))

        with_qjit = qml.qjit(circuit)
        result_with_qjit = with_qjit()
        resources = qml.specs(with_qjit, level="device")()["resources"].gate_types

        with qml.capture.pause():
            result_without_qjit = circuit()
            expected_resources = qml.specs(circuit, level="device")()["resources"].gate_types

        assert resources == expected_resources
        assert qml.math.allclose(result_without_qjit, result_with_qjit)

    @pytest.mark.usefixtures("use_capture_dgraph")
    def test_gphase(self):
        """Test that the decompose lowering pass works with GlobalPhase."""

        @partial(
            qml.transforms.decompose,
            gate_set={"RX", "RY", "GlobalPhase"},
        )
        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def circuit():
            qml.GlobalPhase(0.5)
            qml.ctrl(qml.GlobalPhase, control=0)(0.3)
            qml.ctrl(qml.GlobalPhase, control=0)(phi=0.3, wires=[1, 2])
            return qml.expval(qml.Z(0))

        without_qjit = circuit()
        with_qjit = qml.qjit(circuit)

        assert qml.math.allclose(without_qjit, with_qjit())

        expected_resources = qml.specs(circuit, level="device")()["resources"].gate_types
        resources = qml.specs(with_qjit, level="device")()["resources"].gate_types
        assert resources == expected_resources

    @pytest.mark.usefixtures("use_capture_dgraph")
    def test_multi_qubits(self):
        """Test that the decompose lowering pass works with multi-qubit gates."""

        @partial(
            qml.transforms.decompose,
            gate_set={"RY", "RX", "CNOT", "Hadamard", "GlobalPhase"},
        )
        @qml.qnode(qml.device("lightning.qubit", wires=4))
        def circuit():
            qml.SingleExcitation(0.5, wires=[0, 1])
            qml.SingleExcitationPlus(0.5, wires=[0, 1])
            qml.SingleExcitationMinus(0.5, wires=[0, 1])
            qml.DoubleExcitation(0.5, wires=[0, 1, 2, 3])
            return qml.expval(qml.Z(0))

        without_qjit = circuit()
        with_qjit = qml.qjit(circuit)
        assert qml.math.allclose(without_qjit, with_qjit())

        expected_resources = qml.specs(circuit, level="device")()["resources"].gate_types
        resources = qml.specs(with_qjit, level="device")()["resources"].gate_types
        assert resources == expected_resources

    @pytest.mark.usefixtures("use_capture_dgraph")
    def test_adjoint(self):
        """Test the decompose lowering pass with adjoint operations."""

        @partial(
            qml.transforms.decompose,
            gate_set={"RY", "RX", "CZ", "GlobalPhase"},
        )
        @qml.qnode(qml.device("lightning.qubit", wires=4))
        def circuit():
            qml.adjoint(qml.Hadamard(wires=2))
            qml.adjoint(qml.CNOT(wires=[0, 1]))
            qml.adjoint(qml.RX(0.5, wires=3))
            qml.adjoint(qml.Toffoli(wires=[0, 1, 2]))
            return qml.expval(qml.Z(0))

        without_qjit = circuit()
        with_qjit = qml.qjit(circuit)

        assert qml.math.allclose(without_qjit, with_qjit())

        expected_resources = qml.specs(circuit, level="device")()["resources"].gate_types
        resources = qml.specs(with_qjit, level="device")()["resources"].gate_types
        assert resources == expected_resources

    @pytest.mark.usefixtures("use_capture_dgraph")
    def test_ctrl(self):
        """Test the decompose lowering pass with controlled operations."""

        @partial(
            qml.transforms.decompose,
            gate_set={"RX", "RZ", "H", "CZ", "PauliRot"},
        )
        @qml.qnode(qml.device("lightning.qubit", wires=2))
        def circuit():
            qml.ctrl(qml.Hadamard(wires=1), 0)
            qml.ctrl(qml.RY, control=0)(0.5, 1)
            qml.ctrl(qml.PauliX, control=0)(1)
            return qml.expval(qml.Z(0))

        without_qjit = circuit()
        with_qjit = qml.qjit(circuit)

        assert qml.math.allclose(without_qjit, with_qjit())

        expected_resources = qml.specs(circuit, level="device")()["resources"].gate_types
        resources = qml.specs(with_qjit, level="device")()["resources"].gate_types
        assert resources == expected_resources

    @pytest.mark.usefixtures("use_capture_dgraph")
    def test_template_qft(self):
        """Test the decompose lowering pass with the QFT template."""

        @partial(
            qml.transforms.decompose,
            gate_set={"RX", "RY", "CNOT", "GlobalPhase", "PauliRot"},
        )
        @qml.qnode(qml.device("lightning.qubit", wires=4))
        def circuit():
            qml.QFT(wires=[0, 1, 2, 3])
            return qml.expval(qml.Z(0))

        with_qjit = qml.qjit(circuit)
        result_with_qjit = with_qjit()
        resources = qml.specs(with_qjit, level="device")()["resources"].gate_types

        result_without_qjit = circuit()
        expected_resources = qml.specs(circuit, level="device")()["resources"].gate_types

        assert resources == expected_resources
        assert qml.math.allclose(result_without_qjit, result_with_qjit)

    @pytest.mark.usefixtures("use_capture_dgraph")
    def test_multi_passes(self):
        """Test the decompose lowering pass with multiple passes."""

        @qml.transforms.merge_rotations
        @qml.transforms.cancel_inverses
        @partial(
            qml.transforms.decompose,
            gate_set=frozenset({"RZ", "RY", "CNOT", "GlobalPhase"}),
        )
        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def circuit():
            qml.PauliX(0)
            qml.PauliX(0)
            qml.RX(0.1, wires=0)
            return qml.expval(qml.PauliX(0))

        without_qjit = circuit()
        with_qjit = qml.qjit(circuit)

        assert qml.math.allclose(without_qjit, with_qjit())

        expected_resources = qml.specs(circuit, level="device")()["resources"].gate_types
        resources = qml.specs(with_qjit, level="device")()["resources"].gate_types
        assert resources == expected_resources


if __name__ == "__main__":
    pytest.main(["-x", __file__])
