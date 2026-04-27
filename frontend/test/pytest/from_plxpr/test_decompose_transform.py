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

# pylint: disable=too-many-lines

from contextlib import nullcontext as does_not_raise
from functools import partial

import numpy as np
import pennylane as qp
import pytest
from jax.core import ShapedArray
from pennylane.exceptions import DecompositionError, DecompositionWarning
from pennylane.typing import TensorLike
from pennylane.wires import WiresLike
from pennylane_lightning.lightning_qubit.lightning_qubit import (
    stopping_condition as lightning_stopping_condition,
)

from catalyst.jax_primitives import decomposition_rule
from catalyst.passes import graph_decomposition


def _normalize_gate_types(gate_types):
    """
    TODO: Remove this function once PennyLane tape-based resource counting specs format
    is unified with the updated resource tracking specs format.

    Normalize gate type names by stripping NullQubit suffixes (e.g. 'PauliRot-Phi-w4')
    back to the base name ('PauliRot') and summing counts for matching base names.
    """
    result = {}
    for k, v in gate_types.items():
        base = k.split("-")[0]
        result[base] = result.get(base, 0) + v
    return result


class TestGraphDecomposition:
    """Test the graph-decomposition built-in transform."""

    def test_with_precompiled_rule(self):
        """Test graph-decomposition with precompiled rules are handled correctly."""

        @qp.qjit(capture=True)
        @graph_decomposition(gate_set=[qp.RX, qp.RY, qp.RZ])
        @qp.qnode(qp.device("lightning.qubit", wires=1))
        def circuit(x, y, z):
            qp.Rot(x, y, z, wires=0)
            return qp.expval(qp.Z(0))

        x = 0.5
        y = 0.3
        z = 0.2

        assert qp.math.allclose([0.9553364891256059], circuit(x, y, z))

        expected_resources = {"RY": 1, "RZ": 2}
        resources = qp.specs(circuit, level="device")(x, y, z)["resources"].gate_types
        assert resources == expected_resources

    def test_decompose_multi_qubit_gates_precompiled(self):
        """Test that multi-qubit gates are decomposed correctly using precompiled rules."""

        @qp.qjit(capture=True)
        @graph_decomposition(
            gate_set={"RY", "RX", "CNOT", "Hadamard", "GlobalPhase"},
        )
        @qp.qnode(qp.device("lightning.qubit", wires=4))
        def circuit():
            qp.SingleExcitation(0.5, wires=[0, 1])
            qp.SingleExcitationPlus(0.5, wires=[0, 1])
            qp.SingleExcitationMinus(0.5, wires=[0, 1])
            qp.DoubleExcitation(0.5, wires=[0, 1, 2, 3])
            return qp.expval(qp.Z(0))

        expected_resources = {"GlobalPhase": 6, "RX": 6, "RY": 30, "CNOT": 24, "Hadamard": 12}
        resources = qp.specs(circuit, level="device")()["resources"].gate_types
        assert resources == expected_resources

    def test_alt_decomps(self):
        """Test the conversion of a circuit with a custom decomposition."""

        @decomposition_rule(op_type=qp.CNOT)
        def my_cnot(wires):
            qp.H(wires=wires[1])
            qp.CZ(wires=wires)
            qp.H(wires=wires[1])

        @qp.qjit(capture=True)
        @graph_decomposition(
            gate_set={"H", "CZ", "GlobalPhase"},
            alt_decomps={qp.CNOT: [my_cnot]},
            _builtin_rule_path=None,
        )
        @qp.qnode(qp.device("lightning.qubit", wires=2))
        def circuit():
            qp.H(0)
            qp.CNOT(wires=[0, 1])

            # register custom decomposition rules
            my_cnot(ShapedArray((2,), int))

            return qp.state()

        expected_resources = {"CZ": 1, "Hadamard": 3}
        resources = qp.specs(circuit, level="device")()["resources"].gate_types
        assert resources == expected_resources

    def test_fixed_rules(self):
        """Test the decompose lowering pass with custom decomposition rules."""

        @decomposition_rule(op_type=qp.RY)
        def rz_rx(phi, wires: WiresLike, **__):
            """Decomposition of RY gate using RZ and RX gates."""
            qp.RZ(-np.pi / 2, wires=wires)
            qp.RX(phi, wires=wires)
            qp.RZ(np.pi / 2, wires=wires)

        @decomposition_rule(op_type=qp.Rot)
        def rz_ry_rz(phi, theta, omega, wires: WiresLike, **__):
            """Decomposition of Rot gate using RZ and RY gates."""
            qp.RZ(phi, wires=wires)
            qp.RY(theta, wires=wires)
            qp.RZ(omega, wires=wires)

        @decomposition_rule(op_type=qp.PauliY)
        def ry_gp(wires: WiresLike, **__):
            """Decomposition of PauliY gate using RY and GlobalPhase gates."""
            qp.RY(np.pi, wires=wires)
            qp.GlobalPhase(-np.pi / 2, wires=wires)

        @qp.qjit(capture=True)
        @graph_decomposition(
            gate_set={"RX", "RZ", "GlobalPhase"},
            fixed_decomps={
                qp.RY: rz_rx,
                qp.Rot: rz_ry_rz,
                qp.PauliY: ry_gp,
            },
            _builtin_rule_path=None,
        )
        @qp.qnode(qp.device("lightning.qubit", wires=3))
        def circuit():
            qp.RY(0.5, wires=0)
            qp.Rot(0.2, 0.3, 0.4, wires=1)
            qp.PauliY(wires=2)
            qp.Rot(0.2, 0.3, 0.4, wires=2)
            qp.RY(0.5, wires=1)

            # register custom decomposition rules
            rz_rx(float, int)
            rz_ry_rz(float, float, float, int)
            ry_gp(int)

            return qp.expval(qp.Z(0))

        expected_resources = {"GlobalPhase": 1, "RX": 5, "RZ": 14}
        resources = qp.specs(circuit, level="device")()["resources"].gate_types
        assert resources == expected_resources

    def test_multi_passes(self):
        """Test the graph_decomposition pass with other passes."""

        @qp.qjit(capture=True)
        @qp.transforms.merge_rotations
        @graph_decomposition(
            gate_set={"RZ", "RY", "CNOT", "GlobalPhase"},
        )
        @qp.transforms.cancel_inverses
        @qp.qnode(qp.device("lightning.qubit", wires=1))
        def circuit():
            qp.PauliX(0)
            qp.PauliX(0)
            qp.RX(0.1, wires=0)
            return qp.expval(qp.PauliX(0))

        expected_resources = {"RY": 1, "RZ": 2}
        resources = qp.specs(circuit, level="device")()["resources"].gate_types
        assert resources == expected_resources

    def test_multi_graph_decomposition(self):
        """Test that multiple graph-decomposition builtin transforms can be applied."""

        @decomposition_rule(op_type=qp.PauliX)
        def x_to_rx(wire: int):
            qp.RX(np.pi, wire)

        @decomposition_rule(op_type=qp.PauliY)
        def y_to_ry(wire: int):
            qp.RY(np.pi, wire)

        @decomposition_rule(op_type=qp.Hadamard)
        def h_to_rx_ry(wire: int):
            qp.RX(np.pi / 2, wire)
            qp.RY(np.pi / 2, wire)

        @qp.qjit(capture=True)
        @graph_decomposition(gate_set={qp.Rot})
        @qp.transforms.merge_rotations
        @graph_decomposition(
            gate_set={qp.RX, qp.RY},
            fixed_decomps={qp.PauliX: x_to_rx, qp.PauliY: y_to_ry},
            alt_decomps={qp.H: [h_to_rx_ry]},
        )
        @qp.transforms.cancel_inverses
        @qp.qnode(qp.device("lightning.qubit", wires=2))
        def circuit(x: float, y: float):
            qp.H(0)
            qp.H(0)
            qp.RX(x, wires=0)
            qp.PauliX(0)
            qp.RY(y, wires=0)
            qp.PauliY(0)
            qp.RY(x + y, wires=0)

            # register custom decomposition rules
            x_to_rx(int)
            y_to_ry(int)
            h_to_rx_ry(int)

            return qp.state()

        expected_resources = {"Rot": 2}
        resources = qp.specs(circuit, level="device")(1.23, 4.56)["resources"].gate_types
        assert resources == expected_resources

    @pytest.mark.xfail(
        reason="only quantum.custom gates are currently supported with graph_decomposition"
    )
    def test_multirz(self):
        """Test that TensorLike parameters in MultiRZ are handled correctly in rules."""

        @graph_decomposition(op_type="MultiRZ")
        def custom_multirz(params: TensorLike, wires: WiresLike, **__):
            qp.CNOT(wires=(wires[2], wires[1]))
            qp.CNOT(wires=(wires[1], wires[0]))
            qp.RZ(params, wires=wires[0])
            qp.CNOT(wires=(wires[1], wires[0]))
            qp.CNOT(wires=(wires[2], wires[1]))

        @qp.qjit(capture=True)
        @graph_decomposition(
            gate_set={qp.RY, qp.RX, qp.CNOT},
            fixed_decomps={qp.MultiRZ: custom_multirz},
        )
        @qp.qnode(qp.device("lightning.qubit", wires=3), shots=1000)
        def circuit(x, y):
            qp.MultiRZ(x + y, wires=[0, 1, 2])

            # register custom decomposition rules
            custom_multirz(TensorLike, [int, int, int])

            return qp.expval(qp.Z(0))

        expected_resources = {"RX": 1, "RY": 2, "CNOT": 4}
        resources = qp.specs(circuit, level="device")(0.5, 0.3)["resources"].gate_types
        assert resources == expected_resources

    def test_with_subroutine(self):
        """Test that decompositions can happen inside subroutines."""

        @qp.templates.Subroutine
        def f(x, wires):
            qp.IsingXX(x, wires)

        @qp.qjit(capture=True)
        @graph_decomposition(
            gate_set=qp.gate_sets.ROTATIONS_PLUS_CNOT,
        )
        @qp.qnode(qp.device("lightning.qubit", wires=5))
        def circuit():
            f(0.5, (0, 1))
            f(1.2, (2, 3))
            return qp.expval(qp.Z(0)), qp.expval(qp.Z(2))

        resources = qp.specs(circuit, level="device")().resources.gate_types
        assert resources == {"RX": 2, "CNOT": 4}

        r1, r2 = circuit()
        assert qp.math.allclose(r1, np.cos(0.5))
        assert qp.math.allclose(r2, np.cos(1.2))

    def test_ftqc_rotxzx(self):
        """Test qp.ftqc.RotXZX with alt_decomps."""

        @decomposition_rule(op_type="RotXZX")
        def _xzx_decompose(phi, theta, omega, wires, **__):
            qp.RX(phi, wires=wires)
            qp.RZ(theta, wires=wires)
            qp.RX(omega, wires=wires)

        @qp.qjit(capture=True)
        @graph_decomposition(
            gate_set={"CNOT", "GlobalPhase", "RX", "RZ", "PauliRot"},
            alt_decomps={qp.ftqc.RotXZX: [_xzx_decompose]},
        )
        @qp.qnode(qp.device("lightning.qubit", wires=2))
        def circuit():
            qp.ftqc.RotXZX(0.5, 0.3, 0.7, wires=0)

            _xzx_decompose(float, float, float, int)
            return qp.expval(qp.X(0))

        expected_resources = {"RX": 2, "RZ": 1}
        resources = qp.specs(circuit, level="device")()["resources"].gate_types
        assert resources == expected_resources

    def test_empty_rule(self):
        """Test that a decomposition rule with no ops is handled correctly."""

        @decomposition_rule(op_type="PauliX")
        def empty_decomp(_wire):
            pass

        @qp.qjit(capture=True)
        @graph_decomposition(
            gate_set={"PauliY"},
            fixed_decomps={"PauliX": empty_decomp},
        )
        @qp.qnode(qp.device("lightning.qubit", wires=1))
        def circuit():
            qp.X(0)
            qp.Y(0)

            # register the empty decomposition rule
            empty_decomp(int)

            return qp.expval(qp.Z(0))

        expected_resources = {"PauliY": 1}
        resources = qp.specs(circuit, level="device")()["resources"].gate_types
        assert resources == expected_resources

    @pytest.mark.xfail(
        reason="graph-decomposition supports pre-compiled rules, alt_decomps and fix_decomps"
    )
    def test_ftqc_custom_ops(self):
        """Test that ftqc Ops cannot be decomposed without defining rules."""

        @qp.qjit(capture=True)
        @graph_decomposition(
            gate_set={"CNOT", "GlobalPhase", "RX", "RZ", "PauliRot"},
        )
        @qp.qnode(qp.device("lightning.qubit", wires=2))
        def circuit():
            qp.ftqc.RotXZX(0.5, 0.3, 0.7, wires=0)
            return qp.expval(qp.X(0))

        expected_resources = {"RX": 2, "RZ": 1}
        resources = qp.specs(circuit, level="device")()["resources"].gate_types
        assert resources == expected_resources

    @pytest.mark.xfail(reason="graph-decomposition does not yet support adjoint or ctrl operations")
    def test_adjoint(self):
        """Test the graph_decomposition pass with adjoint operations."""

        @qp.qjit(capture=True)
        @graph_decomposition(
            gate_set={"RY", "RX", "CZ", "GlobalPhase"},
        )
        @qp.qnode(qp.device("lightning.qubit", wires=4))
        def circuit():
            qp.adjoint(qp.Hadamard(wires=2))
            qp.adjoint(qp.CNOT(wires=[0, 1]))
            qp.adjoint(qp.RX(0.5, wires=3))
            qp.adjoint(qp.Toffoli(wires=[0, 1, 2]))
            return qp.expval(qp.Z(0))

        expected_resources = {"GlobalPhase": 24, "CZ": 7, "RX": 25, "RY": 65}
        resources = qp.specs(circuit, level="device")()["resources"].gate_types
        assert resources == expected_resources

    @pytest.mark.xfail(reason="graph-decomposition does not yet support adjoint or ctrl operations")
    def test_ctrl(self):
        """Test the graph_decomposition pass with controlled operations."""

        @qp.qjit(capture=True)
        @graph_decomposition(
            gate_set={"RX", "RZ", "H", "CZ", "PauliRot"},
        )
        @qp.qnode(qp.device("lightning.qubit", wires=2))
        def circuit():
            qp.ctrl(qp.Hadamard(wires=1), 0)
            qp.ctrl(qp.RY, control=0)(0.5, 1)
            qp.ctrl(qp.PauliX, control=0)(1)
            return qp.expval(qp.Z(0))

        expected_resources = {"RX": 1, "RZ": 2, "H": 2, "CZ": 1}
        resources = qp.specs(circuit, level="device")()["resources"].gate_types
        assert resources == expected_resources

    @pytest.mark.xfail(reason="graph-decomposition does not yet support work wires")
    def test_work_wires(self):
        """Test that graph decomposition supports work_wires."""

        @decomposition_rule(op_type=qp.CRX)
        def my_decomp(angle, wires, **_):
            def true_func():
                qp.CNOT(wires)

                with qp.allocate(2, state="any", restored=True) as w:
                    qp.H(w[0])
                    qp.H(w[0])
                    qp.X(w[1])
                    qp.X(w[1])
                return

            def false_func():
                with qp.allocate(1, state="any", restored=False) as w:
                    qp.H(w)

                m = qp.measure(wires[0])
                qp.cond(m, qp.CNOT)(wires)
                return

            qp.cond(angle > 1.2, true_func, false_func)()

        @qp.qjit(capture=True)
        @graph_decomposition(
            gate_set={qp.CNOT, qp.H, qp.X, "Conditional", "MidMeasure"},
            fixed_decomps={qp.CRX: my_decomp},
            num_work_wires=7,
        )
        @qp.qnode(qp.device("lightning.qubit", wires=9))
        def circuit():
            qp.CRX(1.7, wires=[0, 1])
            qp.CRX(-7.2, wires=[0, 1])
            return qp.state()


class TestPlxPRDecomposition:
    """Test the PLxPR-based graph-based decomposition integration with from_plxpr."""

    @pytest.mark.usefixtures("use_capture_dgraph")
    def test_with_multiple_decomps_transforms(self):
        """Test that a circuit with multiple decompositions and transforms can be converted."""

        @qp.qjit(target="mlir")
        @partial(
            qp.transforms.decompose,
            gate_set={"RX", "RY"},
        )
        @partial(
            qp.transforms.decompose,
            gate_set={"NOT", "GlobalPhase"},
        )
        @qp.qnode(qp.device("lightning.qubit", wires=0))
        def circuit(x):
            qp.GlobalPhase(x)
            return qp.expval(qp.PauliX(0))

        with pytest.raises(
            NotImplementedError, match="Multiple decomposition transforms are not yet supported."
        ):
            circuit(0.2)

    @pytest.mark.usefixtures("use_capture_dgraph")
    def test_fallback_warnings(self):
        """Test the fallback to legacy decomposition system with warnings."""

        @qp.qjit
        @partial(qp.transforms.decompose, gate_set={qp.GlobalPhase})
        @qp.qnode(qp.device("lightning.qubit", wires=2))
        def circuit(x):
            qp.Hadamard(x)
            return qp.state()

        # TODO: RZ/RX warnings should not be raised, remove (PL issue #8885)
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
            qp.transforms.decompose,
            gate_set={"RX", "RY"},
        )
        @qp.qnode(qp.device("lightning.qubit", wires=2))
        def circuit():
            return qp.expval(qp.X(0))

        without_qjit = circuit()
        with_qjit = qp.qjit(circuit)

        assert qp.math.allclose(without_qjit, with_qjit())

        expected_resources = qp.specs(circuit, level="device")()["resources"].gate_types
        resources = qp.specs(with_qjit, level="device")()["resources"].gate_types
        assert _normalize_gate_types(resources) == _normalize_gate_types(expected_resources)

    @pytest.mark.usefixtures("use_capture_dgraph")
    def test_alt_decomps(self):
        """Test the conversion of a circuit with a custom decomposition."""

        @qp.register_resources({qp.H: 2, qp.CZ: 1})
        def my_cnot(wires):
            qp.H(wires=wires[1])
            qp.CZ(wires=wires)
            qp.H(wires=wires[1])

        @partial(
            qp.transforms.decompose,
            gate_set={"H", "CZ", "GlobalPhase"},
            alt_decomps={qp.CNOT: [my_cnot]},
        )
        @qp.qnode(qp.device("lightning.qubit", wires=2))
        def circuit():
            qp.H(0)
            qp.CNOT(wires=[0, 1])
            return qp.state()

        qjited_circuit = qp.qjit(circuit)

        expected = np.array([1, 0, 0, 1]) / np.sqrt(2)
        assert qp.math.allclose(qjited_circuit(), expected)

        expected_resources = qp.specs(circuit, level="device")()["resources"].gate_types
        resources = qp.specs(qjited_circuit, level="device")()["resources"].gate_types
        assert _normalize_gate_types(resources) == _normalize_gate_types(expected_resources)

    @pytest.mark.usefixtures("use_capture_dgraph")
    def test_fixed_rules(self):
        """Test the decompose lowering pass with custom decomposition rules."""

        @qp.register_resources({qp.RZ: 2, qp.RX: 1})
        def rz_rx(phi, wires: WiresLike, **__):
            """Decomposition of RY gate using RZ and RX gates."""
            qp.RZ(-np.pi / 2, wires=wires)
            qp.RX(phi, wires=wires)
            qp.RZ(np.pi / 2, wires=wires)

        @qp.register_resources({qp.RZ: 2, qp.RY: 1})
        def rz_ry_rz(phi, theta, omega, wires: WiresLike, **__):
            """Decomposition of Rot gate using RZ and RY gates."""
            qp.RZ(phi, wires=wires)
            qp.RY(theta, wires=wires)
            qp.RZ(omega, wires=wires)

        @qp.register_resources({qp.RY: 1, qp.GlobalPhase: 1})
        def ry_gp(wires: WiresLike, **__):
            """Decomposition of PauliY gate using RY and GlobalPhase gates."""
            qp.RY(np.pi, wires=wires)
            qp.GlobalPhase(-np.pi / 2, wires=wires)

        qp.decomposition.enable_graph()

        @partial(
            qp.transforms.decompose,
            gate_set={"RX", "RZ", "GlobalPhase"},
            fixed_decomps={
                qp.RY: rz_rx,
                qp.Rot: rz_ry_rz,
                qp.PauliY: ry_gp,
            },
        )
        @qp.qnode(qp.device("lightning.qubit", wires=3))
        def circuit():
            qp.RY(0.5, wires=0)
            qp.Rot(0.2, 0.3, 0.4, wires=1)
            qp.PauliY(wires=2)
            qp.Rot(0.2, 0.3, 0.4, wires=2)
            qp.RY(0.5, wires=1)
            qp.PauliX(wires=0)
            return qp.expval(qp.Z(0))

        without_qjit = circuit()
        with_qjit = qp.qjit(circuit)

        assert qp.math.allclose(without_qjit, with_qjit())

        expected_resources = qp.specs(circuit, level="device")()["resources"].gate_types
        resources = qp.specs(with_qjit, level="device")()["resources"].gate_types
        assert _normalize_gate_types(resources) == _normalize_gate_types(expected_resources)

    @pytest.mark.usefixtures("use_capture_dgraph")
    def test_tensorlike(self):
        """Test that TensorLike parameters are handled correctly in rules."""

        @qp.register_resources({qp.RZ: 1, qp.CNOT: 4})
        def custom_multirz(params: TensorLike, wires: WiresLike, **__):
            qp.CNOT(wires=(wires[2], wires[1]))
            qp.CNOT(wires=(wires[1], wires[0]))
            qp.RZ(params, wires=wires[0])
            qp.CNOT(wires=(wires[1], wires[0]))
            qp.CNOT(wires=(wires[2], wires[1]))

        @partial(
            qp.transforms.decompose,
            gate_set={"RY", "RX", qp.CNOT},
            fixed_decomps={qp.MultiRZ: custom_multirz},
        )
        @qp.qnode(qp.device("lightning.qubit", wires=3), shots=1000)
        def circuit(x, y):
            qp.MultiRZ(x + y, wires=[0, 1, 2])
            return qp.expval(qp.Z(0))

        x = 0.5
        y = 0.3

        without_qjit = circuit(x, y)
        with_qjit = qp.qjit(circuit)

        assert qp.math.allclose(without_qjit, with_qjit(x, y))
        expected_resources = qp.specs(circuit, level="device")(x, y)["resources"].gate_types
        resources = qp.specs(with_qjit, level="device")(x, y)["resources"].gate_types
        assert _normalize_gate_types(resources) == _normalize_gate_types(expected_resources)

    @pytest.mark.usefixtures("use_capture_dgraph")
    def test_inordered_params(self):
        """Test that unordered parameters in rules are handled correctly."""

        @partial(qp.transforms.decompose, gate_set=[qp.RX, qp.RY, qp.RZ])
        @qp.qnode(qp.device("lightning.qubit", wires=1))
        def circuit(x, y, z):
            qp.Rot(x, y, z, wires=0)
            return qp.expval(qp.Z(0))

        x = 0.5
        y = 0.3
        z = 0.2

        without_qjit = circuit(x, y, z)
        with_qjit = qp.qjit(circuit)

        assert qp.math.allclose(without_qjit, with_qjit(x, y, z))

        expected_resources = qp.specs(circuit, level="device")(x, y, z)["resources"].gate_types
        resources = qp.specs(with_qjit, level="device")(x, y, z)["resources"].gate_types
        assert _normalize_gate_types(resources) == _normalize_gate_types(expected_resources)

    @pytest.mark.usefixtures("use_capture_dgraph")
    def test_decompose_with_stopping_condition(self):
        """Test that decompose with stopping_condition uses plxpr decomposition correctly.

        When stopping_condition is passed to qp.transforms.decompose, from_plxpr uses
        the plxpr decompose path (no graph), passing stopping_condition to the transform.
        This test ensures that path compiles and produces correct results.
        """

        def stopping_condition(op):
            return op.name == "MultiRZ"

        @partial(
            qp.transforms.decompose,
            gate_set=[qp.RX, qp.RY, qp.RZ],
            stopping_condition=stopping_condition,
        )
        @qp.qnode(qp.device("lightning.qubit", wires=2))
        def circuit(x, y, z):
            qp.Rot(x, y, z, wires=0)
            qp.MultiRZ(0.5, wires=[0, 1])
            return qp.expval(qp.PauliZ(0))

        x, y, z = 0.5, 0.3, 0.2
        without_qjit = circuit(x, y, z)
        with_qjit = qp.qjit(circuit)
        assert qp.math.allclose(without_qjit, with_qjit(x, y, z))

        expected_resources = qp.specs(circuit, level="device")(x, y, z)["resources"].gate_types
        resources = qp.specs(with_qjit, level="device")(x, y, z)["resources"].gate_types
        assert "MultiRZ" in resources
        assert "MultiRZ" in expected_resources
        assert _normalize_gate_types(resources) == _normalize_gate_types(expected_resources)

    @pytest.mark.usefixtures("use_capture_dgraph")
    def test_decompose_with_lightning_stopping_condition(self):
        """Test that decompose with stopping_condition using Lightning's stopping condition."""

        device = qp.device("lightning.qubit", wires=4)

        @partial(
            qp.transforms.decompose,
            gate_set=[qp.CNOT, qp.PauliZ],
            stopping_condition=lightning_stopping_condition,
        )
        @qp.qnode(device)
        def circuit(x):
            qp.PauliRot(x, "XYZZ", wires=[0, 1, 2, 3])
            qp.StatePrep(np.array([1, 0, 0, 0]), wires=range(2))
            return qp.expval(qp.PauliZ(0))

        x = 0.5
        without_qjit = circuit(x)
        with_qjit = qp.qjit(circuit)
        assert qp.math.allclose(without_qjit, with_qjit(x))

        expected_resources = qp.specs(circuit, level="device")(x)["resources"].gate_types
        resources = qp.specs(with_qjit, level="device")(x)["resources"].gate_types
        assert any(k.startswith("PauliRot") for k in expected_resources)
        assert any(k.startswith("PauliRot") for k in resources)
        assert not any(k.startswith("StatePrep") for k in expected_resources)
        assert not any(k.startswith("StatePrep") for k in resources)
        assert _normalize_gate_types(resources) == _normalize_gate_types(expected_resources)

    @pytest.mark.skip(
        reason="inconsistent type and error msg across gcc/clang on arm/x86 for undefined symbols"
    )
    @pytest.mark.usefixtures("use_capture_dgraph")
    def test_gateset_with_rotxzx(self):
        """Test the runtime raises an error if RotXZX is not decomposed."""

        @partial(
            qp.transforms.decompose,
            gate_set={qp.ftqc.RotXZX},
        )
        @qp.qnode(qp.device("lightning.qubit", wires=2))
        def circuit():
            qp.ftqc.RotXZX(0.5, 0.3, 0.7, wires=0)
            return qp.expval(qp.X(0))

        with pytest.raises(
            OSError,
            match="undefined symbol",  # ___catalyst__qis__RotXZX
        ):
            qp.qjit(circuit)()

    @pytest.mark.usefixtures("use_capture_dgraph")
    def test_ftqc_rotxzx(self):
        """Test that FTQC RotXZX decomposition works with from_plxpr."""

        @partial(
            qp.transforms.decompose,
            gate_set={"CNOT", "GlobalPhase", "RX", "RZ", "PauliRot"},
        )
        @qp.qnode(qp.device("lightning.qubit", wires=2))
        def circuit():
            qp.ftqc.RotXZX(0.5, 0.3, 0.7, wires=0)
            qp.ctrl(qp.ftqc.RotXZX(0.4, 0.2, 0.6, wires=1), control=0)
            return qp.expval(qp.X(0))

        without_qjit = circuit()
        with_qjit = qp.qjit(circuit)

        assert qp.math.allclose(without_qjit, with_qjit())

        expected_resources = qp.specs(circuit, level="device")()["resources"].gate_types
        resources = qp.specs(with_qjit, level="device")()["resources"].gate_types
        assert _normalize_gate_types(resources) == _normalize_gate_types(expected_resources)

    @pytest.mark.xfail(reason="unstable global phase numbers", strict=False)
    @pytest.mark.usefixtures("use_capture_dgraph")
    def test_multirz(self):
        """Test that multirz decomposition works with from_plxpr."""

        @partial(
            qp.transforms.decompose,
            gate_set={"X", "Y", "Z", "S", "H", "CNOT", "RZ", "Rot", "GlobalPhase"},
        )
        @qp.qnode(qp.device("lightning.qubit", wires=4))
        def circuit():
            qp.Hadamard(0)
            qp.ctrl(qp.MultiRZ(0.345, wires=[1, 2]), control=0)
            qp.adjoint(qp.MultiRZ(0.25, wires=[1, 2]))
            qp.MultiRZ(0.5, wires=[0, 1])
            qp.MultiRZ(0.5, wires=[0])
            qp.MultiRZ(0.5, wires=[0, 1, 3])
            return qp.expval(qp.X(0))

        with_qjit = qp.qjit(circuit)
        result_with_qjit = with_qjit()
        resources = qp.specs(with_qjit, level="device")()["resources"].gate_types

        with qp.capture.pause():
            result_without_qjit = circuit()
            expected_resources = qp.specs(circuit, level="device")()["resources"].gate_types

        assert _normalize_gate_types(resources) == _normalize_gate_types(expected_resources)
        assert qp.math.allclose(result_without_qjit, result_with_qjit)

    @pytest.mark.usefixtures("use_capture_dgraph")
    def test_gphase(self):
        """Test that the decompose lowering pass works with GlobalPhase."""

        @partial(
            qp.transforms.decompose,
            gate_set={"RX", "RY", "GlobalPhase"},
        )
        @qp.qnode(qp.device("lightning.qubit", wires=1))
        def circuit():
            qp.GlobalPhase(0.5)
            qp.ctrl(qp.GlobalPhase, control=0)(0.3)
            qp.ctrl(qp.GlobalPhase, control=0)(phi=0.3, wires=[1, 2])
            return qp.expval(qp.Z(0))

        without_qjit = circuit()
        with_qjit = qp.qjit(circuit)

        assert qp.math.allclose(without_qjit, with_qjit())

        expected_resources = qp.specs(circuit, level="device")()["resources"].gate_types
        resources = qp.specs(with_qjit, level="device")()["resources"].gate_types
        assert _normalize_gate_types(resources) == _normalize_gate_types(expected_resources)

    @pytest.mark.usefixtures("use_capture_dgraph")
    def test_multi_qubits(self):
        """Test that the decompose lowering pass works with multi-qubit gates."""

        @partial(
            qp.transforms.decompose,
            gate_set={"RY", "RX", "CNOT", "Hadamard", "GlobalPhase"},
        )
        @qp.qnode(qp.device("lightning.qubit", wires=4))
        def circuit():
            qp.SingleExcitation(0.5, wires=[0, 1])
            qp.SingleExcitationPlus(0.5, wires=[0, 1])
            qp.SingleExcitationMinus(0.5, wires=[0, 1])
            qp.DoubleExcitation(0.5, wires=[0, 1, 2, 3])
            return qp.expval(qp.Z(0))

        without_qjit = circuit()
        with_qjit = qp.qjit(circuit)
        assert qp.math.allclose(without_qjit, with_qjit())

        expected_resources = qp.specs(circuit, level="device")()["resources"].gate_types
        resources = qp.specs(with_qjit, level="device")()["resources"].gate_types
        assert _normalize_gate_types(resources) == _normalize_gate_types(expected_resources)

    @pytest.mark.usefixtures("use_capture_dgraph")
    def test_adjoint(self):
        """Test the decompose lowering pass with adjoint operations."""

        @partial(
            qp.transforms.decompose,
            gate_set={"RY", "RX", "CZ", "GlobalPhase"},
        )
        @qp.qnode(qp.device("lightning.qubit", wires=4))
        def circuit():
            qp.adjoint(qp.Hadamard(wires=2))
            qp.adjoint(qp.CNOT(wires=[0, 1]))
            qp.adjoint(qp.RX(0.5, wires=3))
            qp.adjoint(qp.Toffoli(wires=[0, 1, 2]))
            return qp.expval(qp.Z(0))

        without_qjit = circuit()
        with_qjit = qp.qjit(circuit)

        assert qp.math.allclose(without_qjit, with_qjit())

        expected_resources = qp.specs(circuit, level="device")()["resources"].gate_types
        resources = qp.specs(with_qjit, level="device")()["resources"].gate_types
        assert _normalize_gate_types(resources) == _normalize_gate_types(expected_resources)

    @pytest.mark.usefixtures("use_capture_dgraph")
    def test_ctrl(self):
        """Test the decompose lowering pass with controlled operations."""

        @partial(
            qp.transforms.decompose,
            gate_set={"RX", "RZ", "H", "CZ", "PauliRot"},
        )
        @qp.qnode(qp.device("lightning.qubit", wires=2))
        def circuit():
            qp.ctrl(qp.Hadamard(wires=1), 0)
            qp.ctrl(qp.RY, control=0)(0.5, 1)
            qp.ctrl(qp.PauliX, control=0)(1)
            return qp.expval(qp.Z(0))

        without_qjit = circuit()
        with_qjit = qp.qjit(circuit)

        assert qp.math.allclose(without_qjit, with_qjit())

        expected_resources = qp.specs(circuit, level="device")()["resources"].gate_types
        resources = qp.specs(with_qjit, level="device")()["resources"].gate_types
        assert _normalize_gate_types(resources) == _normalize_gate_types(expected_resources)

    @pytest.mark.usefixtures("use_capture_dgraph")
    def test_template_qft(self):
        """Test the decompose lowering pass with the QFT template."""

        @partial(
            qp.transforms.decompose,
            gate_set={"RX", "RY", "CNOT", "GlobalPhase", "PauliRot"},
        )
        @qp.qnode(qp.device("lightning.qubit", wires=4))
        def circuit():
            qp.QFT(wires=[0, 1, 2, 3])
            return qp.expval(qp.Z(0))

        with_qjit = qp.qjit(circuit)
        result_with_qjit = with_qjit()
        resources = qp.specs(with_qjit, level="device")()["resources"].gate_types

        result_without_qjit = circuit()
        expected_resources = qp.specs(circuit, level="device")()["resources"].gate_types

        assert _normalize_gate_types(resources) == _normalize_gate_types(expected_resources)
        assert qp.math.allclose(result_without_qjit, result_with_qjit)

    @pytest.mark.usefixtures("use_capture_dgraph")
    def test_multi_passes(self):
        """Test the decompose lowering pass with multiple passes."""

        @qp.transforms.merge_rotations
        @qp.transforms.cancel_inverses
        @partial(
            qp.transforms.decompose,
            gate_set=frozenset({"RZ", "RY", "CNOT", "GlobalPhase"}),
        )
        @qp.qnode(qp.device("lightning.qubit", wires=1))
        def circuit():
            qp.PauliX(0)
            qp.PauliX(0)
            qp.RX(0.1, wires=0)
            return qp.expval(qp.PauliX(0))

        without_qjit = circuit()
        with_qjit = qp.qjit(circuit)

        assert qp.math.allclose(without_qjit, with_qjit())

        expected_resources = qp.specs(circuit, level="device")()["resources"].gate_types
        resources = qp.specs(with_qjit, level="device")()["resources"].gate_types
        assert _normalize_gate_types(resources) == _normalize_gate_types(expected_resources)

    @pytest.mark.usefixtures("use_capture_dgraph")
    @pytest.mark.parametrize(
        "num_work_wires,expectation",
        [
            (0, pytest.raises(DecompositionError)),
            (2, pytest.raises(DecompositionError)),
            (3, does_not_raise()),
            (7, does_not_raise()),
        ],
    )
    def test_work_wires(self, num_work_wires, expectation):
        """
        Test that graph decomposition raises the correct exception when given an insufficient
        number of work wires, and passes otherwise.
        """

        @qp.register_resources(
            {qp.CNOT: 3, qp.H: 1, qp.X: 1, qp.ops.op_math.Conditional: 2},
            work_wires={
                "borrowed": 2,
                "garbage": 1,
            },
        )
        def my_decomp(angle, wires, **_):
            def true_func():
                qp.CNOT(wires)

                with qp.allocate(2, state="any", restored=True) as w:
                    qp.H(w[0])
                    qp.H(w[0])
                    qp.X(w[1])
                    qp.X(w[1])

                return

            def false_func():
                with qp.allocate(1, state="any", restored=False) as w:
                    qp.H(w)

                m = qp.measure(wires[0])

                qp.cond(m, qp.CNOT)(wires)

                return

            qp.cond(angle > 1.2, true_func, false_func)()

        with expectation:

            @qp.qjit
            @partial(
                qp.transforms.decompose,
                gate_set={qp.CNOT, qp.H, qp.X, "Conditional", "MidMeasure"},
                fixed_decomps={qp.CRX: my_decomp},
                num_work_wires=num_work_wires,
            )
            @qp.qnode(qp.device("lightning.qubit", wires=9))
            def circuit():
                qp.CRX(1.7, wires=[0, 1])
                qp.CRX(-7.2, wires=[0, 1])
                return qp.state()

    def test_decomp_inside_subroutine(self):
        """Test that decompositions can happen inside subroutines."""

        qp.decomposition.enable_graph()

        @qp.templates.Subroutine
        def f(x, wires):
            qp.IsingXX(x, wires)

        @qp.qjit(capture=True)
        @qp.decompose(gate_set=qp.gate_sets.ROTATIONS_PLUS_CNOT)
        @qp.qnode(qp.device("lightning.qubit", wires=5))
        def c():
            f(0.5, (0, 1))
            f(1.2, (2, 3))
            return qp.expval(qp.Z(0)), qp.expval(qp.Z(2))

        resources = qp.specs(c, level="device")().resources.gate_types
        assert resources == {"RX": 2, "CNOT": 4}

        r1, r2 = c()
        assert qp.math.allclose(r1, np.cos(0.5))
        assert qp.math.allclose(r2, np.cos(1.2))

        qp.decomposition.disable_graph()


if __name__ == "__main__":
    pytest.main(["-x", __file__])
