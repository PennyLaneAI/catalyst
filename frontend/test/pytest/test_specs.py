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
"""Tests for qml.specs() Catalyst integration"""

from functools import partial

import pennylane as qml
import pytest
from jax import numpy as jnp
from pennylane.measurements import Shots
from pennylane.resource import CircuitSpecs, SpecsResources

import catalyst
from catalyst import qjit

# pylint:disable = protected-access,attribute-defined-outside-init


def check_specs_header_same(
    actual: CircuitSpecs, expected: CircuitSpecs, skip_level: bool = False
) -> None:
    """Check that two specs dictionaries are the same."""
    assert actual["device_name"] == expected["device_name"]
    assert actual["num_device_wires"] == expected["num_device_wires"]
    if not skip_level:
        assert actual["level"] == expected["level"]
    assert actual["shots"] == expected["shots"]


# TODO: Remove this method once feature parity has been reached, and instead use `==` directly
def check_specs_resources_same(
    actual_res: (
        SpecsResources | list[SpecsResources] | dict[any, SpecsResources | list[SpecsResources]]
    ),
    expected_res: (
        SpecsResources | list[SpecsResources] | dict[any, SpecsResources | list[SpecsResources]]
    ),
) -> None:
    """Helper function to check if 2 resources objects are the same"""
    assert type(actual_res) == type(expected_res)

    if isinstance(actual_res, list):
        assert len(actual_res) == len(expected_res)

        for r1, r2 in zip(actual_res, expected_res):
            check_specs_resources_same(r1, r2)

    elif isinstance(actual_res, dict):
        assert len(actual_res) == len(expected_res)

        for k in actual_res.keys():
            assert k in expected_res
            check_specs_resources_same(actual_res[k], expected_res[k])

    elif isinstance(actual_res, SpecsResources):
        assert actual_res.gate_types == expected_res.gate_types
        assert actual_res.gate_sizes == expected_res.gate_sizes

        assert actual_res.measurements == expected_res.measurements

        assert actual_res.num_allocs == expected_res.num_allocs
        assert actual_res.depth == expected_res.depth
        assert actual_res.num_gates == expected_res.num_gates

    else:
        raise ValueError("Invalid Type")


def check_specs_same(actual: CircuitSpecs, expected: CircuitSpecs):
    """Check that two specs dictionaries are the same."""
    check_specs_header_same(actual, expected)
    check_specs_resources_same(actual["resources"], expected["resources"])


class TestDeviceLevelSpecs:
    """Test qml.specs() at device level"""

    def test_simple(self):
        """Test a simple case of qml.specs() against PennyLane"""

        dev = qml.device("lightning.qubit", wires=1)

        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(wires=0)
            return qml.expval(qml.PauliZ(0))

        pl_specs = qml.specs(circuit, level="device")()
        cat_specs = qml.specs(qjit(circuit), level="device")()

        assert cat_specs["device_name"] == "lightning.qubit"
        check_specs_same(cat_specs, pl_specs)

    def test_complex(self):
        """Test a complex case of qml.specs() against PennyLane"""

        dev = qml.device("lightning.qubit", wires=4)
        U = 1 / jnp.sqrt(2) * jnp.array([[1, 1], [1, -1]], dtype=jnp.complex128)

        @qml.qnode(dev)
        def circuit():
            qml.PauliX(0)
            qml.adjoint(qml.T)(0)
            qml.ctrl(op=qml.S, control=[1], control_values=[1])(0)
            qml.ctrl(op=qml.S, control=[1, 2], control_values=[1, 0])(0)
            qml.ctrl(op=qml.adjoint(qml.Y), control=[2], control_values=[1])(0)
            qml.CNOT([0, 1])

            qml.QubitUnitary(U, wires=0)
            qml.ControlledQubitUnitary(U, control_values=[1], wires=[1, 0])
            qml.adjoint(qml.QubitUnitary(U, wires=0))
            qml.adjoint(qml.ControlledQubitUnitary(U, control_values=[1, 1], wires=[1, 2, 0]))

            return qml.probs()

        pl_specs = qml.specs(circuit, level="device")()
        cat_specs = qml.specs(qjit(circuit), level="device")()

        assert cat_specs["device_name"] == "lightning.qubit"

        # Catalyst will handle Adjoint(PauliY) == PauliY
        assert "CY" in cat_specs["resources"].gate_types
        cat_specs["resources"].gate_types["C(Adjoint(PauliY))"] = cat_specs["resources"].gate_types[
            "CY"
        ]
        del cat_specs["resources"].gate_types["CY"]

        check_specs_same(cat_specs, pl_specs)

    def test_measurements(self):
        """Test that measurements are tracked correctly at device level."""

        dev = qml.device("null.qubit", wires=3)

        @qml.set_shots(1)
        @qml.qnode(dev)
        def circuit():
            return (
                qml.expval(qml.PauliX(0)),
                qml.expval(qml.PauliZ(0)),
                qml.expval(qml.PauliZ(1)),
                qml.probs(),
                qml.probs(wires=[0]),
                qml.sample(),
                qml.counts(),
                qml.counts(wires=[1]),
            )

        pl_specs = qml.specs(circuit, level="device")()
        cat_specs = qml.specs(qjit(circuit), level="device")()

        check_specs_same(cat_specs, pl_specs)

        @qml.qnode(dev)
        def circuit_complex():
            coeffs = [0.2, -0.543]
            obs = [qml.X(0) @ qml.Z(1), qml.Z(0) @ qml.Hadamard(2)]
            ham = qml.ops.LinearCombination(coeffs, obs)
            return (
                qml.expval(qml.PauliZ(0) @ qml.PauliX(1)),
                qml.expval(ham),
                qml.state(),
                qml.var(qml.PauliX(0) @ qml.PauliY(1) @ qml.PauliZ(2)),
            )

        complex_meas_specs = qml.specs(qjit(circuit_complex), level="device")()
        expected_measurements = {
            "expval(Prod(num_terms=2))": 1,
            "expval(Hamiltonian(num_terms=2))": 1,
            "state(all wires)": 1,
            "var(Prod(num_terms=3))": 1,
        }
        assert complex_meas_specs["resources"].measurements == expected_measurements


class TestPassByPassSpecs:
    """Test qml.specs() pass-by-pass specs"""

    @pytest.fixture
    def simple_circuit(self):
        """Fixture for a circuit."""

        @qml.qnode(qml.device("lightning.qubit", wires=2))
        def circ():
            qml.RX(1.0, 0)
            qml.RX(2.0, 0)
            qml.RZ(3.0, 1)
            qml.RZ(4.0, 1)
            qml.Hadamard(0)
            qml.Hadamard(0)
            qml.CNOT([0, 1])
            qml.CNOT([0, 1])
            return qml.probs()

        return circ

    @pytest.mark.usefixtures("use_both_frontend")
    def test_invalid_levels(self, simple_circuit):
        """Test invalid inputs."""

        no_passes = qjit(simple_circuit)
        with pytest.raises(
            check=ValueError,
            match="The 'level' argument to qml.specs for QJIT'd QNodes must be "
            "non-negative, got -1.",
        ):
            qml.specs(no_passes, level=-1)()

        with pytest.raises(check=ValueError, match="Requested specs levels 2"):
            qml.specs(no_passes, level=2)()

        with pytest.raises(check=ValueError, match="Requested specs levels 2, 3"):
            qml.specs(no_passes, level=[2, 3])()

    @pytest.mark.usefixtures("use_both_frontend")
    def test_basic_passes_multi_level(self, simple_circuit):
        """Test that when passes are applied, the circuit resources are updated accordingly."""

        simple_circuit = qml.transforms.cancel_inverses(simple_circuit)
        simple_circuit = qml.transforms.merge_rotations(simple_circuit)

        simple_circuit = qjit(simple_circuit)

        expected = CircuitSpecs(
            device_name="lightning.qubit",
            num_device_wires=2,
            shots=Shots(None),
            level=dict(
                enumerate(
                    (
                        "Before MLIR Passes (MLIR-0)",
                        "cancel-inverses (MLIR-1)",
                        "merge-rotations (MLIR-2)",
                    )
                )
            },
            resources={
                "Before MLIR Passes (MLIR-0)": SpecsResources(
                    gate_types={"RX": 2, "RZ": 2, "Hadamard": 2, "CNOT": 2},
                    gate_sizes={1: 6, 2: 2},
                    measurements={"probs(all wires)": 1},
                    num_allocs=2,
                ),
                "cancel-inverses (MLIR-1)": SpecsResources(
                    gate_types={"RX": 2, "RZ": 2},
                    gate_sizes={1: 4},
                    measurements={"probs(all wires)": 1},
                    num_allocs=2,
                ),
                "merge-rotations (MLIR-2)": SpecsResources(
                    gate_types={"RX": 1, "RZ": 1},
                    gate_sizes={1: 2},
                    measurements={"probs(all wires)": 1},
                    num_allocs=2,
                ),
            },
        )

        actual = qml.specs(simple_circuit, level="all")()

        check_specs_same(actual, expected)

        # Test resources at each level match individual specs calls
        for i, res in enumerate(actual["resources"].values()):
            single_level_specs = qml.specs(simple_circuit, level=i)()
            check_specs_header_same(actual, single_level_specs, skip_level=True)
            check_specs_resources_same(res, single_level_specs["resources"])

    def test_basic_passes_multi_level_with_tapes(self, simple_circuit):
        """Test that when passes are applied, the circuit resources are updated accordingly."""

        @qml.transform
        def dummy_transform(tape):
            return (tape,), lambda res: res[0]

        simple_circuit = dummy_transform(simple_circuit)
        simple_circuit = dummy_transform(simple_circuit)

        simple_circuit = qml.transforms.cancel_inverses(simple_circuit)
        simple_circuit = qml.transforms.merge_rotations(simple_circuit)

        simple_circuit = qjit(simple_circuit)

        expected = CircuitSpecs(
            device_name="lightning.qubit",
            num_device_wires=2,
            shots=Shots(None),
            level={
                i: pass_name
                for i, pass_name in enumerate(
                    (
                        "Before transforms",
                        "dummy_transform",
                        "dummy_transform-2",
                        "Before MLIR Passes (MLIR-0)",
                        "cancel-inverses (MLIR-1)",
                        "merge-rotations (MLIR-2)",
                    )
                )
            ),
            resources={
                "Before transforms": SpecsResources(
                    gate_types={"RX": 2, "RZ": 2, "Hadamard": 2, "CNOT": 2},
                    gate_sizes={1: 6, 2: 2},
                    measurements={"probs(all wires)": 1},
                    num_allocs=2,
                ),
                "dummy_transform": SpecsResources(
                    gate_types={"RX": 2, "RZ": 2, "Hadamard": 2, "CNOT": 2},
                    gate_sizes={1: 6, 2: 2},
                    measurements={"probs(all wires)": 1},
                    num_allocs=2,
                ),
                "dummy_transform-2": SpecsResources(
                    gate_types={"RX": 2, "RZ": 2, "Hadamard": 2, "CNOT": 2},
                    gate_sizes={1: 6, 2: 2},
                    measurements={"probs(all wires)": 1},
                    num_allocs=2,
                ),
                "Before MLIR Passes (MLIR-0)": SpecsResources(
                    gate_types={"RX": 2, "RZ": 2, "Hadamard": 2, "CNOT": 2},
                    gate_sizes={1: 6, 2: 2},
                    measurements={"probs(all wires)": 1},
                    num_allocs=2,
                ),
                "cancel-inverses (MLIR-1)": SpecsResources(
                    gate_types={"RX": 2, "RZ": 2},
                    gate_sizes={1: 4},
                    measurements={"probs(all wires)": 1},
                    num_allocs=2,
                ),
                "merge-rotations (MLIR-2)": SpecsResources(
                    gate_types={"RX": 1, "RZ": 1},
                    gate_sizes={1: 2},
                    measurements={"probs(all wires)": 1},
                    num_allocs=2,
                ),
            },
        )

        actual = qml.specs(simple_circuit, level="all")()

        check_specs_same(actual, expected)

        # Test resources at each level match individual specs calls
        for i, res in enumerate(actual["resources"].values()):
            single_level_specs = qml.specs(simple_circuit, level=i)()
            check_specs_header_same(actual, single_level_specs, skip_level=True)
            check_specs_resources_same(res, single_level_specs["resources"])

    def test_mix_transforms_and_passes(self, simple_circuit):
        """Test using a mix of compiler passes and plain tape transforms"""

        simple_circuit = qml.transforms.cancel_inverses(
            simple_circuit
        )  # Has to be applied as a tape transform
        simple_circuit = qml.transforms.undo_swaps(
            simple_circuit
        )  # No actual swaps to undo, but forces normal tape transform
        simple_circuit = qml.transforms.merge_rotations(
            simple_circuit
        )  # Can be applied as an MLIR pass

        simple_circuit = qjit(simple_circuit)

        actual = qml.specs(simple_circuit, level="all")()
        expected = CircuitSpecs(
            device_name="lightning.qubit",
            num_device_wires=2,
            shots=Shots(None),
            level=dict(
                enumerate(
                    (
                        "Before transforms",
                        "cancel_inverses",
                        "undo_swaps",
                        "Before MLIR Passes (MLIR-0)",
                        "merge-rotations (MLIR-1)",
                    )
                )
            ),
            resources={
                "Before transforms": SpecsResources(
                    gate_types={"RX": 2, "RZ": 2, "Hadamard": 2, "CNOT": 2},
                    gate_sizes={1: 6, 2: 2},
                    measurements={"probs(all wires)": 1},
                    num_allocs=2,
                ),
                "cancel_inverses": SpecsResources(
                    gate_types={"RX": 2, "RZ": 2},
                    gate_sizes={1: 4},
                    measurements={"probs(all wires)": 1},
                    num_allocs=2,
                ),
                "undo_swaps": SpecsResources(
                    gate_types={"RX": 2, "RZ": 2},
                    gate_sizes={1: 4},
                    measurements={"probs(all wires)": 1},
                    num_allocs=2,
                ),
                "Before MLIR Passes (MLIR-0)": SpecsResources(
                    gate_types={"RX": 2, "RZ": 2},
                    gate_sizes={1: 4},
                    measurements={"probs(all wires)": 1},
                    num_allocs=2,
                ),
                "merge-rotations (MLIR-1)": SpecsResources(
                    gate_types={"RX": 1, "RZ": 1},
                    gate_sizes={1: 2},
                    measurements={"probs(all wires)": 1},
                    num_allocs=2,
                ),
            },
        )

        check_specs_same(actual, expected)

    @pytest.mark.usefixtures("use_both_frontend")
    def test_advanced_measurements(self):
        """Test that advanced measurements such as LinearCombination are handled correctly."""

        dev = qml.device("lightning.qubit", wires=7)

        @qml.qnode(dev, shots=10)
        def circ():
            coeffs = [0.2, -0.543]
            obs = [qml.X(0) @ qml.Z(1), qml.Z(0) @ qml.Hadamard(2)]
            ham = qml.ops.LinearCombination(coeffs, obs)

            return (
                qml.expval(ham),
                qml.expval(qml.PauliZ(0) @ qml.PauliZ(1)),
                qml.sample(wires=3),
                qml.sample(),
            )

        # Representations are slightly different from plain PL -- wire counts are missing
        info = qml.specs(qjit(circ), level=0, compute_depth=False)()

        assert info.resources.measurements == {
            "expval(Hamiltonian(num_terms=2))": 1,
            "expval(Prod(num_terms=2))": 1,
            "sample(1 wires)": 1,
            "sample(all wires)": 1,
        }

    def test_split_non_commuting(self):
        """Test that qml.transforms.split_non_commuting works as expected"""

        @qml.transforms.cancel_inverses
        @qml.transforms.split_non_commuting
        @qml.qnode(qml.device("null.qubit", wires=3))
        def circuit():
            qml.H(0)
            qml.X(0)
            qml.X(0)
            return qml.expval(qml.X(0)), qml.expval(qml.Y(0)), qml.expval(qml.Z(0))

        actual = qml.specs(qjit(circuit), level=1)()
        expected = CircuitSpecs(
            device_name="null.qubit",
            num_device_wires=3,
            shots=Shots(None),
            level="split_non_commuting",
            resources=[
                SpecsResources(
                    gate_types={"Hadamard": 1, "PauliX": 2},
                    gate_sizes={1: 3},
                    measurements={"expval(PauliX)": 1},
                    num_allocs=1,
                ),
                SpecsResources(
                    gate_types={"Hadamard": 1, "PauliX": 2},
                    gate_sizes={1: 3},
                    measurements={"expval(PauliY)": 1},
                    num_allocs=1,
                ),
                SpecsResources(
                    gate_types={"Hadamard": 1, "PauliX": 2},
                    gate_sizes={1: 3},
                    measurements={"expval(PauliZ)": 1},
                    num_allocs=1,
                ),
            ],
        )

        check_specs_same(actual, expected)

    @pytest.mark.usefixtures("use_capture")
    def test_subroutine(self):
        """Test qml.specs when there is a Catalyst subroutine"""
        dev = qml.device("lightning.qubit", wires=3)

        @qml.capture.subroutine
        def subroutine():
            qml.Hadamard(wires=0)

        @qml.qjit(autograph=True)
        @qml.qnode(dev)
        def circuit():
            for _ in range(3):
                subroutine()

            return qml.probs()

        actual = qml.specs(circuit, level=0)()
        expected = CircuitSpecs(
            device_name="lightning.qubit",
            num_device_wires=3,
            shots=Shots(None),
            level="Before MLIR Passes (MLIR-0)",
            resources=SpecsResources(
                gate_types={"Hadamard": 3},
                gate_sizes={1: 3},
                measurements={"probs(all wires)": 1},
                num_allocs=3,
            ),
        )

        check_specs_same(actual, expected)

    def test_ppr(self):
        """Test that PPRs are handled correctly."""

        pipeline = [("pipe", ["enforce-runtime-invariants-pipeline"])]

        @qml.qjit(pipelines=pipeline, target="mlir")
        @catalyst.passes.to_ppr
        @qml.qnode(qml.device("null.qubit", wires=2))
        def circ():
            qml.H(0)
            qml.T(0)

        expected = CircuitSpecs(
            device_name="null.qubit",
            num_device_wires=2,
            shots=Shots(None),
            level="to-ppr (MLIR-1)",
            resources=SpecsResources(
                gate_types={"GlobalPhase": 2, "PPR-pi/4-w1": 3, "PPR-pi/8-w1": 1},
                gate_sizes={0: 2, 1: 4},
                measurements={},
                num_allocs=2,
            ),
        )

        actual = qml.specs(circ, level=1)()
        check_specs_same(actual, expected)

    @pytest.mark.usefixtures("use_capture")
    def test_arbitrary_ppr(self):
        """Test that PPRs are handled correctly."""

        @qml.qjit(target="mlir")
        @qml.transforms.decompose_arbitrary_ppr
        @qml.transforms.to_ppr
        @qml.qnode(qml.device("null.qubit", wires=3))
        def circ():
            qml.PauliRot(0.1, pauli_word="XY", wires=[0, 1])

        expected = CircuitSpecs(
            device_name="null.qubit",
            num_device_wires=3,
            shots=Shots(None),
            level="decompose-arbitrary-ppr (MLIR-2)",
            resources=SpecsResources(
                gate_types={
                    "pbc.prepare": 1,
                    "PPM-w3": 1,
                    "PPM-w1": 1,
                    "PPR-pi/2-w1": 1,
                    "PPR-pi/2-w2": 1,
                    "PPR-Phi-w1": 1,
                },
                gate_sizes={1: 4, 2: 1, 3: 1},
                measurements={},
                num_allocs=4,
            ),
        )

        actual = qml.specs(circ, level=2)()
        check_specs_same(actual, expected)


class TestMarkerIntegration:
    """Tests the integration with qml.marker."""

    @pytest.fixture
    def simple_circuit(self):
        """Fixture for a circuit."""

        @qml.qnode(qml.device("lightning.qubit", wires=2))
        def circ():
            qml.RX(1.0, 0)
            qml.RX(2.0, 0)
            qml.RZ(3.0, 1)
            qml.RZ(4.0, 1)
            qml.Hadamard(0)
            qml.Hadamard(0)
            qml.CNOT([0, 1])
            qml.CNOT([0, 1])
            return qml.probs()

        return circ

    def test_marker_with_tape_and_mlir_transforms(self, simple_circuit):
        """Tests that markers can work with both tape and mlir transforms."""

        @qml.transform
        def dummy_transform(tape):
            return (tape,), lambda res: res[0]

        simple_circuit = qml.marker(simple_circuit, "before-transforms")
        simple_circuit = dummy_transform(simple_circuit)
        simple_circuit = dummy_transform(simple_circuit)
        simple_circuit = qml.marker(simple_circuit, "after-tape")
        # Completely relying on cancel inverses being used as an MLIR transform
        simple_circuit = qml.transforms.cancel_inverses(simple_circuit)
        simple_circuit = qml.transforms.cancel_inverses(simple_circuit)
        simple_circuit = qml.marker(simple_circuit, "after-mlir")

        assert len(simple_circuit.compile_pipeline.markers) == 3

        qjit_circuit = qml.qjit(simple_circuit)

        expected = CircuitSpecs(
            device_name="lightning.qubit",
            num_device_wires=2,
            shots=Shots(None),
            level={0: "before-transforms", 2: "after-tape", 5: "after-mlir"},
            resources={
                "before-transforms": SpecsResources(
                    gate_types={"RX": 2, "RZ": 2, "Hadamard": 2, "CNOT": 2},
                    gate_sizes={1: 6, 2: 2},
                    measurements={"probs(all wires)": 1},
                    num_allocs=2,
                ),
                "after-tape": SpecsResources(
                    gate_types={"RX": 2, "RZ": 2, "Hadamard": 2, "CNOT": 2},
                    gate_sizes={1: 6, 2: 2},
                    measurements={"probs(all wires)": 1},
                    num_allocs=2,
                ),
                "after-mlir": SpecsResources(
                    gate_types={"RX": 2, "RZ": 2},
                    gate_sizes={1: 4},
                    measurements={"probs(all wires)": 1},
                    num_allocs=2,
                ),
            },
        )

        actual = qml.specs(qjit_circuit, level=["before-transforms", "after-tape", "after-mlir"])()

        check_specs_same(actual, expected)

    def test_marker_with_tape_and_mlir_transforms_level_all(self, simple_circuit):
        """Tests that markers can work with both tape and mlir transforms when level is 'all'."""

        @qml.transform
        def dummy_transform(tape):
            return (tape,), lambda res: res[0]

        simple_circuit = qml.marker(simple_circuit, "before-transforms")
        simple_circuit = dummy_transform(simple_circuit)
        simple_circuit = dummy_transform(simple_circuit)
        simple_circuit = qml.marker(simple_circuit, "after-tape")
        # Completely relying on cancel inverses being used as an MLIR transform
        simple_circuit = qml.transforms.cancel_inverses(simple_circuit)
        simple_circuit = qml.transforms.cancel_inverses(simple_circuit)
        simple_circuit = qml.marker(simple_circuit, "after-mlir")

        assert len(simple_circuit.compile_pipeline.markers) == 3

        qjit_circuit = qml.qjit(simple_circuit)

        expected = CircuitSpecs(
            device_name="lightning.qubit",
            num_device_wires=2,
            shots=Shots(None),
            level=dict(
                enumerate(
                    (
                        "before-transforms",
                        "dummy_transform",
                        "after-tape",
                        "Before MLIR Passes (MLIR-0)",
                        "cancel-inverses (MLIR-1)",
                        "after-mlir",
                    )
                )
            ),
            resources={
                "before-transforms": SpecsResources(
                    gate_types={"RX": 2, "RZ": 2, "Hadamard": 2, "CNOT": 2},
                    gate_sizes={1: 6, 2: 2},
                    measurements={"probs(all wires)": 1},
                    num_allocs=2,
                ),
                "dummy_transform": SpecsResources(
                    gate_types={"RX": 2, "RZ": 2, "Hadamard": 2, "CNOT": 2},
                    gate_sizes={1: 6, 2: 2},
                    measurements={"probs(all wires)": 1},
                    num_allocs=2,
                ),
                "after-tape": SpecsResources(
                    gate_types={"RX": 2, "RZ": 2, "Hadamard": 2, "CNOT": 2},
                    gate_sizes={1: 6, 2: 2},
                    measurements={"probs(all wires)": 1},
                    num_allocs=2,
                ),
                "Before MLIR Passes (MLIR-0)": SpecsResources(
                    gate_types={"RX": 2, "RZ": 2, "Hadamard": 2, "CNOT": 2},
                    gate_sizes={1: 6, 2: 2},
                    measurements={"probs(all wires)": 1},
                    num_allocs=2,
                ),
                "cancel-inverses (MLIR-1)": SpecsResources(
                    gate_types={"RX": 2, "RZ": 2},
                    gate_sizes={1: 4},
                    measurements={"probs(all wires)": 1},
                    num_allocs=2,
                ),
                "after-mlir": SpecsResources(
                    gate_types={"RX": 2, "RZ": 2},
                    gate_sizes={1: 4},
                    measurements={"probs(all wires)": 1},
                    num_allocs=2,
                ),
            },
        )

        actual = qml.specs(qjit_circuit, level="all")()

        check_specs_same(actual, expected)

    def test_redundant_marker(self, simple_circuit):
        """Test that two markers on the same level generate the same specs."""

        simple_circuit = partial(qml.marker, label="m0")(simple_circuit)
        simple_circuit = qml.transforms.cancel_inverses(simple_circuit)
        simple_circuit = partial(qml.marker, label="m1")(simple_circuit)
        simple_circuit = partial(qml.marker, label="m1-duplicate")(simple_circuit)

        simple_circuit = qjit(simple_circuit)

        expected = CircuitSpecs(
            device_name="lightning.qubit",
            num_device_wires=2,
            shots=Shots(None),
            level={0: "m0", 1: "m1, m1-duplicate"},
            resources={
                "m0": SpecsResources(
                    gate_types={"RX": 2, "RZ": 2, "Hadamard": 2, "CNOT": 2},
                    gate_sizes={1: 6, 2: 2},
                    measurements={"probs(all wires)": 1},
                    num_allocs=2,
                ),
                "m1, m1-duplicate": SpecsResources(
                    gate_types={"RX": 2, "RZ": 2},
                    gate_sizes={1: 4},
                    measurements={"probs(all wires)": 1},
                    num_allocs=2,
                ),
            },
        )

        with pytest.warns(
            UserWarning,
            match="The 'level' argument to qml.specs for QJIT'd QNodes has been sorted to be "
            "in ascending order with no duplicate levels.",
        ):
            actual = qml.specs(simple_circuit, level=["m0", "m1", "m1-duplicate"])()

        check_specs_same(actual, expected)

    def test_marker(self, simple_circuit):
        """Test that qml.marker can be used appropriately."""

        simple_circuit = partial(qml.marker, label="m0")(simple_circuit)
        simple_circuit = qml.transforms.cancel_inverses(simple_circuit)
        simple_circuit = partial(qml.marker, label="m1")(simple_circuit)
        simple_circuit = qml.transforms.merge_rotations(simple_circuit)
        simple_circuit = partial(qml.marker, label="m2")(simple_circuit)

        simple_circuit = qjit(simple_circuit)

        expected = CircuitSpecs(
            device_name="lightning.qubit",
            num_device_wires=2,
            shots=Shots(None),
            level={0: "m0", 1: "m1", 2: "m2"},
            resources={
                "m0": SpecsResources(
                    gate_types={"RX": 2, "RZ": 2, "Hadamard": 2, "CNOT": 2},
                    gate_sizes={1: 6, 2: 2},
                    measurements={"probs(all wires)": 1},
                    num_allocs=2,
                ),
                "m1": SpecsResources(
                    gate_types={"RX": 2, "RZ": 2},
                    gate_sizes={1: 4},
                    measurements={"probs(all wires)": 1},
                    num_allocs=2,
                ),
                "m2": SpecsResources(
                    gate_types={"RX": 1, "RZ": 1},
                    gate_sizes={1: 2},
                    measurements={"probs(all wires)": 1},
                    num_allocs=2,
                ),
            },
        )

        actual = qml.specs(simple_circuit, level=["m0", "m1", "m2"])()

        check_specs_same(actual, expected)


if __name__ == "__main__":
    pytest.main(["-x", __file__])
