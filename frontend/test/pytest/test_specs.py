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
"""Tests for qp.specs() Catalyst integration"""

from functools import partial

import pennylane as qp
import pytest
from jax import numpy as jnp
from pennylane.measurements import Shots
from pennylane.resource import CircuitSpecs, SpecsResources

import catalyst
from catalyst import qjit

# pylint:disable = protected-access,attribute-defined-outside-init,too-many-lines


@qp.transform
def dummy_transform(tape):
    """Returns a tape-only transform that can be used for testing"""
    return (tape,), lambda res: res[0]


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
    """Test qp.specs() at device level"""

    def test_with_passes(self, capture_mode):
        """Test that device-level specs count resources *after* all passes are applied"""

        dev = qp.device("lightning.qubit", wires=2)

        @qjit(capture=capture_mode)
        @qp.transforms.merge_rotations
        @qp.transforms.cancel_inverses
        @qp.qnode(dev)
        def circuit():
            qp.Hadamard(wires=0)
            qp.Hadamard(wires=0)
            qp.CNOT(wires=[0, 1])
            qp.CNOT(wires=[0, 1])
            qp.RX(1.2, wires=0)
            qp.RX(1.2, wires=0)
            return qp.expval(qp.PauliZ(0))

        cat_specs = qp.specs(circuit, level="device")()

        assert cat_specs.resources.num_gates == 1
        assert cat_specs.resources.gate_types == {"RX": 1}
        assert cat_specs.resources.gate_sizes == {1: 1}

    def test_simple(self):
        """Test a simple case of qp.specs() against PennyLane"""

        dev = qp.device("lightning.qubit", wires=1)

        @qp.qnode(dev)
        def circuit():
            qp.Hadamard(wires=0)
            return qp.expval(qp.PauliZ(0))

        pl_specs = qp.specs(circuit, level="device")()
        cat_specs = qp.specs(qjit(circuit), level="device")()

        assert cat_specs["device_name"] == "lightning.qubit"
        check_specs_same(cat_specs, pl_specs)

    def test_complex(self):
        """Test a complex case of qp.specs() against PennyLane"""

        dev = qp.device("lightning.qubit", wires=4)
        U = 1 / jnp.sqrt(2) * jnp.array([[1, 1], [1, -1]], dtype=jnp.complex128)

        @qp.qnode(dev)
        def circuit():
            qp.PauliX(0)
            qp.adjoint(qp.T)(0)
            qp.ctrl(op=qp.S, control=[1], control_values=[1])(0)
            qp.ctrl(op=qp.S, control=[1, 2], control_values=[1, 0])(0)
            qp.ctrl(op=qp.adjoint(qp.Y), control=[2], control_values=[1])(0)
            qp.CNOT([0, 1])

            qp.QubitUnitary(U, wires=0)
            qp.ControlledQubitUnitary(U, control_values=[1], wires=[1, 0])
            qp.adjoint(qp.QubitUnitary(U, wires=0))
            qp.adjoint(qp.ControlledQubitUnitary(U, control_values=[1, 1], wires=[1, 2, 0]))

            return qp.probs()

        pl_specs = qp.specs(circuit, level="device")()
        cat_specs = qp.specs(qjit(circuit), level="device")()

        assert cat_specs["device_name"] == "lightning.qubit"

        # Catalyst will handle Adjoint(PauliY) == PauliY
        assert "CY" in cat_specs["resources"].gate_types
        cat_specs["resources"].gate_types["C(Adjoint(PauliY))"] = cat_specs["resources"].gate_types[
            "CY"
        ]
        del cat_specs["resources"].gate_types["CY"]

        check_specs_same(cat_specs, pl_specs)

    def test_paulirot_and_measure(self):
        """Test that PauliRot and PauliMeasure are tracked at the device level."""

        dev = qp.device("null.qubit", wires=2)

        @qjit(capture=True)
        @qp.qnode(dev)
        def circuit():
            qp.PauliRot(0.42, pauli_word="Y", wires=0)  # arbitrary angle
            qp.PauliRot(jnp.pi / 2, pauli_word="YZ", wires=[0, 1])  # pi/2 angle
            qp.PauliRot(2 * jnp.pi, pauli_word="X", wires=0)  # identity
            qp.pauli_measure("X", wires=0)
            return qp.probs()

        cat_specs = qp.specs(circuit, level="device")()

        assert cat_specs.resources.num_gates == 4
        assert cat_specs.resources.gate_types == {
            "PauliRot-pi/2-w2": 1,
            "PauliRot-identity-w1": 1,
            "PauliRot-Phi-w1": 1,
            "PauliMeasure-w1": 1,
        }
        assert cat_specs.resources.gate_sizes == {1: 3, 2: 1}

    def test_measurements(self):
        """Test that measurements are tracked correctly at device level."""

        dev = qp.device("null.qubit", wires=3)

        @qp.set_shots(1)
        @qp.qnode(dev)
        def circuit():
            return (
                qp.expval(qp.PauliX(0)),
                qp.expval(qp.PauliZ(0)),
                qp.expval(qp.PauliZ(1)),
                qp.probs(),
                qp.probs(wires=[0]),
                qp.sample(),
                qp.counts(),
                qp.counts(wires=[1]),
            )

        pl_specs = qp.specs(circuit, level="device")()
        cat_specs = qp.specs(qjit(circuit), level="device")()

        check_specs_same(cat_specs, pl_specs)

        @qp.qnode(dev)
        def circuit_complex():
            coeffs = [0.2, -0.543]
            obs = [qp.X(0) @ qp.Z(1), qp.Z(0) @ qp.Hadamard(2)]
            ham = qp.ops.LinearCombination(coeffs, obs)
            return (
                qp.expval(qp.PauliZ(0) @ qp.PauliX(1)),
                qp.expval(ham),
                qp.state(),
                qp.var(qp.PauliX(0) @ qp.PauliY(1) @ qp.PauliZ(2)),
            )

        complex_meas_specs = qp.specs(qjit(circuit_complex), level="device")()
        expected_measurements = {
            "expval(Prod(num_terms=2))": 1,
            "expval(Hamiltonian(num_terms=2))": 1,
            "state(all wires)": 1,
            "var(Prod(num_terms=3))": 1,
        }
        assert complex_meas_specs["resources"].measurements == expected_measurements


class TestPassByPassSpecs:
    """Test qp.specs() pass-by-pass specs"""

    @pytest.fixture
    def simple_circuit(self):
        """Fixture for a circuit."""

        @qp.qnode(qp.device("lightning.qubit", wires=2))
        def circ():
            qp.RX(1.0, 0)
            qp.RX(2.0, 0)
            qp.RZ(3.0, 1)
            qp.RZ(4.0, 1)
            qp.Hadamard(0)
            qp.Hadamard(0)
            qp.CNOT([0, 1])
            qp.CNOT([0, 1])
            return qp.probs()

        return circ

    def test_invalid_levels(self, simple_circuit, capture_mode):
        """Test invalid inputs."""

        no_passes = qjit(simple_circuit, capture=capture_mode)
        with pytest.raises(
            check=ValueError,
            match=r"The 'level' argument to .*\.specs for QJIT'd QNodes is out of "
            "bounds, got -5.",
        ):
            qp.specs(no_passes, level=-5)()

        with pytest.raises(
            check=ValueError,
            match=r"The 'level' argument to .*\.specs for QJIT'd "
            "QNodes is out of bounds, got 10.",
        ):
            qp.specs(no_passes, level=10)()

        with pytest.raises(
            check=ValueError,
            match=r"The 'level' argument to .*\.specs for QJIT'd "
            "QNodes is out of bounds, got 10.",
        ):
            qp.specs(no_passes, level=[10, 11])()

    def test_basic_passes_multi_level(self, simple_circuit, capture_mode):
        """Test that when passes are applied, the circuit resources are updated accordingly."""

        simple_circuit = qp.transforms.cancel_inverses(simple_circuit)
        simple_circuit = qp.transforms.merge_rotations(simple_circuit)

        simple_circuit = qjit(simple_circuit, capture=capture_mode)

        expected = CircuitSpecs(
            device_name="lightning.qubit",
            num_device_wires=2,
            shots=Shots(None),
            level=dict(
                enumerate(
                    (
                        "Before MLIR Passes",
                        "cancel-inverses",
                        "merge-rotations",
                    )
                )
            ),
            resources={
                "Before MLIR Passes": SpecsResources(
                    gate_types={"RX": 2, "RZ": 2, "Hadamard": 2, "CNOT": 2},
                    gate_sizes={1: 6, 2: 2},
                    measurements={"probs(all wires)": 1},
                    num_allocs=2,
                ),
                "cancel-inverses": SpecsResources(
                    gate_types={"RX": 2, "RZ": 2},
                    gate_sizes={1: 4},
                    measurements={"probs(all wires)": 1},
                    num_allocs=2,
                ),
                "merge-rotations": SpecsResources(
                    gate_types={"RX": 1, "RZ": 1},
                    gate_sizes={1: 2},
                    measurements={"probs(all wires)": 1},
                    num_allocs=2,
                ),
            },
        )

        actual = qp.specs(simple_circuit, level="all")()

        check_specs_same(actual, expected)

        # Test resources at each level match individual specs calls
        for i, res in enumerate(actual["resources"].values()):
            single_level_specs = qp.specs(simple_circuit, level=i)()
            check_specs_header_same(actual, single_level_specs, skip_level=True)
            check_specs_resources_same(res, single_level_specs["resources"])

    def test_user_level(self, simple_circuit, capture_mode):
        """Test that 'user' level is handled correctly."""

        simple_circuit = qp.transform(pass_name="cancel-inverses")(simple_circuit)
        simple_circuit = qp.transform(pass_name="merge-rotations")(simple_circuit)
        simple_circuit = qp.qjit(simple_circuit, capture=capture_mode)

        specs = qp.specs(simple_circuit, level="user")()
        assert specs.level == "merge-rotations"
        assert specs.resources == SpecsResources(
            gate_types={"RX": 1, "RZ": 1},
            gate_sizes={1: 2},
            measurements={"probs(all wires)": 1},
            num_allocs=2,
        )

    def test_user_level_with_tapes(self, simple_circuit):
        """Test that 'user' level is handled correctly with tape transforms."""

        simple_circuit = qp.transforms.cancel_inverses(simple_circuit)
        simple_circuit = dummy_transform(simple_circuit)  # Force tape transform
        simple_circuit = qp.transform(pass_name="merge-rotations")(simple_circuit)
        simple_circuit = qp.qjit(simple_circuit)

        specs = qp.specs(simple_circuit, level="user")()
        assert specs.level == "merge-rotations"
        assert specs.resources == SpecsResources(
            gate_types={"RX": 1, "RZ": 1},
            gate_sizes={1: 2},
            measurements={"probs(all wires)": 1},
            num_allocs=2,
        )

    def test_duplicate_level_names(self, simple_circuit):
        """Test that duplicate pass names are handled gracefully."""

        # TODO: At some point the names for the tape transform and MLIR pass will be unified
        # Once this happens, this test will need to be updated
        simple_circuit = qp.transforms.cancel_inverses(simple_circuit)
        simple_circuit = dummy_transform(simple_circuit)
        simple_circuit = qp.transform(pass_name="cancel-inverses")(simple_circuit)
        simple_circuit = qp.transform(pass_name="cancel-inverses")(simple_circuit)

        simple_circuit = qjit(simple_circuit)

        canceled_res = SpecsResources(
            gate_types={"RX": 2, "RZ": 2},
            gate_sizes={1: 4},
            measurements={"probs(all wires)": 1},
            num_allocs=2,
        )

        expected = CircuitSpecs(
            device_name="lightning.qubit",
            num_device_wires=2,
            shots=Shots(None),
            level=dict(
                enumerate(
                    (
                        "Before Tape Transforms",
                        "cancel_inverses",
                        "dummy_transform",
                        "Before MLIR Passes",
                        "cancel-inverses",
                        "cancel-inverses-2",
                    )
                )
            ),
            resources={
                "Before Tape Transforms": SpecsResources(
                    gate_types={"RX": 2, "RZ": 2, "Hadamard": 2, "CNOT": 2},
                    gate_sizes={1: 6, 2: 2},
                    measurements={"probs(all wires)": 1},
                    num_allocs=2,
                ),
                "cancel_inverses": canceled_res,
                "dummy_transform": canceled_res,
                "Before MLIR Passes": canceled_res,
                "cancel-inverses": canceled_res,
                "cancel-inverses-2": canceled_res,
            },
        )

        actual = qp.specs(simple_circuit, level="all")()

        check_specs_same(actual, expected)

        # Test resources at each level match individual specs calls
        for i, res in enumerate(actual["resources"].values()):
            single_level_specs = qp.specs(simple_circuit, level=i)()
            check_specs_header_same(actual, single_level_specs, skip_level=True)
            check_specs_resources_same(res, single_level_specs["resources"])

    def test_basic_passes_multi_level_with_tapes(self, simple_circuit):
        """Test that when passes are applied, the circuit resources are updated accordingly."""

        simple_circuit = dummy_transform(simple_circuit)
        simple_circuit = dummy_transform(simple_circuit)

        simple_circuit = qp.transforms.cancel_inverses(simple_circuit)
        simple_circuit = qp.transforms.merge_rotations(simple_circuit)

        simple_circuit = qjit(simple_circuit)

        expected = CircuitSpecs(
            device_name="lightning.qubit",
            num_device_wires=2,
            shots=Shots(None),
            level=dict(
                enumerate(
                    (
                        "Before Tape Transforms",
                        "dummy_transform",
                        "dummy_transform-2",
                        "Before MLIR Passes",
                        "cancel-inverses",
                        "merge-rotations",
                    )
                )
            ),
            resources={
                "Before Tape Transforms": SpecsResources(
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
                "Before MLIR Passes": SpecsResources(
                    gate_types={"RX": 2, "RZ": 2, "Hadamard": 2, "CNOT": 2},
                    gate_sizes={1: 6, 2: 2},
                    measurements={"probs(all wires)": 1},
                    num_allocs=2,
                ),
                "cancel-inverses": SpecsResources(
                    gate_types={"RX": 2, "RZ": 2},
                    gate_sizes={1: 4},
                    measurements={"probs(all wires)": 1},
                    num_allocs=2,
                ),
                "merge-rotations": SpecsResources(
                    gate_types={"RX": 1, "RZ": 1},
                    gate_sizes={1: 2},
                    measurements={"probs(all wires)": 1},
                    num_allocs=2,
                ),
            },
        )

        actual = qp.specs(simple_circuit, level="all")()

        check_specs_same(actual, expected)

        # Test resources at each level match individual specs calls
        for i, res in enumerate(actual["resources"].values()):
            single_level_specs = qp.specs(simple_circuit, level=i)()
            check_specs_header_same(actual, single_level_specs, skip_level=True)
            check_specs_resources_same(res, single_level_specs["resources"])

    def test_mix_transforms_and_passes(self, simple_circuit):
        """Test using a mix of compiler passes and plain tape transforms"""

        simple_circuit = qp.transforms.cancel_inverses(
            simple_circuit
        )  # Has to be applied as a tape transform because of the next transform
        simple_circuit = dummy_transform(simple_circuit)  # Forces normal tape transform
        simple_circuit = qp.transforms.merge_rotations(
            simple_circuit
        )  # Can be applied as an MLIR pass

        simple_circuit = qjit(simple_circuit)

        actual = qp.specs(simple_circuit, level="all")()
        expected = CircuitSpecs(
            device_name="lightning.qubit",
            num_device_wires=2,
            shots=Shots(None),
            level=dict(
                enumerate(
                    (
                        "Before Tape Transforms",
                        "cancel_inverses",
                        "dummy_transform",
                        "Before MLIR Passes",
                        "merge-rotations",
                    )
                )
            ),
            resources={
                "Before Tape Transforms": SpecsResources(
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
                "dummy_transform": SpecsResources(
                    gate_types={"RX": 2, "RZ": 2},
                    gate_sizes={1: 4},
                    measurements={"probs(all wires)": 1},
                    num_allocs=2,
                ),
                "Before MLIR Passes": SpecsResources(
                    gate_types={"RX": 2, "RZ": 2},
                    gate_sizes={1: 4},
                    measurements={"probs(all wires)": 1},
                    num_allocs=2,
                ),
                "merge-rotations": SpecsResources(
                    gate_types={"RX": 1, "RZ": 1},
                    gate_sizes={1: 2},
                    measurements={"probs(all wires)": 1},
                    num_allocs=2,
                ),
            },
        )

        check_specs_same(actual, expected)

    def test_circuit_with_args(self):
        """Test using a mix of compiler passes and plain tape transforms"""

        @qp.qnode(qp.device("lightning.qubit", wires=2))
        def circ(x):
            qp.RX(x * 1.0, 0)
            qp.RX(x * 2.0, 0)
            qp.RZ(x * 3.0, 1)
            qp.RZ(x * 4.0, 1)
            qp.Hadamard(0)
            qp.Hadamard(0)
            qp.CNOT([0, 1])
            qp.CNOT([0, 1])
            return qp.probs()

        circ = qp.transforms.cancel_inverses(
            circ
        )  # Has to be applied as a tape transform because of the next transform
        circ = dummy_transform(circ)  # Forces normal tape transform
        circ = qp.transforms.merge_rotations(circ)  # Can be applied as an MLIR pass

        circ = qjit(circ)

        actual = qp.specs(circ, level="all")(3)
        expected = CircuitSpecs(
            device_name="lightning.qubit",
            num_device_wires=2,
            shots=Shots(None),
            level=dict(
                enumerate(
                    (
                        "Before Tape Transforms",
                        "cancel_inverses",
                        "dummy_transform",
                        "Before MLIR Passes",
                        "merge-rotations",
                    )
                )
            ),
            resources={
                "Before Tape Transforms": SpecsResources(
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
                "dummy_transform": SpecsResources(
                    gate_types={"RX": 2, "RZ": 2},
                    gate_sizes={1: 4},
                    measurements={"probs(all wires)": 1},
                    num_allocs=2,
                ),
                "Before MLIR Passes": SpecsResources(
                    gate_types={"RX": 2, "RZ": 2},
                    gate_sizes={1: 4},
                    measurements={"probs(all wires)": 1},
                    num_allocs=2,
                ),
                "merge-rotations": SpecsResources(
                    gate_types={"RX": 1, "RZ": 1},
                    gate_sizes={1: 2},
                    measurements={"probs(all wires)": 1},
                    num_allocs=2,
                ),
            },
        )

        check_specs_same(actual, expected)

    def test_all_mlir(self, simple_circuit):
        """Test using "all-mlir" level"""

        simple_circuit = qp.transforms.cancel_inverses(simple_circuit)
        simple_circuit = qp.transforms.merge_rotations(
            simple_circuit
        )  # Can be applied as an MLIR pass

        simple_circuit = qjit(simple_circuit)

        actual = qp.specs(simple_circuit, level="all-mlir")()
        expected = CircuitSpecs(
            device_name="lightning.qubit",
            num_device_wires=2,
            shots=Shots(None),
            level={
                0: "Before MLIR Passes",
                1: "cancel-inverses",
                2: "merge-rotations",
            },
            resources={
                "Before MLIR Passes": SpecsResources(
                    gate_types={"RX": 2, "RZ": 2, "Hadamard": 2, "CNOT": 2},
                    gate_sizes={1: 6, 2: 2},
                    measurements={"probs(all wires)": 1},
                    num_allocs=2,
                ),
                "cancel-inverses": SpecsResources(
                    gate_types={"RX": 2, "RZ": 2},
                    gate_sizes={1: 4},
                    measurements={"probs(all wires)": 1},
                    num_allocs=2,
                ),
                "merge-rotations": SpecsResources(
                    gate_types={"RX": 1, "RZ": 1},
                    gate_sizes={1: 2},
                    measurements={"probs(all wires)": 1},
                    num_allocs=2,
                ),
            },
        )

        check_specs_same(actual, expected)

    def test_all_mlir_with_tape_transforms(self, simple_circuit):
        """Test using "all-mlir" level"""

        simple_circuit = qp.transforms.cancel_inverses(
            simple_circuit
        )  # Has to be applied as a tape transform because of the next transform
        simple_circuit = dummy_transform(simple_circuit)  # Forces normal tape transform
        simple_circuit = qp.transforms.merge_rotations(
            simple_circuit
        )  # Can be applied as an MLIR pass

        simple_circuit = qjit(simple_circuit)

        actual = qp.specs(simple_circuit, level="all-mlir")()
        expected = CircuitSpecs(
            device_name="lightning.qubit",
            num_device_wires=2,
            shots=Shots(None),
            level={
                3: "Before MLIR Passes",
                4: "merge-rotations",
            },
            resources={
                "Before MLIR Passes": SpecsResources(
                    gate_types={"RX": 2, "RZ": 2},
                    gate_sizes={1: 4},
                    measurements={"probs(all wires)": 1},
                    num_allocs=2,
                ),
                "merge-rotations": SpecsResources(
                    gate_types={"RX": 1, "RZ": 1},
                    gate_sizes={1: 2},
                    measurements={"probs(all wires)": 1},
                    num_allocs=2,
                ),
            },
        )

        check_specs_same(actual, expected)

    def test_advanced_measurements(self, capture_mode):
        """Test that advanced measurements such as LinearCombination are handled correctly."""

        dev = qp.device("lightning.qubit", wires=7)

        @qjit(capture=capture_mode)
        @qp.qnode(dev, shots=10)
        def circ():
            coeffs = [0.2, -0.543]
            obs = [qp.X(0) @ qp.Z(1), qp.Z(0) @ qp.Hadamard(2)]
            ham = qp.ops.LinearCombination(coeffs, obs)

            return (
                qp.expval(ham),
                qp.expval(qp.PauliZ(0) @ qp.PauliZ(1)),
                qp.sample(wires=3),
                qp.sample(),
            )

        # Representations are slightly different from plain PL -- wire counts are missing
        info = qp.specs(circ, level=0, compute_depth=False)()

        assert info.resources.measurements == {
            "expval(Hamiltonian(num_terms=2))": 1,
            "expval(Prod(num_terms=2))": 1,
            "sample(1 wires)": 1,
            "sample(all wires)": 1,
        }

    def test_conditionals(self, capture_mode):
        """Test that conditionals are handled correctly."""

        @qp.qjit(autograph=True, capture=capture_mode)
        @qp.qnode(qp.device("null.qubit", wires=1))
        def circuit(x):
            if x > 0.5:
                qp.Hadamard(0)
                qp.PauliX(0)
            else:
                qp.PauliX(0)
                if x < 2:
                    qp.PauliX(0)
                else:
                    qp.PauliZ(0)

            return qp.expval(qp.PauliX(0))

        with pytest.warns(
            UserWarning,
            match="Specs was unable to determine the branch of a conditional or switch statement.",
        ):
            actual = qp.specs(circuit, level=0)(3)
        expected = CircuitSpecs(
            device_name="null.qubit",
            num_device_wires=1,
            shots=Shots(None),
            level="Before MLIR Passes",
            resources=SpecsResources(
                gate_types={"Hadamard": 1, "PauliX": 2, "PauliZ": 1},
                gate_sizes={1: 4},
                measurements={"expval(PauliX)": 1},
                num_allocs=1,
            ),
        )

        check_specs_same(actual, expected)

    def test_loops(self, capture_mode):
        """Test that static loops are handled correctly and that resources are counted
        according to the number of iterations (including nested loops)."""

        @qp.qjit(capture=capture_mode)
        @qp.qnode(qp.device("null.qubit", wires=1))
        def circuit():
            for _ in range(5):
                qp.PauliX(0)
                for _ in range(3):
                    qp.Hadamard(0)
            return qp.expval(qp.PauliX(0))

        actual = qp.specs(circuit, level=0)()
        expected = CircuitSpecs(
            device_name="null.qubit",
            num_device_wires=1,
            shots=Shots(None),
            level="Before MLIR Passes",
            resources=SpecsResources(
                gate_types={"Hadamard": 15, "PauliX": 5},
                gate_sizes={1: 20},
                measurements={"expval(PauliX)": 1},
                num_allocs=1,
            ),
        )

        check_specs_same(actual, expected)

    def test_split_non_commuting_tape(self):
        """Test that qp.transforms.split_non_commuting works as expected"""

        @qp.transforms.cancel_inverses
        @qp.transforms.split_non_commuting  # Applies as tape transform
        @qp.qnode(qp.device("null.qubit", wires=3))
        def circuit():
            qp.H(0)
            qp.X(0)
            qp.X(0)
            return qp.expval(qp.X(0)), qp.expval(qp.Y(0)), qp.expval(qp.Z(0))

        actual = qp.specs(qjit(circuit), level=1)()
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

    def test_split_non_commuting_mlir(self):
        """Test that qp.transforms.split_non_commuting works as expected"""

        @qp.transforms.cancel_inverses
        @qp.transform(pass_name="split-non-commuting")  # Applies as MLIR pass
        @qp.qnode(qp.device("null.qubit", wires=3))
        def circuit():
            qp.H(0)
            qp.X(0)
            qp.X(0)
            return qp.expval(qp.X(0)), qp.expval(qp.Y(0)), qp.expval(qp.Z(0))

        actual = qp.specs(qjit(circuit), level=[1, 2])()
        expected = CircuitSpecs(
            device_name="null.qubit",
            num_device_wires=3,
            shots=Shots(None),
            level={1: "split-non-commuting", 2: "cancel-inverses"},
            resources={
                "split-non-commuting": [
                    SpecsResources(
                        gate_types={"Hadamard": 1, "PauliX": 2},
                        gate_sizes={1: 3},
                        measurements={"expval(PauliX)": 1},
                        num_allocs=3,
                    ),
                    SpecsResources(
                        gate_types={"Hadamard": 1, "PauliX": 2},
                        gate_sizes={1: 3},
                        measurements={"expval(PauliY)": 1},
                        num_allocs=3,
                    ),
                    SpecsResources(
                        gate_types={"Hadamard": 1, "PauliX": 2},
                        gate_sizes={1: 3},
                        measurements={"expval(PauliZ)": 1},
                        num_allocs=3,
                    ),
                ],
                "cancel-inverses": [  # The split should remain throughout subsequent passes
                    SpecsResources(
                        gate_types={"Hadamard": 1},
                        gate_sizes={1: 1},
                        measurements={"expval(PauliX)": 1},
                        num_allocs=3,
                    ),
                    SpecsResources(
                        gate_types={"Hadamard": 1},
                        gate_sizes={1: 1},
                        measurements={"expval(PauliY)": 1},
                        num_allocs=3,
                    ),
                    SpecsResources(
                        gate_types={"Hadamard": 1},
                        gate_sizes={1: 1},
                        measurements={"expval(PauliZ)": 1},
                        num_allocs=3,
                    ),
                ],
            },
        )

        check_specs_same(actual, expected)

    def test_subroutine(self):
        """Test qp.specs when there is a Catalyst subroutine"""
        dev = qp.device("lightning.qubit", wires=3)

        @qp.capture.subroutine
        def subroutine():
            qp.Hadamard(wires=0)

        @qp.qjit(autograph=True, capture=True)
        @qp.qnode(dev)
        def circuit():
            qp.PauliX(wires=1)

            for _ in range(3):
                subroutine()

            return qp.probs()

        actual = qp.specs(circuit, level=0)()
        expected = CircuitSpecs(
            device_name="lightning.qubit",
            num_device_wires=3,
            shots=Shots(None),
            level="Before MLIR Passes",
            resources=SpecsResources(
                gate_types={"Hadamard": 3, "PauliX": 1},
                gate_sizes={1: 4},
                measurements={"probs(all wires)": 1},
                num_allocs=3,
            ),
        )

        check_specs_same(actual, expected)

    def test_ppr(self):
        """Test that PPRs are handled correctly."""

        @qp.qjit(target="mlir")
        @catalyst.passes.to_ppr
        @qp.qnode(qp.device("null.qubit", wires=2))
        def circ():
            qp.H(0)
            qp.T(0)

        expected = CircuitSpecs(
            device_name="null.qubit",
            num_device_wires=2,
            shots=Shots(None),
            level="to-ppr",
            resources=SpecsResources(
                gate_types={"GlobalPhase": 2, "PPR-pi/4-w1": 3, "PPR-pi/8-w1": 1},
                gate_sizes={0: 2, 1: 4},
                measurements={},
                num_allocs=2,
            ),
        )

        actual = qp.specs(circ, level=1)()
        check_specs_same(actual, expected)

    def test_arbitrary_ppr(self):
        """Test that PPRs are handled correctly."""

        @qp.qjit(target="mlir", capture=True)
        @qp.transforms.decompose_arbitrary_ppr
        @qp.transforms.to_ppr
        @qp.qnode(qp.device("null.qubit", wires=3))
        def circ():
            qp.PauliRot(0.1, pauli_word="XY", wires=[0, 1])

        expected = CircuitSpecs(
            device_name="null.qubit",
            num_device_wires=3,
            shots=Shots(None),
            level="decompose-arbitrary-ppr",
            resources=SpecsResources(
                gate_types={
                    "pbc.prepare": 1,
                    "PPM-w3": 1,
                    "PPM-w1": 1,
                    "PPR-pi/2-w1": 1,
                    "PPR-pi/2-w2": 1,
                    "PPR-Phi-w1": 1,
                },
                gate_sizes={0: 1, 1: 3, 2: 1, 3: 1},
                measurements={},
                num_allocs=3,
            ),
        )

        actual = qp.specs(circ, level=2)()
        check_specs_same(actual, expected)

    def test_loop_warning(self):
        """Test that a warning is raised when dynamic loops are present in the circuit,
        as resource counting may be inaccurate."""

        @qp.qjit(autograph=True)
        @qp.qnode(qp.device("null.qubit", wires=1))
        def circuit(x):
            for _ in range(x):
                qp.PauliX(0)
            return qp.expval(qp.PauliX(0))

        with pytest.warns(
            UserWarning,
            match="Specs was unable to determine the number of loop iterations.",
        ):
            qp.specs(circuit, level=0)(5)


class TestMarkerIntegration:
    """Tests the integration with qp.marker."""

    @pytest.fixture
    def simple_circuit(self):
        """Fixture for a circuit."""

        @qp.qnode(qp.device("lightning.qubit", wires=2))
        def circ():
            qp.RX(1.0, 0)
            qp.RX(2.0, 0)
            qp.RZ(3.0, 1)
            qp.RZ(4.0, 1)
            qp.Hadamard(0)
            qp.Hadamard(0)
            qp.CNOT([0, 1])
            qp.CNOT([0, 1])
            return qp.probs()

        return circ

    def test_marker_with_tape_and_mlir_transforms(self, simple_circuit):
        """Tests that markers can work with both tape and mlir transforms."""

        simple_circuit = qp.marker(simple_circuit, "before-transforms")
        simple_circuit = dummy_transform(simple_circuit)
        simple_circuit = dummy_transform(simple_circuit)
        simple_circuit = qp.marker(simple_circuit, "after-tape")
        # Completely relying on cancel inverses being used as an MLIR transform
        simple_circuit = qp.transforms.cancel_inverses(simple_circuit)
        simple_circuit = qp.transforms.cancel_inverses(simple_circuit)
        simple_circuit = qp.marker(simple_circuit, "after-mlir")

        assert len(simple_circuit.compile_pipeline.markers) == 3

        qjit_circuit = qp.qjit(simple_circuit)

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

        actual = qp.specs(qjit_circuit, level=["before-transforms", "after-tape", "after-mlir"])()

        check_specs_same(actual, expected)

    def test_marker_with_tape_and_mlir_transforms_level_all(self, simple_circuit):
        """Tests that markers can work with both tape and mlir transforms when level is 'all'."""

        simple_circuit = qp.marker(simple_circuit, "before-transforms")
        simple_circuit = dummy_transform(simple_circuit)
        simple_circuit = dummy_transform(simple_circuit)
        simple_circuit = qp.marker(simple_circuit, "after-tape")
        # Completely relying on cancel inverses being used as an MLIR transform
        simple_circuit = qp.transforms.cancel_inverses(simple_circuit)
        simple_circuit = qp.transforms.cancel_inverses(simple_circuit)
        simple_circuit = qp.marker(simple_circuit, "after-mlir")

        assert len(simple_circuit.compile_pipeline.markers) == 3

        qjit_circuit = qp.qjit(simple_circuit)

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
                        "Before MLIR Passes",
                        "cancel-inverses",
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
                "Before MLIR Passes": SpecsResources(
                    gate_types={"RX": 2, "RZ": 2, "Hadamard": 2, "CNOT": 2},
                    gate_sizes={1: 6, 2: 2},
                    measurements={"probs(all wires)": 1},
                    num_allocs=2,
                ),
                "cancel-inverses": SpecsResources(
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

        actual = qp.specs(qjit_circuit, level="all")()

        check_specs_same(actual, expected)

    def test_redundant_marker(self, simple_circuit, capture_mode):
        """Test that two markers on the same level generate the same specs."""

        simple_circuit = partial(qp.marker, label="m0")(simple_circuit)
        simple_circuit = qp.transforms.cancel_inverses(simple_circuit)
        simple_circuit = partial(qp.marker, label="m1")(simple_circuit)
        simple_circuit = partial(qp.marker, label="m1-duplicate")(simple_circuit)

        simple_circuit = qjit(simple_circuit, capture=capture_mode)

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
            match=r"The 'level' argument to .*\.specs for QJIT'd QNodes has been sorted to be "
            "in ascending order with no duplicate levels.",
        ):
            actual = qp.specs(simple_circuit, level=["m0", "m1", "m1-duplicate"])()

        check_specs_same(actual, expected)

    def test_marker(self, simple_circuit, capture_mode):
        """Test that qp.marker can be used appropriately."""

        simple_circuit = partial(qp.marker, label="m0")(simple_circuit)
        simple_circuit = qp.transforms.cancel_inverses(simple_circuit)
        simple_circuit = partial(qp.marker, label="m1")(simple_circuit)
        simple_circuit = qp.transforms.merge_rotations(simple_circuit)
        simple_circuit = partial(qp.marker, label="m2")(simple_circuit)

        simple_circuit = qjit(simple_circuit, capture=capture_mode)

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

        actual = qp.specs(simple_circuit, level=["m0", "m1", "m2"])()

        check_specs_same(actual, expected)


if __name__ == "__main__":
    pytest.main(["-x", __file__])
