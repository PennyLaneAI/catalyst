# Copyright 2025-2026 Xanadu Quantum Technologies Inc.

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
from pennylane.resource import CircuitSpecs, PBCSpecsResources, SpecsResources

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
    if isinstance(actual_res, list):
        assert len(actual_res) == len(expected_res)

        for r1, r2 in zip(actual_res, expected_res):
            check_specs_resources_same(r1, r2)

    elif isinstance(actual_res, dict):
        assert len(actual_res) == len(expected_res)

        for k in actual_res.keys():
            assert k in expected_res
            check_specs_resources_same(actual_res[k], expected_res[k])

    elif isinstance(actual_res, (SpecsResources, PBCSpecsResources)):
        assert isinstance(expected_res, (SpecsResources, PBCSpecsResources))
        assert type(actual_res) is type(expected_res)
        assert actual_res.quantum_operations == expected_res.quantum_operations
        assert actual_res.measurement_processes == expected_res.measurement_processes
        assert actual_res.num_wires == expected_res.num_wires
        assert actual_res.depth == expected_res.depth
        assert actual_res.total_quantum_operations == expected_res.total_quantum_operations
        if isinstance(actual_res, PBCSpecsResources) and isinstance(
            expected_res, PBCSpecsResources
        ):
            assert actual_res.any_commuting_depth == expected_res.any_commuting_depth
            assert actual_res.qubit_disjoint_depth == expected_res.qubit_disjoint_depth

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

        assert cat_specs.resources.total_quantum_operations == 1
        assert cat_specs.resources.quantum_operations == {"RX": 1}

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

    @pytest.mark.xfail(reason="Broken by changes to tape-based specs. Fixed in PL PR #9988")
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
        cat_ops = cat_specs["resources"].quantum_operations
        assert "CY" in cat_ops
        cat_ops["C(Adjoint(PauliY))"] = cat_ops["CY"]
        del cat_ops["CY"]

        # Catalyst may count doubly-controlled S separately from singly-controlled S
        if "2C(S)" in cat_ops:
            cat_ops["C(S)"] = cat_ops.get("C(S)", 0) + cat_ops.pop("2C(S)")

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

        assert cat_specs.resources.total_quantum_operations == 4
        assert cat_specs.resources.quantum_operations == {
            "PauliRot-pi/2-w2": 1,
            "PauliRot-identity-w1": 1,
            "PauliRot-Phi-w1": 1,
            "PauliMeasure-w1": 1,
        }

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
        assert complex_meas_specs["resources"].measurement_processes == expected_measurements


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
            ValueError,
            match=r"The 'level' argument to .*\.specs for QJIT'd QNodes is out of "
            "bounds, got -5.",
        ):
            qp.specs(no_passes, level=-5)()

        with pytest.raises(
            ValueError,
            match=r"The 'level' argument to .*\.specs for QJIT'd "
            "QNodes is out of bounds, got 10.",
        ):
            qp.specs(no_passes, level=10)()

        with pytest.raises(
            ValueError,
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
                    counts={"RX": 2, "RZ": 2, "Hadamard": 2, "CNOT": 2},
                    measurement_processes={"probs(all wires)": 1},
                    num_wires=2,
                ),
                "cancel-inverses": SpecsResources(
                    counts={"RX": 2, "RZ": 2},
                    measurement_processes={"probs(all wires)": 1},
                    num_wires=2,
                ),
                "merge-rotations": SpecsResources(
                    counts={"RX": 1, "RZ": 1},
                    measurement_processes={"probs(all wires)": 1},
                    num_wires=2,
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
            counts={"RX": 1, "RZ": 1},
            measurement_processes={"probs(all wires)": 1},
            num_wires=2,
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
            counts={"RX": 1, "RZ": 1},
            measurement_processes={"probs(all wires)": 1},
            num_wires=2,
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
            counts={"RX": 2, "RZ": 2},
            measurement_processes={"probs(all wires)": 1},
            num_wires=2,
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
                    counts={"RX": 2, "RZ": 2, "Hadamard": 2, "CNOT": 2},
                    measurement_processes={"probs(all wires)": 1},
                    num_wires=2,
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
                    counts={"RX": 2, "RZ": 2, "Hadamard": 2, "CNOT": 2},
                    measurement_processes={"probs(all wires)": 1},
                    num_wires=2,
                ),
                "dummy_transform": SpecsResources(
                    counts={"RX": 2, "RZ": 2, "Hadamard": 2, "CNOT": 2},
                    measurement_processes={"probs(all wires)": 1},
                    num_wires=2,
                ),
                "dummy_transform-2": SpecsResources(
                    counts={"RX": 2, "RZ": 2, "Hadamard": 2, "CNOT": 2},
                    measurement_processes={"probs(all wires)": 1},
                    num_wires=2,
                ),
                "Before MLIR Passes": SpecsResources(
                    counts={"RX": 2, "RZ": 2, "Hadamard": 2, "CNOT": 2},
                    measurement_processes={"probs(all wires)": 1},
                    num_wires=2,
                ),
                "cancel-inverses": SpecsResources(
                    counts={"RX": 2, "RZ": 2},
                    measurement_processes={"probs(all wires)": 1},
                    num_wires=2,
                ),
                "merge-rotations": SpecsResources(
                    counts={"RX": 1, "RZ": 1},
                    measurement_processes={"probs(all wires)": 1},
                    num_wires=2,
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
                    counts={"RX": 2, "RZ": 2, "Hadamard": 2, "CNOT": 2},
                    measurement_processes={"probs(all wires)": 1},
                    num_wires=2,
                ),
                "cancel_inverses": SpecsResources(
                    counts={"RX": 2, "RZ": 2},
                    measurement_processes={"probs(all wires)": 1},
                    num_wires=2,
                ),
                "dummy_transform": SpecsResources(
                    counts={"RX": 2, "RZ": 2},
                    measurement_processes={"probs(all wires)": 1},
                    num_wires=2,
                ),
                "Before MLIR Passes": SpecsResources(
                    counts={"RX": 2, "RZ": 2},
                    measurement_processes={"probs(all wires)": 1},
                    num_wires=2,
                ),
                "merge-rotations": SpecsResources(
                    counts={"RX": 1, "RZ": 1},
                    measurement_processes={"probs(all wires)": 1},
                    num_wires=2,
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
                    counts={"RX": 2, "RZ": 2, "Hadamard": 2, "CNOT": 2},
                    measurement_processes={"probs(all wires)": 1},
                    num_wires=2,
                ),
                "cancel_inverses": SpecsResources(
                    counts={"RX": 2, "RZ": 2},
                    measurement_processes={"probs(all wires)": 1},
                    num_wires=2,
                ),
                "dummy_transform": SpecsResources(
                    counts={"RX": 2, "RZ": 2},
                    measurement_processes={"probs(all wires)": 1},
                    num_wires=2,
                ),
                "Before MLIR Passes": SpecsResources(
                    counts={"RX": 2, "RZ": 2},
                    measurement_processes={"probs(all wires)": 1},
                    num_wires=2,
                ),
                "merge-rotations": SpecsResources(
                    counts={"RX": 1, "RZ": 1},
                    measurement_processes={"probs(all wires)": 1},
                    num_wires=2,
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
                    counts={"RX": 2, "RZ": 2, "Hadamard": 2, "CNOT": 2},
                    measurement_processes={"probs(all wires)": 1},
                    num_wires=2,
                ),
                "cancel-inverses": SpecsResources(
                    counts={"RX": 2, "RZ": 2},
                    measurement_processes={"probs(all wires)": 1},
                    num_wires=2,
                ),
                "merge-rotations": SpecsResources(
                    counts={"RX": 1, "RZ": 1},
                    measurement_processes={"probs(all wires)": 1},
                    num_wires=2,
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
                    counts={"RX": 2, "RZ": 2},
                    measurement_processes={"probs(all wires)": 1},
                    num_wires=2,
                ),
                "merge-rotations": SpecsResources(
                    counts={"RX": 1, "RZ": 1},
                    measurement_processes={"probs(all wires)": 1},
                    num_wires=2,
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

        assert info.resources.measurement_processes == {
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
                counts={"Hadamard": 1, "PauliX": 2, "PauliZ": 1},
                measurement_processes={"expval(PauliX)": 1},
                num_wires=1,
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
                counts={"Hadamard": 15, "PauliX": 5},
                measurement_processes={"expval(PauliX)": 1},
                num_wires=1,
            ),
        )

        check_specs_same(actual, expected)

    def test_empty_loops(self, capture_mode):
        """Test that empty static loops are handled correctly."""

        @qp.qjit(capture=capture_mode)
        @qp.qnode(qp.device("null.qubit", wires=1))
        def circuit():
            for _ in range(0):
                qp.PauliX(0)
                for _ in range(2, 2):
                    qp.Hadamard(0)
            return qp.expval(qp.PauliX(0))

        actual = qp.specs(circuit, level=0)()
        expected = CircuitSpecs(
            device_name="null.qubit",
            num_device_wires=1,
            shots=Shots(None),
            level="Before MLIR Passes",
            resources=SpecsResources(
                counts={},
                measurement_processes={"expval(PauliX)": 1},
                num_wires=1,
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
                    counts={"Hadamard": 1, "PauliX": 2},
                    measurement_processes={"expval(PauliX)": 1},
                    num_wires=1,
                ),
                SpecsResources(
                    counts={"Hadamard": 1, "PauliX": 2},
                    measurement_processes={"expval(PauliY)": 1},
                    num_wires=1,
                ),
                SpecsResources(
                    counts={"Hadamard": 1, "PauliX": 2},
                    measurement_processes={"expval(PauliZ)": 1},
                    num_wires=1,
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
                        counts={"Hadamard": 1, "PauliX": 2},
                        measurement_processes={"expval(PauliX)": 1},
                        num_wires=3,
                    ),
                    SpecsResources(
                        counts={"Hadamard": 1, "PauliX": 2},
                        measurement_processes={"expval(PauliY)": 1},
                        num_wires=3,
                    ),
                    SpecsResources(
                        counts={"Hadamard": 1, "PauliX": 2},
                        measurement_processes={"expval(PauliZ)": 1},
                        num_wires=3,
                    ),
                ],
                "cancel-inverses": [  # The split should remain throughout subsequent passes
                    SpecsResources(
                        counts={"Hadamard": 1},
                        measurement_processes={"expval(PauliX)": 1},
                        num_wires=3,
                    ),
                    SpecsResources(
                        counts={"Hadamard": 1},
                        measurement_processes={"expval(PauliY)": 1},
                        num_wires=3,
                    ),
                    SpecsResources(
                        counts={"Hadamard": 1},
                        measurement_processes={"expval(PauliZ)": 1},
                        num_wires=3,
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
                counts={"Hadamard": 3, "PauliX": 1},
                measurement_processes={"probs(all wires)": 1},
                num_wires=3,
            ),
        )

        check_specs_same(actual, expected)

    def test_operator2(self):
        """Test that specs works with operator2 classes."""

        # pylint: disable=useless-parent-delegation
        class DummyOp(qp.core.Operator2):
            """Dummy Local Operator."""

            dynamic_argnames = ("phi",)
            wire_argnames = ("reg1", "reg2")
            compilable_argnames = ("metadata",)

            def __init__(self, phi, reg1, reg2, metadata):
                super().__init__(phi, reg1, reg2, metadata)

        @qp.qjit(capture=True, target="mlir")
        @qp.transforms.merge_rotations
        @qp.qnode(qp.device("null.qubit", wires=10))
        def c():
            DummyOp(0.5, (0, 1), (2, 3, 4), metadata="word")
            DummyOp(0.5, (2, 3, 4), (0,), metadata="word")
            return qp.state()

        for level in [0, 1]:
            resources = qp.specs(c, level=level)().resources

            assert resources.quantum_operations == {"DummyOp": 2}

    def test_symbolic_array(self):
        """Test using specs with symbolic_array."""

        @qp.qjit(capture=True, target="mlir")
        @qp.transforms.merge_rotations
        @qp.qnode(qp.device("null.qubit", wires=1))
        def c():
            x = qp.capture.symbolic_array((), float)
            qp.RX(x, 0)
            qp.RX(2 * x, 0)
            return qp.probs()

        counts = qp.specs(c, level=0)().resources.quantum_operations
        assert counts == {"RX": 2}

        counts1 = qp.specs(c, level=1)().resources.quantum_operations
        assert counts1 == {"RX": 1}

        with pytest.raises(catalyst.utils.exceptions.CompileError, match="is a placeholder op"):
            qp.specs(c, level="device")()


class TestSpecsWithPPR:
    """Tests for using qp.specs with PPRs"""

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
            resources=PBCSpecsResources(
                counts={"GlobalPhase": 2, "PPR-pi/4-w1": 3, "PPR-pi/8-w1": 1},
                measurement_processes={},
                num_wires=2,
                any_commuting_depth=3,
                qubit_disjoint_depth=4,
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
            resources=PBCSpecsResources(
                counts={
                    "pbc.prepare": 1,
                    "PPM-w3": 1,
                    "PPM-w1": 1,
                    "PPR-pi/2-w1": 1,
                    "PPR-pi/2-w2": 1,
                    "PPR-Phi-w1": 1,
                },
                measurement_processes={},
                num_wires=4,
                any_commuting_depth=4,
                qubit_disjoint_depth=4,
            ),
        )

        actual = qp.specs(circ, level=2)()
        check_specs_same(actual, expected)


class TestSymbolicSpecs:
    """Tests for using qp.specs with dynamic loops whose bounds are not known at compile time"""

    def test_dynamic_loop(self, capture_mode):
        """Test specs with a dynamic loop that can't be resolved at compile time"""

        @qp.qjit(autograph=True, capture=capture_mode)
        @qp.qnode(qp.device("lightning.qubit", wires=1))
        def circuit(x):
            qp.Hadamard(0)
            qp.PauliX(0)
            for _ in range(x):
                qp.PauliX(0)
            return qp.expval(qp.PauliX(0))

        s = qp.specs(circuit, level=0)(5)
        assert s.level == "Before MLIR Passes"
        assert s.device_name == "lightning.qubit"
        res = s.resources
        assert res.is_symbolic
        assert len(res.vars) == 1

        concrete_res = res.subs({var: 5 for var in res.vars})
        assert isinstance(concrete_res, SpecsResources) and not concrete_res.is_symbolic

        expected_res = SpecsResources(
            counts={"Hadamard": 1, "PauliX": 6},
            measurement_processes={"expval(PauliX)": 1},
            num_wires=1,
        )
        check_specs_resources_same(concrete_res, expected_res)

    def test_dynamic_loop_and_static_loop(self, capture_mode):
        """
        Test specs with a dynamic loop that can't be resolved at compile time and
        a static loop nested inside it
        """

        @qp.qjit(autograph=True, capture=capture_mode)
        @qp.qnode(qp.device("lightning.qubit", wires=1))
        def circuit(x):
            qp.Hadamard(0)
            qp.PauliX(0)
            for _ in range(x):
                qp.PauliX(0)
                for _ in range(3):
                    qp.PauliY(0)
                for _ in range(5):
                    qp.PauliZ(0)

            return qp.expval(qp.PauliX(0))

        s = qp.specs(circuit, level=0)(5)
        assert s.level == "Before MLIR Passes"
        assert s.device_name == "lightning.qubit"

        res = s.resources
        assert res.is_symbolic
        assert len(res.vars) == 1

        concrete_res = res.subs({var: 5 for var in res.vars})
        assert isinstance(concrete_res, SpecsResources) and not concrete_res.is_symbolic

        expected_res = SpecsResources(
            counts={"Hadamard": 1, "PauliX": 6, "PauliY": 15, "PauliZ": 25},
            measurement_processes={"expval(PauliX)": 1},
            num_wires=1,
        )
        check_specs_resources_same(concrete_res, expected_res)

    def test_dynamic_loop_and_static_loop2(self, capture_mode):
        """
        Test specs with a static loop and a dynamic loop that can't be resolved at compile time
        nested inside it
        """

        @qp.qjit(autograph=True, capture=capture_mode)
        @qp.qnode(qp.device("lightning.qubit", wires=1))
        def circuit(x):
            qp.Hadamard(0)
            qp.PauliX(0)
            for _ in range(3):
                qp.PauliZ(0)
                for _ in range(x):
                    qp.PauliX(0)

            return qp.expval(qp.PauliX(0))

        s = qp.specs(circuit, level=0)(5)
        assert s.level == "Before MLIR Passes"
        assert s.device_name == "lightning.qubit"

        res = s.resources
        assert res.is_symbolic
        assert len(res.vars) == 1

        concrete_res = res.subs({var: 5 for var in res.vars})
        assert isinstance(concrete_res, SpecsResources) and not concrete_res.is_symbolic

        expected_res = SpecsResources(
            counts={"Hadamard": 1, "PauliX": 16, "PauliZ": 3},
            measurement_processes={"expval(PauliX)": 1},
            num_wires=1,
        )
        check_specs_resources_same(concrete_res, expected_res)

    def test_nested_dynamic_loop(self, capture_mode):
        """Test specs with a nested dynamic loops that can't be resolved at compile time"""

        @qp.qjit(autograph=True, capture=capture_mode)
        @qp.qnode(qp.device("lightning.qubit", wires=1))
        def circuit(x, y):
            qp.Hadamard(0)
            for _ in range(x):
                qp.PauliX(0)
                for _ in range(y):
                    qp.Hadamard(0)
            return qp.expval(qp.PauliX(0))

        s = qp.specs(circuit, level=0)(5, 3)
        assert s.level == "Before MLIR Passes"
        assert s.device_name == "lightning.qubit"
        res = s.resources
        assert res.is_symbolic
        assert len(res.vars) == 2

        for n in [2, 3]:
            concrete_res = res.subs({var: n for var in res.vars})
            assert isinstance(concrete_res, SpecsResources) and not concrete_res.is_symbolic
            expected_res = SpecsResources(
                counts={"Hadamard": 1 + n * n, "PauliX": n},
                measurement_processes={"expval(PauliX)": 1},
                num_wires=1,
            )
            check_specs_resources_same(concrete_res, expected_res)

    def test_dynamic_loops_multi_level(self, capture_mode):
        """Test smulti-level specs with dynamic loops"""

        @qp.qjit(autograph=True, capture=capture_mode)
        @qp.transforms.cancel_inverses
        @qp.qnode(qp.device("lightning.qubit", wires=1))
        def circuit(x, y):
            qp.Hadamard(0)
            for _ in range(x):
                qp.Hadamard(0)
                qp.PauliX(0)
                for _ in range(y):
                    qp.Hadamard(0)
            return qp.expval(qp.PauliX(0))

        s = qp.specs(circuit, level="all")(3, 5)
        assert s.level == {0: "Before MLIR Passes", 1: "cancel-inverses"}
        assert s.device_name == "lightning.qubit"
        all_res = s.resources

        assert isinstance(all_res, dict)
        for res in all_res.values():
            assert res.is_symbolic
            assert len(res.vars) == 2

        for n in [2, 3]:
            for res in all_res.values():
                concrete_res = res.subs({var: n for var in res.vars})

                check_specs_resources_same(
                    concrete_res,
                    SpecsResources(
                        counts={"Hadamard": n * n + n + 1, "PauliX": n},
                        measurement_processes={"expval(PauliX)": 1},
                        num_wires=1,
                    ),
                )

    def test_symbolic_array_inside_loop(self):
        """Test dynamic loop with symbolic_array in a loop."""

        @qp.qjit(capture=True)
        @qp.qnode(qp.device("null.qubit", wires=1))
        def c(n):

            # pylint: disable=unused-argument
            @qp.for_loop(n)
            def loop(i):
                x = qp.capture.symbolic_array((), float)
                qp.RX(x, 0)

            loop()  # pylint: disable=no-value-for-parameter

            return qp.state()

        r = qp.specs(c, level=0)(2).resources
        assert r.subs({var: 10 for var in r.vars}).quantum_operations["RX"] == 10

    def test_symbolic_array_loop_arguemtn(self):
        """Test dynamic loop with a symbolic array as a loop argument."""

        @qp.qjit(capture=True)
        @qp.qnode(qp.device("null.qubit", wires=1))
        def c(n):

            # pylint: disable=unused-argument
            @qp.for_loop(n)
            def loop(i, x):
                qp.RX(x, 0)
                return x

            y = qp.capture.symbolic_array((), float)
            loop(y)  # pylint: disable=no-value-for-parameter

            return qp.state()

        r = qp.specs(c, level=0)(2).resources
        assert r.subs({var: 10 for var in r.vars}).quantum_operations["RX"] == 10


class TestSymbolicSpecsLoopConcretization:
    """
    Integration tests for the loop concretization feature of the resource analysis pass, which
    resolves nested loops whose inner bounds are the immediately enclosing loop's induction
    variable.
    """

    def test_loop_concretization(self):
        """Test a straightforward nested loop whose inner bound depends on the outer loop var."""
        n = 8

        @qp.qjit(autograph=True)
        @qp.qnode(qp.device("null.qubit", wires=n))
        def circuit():
            for i in range(n):
                for j in range(i):
                    qp.PauliZ(wires=j % 2)
            return qp.expval(qp.X(0))

        resources = qp.specs(circuit, level=0)().resources

        assert resources.quantum_operations["PauliZ"] == 28

    def test_triple_nested_loop_concretization(self):
        """Test 3 nested loops whose bounds depends on the outer loop var."""
        n = 8

        @qp.qjit(autograph=True)
        @qp.qnode(qp.device("null.qubit", wires=n))
        def circuit():
            for i in range(n):  # Runs 8 times total
                for j in range(i):  # Runs 28 times total
                    for k in range(j):  # Runs 56 times total
                        qp.PauliZ(wires=k % 2)
                    qp.PauliX(wires=j % 2)

            return qp.expval(qp.X(0))

        resources = qp.specs(circuit, level=0)().resources

        assert resources.quantum_operations["PauliZ"] == 56
        assert resources.quantum_operations["PauliX"] == 28

    def test_loop_concretization_with_unrelated_middle_loop(self):
        """Test 3 nested loops where the middle loop is unrelated to the other 2."""
        a, b = 4, 3

        @qp.qjit(autograph=True)
        @qp.qnode(qp.device("null.qubit", wires=2))
        def circuit():
            for i in range(a):  # Runs 4 times total
                for j in range(b):  # Runs 12 times total
                    for k in range(i):  # Runs 18 times total
                        qp.PauliZ(wires=k % 2)
                    qp.PauliX(wires=j % 2)

            return qp.expval(qp.X(0))

        resources = qp.specs(circuit, level=0)().resources

        assert resources.quantum_operations["PauliZ"] == 18
        assert resources.quantum_operations["PauliX"] == 12

    def test_loop_concretization_symbolic(self):
        """Test nested dynamic loops."""

        @qp.qjit(autograph=True)
        @qp.qnode(qp.device("null.qubit", wires=8))
        def circuit(n):
            for i in range(n):
                for j in range(i):
                    qp.PauliZ(wires=j % 2)

            return qp.expval(qp.X(0))

        resources = qp.specs(circuit, level=0)(8).resources

        # Current behaviour is that these loops are *NOT* folded like static loops
        assert not isinstance(resources.quantum_operations["PauliZ"], (int, float))
        assert len(resources.quantum_operations["PauliZ"].vars) == 2

    def test_loop_concretization_with_step(self):
        """Test an outer loop with a step != 1."""
        n = 8

        @qp.qjit(autograph=True)
        @qp.qnode(qp.device("null.qubit", wires=n))
        def circuit():
            for i in range(0, n, 2):
                for j in range(i):
                    qp.PauliZ(wires=j % 2)

            return qp.expval(qp.X(0))

        resources = qp.specs(circuit, level=0)().resources

        assert resources.quantum_operations["PauliZ"] == 12

    def test_loop_concretization_with_inner_step(self):
        """Test an inner loop with a step != 1."""
        n = 8

        @qp.qjit(autograph=True)
        @qp.qnode(qp.device("null.qubit", wires=n))
        def circuit():
            for i in range(n):
                for j in range(0, i, 2):
                    qp.PauliZ(wires=j % 2)

            return qp.expval(qp.X(0))

        resources = qp.specs(circuit, level=0)().resources

        assert resources.quantum_operations["PauliZ"] == 16

    def test_loop_concretization_with_lower_bound(self):
        """Test an outer loop with a lower bound."""
        n = 8

        @qp.qjit(autograph=True)
        @qp.qnode(qp.device("null.qubit", wires=n))
        def circuit():
            for i in range(2, n):
                for j in range(i):
                    qp.PauliZ(wires=j % 2)

            return qp.expval(qp.X(0))

        resources = qp.specs(circuit, level=0)().resources

        assert resources.quantum_operations["PauliZ"] == 27

    def test_loop_concretization_with_inner_lower_bound(self):
        """Test an inner loop with a lower bound."""
        n = 8

        @qp.qjit(autograph=True)
        @qp.qnode(qp.device("null.qubit", wires=n))
        def circuit():
            for i in range(n):
                for j in range(1, i):
                    qp.PauliZ(wires=j % 2)

            return qp.expval(qp.X(0))

        resources = qp.specs(circuit, level=0)().resources

        assert resources.quantum_operations["PauliZ"] == 21

    def test_loop_concretization_reverse(self):
        """Test concretization on a decrementing loop."""
        n = 8

        @qp.qjit(autograph=True)
        @qp.qnode(qp.device("null.qubit", wires=n))
        def circuit():
            for i in range(n, 0, -1):
                for j in range(i):
                    qp.PauliZ(wires=j % 2)

            return qp.expval(qp.X(0))

        resources = qp.specs(circuit, level=0)().resources

        # Expect a symbolic value: reverse iteration is not supported for concretization
        assert not isinstance(resources.quantum_operations["PauliZ"], (int, float))

    def test_loop_concretization_static_change(self):
        """Test concretization where the inner loop depends indirectly on the outer loop var."""
        n = 8

        @qp.qjit(autograph=True)
        @qp.qnode(qp.device("null.qubit", wires=n))
        def circuit():
            for i in range(n):
                for j in range(i + 1):  # Note the +1, this is now an expression
                    qp.PauliZ(wires=j % 2)

            return qp.expval(qp.X(0))

        resources = qp.specs(circuit, level=0)().resources

        # Expect a symbolic value: indirect dependency is not supported for concretization
        assert not isinstance(resources.quantum_operations["PauliZ"], (int, float))

    # FIXME: This case is failing
    @pytest.mark.xfail
    def test_loop_concretization_multi_dependency(self):
        """Test concretization with a loop that has 2 direct dependencies from inner loops."""
        n = 8

        @qp.qjit(autograph=True)
        @qp.qnode(qp.device("null.qubit", wires=n))
        def circuit():
            for i in range(n):
                for _ in range(i):
                    for k in range(i):  # Depends on outer-most loop
                        qp.PauliZ(wires=k % 2)

            return qp.expval(qp.X(0))

        resources = qp.specs(circuit, level=0)().resources

        assert resources.quantum_operations.get("PauliZ", 0) == 140

    def test_loop_concretization_combined(self):
        """Test concretization with all different complexities on loop bounds put together."""
        n = 8

        @qp.qjit(autograph=True)
        @qp.qnode(qp.device("null.qubit", wires=n))
        def circuit():
            for i in range(1, n, 2):
                for j in range(1, i):
                    for k in range(0, j, 2):
                        qp.PauliZ(wires=k % 2)

            return qp.expval(qp.X(0))

        resources = qp.specs(circuit, level=0)().resources

        assert resources.quantum_operations["PauliZ"] == 20

    def test_loop_concretization_no_iters(self):
        """Test concretization with a loop that has no iterations."""

        @qp.qjit(autograph=True)
        @qp.qnode(qp.device("null.qubit", wires=1))
        def circuit():
            for i in range(0):
                for j in range(i):
                    qp.PauliZ(wires=j % 2)
            for i in range(2, 2):
                for j in range(i):
                    qp.PauliX(wires=j % 2)

            return qp.expval(qp.X(0))

        resources = qp.specs(circuit, level=0)().resources

        assert resources.quantum_operations.get("PauliZ", 0) == 0
        assert resources.quantum_operations.get("PauliX", 0) == 0


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
                    counts={"RX": 2, "RZ": 2, "Hadamard": 2, "CNOT": 2},
                    measurement_processes={"probs(all wires)": 1},
                    num_wires=2,
                ),
                "after-tape": SpecsResources(
                    counts={"RX": 2, "RZ": 2, "Hadamard": 2, "CNOT": 2},
                    measurement_processes={"probs(all wires)": 1},
                    num_wires=2,
                ),
                "after-mlir": SpecsResources(
                    counts={"RX": 2, "RZ": 2},
                    measurement_processes={"probs(all wires)": 1},
                    num_wires=2,
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
                    counts={"RX": 2, "RZ": 2, "Hadamard": 2, "CNOT": 2},
                    measurement_processes={"probs(all wires)": 1},
                    num_wires=2,
                ),
                "dummy_transform": SpecsResources(
                    counts={"RX": 2, "RZ": 2, "Hadamard": 2, "CNOT": 2},
                    measurement_processes={"probs(all wires)": 1},
                    num_wires=2,
                ),
                "after-tape": SpecsResources(
                    counts={"RX": 2, "RZ": 2, "Hadamard": 2, "CNOT": 2},
                    measurement_processes={"probs(all wires)": 1},
                    num_wires=2,
                ),
                "Before MLIR Passes": SpecsResources(
                    counts={"RX": 2, "RZ": 2, "Hadamard": 2, "CNOT": 2},
                    measurement_processes={"probs(all wires)": 1},
                    num_wires=2,
                ),
                "cancel-inverses": SpecsResources(
                    counts={"RX": 2, "RZ": 2},
                    measurement_processes={"probs(all wires)": 1},
                    num_wires=2,
                ),
                "after-mlir": SpecsResources(
                    counts={"RX": 2, "RZ": 2},
                    measurement_processes={"probs(all wires)": 1},
                    num_wires=2,
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
                    counts={"RX": 2, "RZ": 2, "Hadamard": 2, "CNOT": 2},
                    measurement_processes={"probs(all wires)": 1},
                    num_wires=2,
                ),
                "m1, m1-duplicate": SpecsResources(
                    counts={"RX": 2, "RZ": 2},
                    measurement_processes={"probs(all wires)": 1},
                    num_wires=2,
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
                    counts={"RX": 2, "RZ": 2, "Hadamard": 2, "CNOT": 2},
                    measurement_processes={"probs(all wires)": 1},
                    num_wires=2,
                ),
                "m1": SpecsResources(
                    counts={"RX": 2, "RZ": 2},
                    measurement_processes={"probs(all wires)": 1},
                    num_wires=2,
                ),
                "m2": SpecsResources(
                    counts={"RX": 1, "RZ": 1},
                    measurement_processes={"probs(all wires)": 1},
                    num_wires=2,
                ),
            },
        )

        actual = qp.specs(simple_circuit, level=["m0", "m1", "m2"])()

        check_specs_same(actual, expected)


def test_abstract_array_inputs():
    """Test that AbstractArray and AbstractWires can be used with specs when level!= device."""

    @qp.qjit(capture=True)
    @qp.qnode(qp.device("lightning.qubit", wires=4))
    def c(x, wires):
        @qp.for_loop(x.shape[0])
        def loop(i):
            qp.RX(x[i], wires[i])

        @qp.for_loop(wires.shape[0])
        def loop2(i):
            qp.X(i)

        loop()
        loop2()
        return qp.expval(qp.Z(0))

    s = qp.specs(c, level=0)(qp.typing.AbstractArray((3,), float), qp.typing.Wire[3])
    assert s.resources.quantum_operations["PauliX"] == 3
    assert s.resources.quantum_operations["RX"] == 3


if __name__ == "__main__":
    pytest.main(["-x", __file__])
