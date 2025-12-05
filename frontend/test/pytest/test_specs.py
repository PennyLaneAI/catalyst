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


# TODO: Remove this method once feature pairty has been reached, and instead use `==` directly
def check_specs_resources_same(
    actual_res: SpecsResources | dict[any, SpecsResources],
    expected_res: SpecsResources | dict[any, SpecsResources],
    skip_measurements: bool = False,
) -> None:
    assert type(actual_res) == type(expected_res)

    if not isinstance(actual_res, dict):
        actual_res = {None: actual_res}
        expected_res = {None: expected_res}

    for res1, res2 in zip(actual_res.values(), expected_res.values()):
        assert res1.gate_types == res2.gate_types
        assert res1.gate_sizes == res2.gate_sizes

        # TODO: Measurements are not yet supported in Catalyst device-level specs
        if not skip_measurements:
            assert res1.measurements == res2.measurements

        assert res1.num_allocs == res2.num_allocs
        assert res1.depth == res2.depth
        assert res1.num_gates == res2.num_gates


def check_specs_same(actual: CircuitSpecs, expected: CircuitSpecs, skip_measurements: bool = False):
    """Check that two specs dictionaries are the same."""
    check_specs_header_same(actual, expected)
    check_specs_resources_same(
        actual["resources"], expected["resources"], skip_measurements=skip_measurements
    )


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
        check_specs_same(cat_specs, pl_specs, skip_measurements=True)

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

        check_specs_same(cat_specs, pl_specs, skip_measurements=True)


@pytest.mark.usefixtures("use_both_frontend")
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

    def test_basic_passes_multi_level(self, simple_circuit):
        """Test that when passes are applied, the circuit resources are updated accordingly."""

        if not qml.capture.enabled():
            pytest.xfail("Catalyst transforms display twice when capture not enabled")

        simple_circuit = qml.transforms.cancel_inverses(simple_circuit)
        simple_circuit = qml.transforms.merge_rotations(simple_circuit)

        simple_circuit = qjit(simple_circuit)

        expected = CircuitSpecs(
            device_name="lightning.qubit",
            num_device_wires=2,
            shots=Shots(None),
            level=[
                "Before transforms",
                "Before MLIR Passes (MLIR-0)",
                "cancel-inverses (MLIR-1)",
                "merge-rotations (MLIR-2)",
            ],
            resources={
                "Before transforms": SpecsResources(
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

    def test_marker(self, simple_circuit):
        """Test that qml.marker can be used appropriately."""

        if qml.capture.enabled():
            pytest.xfail("qml.marker is not currently compatible with program capture")

        simple_circuit = partial(qml.marker, level="m0")(simple_circuit)
        simple_circuit = qml.transforms.cancel_inverses(simple_circuit)
        simple_circuit = partial(qml.marker, level="m1")(simple_circuit)
        simple_circuit = qml.transforms.merge_rotations(simple_circuit)
        simple_circuit = partial(qml.marker, level="m2")(simple_circuit)

        simple_circuit = qjit(simple_circuit)

        expected = CircuitSpecs(
            device_name="lightning.qubit",
            num_device_wires=2,
            shots=Shots(None),
            level=["m0", "m1", "m2"],
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

    def test_reprs_match(self):
        """Test that when no transforms are applied to a typical circuit, the "Before Transform"
        and "Before MLIR Passes" representations match."""

        dev = qml.device("lightning.qubit", wires=4)

        @qml.qnode(dev)
        def circuit():
            qml.StatePrep(jnp.array([0, 1]), wires=0)

            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])

            qml.GlobalPhase(jnp.pi / 4)
            qml.MultiRZ(jnp.pi / 2, wires=[1, 2, 3])
            qml.ctrl(qml.T, control=0)(wires=3)

            qml.QubitUnitary(jnp.array([[1, 0], [0, 1j]]), wires=2)

            coeffs = [0.2, -0.543]
            obs = [qml.X(0) @ qml.Z(1), qml.Z(0) @ qml.Hadamard(2)]
            ham = qml.ops.LinearCombination(coeffs, obs)

            return (
                qml.expval(qml.PauliZ(0)),
                qml.expval(ham),
                qml.probs(wires=[0, 1]),
                qml.state(),
            )

        circuit = qjit(circuit)

        specs_all = qml.specs(circuit, level="all")()

        before_transforms = specs_all["resources"]["Before transforms"]
        before_mlir = specs_all["resources"]["Before MLIR Passes (MLIR-0)"]

        check_specs_resources_same(before_transforms, before_mlir)


if __name__ == "__main__":
    pytest.main(["-x", __file__])
