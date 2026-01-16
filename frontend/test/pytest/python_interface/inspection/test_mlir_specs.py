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
"""Unit test module for the mlir_specs function in the Python Compiler inspection module."""

from functools import partial

import jax.numpy as jnp
import pennylane as qml
import pytest

import catalyst
from catalyst.python_interface.inspection import ResourcesResult, mlir_specs

pytestmark = pytest.mark.xdsl


def resources_equal(
    actual: ResourcesResult, expected: ResourcesResult, return_only: bool = False
) -> bool:
    """
    Check if two ResourcesResult objects are equal.
    Ignores certain attributes (e.g. classical_instructions)
    """
    try:

        # actual.device_name == expected.device_name TODO: Don't worry about this one for now
        assert actual.num_allocs == expected.num_allocs
        assert actual.operations == expected.operations
        assert actual.measurements == expected.measurements

        # There should be no remaining unresolved function calls
        assert sum(actual.unresolved_function_calls.values()) == 0

        # Only check that expected function calls are a subset of actual function calls
        #   Other random helper functions may be inserted by the compiler
        for name, count in expected.function_calls.items():
            assert name in actual.function_calls
            assert actual.function_calls[name] == count

    except AssertionError:
        if return_only:
            return False
        raise

    return True


def make_static_resources(
    operations: dict[str, dict[int, int]] | None = None,
    measurements: dict[str, int] | None = None,
    function_calls: dict[str, int] | None = None,
    device_name: str | None = None,
    num_allocs: int = 0,
) -> ResourcesResult:
    """
    Create a ResourcesResult object for testing against
    """
    res = ResourcesResult()
    res.operations = operations or {}
    res.measurements = measurements or {}
    res.function_calls = function_calls or {}
    res.device_name = device_name
    res.num_allocs = num_allocs
    return res


@pytest.mark.usefixtures("use_both_frontend")
class TestMLIRSpecs:
    """Unit tests for the mlir_specs function in the Python Compiler inspection module."""

    @pytest.fixture
    def simple_circuit(self):
        """Fixture for a circuit."""

        @qml.qnode(qml.device("lightning.qubit", wires=2))
        def circ():
            qml.RX(1, 0)
            qml.RX(2.0, 0)
            qml.RZ(3.0, 1)
            qml.RZ(4.0, 1)
            qml.Hadamard(0)
            qml.Hadamard(0)
            qml.CNOT([0, 1])
            qml.CNOT([0, 1])
            return qml.probs()

        return circ

    @pytest.mark.parametrize("level", [3.14, "invalid"])
    def test_invalid_level_type(self, simple_circuit, level):
        """Test that requesting an invalid level type raises an error."""

        simple_circuit = qml.qjit(simple_circuit)

        with pytest.raises(
            ValueError, match="The `level` argument must be an int, a tuple/list of ints, or 'all'."
        ):
            mlir_specs(simple_circuit, level=level)

    @pytest.mark.parametrize("level", [10, -1])
    def test_invalid_int_level(self, simple_circuit, level):
        """Test that requesting an invalid level raises an error."""

        simple_circuit = qml.qjit(simple_circuit)

        with pytest.raises(
            ValueError, match=f"Requested specs level {level} not found in MLIR pass list."
        ):
            mlir_specs(simple_circuit, level=level)

    @pytest.mark.parametrize(
        "level, expected",
        [
            (
                0,
                make_static_resources(
                    operations={"RX": {1: 2}, "RZ": {1: 2}, "Hadamard": {1: 2}, "CNOT": {2: 2}},
                    measurements={"probs(all wires)": 1},
                    num_allocs=2,
                ),
            ),
        ],
    )
    def test_no_passes(self, simple_circuit, level, expected):
        """Test that if no passes are applied, the circuit resources are the original amount."""

        simple_circuit = qml.qjit(simple_circuit)
        res = mlir_specs(simple_circuit, level=level)
        assert resources_equal(res, expected)

    @pytest.mark.parametrize(
        "level, expected",
        [
            (
                0,
                make_static_resources(
                    operations={"RX": {1: 2}, "RZ": {1: 2}, "Hadamard": {1: 2}, "CNOT": {2: 2}},
                    measurements={"probs(all wires)": 1},
                    num_allocs=2,
                ),
            ),
            (
                1,
                make_static_resources(
                    operations={"RX": {1: 2}, "RZ": {1: 2}},
                    measurements={"probs(all wires)": 1},
                    num_allocs=2,
                ),
            ),
            (
                2,
                make_static_resources(
                    operations={"RX": {1: 1}, "RZ": {1: 1}},
                    measurements={"probs(all wires)": 1},
                    num_allocs=2,
                ),
            ),
        ],
    )
    def test_basic_passes(self, simple_circuit, level, expected):
        """Test that when passes are applied, the circuit resources are updated accordingly."""

        if qml.capture.enabled():
            simple_circuit = qml.transforms.cancel_inverses(simple_circuit)
            simple_circuit = qml.transforms.merge_rotations(simple_circuit)
        else:
            simple_circuit = catalyst.passes.cancel_inverses(simple_circuit)
            simple_circuit = catalyst.passes.merge_rotations(simple_circuit)

        simple_circuit = qml.qjit(simple_circuit)
        res = mlir_specs(simple_circuit, level=level)
        assert resources_equal(res, expected)

    def test_basic_passes_level_all(self, simple_circuit):
        """Test that when passes are applied, the circuit resources are updated accordingly."""

        if qml.capture.enabled():
            simple_circuit = qml.transforms.cancel_inverses(simple_circuit)
            simple_circuit = qml.transforms.merge_rotations(simple_circuit)
        else:
            simple_circuit = catalyst.passes.cancel_inverses(simple_circuit)
            simple_circuit = catalyst.passes.merge_rotations(simple_circuit)

        expected = {
            "Before MLIR Passes (MLIR-0)": make_static_resources(
                operations={"RX": {1: 2}, "RZ": {1: 2}, "Hadamard": {1: 2}, "CNOT": {2: 2}},
                measurements={"probs(all wires)": 1},
                num_allocs=2,
            ),
            "cancel-inverses (MLIR-1)": make_static_resources(
                operations={"RX": {1: 2}, "RZ": {1: 2}},
                measurements={"probs(all wires)": 1},
                num_allocs=2,
            ),
            "merge-rotations (MLIR-2)": make_static_resources(
                operations={"RX": {1: 1}, "RZ": {1: 1}},
                measurements={"probs(all wires)": 1},
                num_allocs=2,
            ),
        }

        simple_circuit = qml.qjit(simple_circuit)
        res = mlir_specs(simple_circuit, level="all")

        assert isinstance(res, dict)
        assert len(res) == len(expected)

        for lvl, expected_res in expected.items():
            assert lvl in res.keys()
            assert resources_equal(res[lvl], expected_res)

    def test_basic_passes_multi_level(self, simple_circuit):
        """Test that when passes are applied, the circuit resources are updated accordingly."""

        if qml.capture.enabled():
            simple_circuit = qml.transforms.cancel_inverses(simple_circuit)
            simple_circuit = qml.transforms.merge_rotations(simple_circuit)
        else:
            simple_circuit = catalyst.passes.cancel_inverses(simple_circuit)
            simple_circuit = catalyst.passes.merge_rotations(simple_circuit)

        expected = {
            "Before MLIR Passes (MLIR-0)": make_static_resources(
                operations={"RX": {1: 2}, "RZ": {1: 2}, "Hadamard": {1: 2}, "CNOT": {2: 2}},
                measurements={"probs(all wires)": 1},
                num_allocs=2,
            ),
            "merge-rotations (MLIR-2)": make_static_resources(
                operations={"RX": {1: 1}, "RZ": {1: 1}},
                measurements={"probs(all wires)": 1},
                num_allocs=2,
            ),
        }

        simple_circuit = qml.qjit(simple_circuit)
        res = mlir_specs(simple_circuit, level=[0, 2])

        assert isinstance(res, dict)
        assert len(res) == len(expected)

        for lvl, expected_res in expected.items():
            assert lvl in res.keys()
            assert resources_equal(res[lvl], expected_res)

        with pytest.raises(
            ValueError, match="Requested specs levels 3 not found in MLIR pass list."
        ):
            mlir_specs(simple_circuit, level=[0, 3])

    def test_not_qnode(self):
        """Test that a malformed QNode raises an error."""

        def not_a_qnode():
            pass

        with pytest.raises(
            ValueError,
            match="The provided `qnode` argument does not appear to be a valid QJIT "
            "compiled QNode.",
        ):
            mlir_specs(not_a_qnode, level=0)

    def test_malformed_qnode(self):
        """Test that a QNode without measurements can still be collected."""

        dev = qml.device("lightning.qubit", wires=1)

        @qml.qjit
        @qml.qnode(dev)
        def circ(wire):
            qml.X(0)
            qml.X(wire)

        # TODO: this warning is excessive, remove in the feature (sc-108251)
        with pytest.warns(UserWarning, match="does not yet support dynamic arguments"):
            res = mlir_specs(circ, 0, 1)
        expected = make_static_resources(
            operations={"PauliX": {1: 2}},
            measurements={},
            num_allocs=1,
        )

        assert resources_equal(res, expected)

    @pytest.mark.parametrize(
        "pl_ctrl_flow, iters, autograph",
        [
            (True, 5, False),
            (False, 2, False),
            (True, 3, False),
            (False, 10, True),
        ],
    )
    def test_fixed_loop(self, pl_ctrl_flow, iters, autograph):
        """Test that loop resources are counted correctly."""

        if pl_ctrl_flow:

            @qml.qnode(qml.device("lightning.qubit", wires=2))
            def circ():
                @qml.for_loop(iters)
                def loop_body(i):
                    qml.X(i % 2)

                loop_body()

                return qml.state()

        else:

            @qml.qnode(qml.device("lightning.qubit", wires=2))
            def circ():
                for i in range(iters):
                    qml.X(i % 2)
                return qml.state()

        expected = make_static_resources(
            operations={"PauliX": {1: iters}},
            measurements={"state(all wires)": 1},
            num_allocs=2,
        )

        circ = qml.qjit(autograph=autograph)(circ)
        res = mlir_specs(circ, level=0)
        assert resources_equal(res, expected)

    @pytest.mark.parametrize(
        "pl_ctrl_flow, iters",
        [
            (True, 5),
            (False, 2),
        ],
    )
    def test_dynamic_for_loop(self, pl_ctrl_flow, iters):
        """Test that dynamic for loops emit a warning."""

        if pl_ctrl_flow:

            @qml.qnode(qml.device("lightning.qubit", wires=2))
            def circ(n):
                @qml.for_loop(n)
                def loop_body(i):
                    qml.X(i % 2)

                loop_body()

                return qml.state()

        else:

            @qml.qnode(qml.device("lightning.qubit", wires=2))
            def circ(n):
                for i in range(n):
                    qml.X(i % 2)
                return qml.state()

        expected = make_static_resources(
            operations={"PauliX": {1: 1}},
            measurements={"state(all wires)": 1},
            num_allocs=2,
        )

        circ = qml.qjit(autograph=True)(circ)

        with pytest.warns(
            UserWarning,
            match="Specs was unable to determine the number of loop iterations. "
            "The results will assume the loop runs only once. "
            "This may be fixed in some cases by inlining dynamic arguments.",
        ):
            # TODO: this warning is excessive, remove in the feature (sc-108251)
            with pytest.warns(UserWarning, match="does not yet support dynamic arguments"):
                res = mlir_specs(circ, 0, iters)
            assert resources_equal(res, expected)

    @pytest.mark.parametrize(
        "pl_ctrl_flow, iters",
        [
            (True, 5),
            (False, 2),
        ],
    )
    def test_while_loop(self, pl_ctrl_flow, iters):
        """Test that dynamic while loops emit a warning."""

        if pl_ctrl_flow:

            @qml.qnode(qml.device("lightning.qubit", wires=2))
            def circ(n):
                def loop_cond(i):
                    return i < n

                @qml.while_loop(loop_cond)
                def loop_body(i):
                    qml.X(i % 2)
                    return i + 1

                loop_body(0)

                return qml.state()

        else:

            @qml.qnode(qml.device("lightning.qubit", wires=2))
            def circ(n):
                i = 0
                while i < n:
                    qml.X(i % 2)
                    i += 1
                return qml.state()

        expected = make_static_resources(
            operations={"PauliX": {1: 1}},
            measurements={"state(all wires)": 1},
            num_allocs=2,
        )

        circ = qml.qjit(autograph=True)(circ)

        with pytest.warns(
            UserWarning,
            match="Specs was unable to determine the number of loop iterations. "
            "The results will assume the loop runs only once. "
            "This may be fixed in some cases by inlining dynamic arguments.",
        ):
            # TODO: this warning is excessive, remove in the feature (sc-108251)
            with pytest.warns(UserWarning, match="does not yet support dynamic arguments"):
                res = mlir_specs(circ, 0, iters)
            assert resources_equal(res, expected)

    @pytest.mark.parametrize(
        "pl_ctrl_flow",
        [
            (True),
            (False),
        ],
    )
    def test_cond(self, pl_ctrl_flow):
        """Test that conditions emit a warning."""

        if pl_ctrl_flow:

            @qml.qnode(qml.device("lightning.qubit", wires=2))
            def circ(n):
                qml.cond(n > 0, qml.X, qml.Z)(0)

                return qml.state()

        else:

            @qml.qnode(qml.device("lightning.qubit", wires=2))
            def circ(n):
                if n > 0:
                    qml.X(0)
                else:
                    qml.Z(0)
                return qml.state()

        expected = make_static_resources(
            operations={"PauliX": {1: 1}, "PauliZ": {1: 1}},
            measurements={"state(all wires)": 1},
            num_allocs=2,
        )

        circ = qml.qjit(autograph=True)(circ)

        with pytest.warns(
            UserWarning,
            match="Specs was unable to determine the branch of a conditional or switch statement. "
            "The results will take the maximum resources across all possible branches.",
        ):
            n = 3  # Arbitrary value for n
            # TODO: this warning is excessive, remove in the feature (sc-108251)
            with pytest.warns(UserWarning, match="does not yet support dynamic arguments"):
                res = mlir_specs(circ, 0, n)
            assert resources_equal(res, expected)

    def test_tape_transforms(self):
        """Test that tape transforms are handled correctly."""
        if qml.capture.enabled():
            pytest.xfail("Currently broken with plxpr enabled.")

        @qml.qnode(qml.device("lightning.qubit", wires=2))
        def circ():
            qml.GlobalPhase(0.5)
            qml.GlobalPhase(1.0)
            return qml.expval(qml.PauliZ(0))

        circ = qml.transforms.combine_global_phases(circ)
        circ = qml.qjit(circ)

        expected = make_static_resources(
            operations={"GlobalPhase": {0: 1}},
            measurements={"expval(PauliZ)": 1},
            num_allocs=2,
        )

        res = mlir_specs(circ, level=0)
        assert resources_equal(res, expected)

    def test_stateprep(self):
        """Test that StatePrep operations are handled correctly."""

        @qml.qjit
        @qml.qnode(qml.device("lightning.qubit", wires=3), shots=10)
        def circ():
            qml.StatePrep(jnp.array([1, 0, 0, 0]), wires=[0, 1])
            qml.Hadamard(wires=1)
            qml.Hadamard(wires=2)
            return qml.sample()

        expected = make_static_resources(
            operations={"StatePrep": {2: 1}, "Hadamard": {1: 2}},
            measurements={"sample(all wires)": 1},
            num_allocs=3,
        )

        res = mlir_specs(circ, level=0)
        assert resources_equal(res, expected)

    def test_adjoint(self):
        """Test that adjoint operations are handled correctly."""

        @qml.qnode(qml.device("lightning.qubit", wires=2))
        def circ():
            def subroutine():
                qml.Hadamard(wires=0)
                qml.CNOT(wires=[0, 1])
                qml.RZ(0.789, wires=1)
                qml.adjoint(qml.T)(wires=1)

            qml.adjoint(subroutine)()
            return qml.probs()

        circ = qml.qjit(circ)

        expected = make_static_resources(
            operations={
                "Hadamard": {1: 1},
                "CNOT": {2: 1},
                "Adjoint(RZ)": {1: 1},
                "T": {1: 1},
            },
            measurements={"probs(all wires)": 1},
            num_allocs=2,
        )

        res = mlir_specs(circ, level=0)
        assert resources_equal(res, expected)

    def test_hamiltonian(self):
        """Test that Hamiltonian observables are handled correctly."""

        @qml.qjit
        @qml.qnode(qml.device("lightning.qubit", wires=2))
        def circ(i: int):

            coeffs = [0.2, -0.543]
            obs = [qml.X(0) @ qml.Z(1), qml.Z(i) @ qml.Hadamard(2)]
            ham1 = qml.Hamiltonian([1.0], [qml.Z(0) @ qml.Z(1)])
            ham2 = qml.ops.LinearCombination(coeffs, obs)

            return qml.expval(ham1), qml.expval(ham2)

        expected = make_static_resources(
            operations={},
            measurements={
                "expval(Hamiltonian(num_terms=1))": 1,
                "expval(Hamiltonian(num_terms=2))": 1,
            },
            num_allocs=2,
        )

        # TODO: this warning is excessive, remove in the feature (sc-108251)
        with pytest.warns(UserWarning, match="does not yet support dynamic arguments"):
            res = mlir_specs(circ, level=0, args=(0,))
        assert resources_equal(res, expected)

    def test_ppr(self):
        """Test that PPRs are handled correctly."""

        if not qml.capture.enabled():
            pytest.xfail("to_ppr requires plxpr to be enabled to lower PauliRot")

        pipeline = [("pipe", ["enforce-runtime-invariants-pipeline"])]

        @qml.qjit(pipelines=pipeline, target="mlir")
        @qml.transform(pass_name="to-ppr")
        @qml.qnode(qml.device("null.qubit", wires=2))
        def circ():
            qml.H(0)
            qml.T(0)
            qml.PauliRot(0.1234, pauli_word="Z", wires=0)

        expected = make_static_resources(
            operations={"PPR-pi/4": {1: 3}, "PPR-pi/8": {1: 1}, "PPR-Phi": {1: 1}},
            measurements={},
            num_allocs=2,
        )

        res = mlir_specs(circ, level=1)
        assert resources_equal(res, expected)

    def test_subroutine(self):
        """Test that subroutines are handled correctly."""
        if not qml.capture.enabled():
            pytest.xfail("Subroutine requires plxpr to be enabled.")

        @catalyst.jax_primitives.subroutine
        def extra_function():
            qml.Hadamard(wires=0)

        @qml.qjit(autograph=True)
        @qml.qnode(qml.device("null.qubit", wires=2))
        def circ():
            extra_function()
            return qml.probs()

        expected = make_static_resources(
            operations={"Hadamard": {1: 1}},
            measurements={"probs(all wires)": 1},
            num_allocs=2,
            function_calls={"extra_function": 1},
        )

        res = mlir_specs(circ, level=0)
        assert resources_equal(res, expected)

    @pytest.mark.usefixtures("use_capture_dgraph")
    def test_graph_decomp(self):
        """Test that graph decomposition is handled correctly."""

        @qml.register_resources({qml.H: 2, qml.CZ: 1})
        def my_cnot(wires):
            qml.H(wires=wires[1])
            qml.CZ(wires=wires)
            qml.H(wires=wires[1])

        @qml.qjit
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

        expected_resources = {"CZ": {2: 1}, "Hadamard": {1: 3}}
        resources = mlir_specs(circuit, level=1)
        assert resources.operations == expected_resources


# TODO: In the future, it would be good to add unit tests for specs_collector instead of just
#   integration tests

if __name__ == "__main__":
    pytest.main(["-x", __file__])
