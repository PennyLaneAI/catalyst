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

# pylint: disable=too-many-public-methods

from functools import partial

import jax.numpy as jnp
import pennylane as qp
import pytest

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


@pytest.mark.parametrize(
    "skip_preprocess",
    [pytest.param(True, id="skip_preprocess"), pytest.param(False, id="preprocess")],
)
class TestMLIRSpecs:
    """Unit tests for the mlir_specs function in the Python Compiler inspection module."""

    @pytest.fixture
    def simple_circuit(self):
        """Fixture for a circuit."""

        @qp.qnode(qp.device("lightning.qubit", wires=2))
        def circ():
            qp.RX(1, 0)
            qp.RX(2.0, 0)
            qp.RZ(3.0, 1)
            qp.RZ(4.0, 1)
            qp.Hadamard(0)
            qp.Hadamard(0)
            qp.CNOT([0, 1])
            qp.CNOT([0, 1])
            return qp.probs()

        return circ

    def test_float_in_level_sequence(self, skip_preprocess, simple_circuit, capture_mode):
        """Test that requesting an invalid level type raises an error."""
        if not capture_mode and skip_preprocess:
            pytest.skip(reason="skip_preprocess ignored without program capture.")

        simple_circuit = qp.qjit(
            simple_circuit, skip_preprocess=skip_preprocess, capture=capture_mode
        )

        with pytest.raises(ValueError, match="All elements in 'level' sequence must be integers."):
            mlir_specs(simple_circuit, level=[0, 1.1, 2])

    @pytest.mark.parametrize("level", [3.14, "invalid"])
    def test_invalid_level_type(self, skip_preprocess, simple_circuit, level, capture_mode):
        """Test that requesting an invalid level type raises an error."""
        if not capture_mode and skip_preprocess:
            pytest.skip(reason="skip_preprocess ignored without program capture.")

        simple_circuit = qp.qjit(
            simple_circuit, skip_preprocess=skip_preprocess, capture=capture_mode
        )

        with pytest.raises(
            ValueError, match="The 'level' argument must be an int, a tuple/list of ints, or 'all'."
        ):
            mlir_specs(simple_circuit, level=level)

    @pytest.mark.parametrize("level", [10, -1])
    def test_invalid_int_level(self, skip_preprocess, simple_circuit, level, capture_mode):
        """Test that requesting an invalid level raises an error."""
        if not capture_mode and skip_preprocess:
            pytest.skip(reason="skip_preprocess ignored without program capture.")

        simple_circuit = qp.qjit(
            simple_circuit, skip_preprocess=skip_preprocess, capture=capture_mode
        )

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
    def test_no_passes(self, skip_preprocess, simple_circuit, level, expected, capture_mode):
        """Test that if no passes are applied, the circuit resources are the original amount."""
        if not capture_mode and skip_preprocess:
            pytest.skip(reason="skip_preprocess ignored without program capture.")

        simple_circuit = qp.qjit(
            simple_circuit, skip_preprocess=skip_preprocess, capture=capture_mode
        )
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
    def test_basic_passes(self, skip_preprocess, simple_circuit, level, expected, capture_mode):
        """Test that when passes are applied, the circuit resources are updated accordingly."""
        if not capture_mode and skip_preprocess:
            pytest.skip(reason="skip_preprocess ignored without program capture.")

        simple_circuit = qp.transforms.cancel_inverses(simple_circuit)
        simple_circuit = qp.transforms.merge_rotations(simple_circuit)

        simple_circuit = qp.qjit(
            simple_circuit, skip_preprocess=skip_preprocess, capture=capture_mode
        )
        res = mlir_specs(simple_circuit, level=level)
        assert resources_equal(res, expected)

    def test_basic_passes_level_all(self, skip_preprocess, simple_circuit, capture_mode):
        """Test that when passes are applied, the circuit resources are updated accordingly."""
        if not capture_mode and skip_preprocess:
            pytest.skip(reason="skip_preprocess ignored without program capture.")

        simple_circuit = qp.transforms.cancel_inverses(simple_circuit)
        simple_circuit = qp.transforms.merge_rotations(simple_circuit)

        expected = {
            "Before MLIR Passes": make_static_resources(
                operations={"RX": {1: 2}, "RZ": {1: 2}, "Hadamard": {1: 2}, "CNOT": {2: 2}},
                measurements={"probs(all wires)": 1},
                num_allocs=2,
            ),
            "cancel-inverses": make_static_resources(
                operations={"RX": {1: 2}, "RZ": {1: 2}},
                measurements={"probs(all wires)": 1},
                num_allocs=2,
            ),
            "merge-rotations": make_static_resources(
                operations={"RX": {1: 1}, "RZ": {1: 1}},
                measurements={"probs(all wires)": 1},
                num_allocs=2,
            ),
        }
        if capture_mode and not skip_preprocess:
            # Dummy pass to replace verify_operations
            expected["empty"] = make_static_resources(
                operations={"RX": {1: 1}, "RZ": {1: 1}},
                measurements={"probs(all wires)": 1},
                num_allocs=2,
            )
            # Dummy pass to replace validate_measurements
            expected["empty-2"] = make_static_resources(
                operations={"RX": {1: 1}, "RZ": {1: 1}},
                measurements={"probs(all wires)": 1},
                num_allocs=2,
            )
            # Dummy pass to replace verify_no_state_variance_returns
            expected["empty-3"] = make_static_resources(
                operations={"RX": {1: 1}, "RZ": {1: 1}},
                measurements={"probs(all wires)": 1},
                num_allocs=2,
            )

        simple_circuit = qp.qjit(
            simple_circuit, skip_preprocess=skip_preprocess, capture=capture_mode
        )
        res = mlir_specs(simple_circuit, level="all")

        assert isinstance(res, dict)
        assert len(res) == len(expected)

        for lvl, expected_res in expected.items():
            assert lvl in res.keys()
            assert resources_equal(res[lvl], expected_res)

    def test_basic_passes_multi_level(self, skip_preprocess, simple_circuit, capture_mode):
        """Test that when passes are applied, the circuit resources are updated accordingly."""
        if not capture_mode and skip_preprocess:
            pytest.skip(reason="skip_preprocess ignored without program capture.")

        simple_circuit = qp.transforms.cancel_inverses(simple_circuit)
        simple_circuit = qp.transforms.merge_rotations(simple_circuit)

        expected = {
            "Before MLIR Passes": make_static_resources(
                operations={"RX": {1: 2}, "RZ": {1: 2}, "Hadamard": {1: 2}, "CNOT": {2: 2}},
                measurements={"probs(all wires)": 1},
                num_allocs=2,
            ),
            "merge-rotations": make_static_resources(
                operations={"RX": {1: 1}, "RZ": {1: 1}},
                measurements={"probs(all wires)": 1},
                num_allocs=2,
            ),
        }

        simple_circuit = qp.qjit(
            simple_circuit, skip_preprocess=skip_preprocess, capture=capture_mode
        )
        res = mlir_specs(simple_circuit, level=[0, 2])

        assert isinstance(res, dict)
        assert len(res) == len(expected)

        for lvl, expected_res in expected.items():
            assert lvl in res.keys()
            assert resources_equal(res[lvl], expected_res)

        with pytest.raises(
            ValueError, match="Requested specs levels 20 not found in MLIR pass list."
        ):
            mlir_specs(simple_circuit, level=[0, 20])

    def test_splitting_pass(self, skip_preprocess, capture_mode):
        """Test that when passes are applied, the circuit resources are updated accordingly."""

        @qp.qjit(skip_preprocess=skip_preprocess, capture=capture_mode)
        @qp.transforms.cancel_inverses
        @qp.transform(pass_name="split-non-commuting")
        @qp.qnode(qp.device("null.qubit", wires=2))
        def circuit():
            qp.Hadamard(wires=0)
            qp.Hadamard(wires=0)
            return qp.expval(qp.PauliX(0)), qp.expval(qp.PauliY(0)), qp.expval(qp.PauliZ(0))

        expected = {
            "split-non-commuting": [
                make_static_resources(
                    operations={"Hadamard": {1: 2}},
                    measurements={"expval(PauliX)": 1},
                    num_allocs=2,
                ),
                make_static_resources(
                    operations={"Hadamard": {1: 2}},
                    measurements={"expval(PauliY)": 1},
                    num_allocs=2,
                ),
                make_static_resources(
                    operations={"Hadamard": {1: 2}},
                    measurements={"expval(PauliZ)": 1},
                    num_allocs=2,
                ),
            ],
            "cancel-inverses": [
                make_static_resources(
                    measurements={"expval(PauliX)": 1},
                    num_allocs=2,
                ),
                make_static_resources(
                    measurements={"expval(PauliY)": 1},
                    num_allocs=2,
                ),
                make_static_resources(
                    measurements={"expval(PauliZ)": 1},
                    num_allocs=2,
                ),
            ],
        }

        res = mlir_specs(circuit, level=[1, 2])

        assert isinstance(res, dict)
        assert len(res) == len(expected)

        for lvl, expected_res in expected.items():
            assert lvl in res.keys()
            assert isinstance(res[lvl], list)
            assert len(res[lvl]) == len(expected_res)
            for r, er in zip(res[lvl], expected_res):
                assert resources_equal(r, er)

    def test_not_qnode(self, skip_preprocess, capture_mode):
        """Test that a malformed QNode raises an error."""

        def not_a_qnode():
            pass

        with pytest.raises(
            ValueError,
            match="The provided `qnode` argument does not appear to be a valid QJIT "
            "compiled QNode.",
        ):
            mlir_specs(not_a_qnode, level=0)

    def test_malformed_qnode(self, skip_preprocess, capture_mode):
        """Test that a QNode without measurements can still be collected."""
        if not capture_mode and skip_preprocess:
            pytest.skip(reason="skip_preprocess ignored without program capture.")

        dev = qp.device("lightning.qubit", wires=1)

        @qp.qjit(skip_preprocess=skip_preprocess, capture=capture_mode)
        @qp.qnode(dev)
        def circ(wire):
            qp.X(0)
            qp.X(wire)

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
    def test_fixed_loop(self, skip_preprocess, pl_ctrl_flow, iters, autograph, capture_mode):
        """Test that loop resources are counted correctly."""
        if not capture_mode and skip_preprocess:
            pytest.skip(reason="skip_preprocess ignored without program capture.")

        if pl_ctrl_flow:

            @qp.qnode(qp.device("lightning.qubit", wires=2))
            def circ():
                @qp.for_loop(iters)
                def loop_body(i):
                    qp.X(i % 2)

                loop_body()

                return qp.state()

        else:

            @qp.qnode(qp.device("lightning.qubit", wires=2))
            def circ():
                for i in range(iters):
                    qp.X(i % 2)
                return qp.state()

        expected = make_static_resources(
            operations={"PauliX": {1: iters}},
            measurements={"state(all wires)": 1},
            num_allocs=2,
        )

        circ = qp.qjit(
            circ, autograph=autograph, skip_preprocess=skip_preprocess, capture=capture_mode
        )
        res = mlir_specs(circ, level=0)
        assert resources_equal(res, expected)

    @pytest.mark.parametrize(
        "pl_ctrl_flow, iters",
        [
            (True, 5),
            (False, 2),
        ],
    )
    def test_dynamic_for_loop(self, skip_preprocess, pl_ctrl_flow, iters, capture_mode):
        """Test that dynamic for loops emit a warning."""
        if not capture_mode and skip_preprocess:
            pytest.skip(reason="skip_preprocess ignored without program capture.")

        if pl_ctrl_flow:

            @qp.qnode(qp.device("lightning.qubit", wires=2))
            def circ(n):
                @qp.for_loop(n)
                def loop_body(i):
                    qp.X(i % 2)

                loop_body()

                return qp.state()

        else:

            @qp.qnode(qp.device("lightning.qubit", wires=2))
            def circ(n):
                for i in range(n):
                    qp.X(i % 2)
                return qp.state()

        expected = make_static_resources(
            operations={"PauliX": {1: 1}},
            measurements={"state(all wires)": 1},
            num_allocs=2,
        )

        circ = qp.qjit(circ, autograph=True, skip_preprocess=skip_preprocess, capture=capture_mode)

        with pytest.warns(
            UserWarning,
            match="Specs was unable to determine the number of loop iterations. "
            "The results will assume the loop runs only once. "
            "This may be fixed in some cases by inlining dynamic arguments.",
        ):
            res = mlir_specs(circ, 0, iters)
            assert resources_equal(res, expected)

    @pytest.mark.parametrize(
        "pl_ctrl_flow, iters",
        [
            (True, 5),
            (False, 2),
        ],
    )
    def test_while_loop(self, skip_preprocess, pl_ctrl_flow, iters, capture_mode):
        """Test that dynamic while loops emit a warning."""
        if not capture_mode and skip_preprocess:
            pytest.skip(reason="skip_preprocess ignored without program capture.")

        if pl_ctrl_flow:

            @qp.qnode(qp.device("lightning.qubit", wires=2))
            def circ(n):
                def loop_cond(i):
                    return i < n

                @qp.while_loop(loop_cond)
                def loop_body(i):
                    qp.X(i % 2)
                    return i + 1

                loop_body(0)

                return qp.state()

        else:

            @qp.qnode(qp.device("lightning.qubit", wires=2))
            def circ(n):
                i = 0
                while i < n:
                    qp.X(i % 2)
                    i += 1
                return qp.state()

        expected = make_static_resources(
            operations={"PauliX": {1: 1}},
            measurements={"state(all wires)": 1},
            num_allocs=2,
        )

        circ = qp.qjit(circ, autograph=True, skip_preprocess=skip_preprocess, capture=capture_mode)

        with pytest.warns(
            UserWarning,
            match="Specs was unable to determine the number of loop iterations. "
            "The results will assume the loop runs only once. "
            "This may be fixed in some cases by inlining dynamic arguments.",
        ):
            res = mlir_specs(circ, 0, iters)
            assert resources_equal(res, expected)

    @pytest.mark.parametrize(
        "pl_ctrl_flow",
        [
            (True),
            (False),
        ],
    )
    def test_cond(self, skip_preprocess, pl_ctrl_flow, capture_mode):
        """Test that conditions emit a warning."""
        if not capture_mode and skip_preprocess:
            pytest.skip(reason="skip_preprocess ignored without program capture.")

        if pl_ctrl_flow:

            @qp.qnode(qp.device("lightning.qubit", wires=2))
            def circ(n):
                qp.cond(n > 0, qp.X, qp.Z)(0)

                return qp.state()

        else:

            @qp.qnode(qp.device("lightning.qubit", wires=2))
            def circ(n):
                if n > 0:
                    qp.X(0)
                else:
                    qp.Z(0)
                return qp.state()

        expected = make_static_resources(
            operations={"PauliX": {1: 1}, "PauliZ": {1: 1}},
            measurements={"state(all wires)": 1},
            num_allocs=2,
        )

        circ = qp.qjit(circ, autograph=True, skip_preprocess=skip_preprocess, capture=capture_mode)

        with pytest.warns(
            UserWarning,
            match="Specs was unable to determine the branch of a conditional or switch statement. "
            "The results will take the maximum resources across all possible branches.",
        ):
            n = 3  # Arbitrary value for n
            res = mlir_specs(circ, 0, n)
            assert resources_equal(res, expected)

    def test_tape_transforms(self, skip_preprocess, capture_mode):
        """Test that tape transforms are handled correctly."""
        if capture_mode:
            pytest.xfail("Currently broken with plxpr enabled.")
        if not capture_mode and skip_preprocess:
            pytest.skip(reason="skip_preprocess ignored without program capture.")

        @qp.qnode(qp.device("lightning.qubit", wires=2))
        def circ():
            qp.GlobalPhase(0.5)
            qp.GlobalPhase(1.0)
            return qp.expval(qp.PauliZ(0))

        circ = qp.transforms.combine_global_phases(circ)
        circ = qp.qjit(circ, skip_preprocess=skip_preprocess, capture=capture_mode)

        expected = make_static_resources(
            operations={"GlobalPhase": {0: 1}},
            measurements={"expval(PauliZ)": 1},
            num_allocs=2,
        )

        res = mlir_specs(circ, level=0)
        assert resources_equal(res, expected)

    def test_stateprep(self, skip_preprocess, capture_mode):
        """Test that StatePrep operations are handled correctly."""
        if not capture_mode and skip_preprocess:
            pytest.skip(reason="skip_preprocess ignored without program capture.")

        @qp.qjit(skip_preprocess=skip_preprocess, capture=capture_mode)
        @qp.qnode(qp.device("lightning.qubit", wires=3), shots=10)
        def circ():
            qp.StatePrep(jnp.array([1, 0, 0, 0]), wires=[0, 1])
            qp.Hadamard(wires=1)
            qp.Hadamard(wires=2)
            return qp.sample()

        expected = make_static_resources(
            operations={"StatePrep": {2: 1}, "Hadamard": {1: 2}},
            measurements={"sample(all wires)": 1},
            num_allocs=3,
        )

        res = mlir_specs(circ, level=0)
        assert resources_equal(res, expected)

    def test_adjoint(self, skip_preprocess, capture_mode):
        """Test that adjoint operations are handled correctly."""
        if not capture_mode and skip_preprocess:
            pytest.skip(reason="skip_preprocess ignored without program capture.")

        @qp.qnode(qp.device("lightning.qubit", wires=2))
        def circ():
            def subroutine():
                qp.Hadamard(wires=0)
                qp.CNOT(wires=[0, 1])
                qp.RZ(0.789, wires=1)
                qp.adjoint(qp.T)(wires=1)

            qp.adjoint(subroutine)()
            return qp.probs()

        circ = qp.qjit(circ, skip_preprocess=skip_preprocess, capture=capture_mode)

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

    def test_hamiltonian(self, skip_preprocess, capture_mode):
        """Test that Hamiltonian observables are handled correctly."""
        if not capture_mode and skip_preprocess:
            pytest.skip(reason="skip_preprocess ignored without program capture.")

        @qp.qjit(skip_preprocess=skip_preprocess, capture=capture_mode)
        @qp.qnode(qp.device("lightning.qubit", wires=2))
        def circ(i: int):

            coeffs = [0.2, -0.543]
            obs = [qp.X(0) @ qp.Z(1), qp.Z(i) @ qp.Hadamard(2)]
            ham1 = qp.Hamiltonian([1.0], [qp.Z(0) @ qp.Z(1)])
            ham2 = qp.ops.LinearCombination(coeffs, obs)

            return qp.expval(ham1), qp.expval(ham2)

        expected = make_static_resources(
            operations={},
            measurements={
                "expval(Hamiltonian(num_terms=1))": 1,
                "expval(Hamiltonian(num_terms=2))": 1,
            },
            num_allocs=2,
        )

        res = mlir_specs(circ, level=0, args=(0,))
        assert resources_equal(res, expected)

    def test_ppr(self, skip_preprocess, capture_mode):
        """Test that PPRs are handled correctly."""
        if not capture_mode:
            pytest.xfail("to_ppr requires plxpr to be enabled to lower PauliRot")
        if not capture_mode and skip_preprocess:
            pytest.skip(reason="skip_preprocess ignored without program capture.")

        pipeline = [("pipe", ["enforce-runtime-invariants-pipeline"])]

        @qp.qjit(
            pipelines=pipeline, target="mlir", skip_preprocess=skip_preprocess, capture=capture_mode
        )
        @qp.transform(pass_name="to-ppr")
        @qp.qnode(qp.device("null.qubit", wires=2))
        def circ():
            qp.H(0)
            qp.T(0)
            qp.PauliRot(0.1234, pauli_word="Z", wires=0)

        expected = make_static_resources(
            operations={
                "GlobalPhase": {0: 2},
                "PPR-pi/4": {1: 3},
                "PPR-pi/8": {1: 1},
                "PPR-Phi": {1: 1},
            },
            measurements={},
            num_allocs=2,
        )

        res = mlir_specs(circ, level=1)
        assert resources_equal(res, expected)

    def test_subroutine(self, skip_preprocess, capture_mode):
        """Test that subroutines are handled correctly."""
        if not capture_mode:
            pytest.xfail("Subroutine requires plxpr to be enabled.")
        if not capture_mode and skip_preprocess:
            pytest.skip(reason="skip_preprocess ignored without program capture.")

        @qp.capture.subroutine
        def extra_function():
            qp.Hadamard(wires=0)

        @qp.qjit(autograph=True, skip_preprocess=skip_preprocess, capture=capture_mode)
        @qp.qnode(qp.device("null.qubit", wires=2))
        def circ():
            extra_function()
            return qp.probs()

        expected = make_static_resources(
            operations={"Hadamard": {1: 1}},
            measurements={"probs(all wires)": 1},
            num_allocs=2,
            function_calls={"extra_function": 1},
        )

        res = mlir_specs(circ, level=0)
        assert resources_equal(res, expected)

    @pytest.mark.usefixtures("use_capture_dgraph")
    def test_graph_decomp(self, skip_preprocess):
        """Test that graph decomposition is handled correctly."""
        if not qp.capture.enabled() and skip_preprocess:
            pytest.skip(reason="skip_preprocess ignored without program capture.")

        @qp.register_resources({qp.H: 2, qp.CZ: 1})
        def my_cnot(wires):
            qp.H(wires=wires[1])
            qp.CZ(wires=wires)
            qp.H(wires=wires[1])

        @qp.qjit(skip_preprocess=skip_preprocess)
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

        expected_resources = {"CZ": {2: 1}, "Hadamard": {1: 3}}
        resources = mlir_specs(circuit, level=1)
        assert resources.operations == expected_resources


# TODO: In the future, it would be good to add unit tests for specs_collector instead of just
#   integration tests

if __name__ == "__main__":
    pytest.main(["-x", __file__])
