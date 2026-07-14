# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the fabricate operation."""

import pytest

import pennylane as qp
from pennylane.ops.qubit.fabricate import Fabricate, fabricate, _VALID_INIT_STATES

from catalyst import qjit

_FABRICATE_PIPELINE = [
    (
        "pipe",
        [
            "canonicalize",
            "verify-no-quantum-use-after-free",
            "convert-to-value-semantics",
            "canonicalize",
        ],
    )
]


class TestFabricateOp:
    """Tests for the Fabricate operator and function."""

    @pytest.mark.parametrize("init_state", sorted(_VALID_INIT_STATES))
    def test_fabricate_valid_init_states(self, init_state):
        """Test that valid init states are accepted."""
        op = Fabricate(init_state)
        assert op.init_state == init_state
        assert len(op.wires) == 0

    def test_fabricate_invalid_init_state(self):
        """Test that invalid init states are rejected."""
        with pytest.raises(ValueError, match="not allowed"):
            Fabricate("zero")

    @pytest.mark.parametrize("init_state", sorted(_VALID_INIT_STATES))
    @pytest.mark.usefixtures("use_capture")
    def test_fabricate_function_capture(self, init_state):
        """Test fabricate primitive binding under capture."""
        import jax

        def circuit():
            fabricate(init_state)

        jaxpr = jax.make_jaxpr(circuit)().jaxpr
        assert any(eqn.primitive.name == "fabricate" for eqn in jaxpr.eqns)

    @pytest.mark.usefixtures("enable_graph_decomposition")
    def test_fabricate_mlir_lowering(self):
        """Test that fabricate appears as pbc.fabricate in optimized MLIR."""
        dev = qp.device("null.qubit", wires=1)

        @qjit(pipelines=_FABRICATE_PIPELINE, target="mlir", capture=True)
        @qp.qnode(device=dev)
        def circuit():
            magic = fabricate("magic_conj")
            qp.pauli_measure("Z", wires=[magic])
            qp.deallocate(magic)
            return qp.expval(qp.Z(0))

        mlir = circuit.mlir_opt
        assert "pbc.fabricate" in mlir and "magic_conj" in mlir
        assert "pbc.ref.fabricate" not in mlir

    @pytest.mark.usefixtures("enable_graph_decomposition")
    def test_fabricate_mlir_lowering_capture_false(self):
        """Test fabricate lowering via the legacy tracing pathway."""
        dev = qp.device("null.qubit", wires=2)
        pipe = [
            (
                "pipe",
                [
                    "canonicalize",
                    "verify-no-quantum-use-after-free",
                    "convert-to-value-semantics",
                    "canonicalize",
                ],
            )
        ]

        @qjit(pipelines=pipe, target="mlir", capture=False)
        @qp.qnode(device=dev)
        def circuit():
            magic = fabricate("magic")
            qp.pauli_measure("ZZ", wires=[0, magic])
            qp.deallocate(magic)
            return qp.expval(qp.Z(0))

        mlir = circuit.mlir_opt
        assert "pbc.fabricate" in mlir and "magic" in mlir
        assert "pbc.ref.fabricate" not in mlir
        assert "pbc.ppm" in mlir
        assert "quantum.dealloc_qb" in mlir
