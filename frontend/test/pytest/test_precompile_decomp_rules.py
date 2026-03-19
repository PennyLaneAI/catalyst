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

"""Tests for the decomposition rule precompilation utilities."""

import pennylane as qp
import pytest

from catalyst.compiler import _quantum_opt
from catalyst.utils.precompile_decomposition_rules import (
    BYTECODE_FILE_PATH,
    COMPILER_OPS_FOR_DECOMPOSITION,
    compile_op_decomp_rules,
    get_abstract_args,
    get_compiler_ops,
    precompile_decomp_rules,
)


class TestGetCompilerOps:
    """Tests for get_compiler_ops."""

    def test_get_compiler_ops(self):
        """Test that get_compiler_ops succeeds in finding all compiler ops."""
        ops = get_compiler_ops()

        assert len(ops) == len(COMPILER_OPS_FOR_DECOMPOSITION)

    def test_warning(self):
        """Test that get_compiler_ops warns when an op is not found."""
        with pytest.warns():
            get_compiler_ops(frozenset(["fake_op"]))


class TestGetAbstractArgs:
    """Tests for get_abstract_args."""

    def test_ignore_wires(self):
        """Test that get_abstract_args correctly ignores WiresLike params."""
        assert not get_abstract_args(qp.X)

    def test_missing_ndim_params(self):
        """Test that get_abstract_args correctly handles missing ndim_params properties."""
        assert not get_abstract_args(qp.H)

    def test_0_in_ndim_params(self):
        """Test that get_abstract_args correctly handles 0 values in ndim_params."""
        assert get_abstract_args(qp.RX) == [float]

    def test_multiple_values_in_ndim_params(self):
        """Test that get_abstract_args correctly handles length > 1 ndim_params."""
        assert get_abstract_args(qp.U3) == [float, float, float]

    def test_dimension_failure(self):
        """
        Test that get_abstract_args correctly raises an exception given an op with
        multi-dimensional params.
        """
        with pytest.raises(ValueError, match="Cannot generate arguments"):
            get_abstract_args(qp.ControlledQubitUnitary)


class TestCompileOpDecompRules:
    """Tests for compile_op_decomp_rules."""

    def test_compile_hadamard_rules(self):
        """Test that compile_op_decomp_rules successfully compiles each decomp rule for Hadamard."""
        rules = compile_op_decomp_rules(qp.H)

        assert "_hadamard_to_rz_rx" in rules
        assert "_hadamard_to_rz_ry" in rules

    def test_compile_rx_rules(self):
        """Test that compile_op_decomp_rules successfully compiles each decomp rule for RX gates."""
        rules = compile_op_decomp_rules(qp.RX)

        assert "_rx_to_rot" in rules
        assert "_rx_to_rz_ry" in rules
        assert "_rx_to_ry_cliff" in rules
        assert "_rx_to_rz_cliff" in rules
        assert "_rx_to_ppr" in rules

    def test_fails_with_unknown_wires(self):
        """
        Test that compile_op_decomp_rules warns when the number of wires is unknown.
        """
        with pytest.warns():
            compile_op_decomp_rules(qp.Identity)

    def test_compile_error(self):
        """Test that compile_op_decomp_rules warns when compilation fails."""

        class FakeOp(qp.operation.Operator):
            """Test class with incompatible decomp rule."""

            num_wires = 1
            num_params = 1
            ndim_params = (0,)

        @qp.register_resources({})
        def fake_op_decomp(string):
            qp.PauliRot(2, string)

        qp.add_decomps(FakeOp, fake_op_decomp)

        with pytest.warns(match="Unexpected error"):
            compile_op_decomp_rules(FakeOp)

    def test_unexpected_error(self):
        """Test that compile_op_decomp_rules warns when an unexpected exception is thrown."""

        class FakeOp(qp.operation.Operator):
            """Test class without ndim_params."""

            num_wires = 1
            num_params = 1

        @qp.register_resources({})
        def fake_op_decomp(string):
            qp.PauliRot(2, string)

        qp.add_decomps(FakeOp, fake_op_decomp)

        with pytest.warns(match="Unexpected error"):
            compile_op_decomp_rules(FakeOp)


def test_bytecode_file():
    """Test that the bytecode file is generated correctly."""
    BYTECODE_FILE_PATH.unlink(missing_ok=True)

    precompile_decomp_rules()

    assert BYTECODE_FILE_PATH.exists()

    # NOTE: empty pass is needed to prevent running default pipeline
    rules = _quantum_opt("--empty", str(BYTECODE_FILE_PATH))

    assert "_isingxy_to_h_cy" in rules
    assert "_doublexcit" in rules
    assert "_toffoli_to_ppr" in rules
    assert "_pauliz_to_ps" in rules
    assert "_cphase_to_ppr" in rules
    assert "_crot" in rules
