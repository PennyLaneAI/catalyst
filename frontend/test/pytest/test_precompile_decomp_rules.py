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

import subprocess

import pennylane as qp

from catalyst.utils.precompile_decomposition_rules import (
    BYTECODE_FILE_PATH,
    COMPILER_OPS_FOR_DECOMPOSITION,
    compile_op_decomp_rules,
    get_abstract_args,
    get_compiler_ops,
    precompile_decomp_rules,
)
from catalyst.utils.runtime_environment import DEFAULT_BIN_PATHS


def test_get_compiler_ops():
    """Test that get_compiler_ops succeeds in finding all compiler ops."""
    ops, failures = get_compiler_ops()

    assert len(ops) == len(COMPILER_OPS_FOR_DECOMPOSITION)

    assert failures == 0


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


def test_bytecode_file():
    """Test that the bytecode file is generated correctly."""
    BYTECODE_FILE_PATH.unlink()

    precompile_decomp_rules()

    assert BYTECODE_FILE_PATH.exists()

    rules = subprocess.run(
        (
            f"{DEFAULT_BIN_PATHS['cli']}/quantum-opt",
            BYTECODE_FILE_PATH,
        ),
        capture_output=True,
        check=True,
        text=True,
    ).stdout

    assert "_isingxy_to_h_cy" in rules
    assert "_doublexcit" in rules
    assert "_toffoli_to_ppr" in rules
    assert "_pauliz_to_ps" in rules
    assert "_cphase_to_ppr" in rules
    assert "_crot" in rules
