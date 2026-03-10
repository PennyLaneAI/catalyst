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

"""
Tests for the decomposition rule precompilation utilities.
"""

import pennylane as qp
from pennylane.typing import TensorLike
from pennylane.wires import WiresLike
from precompile import (
    DECOMP_FILE_NAME,
    DEFAULT_RULE_DIR,
    compile_op_decomp_rules,
    get_compiler_ops,
    get_dummy_args,
)

from catalyst.from_plxpr.decompose import COMPILER_OPS_FOR_DECOMPOSITION


def test_get_compiler_ops():
    """
    Test that get_compiler_ops succeeds in finding all compiler ops.
    """
    ops, failures = get_compiler_ops()

    assert len(ops) == len(COMPILER_OPS_FOR_DECOMPOSITION)

    assert failures == 0


class TestGetDummyArgs:
    """
    Tests for get_dummy_args.
    """

    def test_empty_func(self):
        """
        Test that get_dummy_args correctly handles funcs with no args.
        """

        def empty():
            return

        assert not get_dummy_args(empty)

    def test_int_param(self):
        """
        Test that get_dummy_args correctly handles funcs with int params.
        """

        def int_param(x: int):
            return x

        assert get_dummy_args(int_param) == [0]

    def test_float_param(self):
        """
        Test that get_dummy_args correctly handles funcs with float params.
        """

        def float_param(x: float):
            return x * 2

        assert get_dummy_args(float_param) == [0.0]

    def test_tensorlike_param(self):
        """
        Test that get_dummy_args correctly handles funcs with TensorLike params.
        """

        def tensorlike_param(a: TensorLike):
            return 3 + a

        assert get_dummy_args(tensorlike_param) == [0.0]

    def test_str_param(self):
        """
        Test that get_dummy_args correctly handles funcs with str params.
        """

        def str_param(string: str):
            return "hello " + string

        assert get_dummy_args(str_param) == ["XX"]

    def test_ignore_wires(self):
        """
        Test that get_dummy_args correctly ignores WiresLike params.
        """

        def wire_param(wires: WiresLike):
            return wires

        assert not get_dummy_args(wire_param)

    def test_mixed_params(self):
        """
        Test that get_dummy_args correctly handles mixed params.
        """

        def mixed_params(x: int, y: float, z: str, w: WiresLike):
            return int(x - y) * z, w

        assert get_dummy_args(mixed_params) == [0, 0.0, "XX"]

    def test_named_params(self):
        """
        Test that get_dummy_args correctly guesses for named params.
        """

        def pauli_names(pauli_word, pauli_string):
            return pauli_word + " and " + pauli_string

        assert get_dummy_args(pauli_names) == ["XX", "XX"]

        def angle_names(theta, phi, omega):
            return theta + phi + omega

        assert get_dummy_args(angle_names) == [0.0, 0.0, 0.0]


class TestCompileOpDecompRules:
    """
    Tests for compile_op_decomp_rules.
    """

    def test_hadamard(self):
        """
        Test that compile_op_decomp_rules successfully compiles each decomp rule for Hadamards
        """
        rules, successes, failures = compile_op_decomp_rules(qp.H)

        assert "_hadamard_to_rz_rx" in rules
        assert "_hadamard_to_rz_ry" in rules

        assert successes == 2
        assert failures == 0

    def test_rx(self):
        """
        Test that compile_op_decomp_rules successfully compiles each decomp rule for Hadamards
        """
        rules, successes, failures = compile_op_decomp_rules(qp.RX)

        assert "_rx_to_rot" in rules
        assert "_rx_to_rz_ry" in rules
        assert "_rx_to_ry_cliff" in rules
        assert "_rx_to_rz_cliff" in rules
        assert "_rx_to_ppr" in rules

        assert successes == 5
        assert failures == 0


def test_mlir_output():
    """
    Spot checks that the compiled rules appear in the mlir file.
    """

    rules = ""
    with open(DEFAULT_RULE_DIR / DECOMP_FILE_NAME, encoding="utf-8") as mlir_file:
        rules = mlir_file.read()

    assert "_rx_to_rot" in rules
    assert "_hadamard_to_rz_rx" in rules
    assert "_rot_to_rz_ry_rz" in rules
    assert "_cswap" in rules
    assert "_isingxy_to_h_cy" in rules
