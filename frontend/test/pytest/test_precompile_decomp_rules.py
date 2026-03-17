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

from catalyst.utils.precompile_decomposition_rules import (
    COMPILER_OPS_FOR_DECOMPOSITION,
    compile_op_decomp_rules,
    get_abstract_args,
    get_compiler_ops,
)


def test_get_compiler_ops():
    """
    Test that get_compiler_ops succeeds in finding all compiler ops.
    """
    ops, failures = get_compiler_ops()

    assert len(ops) == len(COMPILER_OPS_FOR_DECOMPOSITION)

    assert failures == 0


class TestGetAbstractArgs:
    """
    Tests for get_abstract_args.
    """

    def test_empty_func(self):
        """
        Test that get_abstract_args correctly handles funcs with no args.
        """

        def empty():
            return

        assert not get_abstract_args(empty)

    def test_int_param(self):
        """
        Test that get_abstract_args correctly handles funcs with int params.
        """

        def int_param(x: int):
            return x

        assert get_abstract_args(int_param) == [int]

    def test_float_param(self):
        """
        Test that get_abstract_args correctly handles funcs with float params.
        """

        def float_param(x: float):
            return x * 2

        assert get_abstract_args(float_param) == [float]

    def test_tensorlike_param(self):
        """
        Test that get_abstract_args correctly handles funcs with TensorLike params.
        """

        def tensorlike_param(a: TensorLike):
            return 3 + a

        assert get_abstract_args(tensorlike_param) == [float]

    def test_ignore_wires(self):
        """
        Test that get_abstract_args correctly ignores WiresLike params.
        """

        def wire_param(wires: WiresLike):
            return wires

        assert not get_abstract_args(wire_param)

    def test_mixed_params(self):
        """
        Test that get_abstract_args correctly handles mixed params.
        """

        def mixed_params(x: int, y: float, w: WiresLike):
            return int(x - y), w

        assert get_abstract_args(mixed_params) == [
            int,
            float,
        ]

    def test_named_params(self):
        """
        Test that get_abstract_args correctly guesses for named params.
        """

        def angle_names(theta, phi, omega):
            return theta + phi + omega

        assert get_abstract_args(angle_names) == [float, float, float]


class TestCompileOpDecompRules:
    """
    Tests for compile_op_decomp_rules.
    """

    def test_hadamard(self):
        """
        Test that compile_op_decomp_rules successfully compiles each decomp rule for Hadamards
        """
        rules = compile_op_decomp_rules(qp.H)

        assert "_hadamard_to_rz_rx" in rules
        assert "_hadamard_to_rz_ry" in rules

    def test_rx(self):
        """
        Test that compile_op_decomp_rules successfully compiles each decomp rule for Hadamards
        """
        rules = compile_op_decomp_rules(qp.RX)

        assert "_rx_to_rot" in rules
        assert "_rx_to_rz_ry" in rules
        assert "_rx_to_ry_cliff" in rules
        assert "_rx_to_rz_cliff" in rules
        assert "_rx_to_ppr" in rules
