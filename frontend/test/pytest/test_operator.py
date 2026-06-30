# Copyright 2022-2026 Xanadu Quantum Technologies Inc.

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
Tests for the new Operator2 class.
"""

# pylint: disable = useless-parent-delegation, missing-function-docstring, missing-class-docstring
import pennylane as qp
import pytest


class DummyOp(qp.core.Operator2):

    def __init__(self, wires):
        super().__init__(wires=wires)


def test_hybrid_not_supported_yet():
    """Test that hybrid arguments are not yet supported."""

    class OperatorArgument(qp.core.Operator2):

        hybrid_argnames = ("op",)
        wire_argnames = ()

        def __init__(self, op):
            super().__init__(op)

    with pytest.raises(NotImplementedError):

        @qp.qjit(capture=True)
        @qp.qnode(qp.device("null.qubit", wires=3))
        def c():
            OperatorArgument(DummyOp(0))
            return qp.state()


def test_static_argnames():
    """Test that static arguments are not yet supported."""

    class StaticArgsOp(qp.core.Operator2):

        static_argnames = ("thing",)

        def __init__(self, thing, wires):
            super().__init__(thing, wires)

    with pytest.raises(NotImplementedError):

        @qp.qjit(capture=True)
        @qp.qnode(qp.device("null.qubit", wires=2))
        def c():
            StaticArgsOp("hello", 0)
            return qp.state()
