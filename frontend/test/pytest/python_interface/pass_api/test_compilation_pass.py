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
"""Tests for CompilationPass."""

import pytest

from xdsl.ir import Operation

from catalyst.python_interface.dialects import quantum
from catalyst.python_interface.pass_api import CompilationPass
from catalyst.python_interface.conversion import parse_generic_to_xdsl_module


def create_test_pass(greedy: bool, recursive: bool) -> CompilationPass:
    """Helper to create a compilation pass for testing"""

    class MyPass(CompilationPass):
        """Compilation pass for testing."""

        name = "my-pass"

        def action(self, op: quantum.CustomOp, rewriter):
            """Default action."""
            assert isinstance(op, quantum.CustomOp)
            print(f"Default action: found gate {op.name}.")

    MyPass.greedy = greedy
    MyPass.recursive = recursive

    @MyPass.add_action
    def gate_action(self, op: quantum.CustomOp, rewriter):
        """Action on gates."""
        assert isinstance(op, quantum.CustomOp)
        print(f"Found gate {op.gate_name}.")

    @MyPass.add_action
    def mcm_action(self, op: quantum.MeasureOp, rewriter):
        """Action on mcms."""
        assert isinstance(op, quantum.MeasureOp)
        print("Found MCM.")

    @MyPass.add_action
    def insert_extract_action(self, op: quantum.InsertOp | quantum.ExtractOp, rewriter):
        """Action on qubit inserts and extracts."""
        assert isinstance(op, (quantum.InsertOp, quantum.ExtractOp))
        if isinstance(op, quantum.InsertOp):
            print("Found insert.")
        else:
            print("Found extract.")

    return MyPass


class TestCompilationPass:
    """Unit tests for CompilationPass when greedy=True, recursive=True."""

    _pass = create_test_pass(greedy=True, recursive=True)


class TestCompilationPassNotGreedy:
    """Unit tests for CompilationPass when greedy=False, recursive=True."""

    _pass = create_test_pass(greedy=False, recursive=True)


class TestCompilationPassNotRecursive:
    """Unit tests for CompilationPass when greedy=True, recursive=False."""

    _pass = create_test_pass(greedy=True, recursive=False)


class TestCompilationPassNotGreedyNotRecursive:
    """Unit tests for CompilationPass when greedy=False, recursive=False."""

    _pass = create_test_pass(greedy=False, recursive=False)
