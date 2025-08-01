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

"""Testing interface around main plugin functionality"""

from pathlib import Path
from tempfile import NamedTemporaryFile

import pennylane as qml
import pytest
from jax.interpreters.mlir import ir

import catalyst
from catalyst import qjit


def test_path_does_not_exists():
    """Test what happens when a pass_plugin is given an path that does not exist"""

    with pytest.raises(FileNotFoundError, match="does not exist"):
        catalyst.passes.apply_pass_plugin(
            "this-path-does-not-exist", "this-pass-also-doesnt-exists"
        )

    with pytest.raises(FileNotFoundError, match="does not exist"):
        catalyst.passes.apply_pass_plugin(
            Path("this-path-does-not-exist"), "this-pass-also-doesnt-exists"
        )


def test_pass_can_aot_compile():
    """Can we AOT compile when using apply_pass?"""

    @qjit(target="mlir")
    @catalyst.passes.apply_pass("some-pass")
    @qml.qnode(qml.device("null.qubit", wires=1))
    def example():
        return qml.state()

    assert example.mlir


@pytest.mark.skip()
def test_pass_plugin_can_aot_compile():
    """Can we AOT compile when using apply_pass_plugin?

    We can't properly test this because tmp needs to be a valid MLIR plugin.
    And therefore can only be tested when a valid MLIR plugin exists in the path.
    """

    with NamedTemporaryFile() as tmp:

        @qjit(target="mlir")
        @catalyst.passes.apply_pass_plugin(Path(tmp.name), "some-pass")
        @qml.qnode(qml.device("null.qubit", wires=1))
        def example():
            return qml.state()

        assert example.mlir


def test_get_options():
    """
    Test get_options from Pass

    ApplyRegisteredPassOp expects options to be a dictionary from strings to attributes.
    See https://github.com/llvm/llvm-project/pull/143159
    """
    with ir.Context(), ir.Location.unknown():
        options = catalyst.passes.Pass("example-pass", "single-option").get_options()
        assert isinstance(options, dict)
        assert isinstance(options["single-option"], ir.BoolAttr)
        assert options["single-option"].value == True

        options = catalyst.passes.Pass("example-pass", "an-option", "bn-option").get_options()
        assert isinstance(options, dict)
        assert isinstance(options["an-option"], ir.BoolAttr)
        assert options["an-option"].value == True
        assert isinstance(options["bn-option"], ir.BoolAttr)
        assert options["bn-option"].value == True

        options = catalyst.passes.Pass("example-pass", option=True).get_options()
        assert isinstance(options, dict)
        assert isinstance(options["option"], ir.BoolAttr)
        assert options["option"].value == True


@pytest.mark.skip(reason="xdsl not installed in ci cd yet")
def test_xdsl_plugin():
    """Here, we just test that we are able to run."""

    @catalyst.qjit
    @catalyst.passes.apply_pass("catalyst_xdsl_plugin.remove-chained-self-inverse")
    @qml.qnode(qml.device("null.qubit", wires=1))
    def example():
        return qml.state()

    example()
