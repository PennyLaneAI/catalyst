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

import pennylane as qp
import pytest
from jax.interpreters.mlir import ir
from pennylane.transforms.core import BoundTransform

from catalyst import qjit
from catalyst.jax_primitives_utils import _lowered_options


def test_pass_can_aot_compile():
    """Can we AOT compile when using qp.transform(pass_name=...)?"""

    @qjit(target="mlir")
    @qp.transform(pass_name="some-pass")
    @qp.qnode(qp.device("null.qubit", wires=1))
    def example():
        return qp.state()

    assert example.mlir


@pytest.mark.skip()
def test_pass_plugin_can_aot_compile():
    """Can we AOT compile when using pass_plugins with qp.transform?

    We can't properly test this because tmp needs to be a valid MLIR plugin.
    And therefore can only be tested when a valid MLIR plugin exists in the path.
    """

    with NamedTemporaryFile() as tmp:

        @qjit(target="mlir", pass_plugins=[Path(tmp.name)])
        @qp.transform(pass_name="some-pass")
        @qp.qnode(qp.device("null.qubit", wires=1))
        def example():
            return qp.state()

        assert example.mlir


def test_get_options():
    """
    Test lowered options from BoundTransform

    ApplyRegisteredPassOp expects options to be a dictionary from strings to attributes.
    See https://github.com/llvm/llvm-project/pull/143159
    """
    with ir.Context(), ir.Location.unknown():
        options = _lowered_options(qp.transform(pass_name="example-pass")("single-option"))
        assert isinstance(options, ir.DictAttr)
        assert isinstance(options["single-option"], ir.BoolAttr)
        assert options["single-option"].value == True

        options = _lowered_options(qp.transform(pass_name="example-pass")("an-option", "bn-option"))
        assert isinstance(options, ir.DictAttr)
        assert isinstance(options["an-option"], ir.BoolAttr)
        assert options["an-option"].value == True
        assert isinstance(options["bn-option"], ir.BoolAttr)
        assert options["bn-option"].value == True

        options = _lowered_options(qp.transform(pass_name="example-pass")(option=True))
        assert isinstance(options, ir.DictAttr)
        assert isinstance(options["option"], ir.BoolAttr)
        assert options["option"].value == True
