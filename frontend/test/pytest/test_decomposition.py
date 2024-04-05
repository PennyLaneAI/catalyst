# Copyright 2022-2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pennylane as qml
import pytest
from jax import numpy as jnp

from catalyst import CompileError, ctrl, measure, qjit
from catalyst.compiler import get_lib_path

# This is used just for internal testing
from catalyst.pennylane_extensions import qfunc


class TestControlledDecomposition:
    """Test behaviour around the decomposition of the `Controlled` class."""

    def test_no_matrix(self, backend):
        """Test that controlling an operation without a matrix method raises an error."""
        dev = qml.device(backend, wires=4)

        class OpWithNoMatrix(qml.operation.Operation):
            num_wires = qml.operation.AnyWires

            def matrix(self):
                raise NotImplementedError()

        @qml.qnode(dev)
        def f():
            ctrl(OpWithNoMatrix(wires=[0, 1]), control=[2, 3])
            return qml.probs()

        with pytest.raises(CompileError, match="could not be decomposed, it might be unsupported"):
            qjit(f, target="jaxpr")


if __name__ == "__main__":
    pytest.main(["-x", __file__])
