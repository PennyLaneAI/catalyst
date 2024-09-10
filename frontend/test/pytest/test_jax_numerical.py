# Copyright 2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Test that numerical jax functions produce correct results when compiled with qml.qjit"""

import numpy as np
import pennylane as qml
import pytest
from jax import numpy as jnp
from jax import scipy as jsp

from catalyst import qjit


class TestExpmAndSolve:
    """Test that `jax.scipy.linalg.expm` and `jax.scipy.linalg.solve` can run together
    in the same function scope but from different qjit blocks.

    Also test that their results are numerically correct when qjit compiled.
    """


class TestArgsortNumerical:
    """Test jax.numpy.argsort sort arrays correctly when being qjit compiled"""

    @pytest.mark.parametrize(
        "inp",
        [
            jnp.array([1.2, 0.1, 2.7, 0.6]),
            jnp.array([-1.2, -0.1, -2.7, -0.6]),
            jnp.array([[0.1, 0.2], [5.3, 1.2]]),
            jnp.array([[1, 2], [-3, -4]]),
            jnp.array([[1.0, -1.0, 1.0], [1.0, -1.0, -1.0]]),
        ],
    )
    def test_expm_numerical(self, inp):
        """jax.numpy.argsort sort arrays correctly when being qjit compiled"""

        @qjit
        def f(x):
            return jnp.argsort(x)

        observed = f(inp)
        expected = jnp.argsort(inp)

        assert np.allclose(observed, expected)


if __name__ == "__main__":
    pytest.main(["-x", __file__])
