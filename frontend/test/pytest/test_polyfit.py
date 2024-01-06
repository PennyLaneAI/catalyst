# Copyright 2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Test integration for the lowering of MHLO scatter."""

import jax
import numpy as np
import pytest
from jax import numpy as jnp

from catalyst import qjit


@pytest.mark.parametrize(
    "x, y, deg",
    [
        (jnp.array([1.0, 2.0, 3.0]), jnp.array([1.0, 4.0, 9.0]), 2),
        (jnp.array([1.0, 2.0, 3.0, 4.0]), jnp.array([1.0, 8.0, 27.0, 64.0]), 3),
    ],
)
def test_polyfit(x, y, deg):
    """Test that polyfit from Jax produces same results qjitted or not."""

    @qjit
    def polyfit_qjit(x, y):
        return jax.numpy.polyfit(x, y, deg)

    res = polyfit_qjit(x, y)
    res_expected = jax.numpy.polyfit(x, y, deg)
    assert np.allclose(res, res_expected)


if __name__ == "__main__":
    pytest.main(["-x", __file__])
