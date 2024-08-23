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


import pytest
from jax import make_jaxpr

from catalyst.jax_primitives import _get_call_jaxpr


def test_get_call_jaxpr():
    """Test _get_call_jaxpr raises AsserionError if no function primitive exists."""

    def f(x):
        return x * x

    jaxpr = make_jaxpr(f)(2.0)
    with pytest.raises(AssertionError, match="No call_jaxpr found in the JAXPR"):
        _ = _get_call_jaxpr(jaxpr)


if __name__ == "__main__":
    pytest.main(["-x", __file__])
