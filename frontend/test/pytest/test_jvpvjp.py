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

import jax
import jax.numpy as jnp
import numpy as np
import pennylane as qml
import pytest

from typing import Union, TypeVar, Tuple, Any
from jax import linearize as J_jvp, vjp as J_vjp
from catalyst import CompileError, cond, for_loop, grad, qjit, jvp as C_jvp, vjp as C_vjp
from numpy.testing import assert_allclose


X = TypeVar('X')
T = TypeVar('T')
def sum_if_tuples(x:Union[X, Tuple[Tuple[T]]]) -> Union[X, Tuple[T]]:
    return sum(x,()) if isinstance(x, tuple) and all(isinstance(i, tuple) for i in x) else x


testvec = [
    (lambda x: jnp.stack([1*x, 2*x, 3*x]),
     [jnp.zeros([4], dtype=float)],
     [jnp.ones([4], dtype=float)]),

    (lambda x1,x2: (
         1*jnp.reshape(x1,[6]) + 2*jnp.reshape(x2,[6]), \
         jnp.stack([
               3*jnp.reshape(x1,[6]) + 4*jnp.reshape(x2,[6]),
               3*jnp.reshape(x1,[6]) + 4*jnp.reshape(x2,[6])])),
     [jnp.zeros([3,2], dtype=float), jnp.zeros([2,3], dtype=float)],
     [jnp.ones([3,2], dtype=float), jnp.ones([2,3], dtype=float)]),

    (lambda x: (x, jnp.stack([1*x, 2*x, 3*x])),
     [jnp.zeros([4], dtype=float)],
     [jnp.ones([4], dtype=float)]),
]


@pytest.mark.parametrize("f, x, t", testvec)
def test_jvp_against_jax(f:callable, x:list, t:list):
    """ Numerically tests Catalyst's jvp against the JAX version. """

    @qjit
    def C_workflow():
        return C_jvp(f, x, t, method='fd', argnum=list(range(len(x))))

    @jax.jit
    def J_workflow():
        y,ft = J_jvp(f, *x)
        return sum_if_tuples((y,ft(*t)))

    r1 = C_workflow()
    r2 = J_workflow()
    print(r1)
    print(r2)

    for a,b in zip(r1,r2):
        assert_allclose(a, b, rtol=1e-6, atol=1e-6)



if __name__ == "__main__":
    pytest.main(["-x", __file__])
