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
"""
This file contains a couple of tests for the capture of measurement primitives into jaxpr.
"""

# pylint: disable=line-too-long

import jax

from catalyst.jax_primitives import compbasis_p, expval_p, probs_p, state_p, var_p


def test_expval():
    """Test that the expval primitive can be captured by jaxpr."""

    def f():
        obs = compbasis_p.bind()
        return expval_p.bind(obs, shape=(1,))

    jaxpr = jax.make_jaxpr(f)()
    assert jaxpr.eqns[1].primitive == expval_p
    assert jaxpr.eqns[1].params == {"shape": (1,)}
    assert jaxpr.eqns[1].outvars[0].aval.shape == ()


def test_var():
    """Test that the var primitive can be captured by jaxpr."""

    def f():
        obs = compbasis_p.bind()
        return var_p.bind(obs, shape=(1,))

    jaxpr = jax.make_jaxpr(f)()
    assert jaxpr.eqns[1].primitive == var_p
    assert jaxpr.eqns[1].params == {"shape": (1,)}
    assert jaxpr.eqns[1].outvars[0].aval.shape == ()


def test_probs():
    """Test that the var primitive can be captured by jaxpr."""

    def f():
        obs = compbasis_p.bind()
        return probs_p.bind(obs, static_shape=(1,))

    jaxpr = jax.make_jaxpr(f)()
    assert jaxpr.eqns[1].primitive == probs_p
    assert jaxpr.eqns[1].params == {"static_shape": (1,)}
    assert jaxpr.eqns[1].outvars[0].aval.shape == (1,)


def test_state():
    """Test that the state primitive can be captured by jaxpr."""

    def f():
        obs = compbasis_p.bind()
        return state_p.bind(obs, static_shape=(1,))

    jaxpr = jax.make_jaxpr(f)()
    assert jaxpr.eqns[1].primitive == state_p
    assert jaxpr.eqns[1].params == {"static_shape": (1,)}
    assert jaxpr.eqns[1].outvars[0].aval.shape == (1,)
