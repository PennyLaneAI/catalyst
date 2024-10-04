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
import jax

from catalyst.jax_primitives import (
    compbasis_p,
    counts_p,
    expval_p,
    probs_p,
    sample_p,
    state_p,
    var_p,
)


def test_sample():
    """Test that the sample primitive can be captured into jaxpr."""

    def f():
        obs = compbasis_p.bind()
        return sample_p.bind(obs, shots=5, shape=(5, 0))

    jaxpr = jax.make_jaxpr(f)().jaxpr
    assert jaxpr.eqns[1].primitive == sample_p
    assert jaxpr.eqns[1].params == {"shape": (5, 0), "shots": 5}
    assert jaxpr.eqns[1].outvars[0].aval.shape == (5, 0)


def test_sample_dynamic():
    """Test that the sample primitive can be captured into jaxpr
    when using tracers."""

    shots = 5

    def f(shots):
        obs = compbasis_p.bind()
        return sample_p.bind(obs, shots=shots, shape=(shots, 0))

    jaxpr = jax.make_jaxpr(f)(shots).jaxpr
    shots_tracer, shape_value = jaxpr.eqns[1].params.values()

    assert jaxpr.eqns[1].primitive == sample_p
    assert isinstance(shots_tracer, jax._src.interpreters.partial_eval.DynamicJaxprTracer)
    assert isinstance(shape_value[0], jax._src.interpreters.partial_eval.DynamicJaxprTracer)
    assert shape_value[1] == 0
    assert isinstance(jaxpr.eqns[1].outvars[0].aval.shape[0], jax._src.core.Var)
    assert jaxpr.eqns[1].outvars[0].aval.shape[1] == 0


def test_counts():
    """Test that the counts primitive can be captured by jaxpr."""

    def f():
        obs = compbasis_p.bind()
        return counts_p.bind(obs, shots=5, shape=(1,))

    jaxpr = jax.make_jaxpr(f)()
    assert jaxpr.eqns[1].primitive == counts_p
    assert jaxpr.eqns[1].params == {"shape": (1,), "shots": 5}
    assert jaxpr.eqns[1].outvars[0].aval.shape == (1,)
    assert jaxpr.eqns[1].outvars[1].aval.shape == (1,)


def test_expval():
    """Test that the expval primitive can be captured by jaxpr."""

    def f():
        obs = compbasis_p.bind()
        return expval_p.bind(obs, shots=5, shape=(1,))

    jaxpr = jax.make_jaxpr(f)()
    assert jaxpr.eqns[1].primitive == expval_p
    assert jaxpr.eqns[1].params == {"shape": (1,), "shots": 5}
    assert jaxpr.eqns[1].outvars[0].aval.shape == ()


def test_var():
    """Test that the var primitive can be captured by jaxpr."""

    def f():
        obs = compbasis_p.bind()
        return var_p.bind(obs, shots=5, shape=(1,))

    jaxpr = jax.make_jaxpr(f)()
    assert jaxpr.eqns[1].primitive == var_p
    assert jaxpr.eqns[1].params == {"shape": (1,), "shots": 5}
    assert jaxpr.eqns[1].outvars[0].aval.shape == ()


def test_probs():
    """Test that the var primitive can be captured by jaxpr."""

    def f():
        obs = compbasis_p.bind()
        return probs_p.bind(obs, shots=5, shape=(1,))

    jaxpr = jax.make_jaxpr(f)()
    assert jaxpr.eqns[1].primitive == probs_p
    assert jaxpr.eqns[1].params == {"shape": (1,), "shots": 5}
    assert jaxpr.eqns[1].outvars[0].aval.shape == (1,)


def test_state():
    """Test that the state primitive can be captured by jaxpr."""

    def f():
        obs = compbasis_p.bind()
        return state_p.bind(obs, shots=5, shape=(1,))

    jaxpr = jax.make_jaxpr(f)()
    assert jaxpr.eqns[1].primitive == state_p
    assert jaxpr.eqns[1].params == {"shape": (1,), "shots": 5}
    assert jaxpr.eqns[1].outvars[0].aval.shape == (1,)
