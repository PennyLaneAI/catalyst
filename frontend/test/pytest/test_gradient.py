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

import pytest

from catalyst import qjit, grad, cond, for_loop
import pennylane as qml
import numpy as np
from jax import numpy as jnp
import jax


@pytest.mark.parametrize("inp", [(1.0), (2.0), (3.0), (4.0)])
def test_finite_diff(inp):
    def f(x):
        qml.RX(x, wires=0)
        return qml.expval(qml.PauliY(0))

    @qjit()
    def compiled_grad_default(x: float):
        g = qml.qnode(qml.device("lightning.qubit", wires=1))(f)
        h = grad(g)
        return h(x)

    def interpretted_grad_default(x):
        device = qml.device("default.qubit", wires=1)
        g = qml.QNode(f, device, diff_method="finite-diff")
        h = qml.grad(g, argnum=0)
        return h(x)

    assert np.allclose(compiled_grad_default(inp), interpretted_grad_default(inp))


@pytest.mark.parametrize("inp", [(1.0), (2.0), (3.0), (4.0)])
def test_finite_diff_mul(inp):
    def f(x):
        qml.RX(3 * x, wires=0)
        return qml.expval(qml.PauliY(0))

    @qjit()
    def compiled_grad_default(x: float):
        g = qml.qnode(qml.device("lightning.qubit", wires=1))(f)
        h = grad(g)
        return h(x)

    def interpretted_grad_default(x):
        device = qml.device("default.qubit", wires=1)
        g = qml.QNode(f, device, diff_method="finite-diff")
        h = qml.grad(g, argnum=0)
        return h(x)

    assert np.allclose(compiled_grad_default(inp), interpretted_grad_default(inp))


@pytest.mark.parametrize("inp", [(1.0), (2.0), (3.0), (4.0)])
def test_finite_diff_in_loop(inp):
    @qml.qnode(qml.device("lightning.qubit", wires=1))
    def f(x):
        qml.RX(3 * x, wires=0)
        return qml.expval(qml.PauliY(0))

    @qjit()
    @qml.qnode(qml.device("lightning.qubit", wires=1))
    def compiled_grad_default(params, ntrials):
        diff = grad(f, argnum=0, method="fd")

        def fn(i, g):
            return diff(params)

        return for_loop(0, ntrials, 1)(fn)(params)[0]

    def interpretted_grad_default(x):
        device = qml.device("default.qubit", wires=1)
        g = qml.QNode(f, device, diff_method="finite-diff")
        h = qml.grad(g, argnum=0)
        return h(x)

    assert np.allclose(compiled_grad_default(inp, 5), interpretted_grad_default(inp))


@pytest.mark.parametrize("inp", [(1.0), (2.0), (3.0), (4.0)])
def test_adj(inp):
    def f(x):
        qml.RX(x, wires=0)
        return qml.expval(qml.PauliY(0))

    @qjit()
    def compiled(x: float):
        g = qml.qnode(qml.device("lightning.qubit", wires=1))(f)
        h = grad(g, method="adj")
        return h(x)

    def interpreted(x):
        device = qml.device("default.qubit", wires=1)
        g = qml.QNode(f, device, diff_method="backprop")
        h = qml.grad(g, argnum=0)
        return h(x)

    assert np.allclose(compiled(inp), interpreted(inp))


@pytest.mark.parametrize("inp", [(1.0), (2.0), (3.0), (4.0)])
def test_adj_mult(inp):
    def f(x):
        qml.RX(x * 2, wires=0)
        return qml.expval(qml.PauliY(0))

    @qjit()
    def compiled(x: float):
        g = qml.qnode(qml.device("lightning.qubit", wires=1))(f)
        h = grad(g, method="adj")
        return h(x)

    def interpreted(x):
        device = qml.device("default.qubit", wires=1)
        g = qml.QNode(f, device, diff_method="backprop")
        h = qml.grad(g, argnum=0)
        return h(x)

    assert np.allclose(compiled(inp), interpreted(inp))


@pytest.mark.parametrize("inp", [(1.0), (2.0), (3.0), (4.0)])
def test_adj_in_loop(inp):
    @qml.qnode(qml.device("lightning.qubit", wires=1))
    def f(x):
        qml.RX(3 * x, wires=0)
        return qml.expval(qml.PauliY(0))

    @qjit()
    @qml.qnode(qml.device("lightning.qubit", wires=1))
    def compiled_grad_default(params, ntrials):
        diff = grad(f, argnum=0, method="adj")

        def fn(i, g):
            return diff(params)

        return for_loop(0, ntrials, 1)(fn)(params)[0]

    def interpretted_grad_default(x):
        device = qml.device("default.qubit", wires=1)
        g = qml.QNode(f, device, diff_method="backprop")
        h = qml.grad(g, argnum=0)
        return h(x)

    assert np.allclose(compiled_grad_default(inp, 5), interpretted_grad_default(inp))


@pytest.mark.parametrize("inp", [(1.0), (2.0), (3.0), (4.0)])
def test_ps(inp):
    def f(x):
        qml.RX(x * 2, wires=0)
        return qml.expval(qml.PauliY(0))

    @qjit()
    def compiled(x: float):
        g = qml.qnode(qml.device("lightning.qubit", wires=1))(f)
        h = grad(g, method="ps")
        return h(x)

    def interpreted(x):
        device = qml.device("default.qubit", wires=1)
        g = qml.QNode(f, device, diff_method="backprop")
        h = qml.grad(g, argnum=0)
        return h(x)

    assert np.allclose(compiled(inp), interpreted(inp))


@pytest.mark.parametrize("inp", [(1.0), (2.0), (3.0), (4.0)])
def test_ps_conditionals(inp):
    def f_compiled(x, y):
        @cond(y > 1.5)
        def true_path():
            qml.RX(x * 2, wires=0)

        @true_path.otherwise
        def false_path():
            qml.RX(x, wires=0)

        true_path()
        return qml.expval(qml.PauliY(0))

    def f_interpreted(x, y):
        if y > 1.5:
            qml.RX(x * 2, wires=0)
        else:
            qml.RX(x, wires=0)
        return qml.expval(qml.PauliY(0))

    @qjit()
    def compiled(x: float, y: float):
        g = qml.qnode(qml.device("lightning.qubit", wires=1))(f_compiled)
        h = grad(g, method="ps", argnum=0)
        return h(x, y)

    def interpreted(x, y):
        device = qml.device("default.qubit", wires=1)
        g = qml.QNode(f_interpreted, device, diff_method="backprop")
        h = qml.grad(g, argnum=0)
        return h(x, y)

    assert np.allclose(compiled(inp, 0.0), interpreted(inp, 0.0))
    assert np.allclose(compiled(inp, 2.0), interpreted(inp, 2.0))


@pytest.mark.parametrize("inp", [(1.0), (2.0), (3.0), (4.0)])
def test_ps_for_loops(inp):
    def f_compiled(x, y):
        @for_loop(0, y, 1)
        def loop_fn(i):
            qml.RX(x * i * 1.5, wires=0)

        loop_fn()
        return qml.expval(qml.PauliY(0))

    def f_interpreted(x, y):
        for i in range(0, y, 1):
            qml.RX(x * i * 1.5, wires=0)
        return qml.expval(qml.PauliY(0))

    @qjit()
    def compiled(x: float, y: int):
        g = qml.qnode(qml.device("lightning.qubit", wires=1))(f_compiled)
        h = grad(g, method="ps", argnum=0)
        return h(x, y)

    def interpreted(x, y):
        device = qml.device("default.qubit", wires=1)
        g = qml.QNode(f_interpreted, device, diff_method="backprop")
        h = qml.grad(g, argnum=0)
        return h(x, y)

    assert np.allclose(compiled(inp, 1), interpreted(inp, 1))
    assert np.allclose(compiled(inp, 2), interpreted(inp, 2))
    assert np.allclose(compiled(inp, 3), interpreted(inp, 3))
    assert np.allclose(compiled(inp, 4), interpreted(inp, 4))


@pytest.mark.parametrize("inp", [(1.0), (2.0), (3.0), (4.0)])
def test_ps_for_loops_entangled(inp):
    def f_compiled(x, y, z):
        qml.RX(x, wires=0)
        qml.Hadamard(wires=0)

        @for_loop(1, y, 1)
        def loop_fn(i):
            qml.RX(x, wires=i)
            qml.CNOT(wires=[0, i])

        loop_fn()
        return qml.expval(qml.PauliY(z))

    def f_interpreted(x, y, z):
        qml.RX(x, wires=0)
        qml.Hadamard(wires=0)
        for i in range(1, y, 1):
            qml.RX(x, wires=i)
            qml.CNOT(wires=[0, i])
        return qml.expval(qml.PauliY(z))

    @qjit()
    def compiled(x: float, y: int, z: int):
        g = qml.qnode(qml.device("lightning.qubit", wires=3))(f_compiled)
        h = grad(g, method="ps", argnum=0)
        return h(x, y, z)

    def interpreted(x, y, z):
        device = qml.device("default.qubit", wires=3)
        g = qml.QNode(f_interpreted, device, diff_method="backprop")
        h = qml.grad(g, argnum=0)
        return h(x, y, z)

    assert np.allclose(compiled(inp, 1, 1), interpreted(inp, 1, 1))
    assert np.allclose(compiled(inp, 2, 2), interpreted(inp, 2, 2))


@pytest.mark.parametrize("inp", [(1.0), (2.0), (3.0), (4.0)])
def test_ps_qft(inp):
    def qft_compiled(x, n, z):
        # Input state: equal superposition
        @for_loop(0, n, 1)
        def init(i):
            qml.Hadamard(wires=i)

        # QFT
        @for_loop(0, n, 1)
        def qft(i):
            qml.Hadamard(wires=i)

            @for_loop(i + 1, n, 1)
            def inner(j):
                qml.RY(x, wires=j)
                qml.ControlledPhaseShift(jnp.pi / 2 ** (n - j + 1), [i, j])

            inner()

        init()
        qft()

        # Expected output: |100...>
        return qml.expval(qml.PauliZ(z))

    def qft_interpreted(x, n, z):
        # Input state: equal superposition
        for i in range(0, n, 1):
            qml.Hadamard(wires=i)

        for i in range(0, n, 1):
            qml.Hadamard(wires=i)

            for j in range(i + 1, n, 1):
                qml.RY(x, wires=j)
                qml.ControlledPhaseShift(jnp.pi / 2 ** (n - j + 1), [i, j])

        return qml.expval(qml.PauliZ(z))

    @qjit()
    def compiled(x: float, y: int, z: int):
        g = qml.qnode(qml.device("lightning.qubit", wires=3))(qft_compiled)
        h = grad(g, method="ps", argnum=0)
        return h(x, y, z)

    def interpreted(x, y, z):
        device = qml.device("default.qubit", wires=3)
        g = qml.QNode(qft_interpreted, device, diff_method="backprop")
        h = qml.grad(g, argnum=0)
        return h(x, y, z)

    assert np.allclose(compiled(inp, 2, 2), interpreted(inp, 2, 2))


@pytest.mark.parametrize("inp", [(1.0), (2.0), (3.0), (4.0)])
def test_finite_diff_h(inp):
    def f(x):
        qml.RX(x, wires=0)
        return qml.expval(qml.PauliY(0))

    @qjit()
    def compiled_grad_h(x: float):
        g = qml.qnode(qml.device("lightning.qubit", wires=1))(f)
        h = grad(g, h=0.1)
        return h(x)

    def interpretted_grad_h(x):
        device = qml.device("default.qubit", wires=1)
        g = qml.QNode(f, device, diff_method="finite-diff", h=0.1)
        h = qml.grad(g, argnum=0)
        return h(x)

    assert np.allclose(compiled_grad_h(inp), interpretted_grad_h(inp))


@pytest.mark.parametrize("inp", [(1.0), (2.0), (3.0), (4.0)])
def test_finite_diff_argnum(inp):
    def f2(x, y):
        qml.RX(x**y, wires=0)
        return qml.expval(qml.PauliY(0))

    @qjit()
    def compiled_grad_argnum(x: float):
        g = qml.qnode(qml.device("lightning.qubit", wires=1))(f2)
        h = grad(g, argnum=1)
        return h(x, 2.0)

    def interpretted_grad_argnum(x):
        device = qml.device("default.qubit", wires=1)
        g = qml.QNode(f2, device, diff_method="finite-diff")
        h = qml.grad(g, argnum=1)
        return h(x, 2.0)

    assert np.allclose(compiled_grad_argnum(inp), interpretted_grad_argnum(inp))


@pytest.mark.parametrize("inp", [(1.0), (2.0), (3.0), (4.0)])
def test_finite_diff_argnum_list(inp):
    def f2(x, y):
        qml.RX(x**y, wires=0)
        return qml.expval(qml.PauliY(0))

    @qjit()
    def compiled_grad_argnum_list(x: float):
        g = qml.qnode(qml.device("lightning.qubit", wires=1))(f2)
        h = grad(g, argnum=[1])
        return h(x, 2.0)

    def interpretted_grad_argnum_list(x):
        device = qml.device("default.qubit", wires=1)
        g = qml.QNode(f2, device, diff_method="finite-diff")
        h = qml.grad(g, argnum=[1])
        # Slightly different behaviour. If argnum is a list
        # it doesn't matter if it is a single number,
        # the return value will be a n-tuple of size of the
        # argnum list.
        return h(x, 2.0)[0]

    assert np.allclose(compiled_grad_argnum_list(inp), interpretted_grad_argnum_list(inp))


@pytest.mark.parametrize("inp", [(1.0), (2.0), (3.0), (4.0)])
def test_finite_grad_range_change(inp):
    def f2(x, y):
        qml.RX(x**y, wires=0)
        return qml.expval(qml.PauliY(0))

    @qjit()
    def compiled_grad_range_change(x: float):
        g = qml.qnode(qml.device("lightning.qubit", wires=1))(f2)
        h = grad(g, argnum=[0, 1])
        return h(x, 2.0)

    def interpretted_grad_range_change(x):
        device = qml.device("default.qubit", wires=1)
        g = qml.QNode(f2, device, diff_method="finite-diff")
        h = qml.grad(g, argnum=[0, 1])
        return h(x, 2.0)

    assert np.allclose(compiled_grad_range_change(inp), interpretted_grad_range_change(inp))


@pytest.mark.parametrize("inp", [(1.0), (2.0), (3.0), (4.0)])
def test_ps_grad_range_change(inp):
    def f2(x, y):
        qml.RX(x**y, wires=0)
        return qml.expval(qml.PauliY(0))

    @qjit()
    def compiled_grad_range_change(x: float):
        g = qml.qnode(qml.device("lightning.qubit", wires=1))(f2)
        h = grad(g, method="ps", argnum=[0, 1])
        return h(x, 2.0)

    def interpretted_grad_range_change(x):
        device = qml.device("default.qubit", wires=1)
        g = qml.QNode(f2, device, diff_method="backprop")
        h = qml.grad(g, argnum=[0, 1])
        return h(x, 2.0)

    assert np.allclose(compiled_grad_range_change(inp), interpretted_grad_range_change(inp))


@pytest.mark.parametrize("inp", [(1.0), (2.0), (3.0), (4.0)])
def test_ps_tensorinp(inp):
    def f2(x, y):
        qml.RX(x[0] ** y, wires=0)
        return qml.expval(qml.PauliY(0))

    @qjit()
    def compiled(x: jax.core.ShapedArray([1], float)):
        g = qml.qnode(qml.device("lightning.qubit", wires=1))(f2)
        h = grad(g, method="ps", argnum=[0, 1])
        return h(x, 2.0)

    def interpretted(x):
        device = qml.device("default.qubit", wires=1)
        g = qml.QNode(f2, device, diff_method="backprop")
        h = qml.grad(g, argnum=[0, 1])
        return h(x, 2.0)

    for dydx_c, dydx_i in zip(compiled(jnp.array([inp])), interpretted(np.array([inp]))):
        assert np.allclose(dydx_c, dydx_i)


@pytest.mark.parametrize("inp", [(1.0), (2.0), (3.0), (4.0)])
def test_adjoint_grad_range_change(inp):
    def f2(x, y):
        qml.RX(x**y, wires=0)
        return qml.expval(qml.PauliY(0))

    @qjit()
    def compiled_grad_range_change(x: float):
        g = qml.qnode(qml.device("lightning.qubit", wires=1))(f2)
        h = grad(g, method="adj", argnum=[0, 1])
        return h(x, 2.0)

    def interpretted_grad_range_change(x):
        device = qml.device("default.qubit", wires=1)
        g = qml.QNode(f2, device, diff_method="backprop")
        h = qml.grad(g, argnum=[0, 1])
        return h(x, 2.0)

    assert np.allclose(compiled_grad_range_change(inp), interpretted_grad_range_change(inp))


@pytest.mark.parametrize("method", [("ps"), ("adj")])
def test_assert_no_higher_order_without_ps(method):
    def f(x):
        qml.RX(x, wires=0)
        return qml.expval(qml.PauliY(0))

    with pytest.raises(ValueError, match="higher order derivatives"):

        @qjit()
        def workflow(x: float):
            g = qml.qnode(qml.device("lightning.qubit", wires=1))(f)
            h = grad(g, method=method)
            i = grad(h, method=method)
            return i(x)


def test_assert_no_non_func_gradients():
    with pytest.raises(TypeError, match="something other than a function"):

        @qjit()
        def workflow():
            def _f(x):
                return x + x

            return grad(_f, method="fd")(1.0)


def test_assert_no_non_single_expression_gradients():
    with pytest.raises(TypeError, match="is not well defined for non-single Jax equations"):

        @qjit()
        def workflow():
            def _f(x):
                return x

            return grad(_f, method="fd")(1.0)


@pytest.mark.parametrize("inp", [(1.0), (2.0), (3.0), (4.0)])
def test_finite_diff_higher_order(inp):
    def f(x):
        qml.RX(x, wires=0)
        return qml.expval(qml.PauliY(0))

    @qjit()
    def compiled_grad2_default(x: float):
        g = qml.qnode(qml.device("lightning.qubit", wires=1))(f)
        h = grad(g)
        i = grad(h)
        return i(x)

    def interpretted_grad2_default(x):
        device = qml.device("default.qubit", wires=1)
        g = qml.QNode(f, device, diff_method="backprop", max_diff=2)
        h = qml.grad(g, argnum=0)
        i = qml.grad(h, argnum=0)
        return i(x)

    assert np.allclose(compiled_grad2_default(inp), interpretted_grad2_default(inp), rtol=0.1)


@pytest.mark.parametrize("inp", [([1.0, 2.0])])
def test_jax_consts(inp):
    def circuit(params):
        qml.CRX(params[0], wires=[0, 1])
        qml.CRX(params[0], wires=[0, 2])
        h_coeffs = np.array([0.2, -0.53])
        h_obs = [qml.PauliX(0) @ qml.PauliZ(1), qml.PauliZ(0) @ qml.Hadamard(2)]
        return qml.expval(qml.Hamiltonian(h_coeffs, h_obs))

    @qjit()
    def compile_grad(params):
        g = qml.qnode(qml.device("lightning.qubit", wires=3))(circuit)
        h = grad(g)
        return h(params)

    def interpret_grad(params):
        device = qml.device("default.qubit", wires=3)
        g = qml.QNode(circuit, device, diff_method="backprop")
        h = qml.grad(g, argnum=0)
        return h(params)

    assert np.allclose(compile_grad(jnp.array(inp)), interpret_grad(inp))


if __name__ == "__main__":
    pytest.main(["-x", __file__])
