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

"""Test PyTree support in Catalyst."""

from typing import Iterable

import jax
import jax.numpy as jnp
import numpy as np
import pennylane as qml
import pytest
from jax._src.tree_util import tree_flatten

from catalyst import adjoint, cond, for_loop, grad, measure, qjit


class TestPyTreesReturnValues:
    """Test QJIT workflows with different return value data-types."""

    def test_return_value_float(self, backend):
        """Test constant."""

        @qml.qnode(qml.device(backend, wires=2))
        def circuit1(params):
            qml.RX(params[0], wires=0)
            qml.RX(params[1], wires=1)
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        jitted_fn = qjit(circuit1)

        params = [0.4, 0.8]
        expected = 0.64170937
        result = jitted_fn(params)
        assert jnp.allclose(result, expected)

        @qml.qnode(qml.device(backend, wires=2))
        def circuit2():
            return measure(0)

        jitted_fn = qjit(circuit2)

        result = jitted_fn()
        assert not result

    def test_return_value_arrays(self, backend):
        """Test arrays."""

        @qml.qnode(qml.device(backend, wires=2))
        def circuit1(params):
            qml.RX(params[0], wires=0)
            qml.RX(params[1], wires=1)
            return qml.state()

        jitted_fn = qjit(circuit1)

        params = [0.4, 0.8]
        result = jitted_fn(params)
        ip_result = circuit1(params)
        assert jnp.allclose(result, ip_result)

        @qml.qnode(qml.device(backend, wires=2))
        def circuit2(params):
            qml.RX(params[0], wires=0)
            qml.RX(params[1], wires=1)
            return [jnp.pi, qml.state()]

        jitted_fn = qjit(circuit2)

        params = [0.4, 0.8]
        result = jitted_fn(params)
        assert isinstance(result, list)
        assert jnp.allclose(result[0], jnp.pi)
        assert jnp.allclose(result[1], ip_result)

    def test_return_value_tuples(self, backend, tol_stochastic):
        """Test tuples."""

        @qml.qnode(qml.device(backend, wires=2))
        def circuit1():
            qml.Hadamard(0)
            qml.CNOT(wires=[0, 1])
            m0 = measure(0)
            m1 = measure(1)
            return (m0, m1)

        jitted_fn = qjit(circuit1)

        result = jitted_fn()
        assert isinstance(result, tuple)

        @qml.qnode(qml.device(backend, wires=2))
        def circuit2():
            qml.Hadamard(0)
            qml.CNOT(wires=[0, 1])
            m0 = measure(0)
            m1 = measure(1)
            return (((m0, m1), m0 + m1), m0 * m1)

        jitted_fn = qjit(circuit2)
        result = jitted_fn()
        assert isinstance(result, tuple)
        assert isinstance(result[0], tuple)
        assert isinstance(result[0][0], tuple)
        assert result[0][0][0] + result[0][0][1] == result[0][1]
        assert result[0][0][0] * result[0][0][1] == result[1]

        @qml.qnode(qml.device(backend, wires=2, shots=1000))
        def circuit3(params):
            qml.RX(params[0], wires=0)
            qml.RX(params[1], wires=1)
            return (
                qml.counts(),
                qml.expval(qml.PauliZ(0)),
            )

        params = [0.5, 0.6]
        expected_expval = 0.87758256

        jitted_fn = qjit(circuit3)
        result = jitted_fn(params)
        assert isinstance(result, tuple)
        assert isinstance(result[0], tuple)
        assert jnp.allclose(result[1], expected_expval, atol=tol_stochastic, rtol=tol_stochastic)

        @qml.qnode(qml.device(backend, wires=2, shots=None))
        def circuit4(params):
            qml.RX(params[0], wires=0)
            qml.RX(params[1], wires=1)
            return (
                qml.state(),
                qml.expval(qml.PauliZ(0)),
            )

        params = [0.5, 0.6]
        expected_expval = 0.87758256

        jitted_fn = qjit(circuit4)
        result = jitted_fn(params)
        assert isinstance(result, tuple)
        assert len(result[0]) == 4
        assert jnp.allclose(result[1], expected_expval)

        @qjit
        def workflow(x):
            def _f(x):
                return (2 * x, 3 * x)

            return _f(x)

        result = workflow(2.0)
        assert isinstance(result, tuple)
        assert result[0] == 4.0
        assert result[1] == 6.0

    def test_return_value_hybrid(self, backend):
        """Test tuples."""

        @qml.qnode(qml.device(backend, wires=2))
        def circuit1():
            qml.Hadamard(0)
            qml.CNOT(wires=[0, 1])
            return qml.var(qml.PauliZ(1))

        @qjit
        def workflow1(param):
            a = circuit1()
            return (a, [jnp.sin(param) ** 2, jnp.cos(param) ** 2], a)

        result = workflow1(1.27)
        assert isinstance(result, tuple)
        assert jnp.allclose(result[0], result[2])
        assert jnp.allclose(result[1][0] + result[1][1], 1.0)

    def test_return_value_cond(self, backend):
        """Test conditionals."""

        # QFunc Path.
        @qjit
        @qml.qnode(qml.device(backend, wires=1))
        def circuit1(n):
            @cond(n > 4)
            def cond_fn():
                return n**2, (n**3, n**4)

            @cond_fn.otherwise
            def else_fn():
                return n, (n**2, n**3)

            return cond_fn()

        res2 = circuit1(2)
        assert res2[0] == 2
        assert res2[1] == (4, 8)

        res5 = circuit1(5)
        assert res5[0] == 25
        assert res5[1] == (125, 625)

        # Classical Path.
        @qjit
        def circuit2(n):
            @cond(n > 4)
            def cond_fn():
                return n**2, (n**3, n**4)

            @cond_fn.otherwise
            def else_fn():
                return n, (n**2, n**3)

            return {
                "cond": cond_fn(),
                "const": n,
                "classical": n * jnp.pi,
            }

        res2 = circuit2(2)
        assert res2["cond"][1] == (4, 8)
        assert res2["const"] == 2

        res5 = circuit2(5)
        assert res5["cond"][1] == (125, 625)
        assert res5["const"] == 5

    def test_return_value_dict(self, backend, tol_stochastic):
        """Test dictionaries."""

        @qml.qnode(qml.device(backend, wires=2))
        def circuit1(params):
            qml.RX(params[0], wires=0)
            qml.RX(params[1], wires=1)
            return {
                "w0": qml.expval(qml.PauliZ(0)),
                "w1": qml.expval(qml.PauliZ(1)),
            }

        jitted_fn = qjit(circuit1)

        params = [0.2, 0.6]
        expected = {"w0": 0.98006658, "w1": 0.82533561}
        result = jitted_fn(params)
        assert isinstance(result, dict)
        assert jnp.allclose(result["w0"], expected["w0"])
        assert jnp.allclose(result["w1"], expected["w1"])

        @qml.qnode(qml.device(backend, wires=2, shots=1000))
        def circuit2(params):
            qml.RX(params[0], wires=0)
            qml.RX(params[1], wires=1)
            return {
                "counts": qml.counts(),
                "expval": {
                    "z0": qml.expval(qml.PauliZ(0)),
                },
            }

        params = [0.5, 0.6]
        expected_expval = 0.87758256

        jitted_fn = qjit(circuit2)
        result = jitted_fn(params)
        assert isinstance(result, dict)
        assert isinstance(result["counts"], tuple)
        assert jnp.allclose(
            result["expval"]["z0"], expected_expval, atol=tol_stochastic, rtol=tol_stochastic
        )

        @qml.qnode(qml.device(backend, wires=2, shots=None))
        def circuit3(params):
            qml.RX(params[0], wires=0)
            qml.RX(params[1], wires=1)
            return {
                "state": qml.state(),
                "expval": {
                    "z0": qml.expval(qml.PauliZ(0)),
                },
            }

        params = [0.5, 0.6]
        expected_expval = 0.87758256

        jitted_fn = qjit(circuit3)
        result = jitted_fn(params)
        assert isinstance(result, dict)
        assert len(result["state"]) == 4
        assert jnp.allclose(result["expval"]["z0"], expected_expval)

        @qjit
        def workflow1(param):
            return {"w": jnp.sin(param), "q": jnp.cos(param)}

        result = workflow1(jnp.pi / 2)
        assert isinstance(result, dict)
        assert result["w"] == 1


class TestPyTreesFuncArgs:
    """Test QJIT workflows with PyTrees as function arguments."""

    def test_args_dict(self, backend):
        """Test arguments dict."""

        @qml.qnode(qml.device(backend, wires=2))
        def circuit1(params):
            qml.RX(params["a"][0], wires=0)
            qml.RX(params["b"][0], wires=1)
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1)), params["a"][0]

        jitted_fn = qjit(circuit1)

        params = {
            "a": [0.4, 0.6],
            "b": [0.8],
        }
        expected = 0.64170937
        result = jitted_fn(params)

        assert jnp.allclose(result[0], expected)
        assert jnp.allclose(result[1], params["a"][0])

        @qml.qnode(qml.device(backend, wires=2))
        def circuit2(params):
            qml.RX(params["a"]["c"][0], wires=0)
            qml.RX(params["b"][0], wires=1)
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1)), params["a"]

        jitted_fn = qjit(circuit2)

        params = {
            "a": {"c": (0.4, 0.6)},
            "b": [0.8],
        }
        expected = 0.64170937
        result = jitted_fn(params)

        assert isinstance(result, tuple)
        assert jnp.allclose(result[0], expected)
        assert isinstance(result[1]["c"], tuple)
        assert jnp.allclose(result[1]["c"][0], params["a"]["c"][0])
        assert jnp.allclose(result[1]["c"][1], params["a"]["c"][1])

        @qml.qnode(qml.device(backend, wires=2))
        def circuit3(params1, params2):
            qml.RX(params1["a"][0] * params2[0], wires=0)
            qml.RX(params1["b"][0] * params2[1], wires=1)
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

    def test_promotion_unneeded(self, backend):
        """Test arguments list of lists."""

        @qml.qnode(qml.device(backend, wires=2))
        def circuit1(params):
            qml.RX(params["a"][0], wires=0)
            qml.RX(params["b"][0], wires=1)
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1)), params["a"][0]

        jitted_fn = qjit(circuit1)

        params = {
            "a": [0.4, 0.6],
            "b": [0.8],
        }

        jitted_fn(params)
        jitted_fn(params)

    def test_promotion_needed(self, backend):
        """Test arguments list of lists."""

        @qml.qnode(qml.device(backend, wires=2))
        def circuit1(params):
            qml.RX(params["a"][0], wires=0)
            qml.RX(params["b"][0], wires=1)
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1)), params["a"][0]

        jitted_fn = qjit(circuit1)

        params = {
            "a": [0.4, 0.6],
            "b": [0.8],
        }
        result = jitted_fn(params)
        params = {
            "a": [4, 6],
            "b": [8],
        }
        result = jitted_fn(params)

    def test_args_workflow(self, backend):
        """Test arguments with workflows."""

        @qjit
        def workflow1(params1, params2):
            """A classical workflow"""
            res1 = params1["a"][0][0] + params2[1]
            return jnp.sin(res1)

        params1 = {
            "a": [[0.1], 0.2],
        }
        params2 = (0.6, 0.8)
        expected = 0.78332691
        result = workflow1(params1, params2)
        assert jnp.allclose(result, expected)

        @qjit
        def workflow2(params1, params2):
            """A hybrid workflow"""

            @qml.qnode(qml.device(backend, wires=2))
            def circuit(params):
                qml.RX(params["a"][0], wires=0)
                qml.RX(params["b"][0], wires=1)
                return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

            res1 = circuit(params1)
            return jnp.sin(res1), res1 * params2[1]

        params1 = {
            "a": [0.4, 0.6],
            "b": [0.8],
        }
        params2 = (0.6, 2.8)

        res1, res2 = workflow2(params1, params2)
        assert jnp.allclose(res1, 0.59856565)
        assert jnp.allclose(res2, 1.79678625)

    def test_args_grad(self, backend):
        """Test arguments with the grad operation."""

        @qml.qnode(qml.device(backend, wires=2))
        def circuit1(params):
            qml.RX(params["a"][0], wires=0)
            qml.RX(params["b"][1], wires=1)
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        @qjit
        def workflow1(params):
            g = qml.qnode(qml.device(backend, wires=1))(circuit1)
            h = grad(g)
            return h(params)

        params = {
            "a": [0.4, 0.6],
            "b": [0.8, 0.6],
        }
        result = workflow1(params)
        expected = {"a": [-0.32140083, 0.0], "b": [0.0, -0.52007016]}
        result_flatten, tree = tree_flatten(result)
        result_flatten_expected, tree_expected = tree_flatten(expected)
        assert np.allclose(result_flatten, result_flatten_expected)
        assert tree == tree_expected

        @qml.qnode(qml.device(backend, wires=2))
        def circuit2(params):
            qml.RX(params["a"][0], wires=0)
            qml.RX(params["b"][0], wires=1)
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        @qjit
        def workflow2(params):
            g = qml.qnode(qml.device(backend, wires=1))(circuit2)
            h = grad(g)
            return h(params)

        params = {
            "a": (0.4, 0.6),
            "b": [0.6],
        }
        result = workflow2(params)
        expected = {"a": (-0.32140083, 0.0), "b": [-0.52007016]}
        result_flatten, tree = tree_flatten(result)
        result_flatten_expected, tree_expected = tree_flatten(expected)
        assert np.allclose(result_flatten, result_flatten_expected)
        assert tree == tree_expected

    @pytest.mark.parametrize("inp", [(np.array([0.2, 0.5])), (jnp.array([0.2, 0.5]))])
    def test_args_control_flow(self, backend, inp):
        """Test arguments with control-flows operations."""

        @qjit
        @qml.qnode(qml.device(backend, wires=2))
        def circuit1(n, params):
            @for_loop(0, n, 1)
            def loop(i):
                qml.RX(params[i], wires=i)
                return ()

            loop()
            return qml.state()

        circuit1(1, inp)

    def test_args_used_in_measure(self, backend):
        """Argument is used directly in measurement"""

        @qjit
        @qml.qnode(qml.device(backend, wires=2))
        def circuit(dictionary):
            """q0 = 1; q1 = 0;"""
            qml.RX(jnp.pi, wires=0)
            return measure(dictionary["wire"])

        result = circuit({"wire": 0})
        assert jnp.allclose(result, True)
        result = circuit({"wire": 1})
        assert jnp.allclose(result, False)

    def test_args_used_indirectly_in_measure(self, backend):
        """Argument is used indirectly in measurement"""

        @qjit
        @qml.qnode(qml.device(backend, wires=2))
        def circuit(dictionary):
            """q0 = 1; q1 = 0;"""
            qml.RX(jnp.pi, wires=0)
            return measure((dictionary["wire"] + 1) % 2)

        result = circuit({"wire": 0})
        assert jnp.allclose(result, False)
        result = circuit({"wire": 1})
        assert jnp.allclose(result, True)

    def test_dev_wires_have_pytree(self, backend):
        """Device wires are pytree-compatible."""

        def subroutine(wires):
            for wire in wires:
                qml.PauliX(wire)

        dev = qml.device(backend, wires=3)

        @qjit
        @qml.qnode(dev)
        def test_function():
            adjoint(subroutine)(dev.wires)
            return qml.probs()

        test_function()


class TestAuxiliaryData:
    """Test PyTrees with Auxiliary data."""

    def test_auxiliary_data(self):
        """Make sure that we are able to return arbitrary PyTrees.

        The example below was taken from:
        https://jax.readthedocs.io/en/latest/jax-101/05.1-pytrees.html
        """

        class MyContainer:
            """A named container."""

            def __init__(self, name: str, a: int):
                self.name = name
                self.a = a

        def flatten_MyContainer(container) -> tuple[Iterable[int], str]:
            """Returns an iterable over container contents, and aux data."""
            flat_contents = [container.a]

            # we don't want the name to appear as a child, so it is auxiliary data.
            # auxiliary data is usually a description of the structure of a node,
            # e.g., the keys of a dict -- anything that isn't a node's children.
            aux_data = container.name
            return flat_contents, aux_data

        def unflatten_MyContainer(aux_data: str, flat_contents: Iterable[int]) -> MyContainer:
            """Converts aux data and the flat contents into a MyContainer."""
            return MyContainer(aux_data, *flat_contents)

        jax.tree_util.register_pytree_node(MyContainer, flatten_MyContainer, unflatten_MyContainer)

        @qjit
        def classical(x):
            return MyContainer("aux_data", x * x)

        result = classical(2)
        assert result.name == "aux_data"
        assert result.a == 4


class TestPyTreesQmlCounts:
    """Test QJIT workflows when using qml.counts in a return expression."""

    def test_pytree_qml_counts_simple(self):
        """Test if a single qml.counts() can be used and output correctly."""
        dev = qml.device("lightning.qubit", wires=1, shots=20)

        @qjit
        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x, wires=0)
            return {"1": qml.counts()}

        observed = circuit(0.5)
        expected = {"1": (jnp.array((0, 1), dtype=jnp.int64), jnp.array((0, 3), dtype=jnp.int64))}

        _, expected_shape = tree_flatten(expected)
        _, observed_shape = tree_flatten(observed)
        assert expected_shape == observed_shape

    def test_pytree_qml_counts_nested(self):
        """Test if nested qml.counts() can be used and output correctly."""
        dev = qml.device("lightning.qubit", wires=1, shots=20)

        @qjit
        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x, wires=0)
            return {"1": qml.counts()}, {"2": qml.expval(qml.Z(0))}

        observed = circuit(0.5)
        expected = (
            {"1": (jnp.array((0, 1), dtype=jnp.int64), jnp.array((0, 3), dtype=jnp.int64))},
            {"2": jnp.array(-1, dtype=jnp.float64)},
        )

        _, expected_shape = tree_flatten(expected)
        _, observed_shape = tree_flatten(observed)
        assert expected_shape == observed_shape

        @qjit
        @qml.qnode(dev)
        def circuit2(x):
            qml.RX(x, wires=0)
            return [{"1": qml.expval(qml.Z(0))}, {"2": qml.counts()}], {"3": qml.expval(qml.Z(0))}

        observed = circuit2(0.5)
        expected = (
            [
                {"1": jnp.array(-1, dtype=jnp.float64)},
                {"2": (jnp.array((0, 1), dtype=jnp.int64), jnp.array((0, 3), dtype=jnp.int64))},
            ],
            {"3": jnp.array(-1, dtype=jnp.float64)},
        )
        _, expected_shape = tree_flatten(expected)
        _, observed_shape = tree_flatten(observed)
        assert expected_shape == observed_shape

    def test_pytree_qml_counts_2_nested(self):
        """Test if multiple nested qml.counts() can be used and output correctly."""
        dev = qml.device("lightning.qubit", wires=1, shots=20)

        @qjit
        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x, wires=0)
            return [{"1": qml.expval(qml.Z(0))}, {"2": qml.counts()}], [
                {"3": qml.expval(qml.Z(0))},
                {"4": qml.counts()},
            ]

        observed = circuit(0.5)
        expected = (
            [
                {"1": jnp.array(-1, dtype=jnp.float64)},
                {"2": (jnp.array((0, 1), dtype=jnp.int64), jnp.array((0, 3), dtype=jnp.int64))},
            ],
            [
                {"3": jnp.array(-1, dtype=jnp.float64)},
                {"4": (jnp.array((0, 1), dtype=jnp.int64), jnp.array((0, 3), dtype=jnp.int64))},
            ],
        )
        _, expected_shape = tree_flatten(expected)
        _, observed_shape = tree_flatten(observed)
        assert expected_shape == observed_shape

        @qjit
        @qml.qnode(dev)
        def circuit2(x):
            qml.RX(x, wires=0)
            return [{"1": qml.expval(qml.Z(0))}, {"2": qml.counts()}], [
                {"3": qml.counts()},
                {"4": qml.expval(qml.Z(0))},
            ]

        observed = circuit2(0.5)
        expected = (
            [
                {"1": jnp.array(-1, dtype=jnp.float64)},
                {"2": (jnp.array((0, 1), dtype=jnp.int64), jnp.array((0, 3), dtype=jnp.int64))},
            ],
            [
                {"3": (jnp.array((0, 1), dtype=jnp.int64), jnp.array((0, 3), dtype=jnp.int64))},
                {"4": jnp.array(-1, dtype=jnp.float64)},
            ],
        )
        _, expected_shape = tree_flatten(expected)
        _, observed_shape = tree_flatten(observed)
        assert expected_shape == observed_shape

    def test_pytree_qml_counts_longer(self):
        """Test if 3 differently nested qml.counts() can be used and output correctly."""
        dev = qml.device("lightning.qubit", wires=1, shots=20)

        @qjit
        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x, wires=0)
            return [
                [{"1": qml.expval(qml.Z(0))}, {"2": qml.counts()}],
                [{"3": qml.expval(qml.Z(0))}, {"4": qml.counts()}],
                {"5": qml.expval(qml.Z(0))},
                {"6": qml.counts()},
            ]

        observed = circuit(0.5)
        expected = [
            [
                {"1": jnp.array(-1, dtype=jnp.float64)},
                {"2": (jnp.array((0, 1), dtype=jnp.int64), jnp.array((0, 3), dtype=jnp.int64))},
            ],
            [
                {"3": jnp.array(-1, dtype=jnp.float64)},
                {"4": (jnp.array((0, 1), dtype=jnp.int64), jnp.array((0, 3), dtype=jnp.int64))},
            ],
            {"5": jnp.array(-1, dtype=jnp.float64)},
            {"6": (jnp.array((0, 1), dtype=jnp.int64), jnp.array((0, 3), dtype=jnp.int64))},
        ]
        _, expected_shape = tree_flatten(expected)
        _, observed_shape = tree_flatten(observed)
        assert expected_shape == observed_shape

    def test_pytree_qml_counts_mcm(self):
        """Test qml.counts() with mid circuit measurement."""
        dev = qml.device("lightning.qubit", wires=1, shots=20)

        @qml.qjit
        @qml.qnode(dev, mcm_method="one-shot", postselect_mode=None)
        def circuit(x):
            qml.RX(x, wires=0)
            measure(0, postselect=1)
            return {"hi": qml.counts()}, {"bye": qml.expval(qml.Z(0))}, {"hi": qml.counts()}

        observed = circuit(0.5)
        expected = (
            {"hi": (jnp.array((0, 1), dtype=jnp.int64), jnp.array((0, 3), dtype=jnp.int64))},
            {"bye": jnp.array(-1, dtype=jnp.float64)},
            {"hi": (jnp.array((0, 1), dtype=jnp.int64), jnp.array((0, 3), dtype=jnp.int64))},
        )
        _, expected_shape = tree_flatten(expected)
        _, observed_shape = tree_flatten(observed)
        assert expected_shape == observed_shape


if __name__ == "__main__":
    pytest.main(["-x", __file__])
