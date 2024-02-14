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

"""Test QJIT compatibility of jax.vmap and catalyst.vmap."""

import jax
import jax.numpy as jnp
import pennylane as qml
import pytest

from catalyst import qjit, vmap


class TestVectorizeMap:
    """Test QJIT compatibility with JAX vectorization."""

    @pytest.mark.parametrize("vmap_fn", (jax.vmap, vmap))
    def test_simple_classical(self, vmap_fn):
        """
        Test jax.vmap and catalyst.vmap for a classical program inside and outside qjit.
        """

        def workflow(axes_dct):
            return axes_dct["x"] + axes_dct["y"]

        excepted = jnp.array([1, 2, 3, 4, 5])

        # Outside qjit
        result_out = vmap_fn(qjit(workflow), in_axes=({"x": None, "y": 0},))(
            {"x": 1, "y": jnp.arange(5)}
        )
        assert jnp.allclose(result_out, excepted)

        # Inside qjit
        result_in = qjit(vmap_fn(workflow, in_axes=({"x": None, "y": 0},)))(
            {"x": 1, "y": jnp.arange(5)}
        )
        assert jnp.allclose(result_in, excepted)

    @pytest.mark.parametrize("vmap_fn", (jax.vmap, vmap))
    def test_simple_circuit(self, vmap_fn, backend):
        """Test a basic use case of jax.vmap and catalyst.vmap on top of qjit."""

        @qjit
        @qml.qnode(qml.device(backend, wires=2))
        def circuit(x: jax.core.ShapedArray((3,), dtype=float)):
            qml.RX(jnp.pi * x[0], wires=0)
            qml.RY(x[1] ** 2, wires=0)
            qml.RX(x[1] * x[2], wires=0)
            return qml.expval(qml.PauliZ(0))

        def cost_fn(x):
            result = circuit(x)
            return jnp.cos(result) ** 2

        x = jnp.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        result = vmap_fn(cost_fn)(x)

        assert len(result) == 2
        assert jnp.allclose(result[0], cost_fn(x[0]))
        assert jnp.allclose(result[1], cost_fn(x[1]))

    def test_unsupported_jax_vmap(self, backend):
        """Test the QJIT incompatibility of jax.vmap."""

        @qjit
        def workflow(x):
            @qml.qnode(qml.device(backend, wires=1))
            def circuit(x):
                qml.RX(jnp.pi * x[0], wires=0)
                qml.RY(x[1] ** 2, wires=0)
                qml.RX(x[1] * x[2], wires=0)
                return qml.expval(qml.PauliZ(0))

            res = jax.vmap(circuit)(x)
            return res

        x = jnp.array(
            [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
                [0.7, 0.8, 0.9],
            ]
        )

        with pytest.raises(NotImplementedError, match="Batching rule for 'qinst' not implemented"):
            workflow(x)

    def test_vmap_circuit_inside(self, backend):
        """Test catalyst.vmap of a hybrid workflow inside QJIT."""

        @qjit
        def workflow(x):
            @qml.qnode(qml.device(backend, wires=1))
            def circuit(x):
                qml.RX(jnp.pi * x[0], wires=0)
                qml.RY(x[1] ** 2, wires=0)
                qml.RX(x[1] * x[2], wires=0)
                return qml.expval(qml.PauliZ(0))

            res1 = vmap(circuit)(x)
            res2 = vmap(circuit, in_axes=0)(x)
            res3 = vmap(circuit, in_axes=(0,))(x)
            res4 = vmap(circuit, in_axes=None)(x[0])
            return res1, res2, res3, res4

        x = jnp.array(
            [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
                [0.7, 0.8, 0.9],
            ]
        )

        result = workflow(x)
        excepted = jnp.array([0.93005586, 0.00498127, -0.88789978])
        assert jnp.allclose(result[0], excepted)
        assert jnp.allclose(result[1], excepted)
        assert jnp.allclose(result[2], excepted)
        assert jnp.allclose(result[3], excepted[0])

    def test_vmap_nonzero_axes(self, backend):
        """Test catalyst.vmap of a hybrid workflow inside QJIT with axes > 0."""

        @qjit
        def workflow(x):
            @qml.qnode(qml.device(backend, wires=1))
            def circuit(x):
                qml.RX(jnp.pi * x[0], wires=0)
                qml.RY(x[1] ** 2, wires=0)
                qml.RX(x[1] * x[2], wires=0)
                return qml.expval(qml.PauliZ(0))

            res1 = vmap(circuit, in_axes=1)(x)
            res2 = vmap(circuit, in_axes=(1,))(x)
            return res1, res2

        x = jnp.array(
            [
                [0.1, 0.4],
                [0.2, 0.5],
                [0.3, 0.6],
            ]
        )

        result = workflow(x)
        excepted = jnp.array([0.93005586, 0.00498127])
        assert jnp.allclose(result[0], excepted)
        assert jnp.allclose(result[1], excepted)

    def test_vmap_failed_nonint_check(self, backend):
        """Test catalyst.vmap with invalid type of in_axes."""

        def workflow(x):
            @qml.qnode(qml.device(backend, wires=1))
            def circuit(x):
                qml.RX(jnp.pi * x, wires=0)
                return qml.expval(qml.PauliZ(0))

            res = vmap(circuit, in_axes=(0.1, None))(x)
            return res

        with pytest.raises(
            ValueError,
            match="Invalid 'in_axes'; it can be an int or a tuple of "
            "PyTrees with integer leaves",
        ):
            qjit(workflow)(0.1)

    def test_vmap_failed_len_check(self, backend):
        """Test catalyst.vmap with invalid length of in_axes and args."""

        def workflow(x):
            @qml.qnode(qml.device(backend, wires=1))
            def circuit(x):
                qml.RX(jnp.pi * x, wires=0)
                return qml.expval(qml.PauliZ(0))

            res = vmap(circuit, in_axes=(0, None))(x)
            return res

        with pytest.raises(
            ValueError,
            match="Invalid 'in_axes'; it can be an int or a tuple "
            "of PyTrees corresponding to the arguments",
        ):
            qjit(workflow)(0.1)

    def test_vmap_failed_invalid_out_axes(self, backend):
        """Test catalyst.vmap with invalid out_axes."""

        def workflow(x):
            @qml.qnode(qml.device(backend, wires=1))
            def circuit(x):
                qml.RX(jnp.pi * x[0], wires=0)
                qml.RY(x[1] ** 2, wires=0)
                qml.RX(x[1] * x[2], wires=0)
                return qml.expval(qml.PauliZ(0))

            return vmap(circuit, out_axes=(0, 1))(x)

        x = jnp.array(
            [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
                [0.7, 0.8, 0.9],
            ]
        )

        with pytest.raises(
            ValueError,
            match="Invalid 'out_axes'; it can be an int or a tuple "
            "of PyTrees corresponding to the ",
        ):
            qjit(workflow)(x)

    def test_vmap_tuple_in_axes(self, backend):
        """Test catalyst.vmap of a hybrid workflow inside QJIT with a tuple in_axes."""

        @qjit
        def workflow(x, y, z):
            @qml.qnode(qml.device(backend, wires=1))
            def circuit(x, y):
                qml.RX(jnp.pi * x[0] + y - y, wires=0)
                qml.RY(x[1] ** 2, wires=0)
                qml.RX(x[1] * x[2], wires=0)
                return qml.expval(qml.PauliZ(0))

            def workflow2(x, y):
                return circuit(x, y) * y

            def workflow3(y, x):
                return circuit(x, y) * y

            def workflow4(y, x, z):
                return circuit(x, y) * y * z

            res1 = vmap(workflow2, in_axes=(0, None))(x, y)
            res2 = vmap(workflow2, in_axes=[0, None])(x, y)
            res3 = vmap(workflow3, in_axes=(None, 0))(y, x)
            res4 = vmap(workflow4, in_axes=(None, 0, None))(y, x, z)
            res5 = vmap(workflow4, in_axes=(None, 0, None), axis_size=2)(y, x, z)
            return res1, res2, res3, res4, res5

        y = jnp.pi
        x = jnp.array(
            [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
                [0.7, 0.8, 0.9],
            ]
        )

        result = workflow(x, y, 1)
        excepted = jnp.array([0.93005586, 0.00498127, -0.88789978]) * y
        assert jnp.allclose(result[0], excepted)
        assert jnp.allclose(result[1], excepted)
        assert jnp.allclose(result[2], excepted)
        assert jnp.allclose(result[3], excepted)
        assert jnp.allclose(result[4], excepted[:2])

    def test_vmap_tuple_in_axes_multiple_nonuniform(self, backend):
        """Test expected ValueError with non-uniform batch sizes."""

        @qjit
        def workflow(x, y):
            @qml.qnode(qml.device(backend, wires=1))
            def circuit(x, y):
                qml.Identity(wires=0)
                return x + y

            return vmap(circuit, in_axes=(0, 0))(x, y)

        x = jnp.array(
            [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
            ]
        )

        y1 = jnp.array([jnp.pi, jnp.pi / 2, jnp.pi / 4])
        y2 = [jnp.pi, jnp.pi / 2]  # TODO: support Python lists in vmap.

        with pytest.raises(
            ValueError,
            match="Invalid batch sizes; expected to get a uniform batch sizes",
        ):
            qjit(workflow)(x, y1)

        with pytest.raises(
            ValueError,
            match="Invalid batch sizes; expected to get a uniform batch sizes",
        ):
            qjit(workflow)(x, y2)

    def test_vmap_tuple_in_axes_multiple(self, backend):
        """Test catalyst.vmap of a hybrid workflow inside QJIT with a tuple in_axes
        and multiple non-zero axes."""

        @qjit
        def workflow(x, y, z):
            @qml.qnode(qml.device(backend, wires=1))
            def circuit(x, y):
                qml.RX(jnp.pi * x[0] + y, wires=0)
                qml.RY(x[1] ** 2, wires=0)
                qml.RX(x[1] * x[2], wires=0)
                return qml.expval(qml.PauliZ(0))

            def workflow2(x, y):
                return circuit(x, y)

            def workflow3(y, x):
                return circuit(x, y)

            def workflow4(y, x, z):
                return circuit(x, y) * z

            res1 = vmap(workflow2, in_axes=(0, 0))(x, y)
            res2 = vmap(workflow2, in_axes=[0, 0])(x, y)
            res3 = vmap(workflow3, in_axes=(0, 0))(y, x)
            res4 = vmap(workflow4, in_axes=(0, 0, None))(y, x, z)
            res5 = vmap(workflow4, in_axes=(0, 0, None), axis_size=2)(y, x, z)
            return res1, res2, res3, res4, res5

        y = jnp.array([jnp.pi, jnp.pi / 2, jnp.pi / 4])
        x = jnp.array(
            [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
                [0.7, 0.8, 0.9],
            ]
        )

        result = workflow(x, y, 1)
        excepted = jnp.array([-0.93005586, -0.97165424, -0.6987465])
        assert jnp.allclose(result[0], excepted)
        assert jnp.allclose(result[1], excepted)
        assert jnp.allclose(result[2], excepted)
        assert jnp.allclose(result[3], excepted)
        assert jnp.allclose(result[4], excepted[:2])

    def test_vmap_pytree_in_axes(self, backend):
        """Test catalyst.vmap of a hybrid workflow inside QJIT with a PyTree in_axes."""

        @qjit
        def workflow(x, y, z):
            @qml.qnode(qml.device(backend, wires=1))
            def circuit(x, y):
                qml.RX(jnp.pi * x["arr"][0] + y - y, wires=0)
                qml.RY(x["arr"][1] ** 2, wires=0)
                qml.RX(x["arr"][1] * x["arr"][2], wires=0)
                return qml.expval(qml.PauliZ(0))

            def workflow2(x, y):
                return circuit(x, y) * y

            def workflow3(y, x):
                return circuit(x, y) * y

            def workflow4(y, x, z):
                return circuit(x, y) * y * z

            res1 = vmap(workflow2, in_axes=({"arr": 0, "foo": None}, None))(x, y)
            res2 = vmap(workflow2, in_axes=[{"arr": 0, "foo": None}, None])(x, y)
            res3 = vmap(workflow3, in_axes=(None, {"arr": 0, "foo": None}))(y, x)
            res4 = vmap(workflow4, in_axes=(None, {"arr": 0, "foo": None}, None))(y, x, z)
            return res1, res2, res3, res4

        y = jnp.pi
        x = {
            "arr": jnp.array(
                [
                    [0.1, 0.2, 0.3],
                    [0.4, 0.5, 0.6],
                    [0.7, 0.8, 0.9],
                ]
            ),
            "foo": None,
        }

        result = workflow(x, y, 1)
        excepted = jnp.array([0.93005586, 0.00498127, -0.88789978]) * y
        assert jnp.allclose(result[0], excepted)
        assert jnp.allclose(result[1], excepted)
        assert jnp.allclose(result[2], excepted)
        assert jnp.allclose(result[3], excepted)

    def test_vmap_circuit_return_tensor(self, backend):
        """Test catalyst.vmap of a hybrid workflow inside QJIT returning tensors."""

        @qjit
        def workflow(x):
            @qml.qnode(qml.device(backend, wires=1))
            def circuit(x):
                qml.RX(jnp.pi * x[0], wires=0)
                qml.RY(x[1] ** 2, wires=0)
                qml.RX(x[1] * x[2], wires=0)
                return qml.state()

            res1 = vmap(circuit)(x)
            res2 = vmap(circuit, out_axes=0)(x)
            return res1, res2

        x = jnp.array([[0.1, 0.2, 0.3], [0.7, 0.8, 0.9]])

        result = workflow(x)
        excepted = jnp.array(
            [
                [0.98235508 + 0.00253459j, 0.0198374 - 0.18595308j],
                [0.10537427 + 0.2120056j, 0.23239136 - 0.94336851j],
            ]
        )
        assert jnp.allclose(result[0], excepted)
        assert jnp.allclose(result[1], excepted)

    def test_vmap_circuit_return_tensor_out_axes(self, backend):
        """Test catalyst.vmap of a hybrid workflow inside QJIT with out_axes."""

        @qjit
        def workflow(x):
            @qml.qnode(qml.device(backend, wires=1))
            def circuit(x):
                qml.RX(jnp.pi * x[0], wires=0)
                qml.RY(x[1] ** 2, wires=0)
                qml.RX(x[1] * x[2], wires=0)
                return qml.state()

            res1 = vmap(circuit)(x)
            res2 = vmap(circuit, out_axes=1)(x)
            return res1, res2

        x = jnp.array([[0.1, 0.2, 0.3], [0.7, 0.8, 0.9]])

        result = workflow(x)
        excepted = jnp.array(
            [
                [0.98235508 + 0.00253459j, 0.0198374 - 0.18595308j],
                [0.10537427 + 0.2120056j, 0.23239136 - 0.94336851j],
            ]
        )
        assert jnp.allclose(result[0], excepted)
        assert jnp.allclose(jnp.transpose(result[1], (1, 0)), excepted)

    def test_vmap_circuit_return_tensor_pytree(self, backend):
        """Test catalyst.vmap of a hybrid workflow inside QJIT returning PyTrees."""

        @qjit
        def workflow(x):
            @qml.qnode(qml.device(backend, wires=1))
            def circuit(x):
                qml.RX(jnp.pi * x[0], wires=0)
                qml.RY(x[1] ** 2, wires=0)
                qml.RX(x[1] * x[2], wires=0)
                return qml.state(), qml.probs(0)

            res1 = vmap(circuit)(x)
            return res1

        x = jnp.array([[0.1, 0.2, 0.3], [0.7, 0.8, 0.9]])
        expected_state = jnp.array(
            [
                [0.98235508 + 0.00253459j, 0.0198374 - 0.18595308j],
                [0.10537427 + 0.2120056j, 0.23239136 - 0.94336851j],
            ]
        )
        expected_probs = jnp.array([[0.96502793, 0.03497207], [0.05605011, 0.94394989]])
        result = workflow(x)
        assert isinstance(result, tuple)
        assert jnp.allclose(result[0], expected_state)
        assert jnp.allclose(result[1], expected_probs)

    def test_vmap_circuit_return_tensor_pytree_dict(self, backend):
        """Test catalyst.vmap of a hybrid workflow inside QJIT returning PyTrees."""

        @qjit
        def workflow(x):
            @qml.qnode(qml.device(backend, wires=1))
            def circuit(x):
                qml.RX(jnp.pi * x[0], wires=0)
                qml.RY(x[1] ** 2, wires=0)
                qml.RX(x[1] * x[2], wires=0)
                return {
                    "a": qml.state(),
                    "b": {"c": qml.probs(0), "d": qml.expval(qml.PauliZ(0)), "e": x},
                }

            res1 = vmap(circuit)(x)
            return res1

        x = jnp.array([[0.1, 0.2, 0.3], [0.7, 0.8, 0.9]])
        expected_state = jnp.array(
            [
                [0.98235508 + 0.00253459j, 0.0198374 - 0.18595308j],
                [0.10537427 + 0.2120056j, 0.23239136 - 0.94336851j],
            ]
        )
        expected_probs = jnp.array([[0.96502793, 0.03497207], [0.05605011, 0.94394989]])
        expected_expval = jnp.array([0.93005586, -0.88789978])
        result = workflow(x)
        assert isinstance(result, dict)
        assert jnp.allclose(result["a"], expected_state)
        assert jnp.allclose(result["b"]["c"], expected_probs)
        assert jnp.allclose(result["b"]["d"], expected_expval)
        assert jnp.allclose(result["b"]["e"], x)
