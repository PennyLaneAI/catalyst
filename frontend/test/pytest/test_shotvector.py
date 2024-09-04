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

"""Integration tests for the runtime assertion feature."""

import re

import jax.numpy as jnp
import pennylane as qml
import pytest

from catalyst import qjit


class TestShotVector:
    """Test shot-vector"""

    @pytest.mark.parametrize("shots", [((3, 4),), (3,) * 4, (3, 3, 3, 3), [3, 3, 3, 3]])
    def test_return_format_and_shape(self, shots):
        """Test shot-vector as parameter with single sample measurment"""

        @qjit
        @qml.qnode(qml.device("lightning.qubit", wires=1, shots=shots))
        def circuit():
            qml.Hadamard(0)
            return qml.sample()

        assert type(circuit()) == tuple
        assert len(circuit()) == 4
        assert jnp.shape(circuit()) == (4, 3, 1)

    @pytest.mark.parametrize("shots", [((3, 4),), (3,) * 4, (3, 3, 3, 3), [3, 3, 3, 3]])
    def test_multiple_sample_measurement(self, shots):
        """Test shot-vector with mulitple samples measurment"""

        dev = qml.device("lightning.qubit", wires=1, shots=shots)

        @qjit
        @qml.qnode(dev)
        def circuit_list():
            qml.Hadamard(0)
            return [qml.sample(), qml.sample()]

        assert len(circuit_list()) == 2
        assert jnp.shape(circuit_list()[0]) == (4, 3, 1)
        assert jnp.shape(circuit_list()[1]) == (4, 3, 1)

        @qjit
        @qml.qnode(dev)
        def circuit_dict():
            qml.X(0)
            return {"first": qml.sample(), "second": qml.sample()}

        assert list(circuit_dict().keys()) == ["first", "second"]
        assert jnp.shape(circuit_dict()["first"]) == (4, 3, 1)
        assert jnp.shape(circuit_dict()["second"]) == (4, 3, 1)

    def test_shot_vector_with_mixes_shots_and_without_copies(self):
        """Test shot-vector with mixes shots and without copies"""

        dev = qml.device("lightning.qubit", wires=1, shots=((20, 5), 100, (101, 2)))

        @qjit
        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(0)
            return qml.sample()

        assert type(circuit()) == tuple
        assert len(circuit()) == 8

        for i in range(5):
            assert jnp.shape(circuit()[i]) == (20, 1)
        assert jnp.shape(circuit()[5]) == (100, 1)
        assert jnp.shape(circuit()[6]) == (101, 1)
        assert jnp.shape(circuit()[7]) == (101, 1)

    def test_shot_vector_with_different_measurement(self):
        """Test a NotImplementedError is raised when using a shot-vector with a measurement that is not qml.sample()"""

        dev = qml.device("lightning.qubit", wires=1, shots=((3, 4)))

        with pytest.raises(
            NotImplementedError,
            match=re.escape(
                "Measurement expval is not supported a shot-vector. Use qml.sample() instead."
            ),
        ):

            @qjit
            @qml.qnode(dev)
            def circuit():
                qml.Hadamard(0)
                return qml.expval(qml.Z(0))

            circuit()

        with pytest.raises(
            NotImplementedError,
            match=re.escape(
                "Measurement var is not supported a shot-vector. Use qml.sample() instead."
            ),
        ):

            @qjit
            @qml.qnode(dev)
            def circuit():
                qml.Hadamard(0)
                return qml.var(qml.Z(0))

            circuit()

        with pytest.raises(
            NotImplementedError,
            match=re.escape(
                "Measurement probs is not supported a shot-vector. Use qml.sample() instead."
            ),
        ):

            @qjit
            @qml.qnode(dev)
            def circuit():
                qml.Hadamard(0)
                return qml.probs(wires=[0])

            circuit()

    def test_shot_vector_with_complex_container_sample(self):
        """Test shot-vector with complex container sample"""
        
        dev = qml.device("lightning.qubit", wires=1, shots=((3, 4),))

        @qjit
        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(0)
            return {"first": qml.sample(), "second": [100, qml.sample()], "third": (qml.sample(), qml.sample())}

        assert list(circuit().keys()) == ["first", "second", "third"]
        assert jnp.shape(circuit()["first"]) == (4, 3, 1)
        assert circuit()["second"][0] == 100
        assert jnp.shape(circuit()["second"][1]) == (4, 3, 1)
        assert jnp.shape(circuit()["third"]) == (2, 4, 3, 1)

if __name__ == "__main__":
    pytest.main(["-x", __file__])
