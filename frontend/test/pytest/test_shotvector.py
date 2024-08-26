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

import jax.numpy as jnp
import pennylane as qml
import pytest

from catalyst import qjit


class TestShotVector:
    """Test shot-vector"""

    @pytest.mark.parametrize("shots", [((3, 4),), (3,) * 4, (3, 3, 3, 3), [3, 3, 3, 3]])
    def test_return_format_and_shape(self, shots, backend):
        """Test shot-vector as parameter with single sample measurment"""

        @qjit
        @qml.qnode(qml.device(backend, wires=1, shots=shots))
        def circuit():
            qml.Hadamard(0)
            return qml.sample()

        assert type(circuit()) == tuple
        assert len(circuit()) == 4
        assert jnp.shape(circuit()) == (4, 3, 1)

    @pytest.mark.parametrize("shots", [((3, 4),), (3,) * 4, (3, 3, 3, 3), [3, 3, 3, 3]])
    def test_multiple_sample_measurement(self, shots, backend):
        """Test shot-vector with mulitple samples measurment"""

        dev = qml.device(backend, wires=1, shots=shots)

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


if __name__ == "__main__":
    pytest.main(["-x", __file__])
