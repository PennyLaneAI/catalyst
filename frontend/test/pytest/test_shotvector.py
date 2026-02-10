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
    def test_return_format_and_shape(self, shots):
        """Test shot-vector as parameter with single sample measurment"""

        @qjit
        @qml.set_shots(shots)
        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def circuit():
            qml.Hadamard(0)
            return qml.sample()

        assert type(circuit()) == tuple
        assert len(circuit()) == 4
        assert jnp.array(circuit()).shape == (4, 3, 1)

    @pytest.mark.parametrize("mcm_method", ["single-branch-statistics", "one-shot"])
    @pytest.mark.parametrize("shots", [((3, 4),), (3,) * 4, (3, 3, 3, 3), [3, 3, 3, 3]])
    def test_multiple_sample_measurement(self, shots, mcm_method):
        """Test shot-vector with mulitple samples measurment"""

        dev = qml.device("lightning.qubit", wires=1)

        @qjit
        @qml.set_shots(shots)
        @qml.qnode(dev, mcm_method=mcm_method)
        def circuit_list():
            qml.Hadamard(0)
            return [qml.sample(), qml.sample()]

        assert len(circuit_list()) == 2
        assert jnp.array(circuit_list()[0]).shape == (4, 3, 1)
        assert jnp.array(circuit_list()[1]).shape == (4, 3, 1)

        @qjit
        @qml.set_shots(shots)
        @qml.qnode(dev, mcm_method=mcm_method)
        def circuit_dict():
            qml.X(0)
            return {"first": qml.sample(), "second": qml.sample()}

        assert list(circuit_dict().keys()) == ["first", "second"]
        assert jnp.array(circuit_dict()["first"]).shape == (4, 3, 1)
        assert jnp.array(circuit_dict()["second"]).shape == (4, 3, 1)

    @pytest.mark.parametrize("mcm_method", ["single-branch-statistics", "one-shot"])
    def test_shot_vector_with_mixes_shots_and_without_copies(self, mcm_method):
        # pylint: disable=unsubscriptable-object
        """Test shot-vector with mixes shots and without copies"""

        dev = qml.device("lightning.qubit", wires=1)

        @qjit
        @qml.set_shots(((20, 5), 100, (101, 2)))
        @qml.qnode(dev, mcm_method=mcm_method)
        def circuit():
            qml.Hadamard(0)
            return qml.sample()

        assert type(circuit()) == tuple
        assert len(circuit()) == 8

        for i in range(5):
            assert jnp.array(circuit()[i]).shape == (20, 1)
        assert jnp.array(circuit()[5]).shape == (100, 1)
        assert jnp.array(circuit()[6]).shape == (101, 1)
        assert jnp.array(circuit()[7]).shape == (101, 1)

    @pytest.mark.parametrize(
        "measurement",
        [
            (lambda wires: qml.expval(qml.Z(wires)), "ExpectationMP"),
            (lambda wires: qml.var(qml.Z(wires)), "VarianceMP"),
            (lambda wires: qml.probs(wires=wires), "ProbabilityMP"),
        ],
    )
    @pytest.mark.parametrize("mcm_method", ["single-branch-statistics", "one-shot"])
    def test_shot_vector_with_different_measurement(self, measurement, mcm_method):
        """Test a NotImplementedError is raised when using a shot-vector with a measurement that is not qml.sample()"""

        dev = qml.device("lightning.qubit", wires=1)

        @qml.set_shots(((3, 4)))
        @qml.qnode(dev, mcm_method=mcm_method)
        def circuit():
            qml.Hadamard(0)
            return measurement[0](0)

        if measurement[1] == "VarianceMP" and mcm_method == "one-shot":
            with pytest.raises(
                NotImplementedError, match=r"qml.var\(\) cannot be used on observables"
            ):
                qjit(circuit)()
        else:
            with pytest.raises(
                NotImplementedError, match="measurement process does not support shot-vectors"
            ):
                qjit(circuit)()

    def test_shot_vector_with_complex_container_sample(self):
        """Test shot-vector with complex container sample"""

        dev = qml.device("lightning.qubit", wires=1)

        @qjit
        @qml.set_shots(((3, 4),))
        @qml.qnode(dev, mcm_method="single-branch-statistics")
        def circuit():
            qml.Hadamard(0)
            return {
                "first": qml.sample(),
                "second": [100, qml.sample()],
                "third": (qml.sample(), qml.sample()),
            }

        assert list(circuit().keys()) == ["first", "second", "third"]
        assert jnp.array(circuit()["first"]).shape == (4, 3, 1)
        assert circuit()["second"][0] == 100
        assert jnp.array(circuit()["second"][1]).shape == (4, 3, 1)
        assert jnp.array(circuit()["third"]).shape == (2, 4, 3, 1)


if __name__ == "__main__":
    pytest.main(["-x", __file__])
