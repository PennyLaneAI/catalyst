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

"""Test integration for catalyst.mitigate_with_zne."""

import jax
import numpy as np
import pennylane as qml
import pytest
from pennylane.transforms import exponential_extrapolate

import catalyst
from catalyst.api_extensions.error_mitigation import polynomial_extrapolation

quadratic_extrapolation = polynomial_extrapolation(2)


def skip_if_exponential_extrapolation_unstable(circuit_param, extrapolation_func):
    """skip test if exponential extrapolation will be unstable"""
    if circuit_param < 0.3 and extrapolation_func == exponential_extrapolate:
        pytest.skip("Exponential extrapolation unstable in this region.")


@pytest.mark.parametrize("params", [0.1, 0.2, 0.3, 0.4, 0.5])
@pytest.mark.parametrize("extrapolation", [quadratic_extrapolation, exponential_extrapolate])
@pytest.mark.parametrize("folding", ["global", "all"])
def test_single_measurement(params, extrapolation, folding):
    """Test that without noise the same results are returned for single measurements."""
    skip_if_exponential_extrapolation_unstable(params, extrapolation)

    dev = qml.device("lightning.qubit", wires=2)

    @qml.qnode(device=dev)
    def circuit(x):
        qml.Hadamard(wires=0)
        qml.RZ(x, wires=0)
        qml.RZ(x, wires=0)
        qml.CNOT(wires=[1, 0])
        qml.Hadamard(wires=1)
        return qml.expval(qml.PauliY(wires=0))

    @catalyst.qjit
    def mitigated_qnode(args):
        return catalyst.mitigate_with_zne(
            circuit,
            scale_factors=jax.numpy.array([1, 2, 3]),
            extrapolate=extrapolation,
            folding=folding,
        )(args)

    assert np.allclose(mitigated_qnode(params), circuit(params))


@pytest.mark.parametrize("params", [0.1, 0.2, 0.3, 0.4, 0.5])
@pytest.mark.parametrize("extrapolation", [quadratic_extrapolation, exponential_extrapolate])
@pytest.mark.parametrize("folding", ["global", "all"])
def test_multiple_measurements(params, extrapolation, folding):
    """Test that without noise the same results are returned for multiple measurements"""
    skip_if_exponential_extrapolation_unstable(params, extrapolation)

    dev = qml.device("lightning.qubit", wires=2)

    @qml.qnode(device=dev)
    def circuit(x):
        qml.Hadamard(wires=0)
        qml.RZ(x, wires=0)
        qml.RZ(x, wires=0)
        qml.CNOT(wires=[1, 0])
        qml.Hadamard(wires=1)
        return qml.expval(qml.PauliY(wires=0)), qml.expval(qml.PauliY(wires=1))

    @catalyst.qjit
    def mitigated_qnode(args):
        return catalyst.mitigate_with_zne(
            circuit,
            scale_factors=jax.numpy.array([1, 2, 3]),
            extrapolate=extrapolation,
            folding=folding,
        )(args)

    assert np.allclose(mitigated_qnode(params), circuit(params))


@pytest.mark.parametrize("params", [0.1, 0.2, 0.3, 0.4, 0.5])
@pytest.mark.parametrize("folding", ["global", "all"])
def test_single_measurement_control_flow(params, folding):
    """Test that without noise the same results are returned for single measurement and with
    control flow."""
    dev = qml.device("lightning.qubit", wires=2)

    @qml.qnode(device=dev)
    def circuit(x, n):
        @catalyst.for_loop(0, n, 1)
        def loop_0(i):  # pylint: disable=unused-argument
            qml.RX(x, wires=0)

        loop_0()

        qml.Hadamard(wires=0)
        qml.RZ(x, wires=0)
        qml.RZ(x, wires=0)
        qml.CNOT(wires=[1, 0])
        qml.Hadamard(wires=1)

        @catalyst.for_loop(0, n, 1)
        def loop_1(i):  # pylint: disable=unused-argument
            qml.RX(x, wires=0)

        loop_1()
        return qml.expval(qml.PauliY(wires=0))

    @catalyst.qjit
    def mitigated_qnode(args, n):
        return catalyst.mitigate_with_zne(
            circuit, scale_factors=jax.numpy.array([1, 2, 3]), folding=folding
        )(args, n)

    assert np.allclose(mitigated_qnode(params, 3), catalyst.qjit(circuit)(params, 3))


def test_not_qnode_error():
    """Test that when applied not on a QNode the transform raises an error."""

    def circuit(x):
        return jax.numpy.sin(x)

    @catalyst.qjit
    def mitigated_function(args):
        return catalyst.mitigate_with_zne(circuit, scale_factors=jax.numpy.array([1, 2, 3]))(args)

    with pytest.raises(TypeError, match="A QNode is expected, got the classical function"):
        mitigated_function(0.1)


@pytest.mark.parametrize("extrapolation", [quadratic_extrapolation, exponential_extrapolate])
def test_dtype_error(extrapolation):
    """Test that an error is raised when multiple results do not have the same dtype."""
    dev = qml.device("lightning.qubit", wires=2)

    @qml.qnode(device=dev)
    def circuit(x):
        qml.Hadamard(wires=0)
        qml.RZ(x, wires=0)
        qml.RZ(x, wires=0)
        qml.CNOT(wires=[1, 0])
        qml.Hadamard(wires=1)
        return qml.expval(qml.PauliY(wires=0)), 1

    @catalyst.qjit
    def mitigated_qnode(args):
        return catalyst.mitigate_with_zne(
            circuit, scale_factors=jax.numpy.array([1, 2, 3]), extrapolate=extrapolation
        )(args)

    with pytest.raises(
        TypeError, match="All expectation and classical values dtypes must match and be float."
    ):
        mitigated_qnode(0.1)


@pytest.mark.parametrize("extrapolation", [quadratic_extrapolation, exponential_extrapolate])
def test_dtype_not_float_error(extrapolation):
    """Test that an error is raised when results are not float."""
    dev = qml.device("lightning.qubit", wires=2)

    @qml.qnode(device=dev)
    def circuit(x):
        qml.Hadamard(wires=0)
        qml.RZ(x, wires=0)
        qml.RZ(x, wires=0)
        qml.CNOT(wires=[1, 0])
        qml.Hadamard(wires=1)
        return 1

    @catalyst.qjit
    def mitigated_qnode(args):
        return catalyst.mitigate_with_zne(
            circuit, scale_factors=jax.numpy.array([1, 2, 3]), extrapolate=extrapolation
        )(args)

    with pytest.raises(
        TypeError, match="All expectation and classical values dtypes must match and be float."
    ):
        mitigated_qnode(0.1)


@pytest.mark.parametrize("extrapolation", [quadratic_extrapolation, exponential_extrapolate])
def test_shape_error(extrapolation):
    """Test that an error is raised when results have shape."""
    dev = qml.device("lightning.qubit", wires=2)

    @qml.qnode(device=dev)
    def circuit(x):
        qml.Hadamard(wires=0)
        qml.RZ(x, wires=0)
        qml.RZ(x, wires=0)
        qml.CNOT(wires=[1, 0])
        qml.Hadamard(wires=1)
        return qml.probs(wires=0)

    @catalyst.qjit
    def mitigated_qnode(args):
        return catalyst.mitigate_with_zne(
            circuit, scale_factors=jax.numpy.array([1, 2, 3]), extrapolate=extrapolation
        )(args)

    with pytest.raises(
        TypeError, match="Only expectations values and classical scalar values can be returned"
    ):
        mitigated_qnode(0.1)


def test_folding_type_not_supported():
    """Test that value of folding argument is from allowed list"""
    dev = qml.device("lightning.qubit", wires=2)

    @qml.qnode(device=dev)
    def circuit():
        return 0.0

    def mitigated_qnode():
        return catalyst.mitigate_with_zne(
            circuit, scale_factors=[], folding="bad-folding-type-value"
        )()

    with pytest.raises(ValueError, match="Folding type must be"):
        catalyst.qjit(mitigated_qnode)


def test_folding_type_not_implemented():
    """Test value of folding argument supported but not yet developed"""
    dev = qml.device("lightning.qubit", wires=2)

    @qml.qnode(device=dev)
    def circuit():
        return 0.0

    def mitigated_qnode():
        return catalyst.mitigate_with_zne(circuit, scale_factors=[], folding="random")()

    with pytest.raises(NotImplementedError):
        catalyst.qjit(mitigated_qnode)


@pytest.mark.parametrize("params", [0.1, 0.2, 0.3, 0.4, 0.5])
@pytest.mark.parametrize("extrapolation", [quadratic_extrapolation, exponential_extrapolate])
@pytest.mark.parametrize("folding", ["global", "all"])
def test_zne_usage_patterns(params, extrapolation, folding):
    """Test usage patterns of catalyst.zne."""
    skip_if_exponential_extrapolation_unstable(params, extrapolation)

    dev = qml.device("lightning.qubit", wires=2)

    @qml.qnode(device=dev)
    def fn(x):
        qml.Hadamard(wires=0)
        qml.RZ(x, wires=0)
        qml.RZ(x, wires=0)
        qml.CNOT(wires=[1, 0])
        qml.Hadamard(wires=1)
        return qml.expval(qml.PauliY(wires=0))

    @catalyst.qjit
    def mitigated_qnode_fn_as_argument(args):
        return catalyst.mitigate_with_zne(
            fn, scale_factors=jax.numpy.array([1, 2, 3]), extrapolate=extrapolation, folding=folding
        )(args)

    @catalyst.qjit
    def mitigated_qnode_partial(args):
        return catalyst.mitigate_with_zne(
            scale_factors=jax.numpy.array([1, 2, 3]), extrapolate=extrapolation, folding=folding
        )(fn)(args)

    assert np.allclose(mitigated_qnode_fn_as_argument(params), fn(params))
    assert np.allclose(mitigated_qnode_partial(params), fn(params))


def test_zne_with_jax_polyfit():
    """test mitigate_with_zne works with jax polyfit"""
    dev = qml.device("lightning.qubit", wires=2)

    @qml.qnode(device=dev)
    def circuit():
        qml.Hadamard(wires=0)
        qml.RZ(0.4, wires=0)
        qml.RZ(0.3, wires=0)
        qml.CNOT(wires=[1, 0])
        qml.Hadamard(wires=1)
        return qml.expval(qml.PauliY(wires=0))

    def jax_extrapolation(scale_factors, results):
        return jax.numpy.polyfit(scale_factors, results, 2)[-1]

    @catalyst.qjit
    def mitigated_qnode():
        return catalyst.mitigate_with_zne(
            circuit, scale_factors=jax.numpy.array([1, 2, 3]), extrapolate=jax_extrapolation
        )()

    assert np.allclose(mitigated_qnode(), circuit())


def test_zne_with_extrap_kwargs():
    """test mitigate_with_zne with keyword arguments for extrapolation function"""
    dev = qml.device("lightning.qubit", wires=2)

    @qml.qnode(device=dev)
    def circuit():
        qml.Hadamard(wires=0)
        qml.RZ(0.1, wires=0)
        qml.RZ(0.2, wires=0)
        qml.CNOT(wires=[1, 0])
        qml.Hadamard(wires=1)
        return qml.expval(qml.PauliY(wires=0))

    @catalyst.qjit
    def mitigated_qnode():
        return catalyst.mitigate_with_zne(
            circuit,
            scale_factors=jax.numpy.array([1, 2, 3]),
            extrapolate=qml.transforms.poly_extrapolate,
            extrapolate_kwargs={"order": 2},
        )()

    assert np.allclose(mitigated_qnode(), circuit())


def test_exponential_extrapolation_with_kwargs():
    """test mitigate_with_zne with keyword arguments for exponential extrapolation function"""
    dev = qml.device("lightning.qubit", wires=2)

    @qml.qnode(device=dev)
    def circuit():
        qml.Hadamard(wires=0)
        qml.RZ(0.1, wires=0)
        qml.RZ(0.2, wires=0)
        qml.CNOT(wires=[1, 0])
        qml.Hadamard(wires=1)
        return qml.expval(qml.PauliY(wires=0))

    @catalyst.qjit
    def mitigated_qnode():
        return catalyst.mitigate_with_zne(
            circuit,
            scale_factors=jax.numpy.array([1, 2, 3]),
            extrapolate=qml.transforms.exponential_extrapolate,
            extrapolate_kwargs={"asymptote": 3},
        )()

    assert np.allclose(mitigated_qnode(), circuit())


if __name__ == "__main__":
    pytest.main(["-x", __file__])
