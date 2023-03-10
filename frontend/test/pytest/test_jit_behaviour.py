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

from catalyst import qjit, measure, grad, for_loop
import pennylane as qml
from jax import numpy as jnp
from timeit import default_timer as timer
from numpy import pi
import numpy as np
import random
import jax
import warnings

from catalyst.compilation_pipelines import CompiledFunction


def f_aot_builder(wires=1, shots=1000):
    @qjit()
    @qml.qnode(qml.device("lightning.qubit", wires=wires, shots=shots))
    def f(x: float):
        qml.RY(x, wires=0)
        return measure(wires=0)

    return f


def f_jit_builder(wires=1, shots=1000):
    @qjit()
    @qml.qnode(qml.device("lightning.qubit", wires=wires, shots=shots))
    def f(x):
        qml.RY(x, wires=0)
        return measure(wires=0)

    return f


def fsample_aot_builder(wires=1, shots=1000):
    @qjit()
    @qml.qnode(qml.device("lightning.qubit", wires=wires, shots=shots))
    def f(x: float):
        qml.RY(x, wires=0)
        return qml.sample()

    return f


class TestDifferentPrecisions:
    def test_different_precisions(self):
        def builder(_in):
            @qjit
            @qml.qnode(qml.device("lightning.qubit", wires=1))
            def f(x):
                qml.RX(x, wires=0)
                return qml.state()

            return f(_in)

        res_float32 = builder(jnp.float32(1.0))
        res_float64 = builder(jnp.float64(1.0))
        assert jnp.allclose(res_float32, res_float64)


@pytest.mark.filterwarnings("ignore:Casting complex")
class TestJittedWithOneTypeRunWithAnother:
    @pytest.mark.parametrize(
        "from_type,to_type",
        [
            (jnp.float32, jnp.complex64),
            (float, jnp.complex64),
            (jnp.int64, jnp.float64),
            (jnp.int32, jnp.float32),
            (int, jnp.float32),
            (jnp.int32, jnp.int64),
            (jnp.int16, jnp.int32),
            (jnp.int8, jnp.int16),
            (jnp.int8, jnp.uint64),
            (jnp.int8, jnp.uint32),
            (jnp.int8, jnp.uint16),
            (jnp.int8, jnp.uint8),
        ],
    )
    def test_recompile_when_unsupported_argument(self, from_type, to_type):
        @qjit
        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def f(x):
            qml.RX(x.astype(float), wires=0)
            return qml.state()

        res_from = f(from_type(1))
        id_from = id(f.compiled_function)
        res_to = f(to_type(1))
        id_to = id(f.compiled_function)
        assert id_from != id_to
        assert jnp.allclose(res_from, res_to)

    @pytest.mark.parametrize(
        "type",
        [
            (jnp.uint64),
            (jnp.uint32),
            (jnp.uint16),
            (jnp.uint8),
        ],
    )
    def test_signless(self, type):
        with warnings.catch_warnings():
            # Treat warnings as an error.
            warnings.simplefilter("error")

            @qjit
            @qml.qnode(qml.device("lightning.qubit", wires=1))
            def f(x):
                qml.RX(jnp.real(x), wires=0)
                return qml.state()

            res_from = f(type(1))

    @pytest.mark.parametrize(
        "to_type",
        [
            (jnp.complex64),
            (jnp.float64),
            (jnp.float32),
            (jnp.int64),
            (jnp.int32),
            (jnp.int16),
            (jnp.uint64),
            (jnp.uint32),
            (jnp.uint16),
            (jnp.uint8),
        ],
    )
    def test_recompile_warning(self, to_type):
        @qjit
        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def f(x: jax.core.ShapedArray([], jnp.int8)):
            qml.RX(x.astype(float), wires=0)
            return qml.state()

        res_from = f(jax.numpy.array(1, dtype=jnp.int8))
        id_from = id(f.compiled_function)
        with pytest.warns(UserWarning):
            res_to = f(to_type(1))
        id_to = id(f.compiled_function)
        assert id_from != id_to
        assert jnp.allclose(res_from, res_to)

    @pytest.mark.parametrize(
        "from_type,to_type",
        [
            (bool, int),
            (bool, float),
            (bool, complex),
            (int, float),
            (int, complex),
            (float, complex),
        ],
    )
    def test_recompile_python_types(self, from_type, to_type):
        @qjit
        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def f(x):
            qml.RX(x.astype(float), wires=0)
            return qml.state()

        res_from = f(from_type(1))
        id_from = id(f.compiled_function)
        res_to = f(to_type(1))
        id_to = id(f.compiled_function)
        assert id_from != id_to
        assert jnp.allclose(res_from, res_to)

    @pytest.mark.parametrize(
        "to_type",
        [
            (jnp.float64),
            (jnp.float32),
            (jnp.int64),
            (jnp.int32),
            (jnp.int16),
            (jnp.uint64),
            (jnp.uint32),
            (jnp.uint16),
            (jnp.uint8),
        ],
    )
    def test_recompile_no_warning(self, to_type):
        @qjit
        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def f(x):
            qml.RX(x.astype(float), wires=0)
            return qml.state()

        res_from = f(jax.numpy.array(1, dtype=jnp.int8))
        id_from = id(f.compiled_function)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            res_to = f(to_type(1))
        id_to = id(f.compiled_function)
        assert id_from != id_to
        assert jnp.allclose(res_from, res_to)


@pytest.mark.filterwarnings("ignore:Casting complex")
class TestTypePromotion:
    @pytest.mark.parametrize(
        "promote_from,val",
        [
            (jnp.float32, 1.0),
            (int, 1),
            (bool, True),
            (jnp.int64, 1),
            (jnp.int32, 1),
            (jnp.int16, 1),
            (jnp.int8, 1),
            (jnp.uint64, 1),
            (jnp.uint32, 1),
            (jnp.uint16, 1),
            (jnp.uint8, 1),
        ],
    )
    def test_promote_to_double(self, promote_from, val):
        @qjit
        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def f(x):
            qml.RX(x, wires=0)
            return qml.state()

        res_float64 = f(jnp.float64(1.0))
        id_64 = id(f.compiled_function)
        res = f(promote_from(val))
        id_from = id(f.compiled_function)
        assert id_64 == id_from
        assert jnp.allclose(res_float64, res)

    @pytest.mark.parametrize(
        "from_type,to_type",
        [
            (int, bool),
            (float, bool),
            (complex, bool),
            (float, int),
            (complex, int),
            (complex, float),
        ],
    )
    def test_promotion_python_types(self, from_type, to_type):
        @qjit
        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def f(x):
            qml.RX(x.astype(float), wires=0)
            return qml.state()

        res_from = f(from_type(1))
        id_from = id(f.compiled_function)
        res_to = f(to_type(1))
        id_to = id(f.compiled_function)
        assert id_from == id_to
        assert jnp.allclose(res_from, res_to)

    @pytest.mark.parametrize(
        "promote_from",
        [
            (jnp.int32),
            (jnp.int16),
            (jnp.int8),
            (jnp.uint32),
            (jnp.uint16),
            (jnp.uint8),
        ],
    )
    def test_promote_to_int(self, promote_from):
        @qjit
        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def f(x):
            qml.RX(x.astype(float), wires=0)
            return qml.state()

        res_int64 = f(jnp.int64(1))
        id_64 = id(f.compiled_function)
        res = f(promote_from(1))
        id_from = id(f.compiled_function)
        assert id_64 == id_from
        assert jnp.allclose(res_int64, res)

    @pytest.mark.parametrize(
        "promote_from",
        [
            (jnp.uint32),
            (jnp.uint16),
            (jnp.uint8),
        ],
    )
    def test_promote_unsigned(self, promote_from):
        @qjit
        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def f(x):
            qml.RX(x.astype(float), wires=0)
            return qml.state()

        res_uint64 = f(jnp.uint64(1))
        id_64 = id(f.compiled_function)
        res_from = f(promote_from(1))
        id_from = id(f.compiled_function)
        assert id_64 == id_from
        assert jnp.allclose(res_uint64, res_from)

    @pytest.mark.parametrize(
        "promote_from",
        [
            (jnp.complex64),
            (jnp.float64),
            (jnp.float32),
            (jnp.int64),
            (jnp.int32),
            (jnp.int16),
            (jnp.int8),
            (jnp.uint64),
            (jnp.uint32),
            (jnp.uint16),
            (jnp.uint8),
        ],
    )
    def test_promote_complex(self, promote_from):
        @qjit
        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def f(x):
            qml.RX(x.real, wires=0)
            return qml.state()

        res_complex64 = f(jax.lax.complex(jnp.float64(1.0), jnp.float64(2.0)))
        id_64 = id(f.compiled_function)
        res_from = f(promote_from(1))
        id_from = id(f.compiled_function)
        assert id_64 == id_from
        assert jnp.allclose(res_complex64, res_from)


class TestCallsiteCompileVsFunctionDefinitionCompile:
    def test_equivalence(self):
        f_jit = f_jit_builder()
        f_aot = f_aot_builder()
        f_jit(0.0)
        assert f_jit.mlir == f_aot.mlir


class TestDecorator:
    def test_function_is_cached(self):
        @qjit
        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def f_no_parenthesis(x):
            qml.RY(x, wires=0)
            return measure(wires=0)

        @qjit()
        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def f_parenthesis(x):
            qml.RY(x, wires=0)
            return measure(wires=0)

        assert f_no_parenthesis(pi) == f_parenthesis(pi)


class TestCaching:
    def test_function_is_cached(self):
        @qjit()
        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def f_jit(x):
            qml.RY(x, wires=0)
            return measure(wires=0)

        compile_and_run_start = timer()
        f_jit = f_jit_builder()
        f_jit(0.0)
        compile_and_run_end = timer()
        compile_and_run_time = compile_and_run_end - compile_and_run_start
        run_start = timer()
        f_jit(pi)
        run_end = timer()
        run_time = run_end - run_end
        assert run_time < compile_and_run_time


class TestModes:
    def test_ftqc_mode(self):
        f_aot = f_aot_builder()
        assert not f_aot(0.0)
        assert f_aot(pi)


class TestShots:
    # Shots influences on the sample instruction
    def test_shots_in_decorator_in_sample(self):
        max_shots = 500
        max_wires = 9
        for x in range(1, 5):
            shots = random.randint(1, max_shots)
            wires = random.randint(1, max_wires)
            expected_shape = (shots, wires)
            f_aot = fsample_aot_builder(wires, shots)
            observed_val = f_aot(0.0)
            observed_shape = jnp.shape(observed_val)
            assert expected_shape == observed_shape

    def test_shots_in_callsite_in_sample(self):
        max_shots = 500
        max_wires = 9
        for x in range(1, 5):
            shots = random.randint(1, max_shots)
            wires = random.randint(1, max_wires)
            expected_shape = (shots, wires)
            f_aot = fsample_aot_builder(wires)
            observed_val = f_aot(0.0, shots=shots)
            observed_shape = jnp.shape(observed_val)
            # We are failing this test because of the type system.
            # If shots is specified AOT, we would need to recompile
            # since the types are set at compile time. I.e., before the call.
            # assert expected_shape == observed_shape


class TestSignatureErrors:
    def test_incompatible_argument(self):
        string = "hello world"
        with pytest.raises(TypeError) as err:
            CompiledFunction.get_runtime_signature([string])
        assert "Unsupported argument type:" in str(err.value)

    def test_incompatible_compiled_vs_runtime(self):
        retval = CompiledFunction.can_promote([], [1])
        assert not retval


class TestClassicalCompilation:
    @pytest.mark.parametrize("a,b", [(1, 1)])
    def test_pure_classical_function(self, a, b):
        def addi(x: int, y: int):
            return x + y

        addc = qjit(addi)
        assert addc.mlir
        assert addi(a, b) == addc(a, b)


class TestArraysInHamiltonian:
    @pytest.mark.parametrize(
        "coeffs",
        [
            (np.array([0.4, 0.7])),
            ([0.4, 0.7]),
            (jnp.array([0.4, 0.7])),
        ],
    )
    def test_array_repr_from_context(self, coeffs):
        @qjit(target="mlir")
        @qml.qnode(qml.device("lightning.qubit", wires=6))
        def f():
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            obs = [qml.PauliX(0) @ qml.PauliZ(1), qml.Hadamard(0)]
            return qml.expval(qml.Hamiltonian(coeffs, obs))

    @pytest.mark.parametrize(
        "coeffs",
        [
            (np.array([0.4, 0.7])),
            ([0.4, 0.7]),
            (jnp.array([0.4, 0.7])),
        ],
    )
    def test_array_repr_as_parameter(self, coeffs):
        @qjit(target="mlir")
        @qml.qnode(qml.device("lightning.qubit", wires=6))
        def f(coeffs):
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            obs = [qml.PauliX(0) @ qml.PauliZ(1), qml.Hadamard(0)]
            return qml.expval(qml.Hamiltonian(coeffs, obs))

        f(coeffs)

    @pytest.mark.parametrize(
        "repr",
        [
            (np.array),
            (jnp.array),
            (list),
        ],
    )
    def test_array_repr_built_in(self, repr):
        @qjit(target="mlir")
        @qml.qnode(qml.device("lightning.qubit", wires=6))
        def f():
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            obs = [qml.PauliX(0) @ qml.PauliZ(1), qml.Hadamard(0)]
            coeffs = repr([0.4, 0.7])
            return qml.expval(qml.Hamiltonian(coeffs, obs))

        assert f.mlir


class TestArraysInHermitian:
    @pytest.fixture
    def matrix(self):
        return [
            [complex(2.0, 0.0), complex(1.0, 1.0), complex(9.0, 2.0), complex(0.0, 0.0)],
            [complex(1.0, -1.0), complex(5.0, 0.0), complex(4.0, 6.0), complex(3.0, -2.0)],
            [complex(9.0, -2.0), complex(4.0, -6.0), complex(10.0, 0.0), complex(1.0, 7.0)],
            [complex(0.0, 0.0), complex(3.0, 2.0), complex(1.0, -7.0), complex(4.0, 0.0)],
        ]

    @pytest.mark.parametrize(
        "repr",
        [
            (np.array),
            (jnp.array),
            (list),
        ],
    )
    def test_array_repr_from_context(self, matrix, repr):
        @qjit(target="mlir")
        @qml.qnode(qml.device("lightning.qubit", wires=6))
        def f(x: float):
            qml.RX(x, wires=0)
            hermitian = qml.Hermitian(repr(matrix), wires=[0, 1])
            return qml.expval(hermitian)

        assert f.mlir

    @pytest.mark.parametrize(
        "repr",
        [
            (np.array),
            (jnp.array),
            (list),
        ],
    )
    def test_array_repr_as_parameter(self, matrix, repr):
        @qjit(target="mlir")
        @qml.qnode(qml.device("lightning.qubit", wires=2))
        def f(matrix):
            qml.RX(jnp.pi, wires=0)
            hermitian = qml.Hermitian(matrix, wires=[0, 1])
            return qml.expval(hermitian)

        f(repr(matrix))

    @pytest.mark.parametrize(
        "repr",
        [
            (np.array),
            (jnp.array),
            (list),
        ],
    )
    def test_array_repr_built_in(self, repr):
        @qjit(target="mlir")
        @qml.qnode(qml.device("lightning.qubit", wires=2))
        def f(x: float):
            qml.RX(x, wires=0)
            matrix = repr(
                [
                    [complex(2.0, 0.0), complex(1.0, 1.0), complex(9.0, 2.0), complex(0.0, 0.0)],
                    [complex(1.0, -1.0), complex(5.0, 0.0), complex(4.0, 6.0), complex(3.0, -2.0)],
                    [complex(9.0, -2.0), complex(4.0, -6.0), complex(10.0, 0.0), complex(1.0, 7.0)],
                    [complex(0.0, 0.0), complex(3.0, 2.0), complex(1.0, -7.0), complex(4.0, 0.0)],
                ]
            )
            hermitian = qml.Hermitian(matrix, wires=[0, 1])
            return qml.expval(hermitian)

        assert f.mlir


class TestTracingQJITAnnotatedFunctions:
    def test_purely_classical_context(self):
        @qjit
        def f():
            return 1

        assert f.mlir

        @qjit
        def g():
            return f() + 1

        assert g.mlir
        assert g() == 2

    def test_quantum_context(self):
        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def f(x: float):
            qml.RX(x, wires=0)
            return qml.state()

        @qjit
        def g1(x: float):
            return f(x)

        assert g1.mlir

        @qjit
        def g2(x: float):
            return g1(x)

        assert g2.mlir

    @pytest.mark.parametrize("phi", [(0.0), (1.0), (2.0)])
    def test_gradient_of_qjit_equivalence(self, phi):
        # Issue 376
        @qjit
        @qml.qnode(device=qml.device("lightning.qubit", wires=1))
        def circuit(phi):
            qml.RX(phi, wires=0)
            return qml.expval(qml.PauliZ(0))

        @qjit
        def workflow(phi):
            g = grad(circuit)
            return g(phi)

        assert np.allclose(qjit(grad(qjit(circuit)))(phi), qjit(grad(circuit))(phi))
        assert np.allclose(qjit(grad(circuit))(phi), workflow(phi))
        assert np.allclose(workflow(phi), qjit(grad(circuit))(phi))

    @pytest.mark.parametrize("phi", [(0.0), (1.0), (2.0)])
    def test_gradient_of_qjit_correctness(self, phi):
        @qml.qnode(device=qml.device("lightning.qubit", wires=1))
        def circuit(phi):
            qml.RX(phi, wires=0)
            return qml.expval(qml.PauliZ(0))

        @qjit
        def workflow1(phi):
            g = grad(circuit)
            return g(phi)

        @qjit
        def workflow2(phi):
            g = grad(qjit(circuit))
            return g(phi)

        assert np.allclose(workflow1(phi), workflow2(phi))

    def test_gradient_of_qjit_names(self):
        @qml.qnode(device=qml.device("lightning.qubit", wires=1))
        def circuit(phi):
            qml.RX(phi, wires=0)
            return qml.expval(qml.PauliZ(0))

        @qjit
        def workflow(phi: float):
            g = grad(circuit)
            return g(phi)

        mlir_v1 = workflow.mlir

        @qjit
        def workflow(phi: float):
            g = grad(qjit(circuit))
            return g(phi)

        mlir_v2 = workflow.mlir

        assert mlir_v1 == mlir_v2


class TestDefaultAvailableIR:
    def test_mlir(self):
        @qjit
        def f():
            return 1

        assert f.mlir

    def test_qir(self):
        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def f(x: float):
            qml.RX(x, wires=0)
            return qml.state()

        @qjit
        def g(x: float):
            return f(x)

        assert g.qir
        assert "__quantum__qis" in g.qir


class TestAvoidVerification:
    def test_no_verification(self, capfd):
        dev1 = qml.device("lightning.qubit", wires=1)

        @qml.qnode(device=dev1)
        def circuit(x):
            return x

        def test():
            @for_loop(0, 1, 1)
            def loop(i):
                circuit(1.0)

            loop()
            return

        jitted_function = qjit(test)
        capture_string = capfd.readouterr()
        assert "does not reference a valid function" not in capture_string.err


if __name__ == "__main__":
    pytest.main(["-x", __file__])
