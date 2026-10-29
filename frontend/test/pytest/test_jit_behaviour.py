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

import os
import random
import warnings
from copy import deepcopy
from functools import partial
from timeit import default_timer as timer

import jax
import numpy as np
import pennylane as qp
import pytest
from jax import numpy as jnp
from numpy import pi

from catalyst import QJIT, for_loop, grad, measure, qjit
from catalyst.jax_primitives import _scalar_abstractify
from catalyst.tracing.type_signatures import (
    TypeCompatibility,
    get_abstract_signature,
    params_are_annotated,
    typecheck_signatures,
)
from catalyst.utils.exceptions import CompileError

# pylint: disable=too-many-lines


def f_aot_builder(backend, wires=1, shots=1000):
    """Test AOT builder."""

    @qjit
    @qp.set_shots(shots)
    @qp.qnode(qp.device(backend, wires=wires))
    def f(x: float) -> bool:
        qp.RY(x, wires=0)
        return measure(wires=0)

    return f


def f_jit_builder(backend, wires=1, shots=1000):
    """Test JIT builder."""

    @qjit
    @qp.set_shots(shots)
    @qp.qnode(qp.device(backend, wires=wires))
    def f(x):
        qp.RY(x, wires=0)
        return measure(wires=0)

    return f


def fsample_aot_builder(backend, wires=1, shots=1000):
    """Test AOT builder with the sample measurement process."""

    @qjit
    @qp.set_shots(shots)
    @qp.qnode(qp.device(backend, wires=wires))
    def f(x: float):
        qp.RY(x, wires=0)
        return qp.sample()

    return f


class TestDifferentPrecisions:
    def test_different_precisions(self, backend):
        """Test different precisions."""

        def builder(_in):
            @qjit
            @qp.qnode(qp.device(backend, wires=1))
            def f(x):
                qp.RX(x, wires=0)
                return qp.state()

            return f(_in)

        res_float32 = builder(jnp.float32(1.0))
        res_float64 = builder(jnp.float64(1.0))
        assert jnp.allclose(res_float32, res_float64)


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
    def test_recompile_when_unsupported_argument(self, from_type, to_type, backend):
        """Test recompile when unsupported argument."""

        @qjit
        @qp.qnode(qp.device(backend, wires=1))
        def f(x):
            if x.dtype == jnp.dtype(jnp.complex64):
                x = jnp.real(x)
            qp.RX(x.astype(float), wires=0)
            return qp.state()

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
    def test_signless(self, type, backend):
        """Test signless."""

        with warnings.catch_warnings():
            # Treat warnings as an error.
            warnings.simplefilter("error")

            @qjit
            @qp.qnode(qp.device(backend, wires=1))
            def f(x):
                qp.RX(jnp.real(x), wires=0)
                return qp.state()

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
    def test_recompile_warning(self, to_type, backend):
        """Test recompile warning."""

        @qjit
        @qp.qnode(qp.device(backend, wires=1))
        def f(x: jax.core.ShapedArray([], jnp.int8)):
            if x.dtype == jnp.dtype(jnp.complex64):
                x = jnp.real(x)
            qp.RX(x.astype(float), wires=0)
            return qp.state()

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
    def test_recompile_python_types(self, from_type, to_type, backend):
        """Test recompile python types."""

        @qjit
        @qp.qnode(qp.device(backend, wires=1))
        def f(x):
            if x.dtype == jnp.dtype(jnp.complex128):
                x = jnp.real(x)
            qp.RX(x.astype(float), wires=0)
            return qp.state()

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
    def test_recompile_no_warning(self, to_type, backend):
        """Test recompile no warning."""

        @qjit
        @qp.qnode(qp.device(backend, wires=1))
        def f(x):
            qp.RX(x.astype(float), wires=0)
            return qp.state()

        res_from = f(jax.numpy.array(1, dtype=jnp.int8))
        id_from = id(f.compiled_function)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            res_to = f(to_type(1))
        id_to = id(f.compiled_function)
        assert id_from != id_to
        assert jnp.allclose(res_from, res_to)


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
    def test_promote_to_double(self, promote_from, val, backend):
        """Test promote to double."""

        @qjit
        @qp.qnode(qp.device(backend, wires=1))
        def f(x):
            qp.RX(x, wires=0)
            return qp.state()

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
    def test_promotion_python_types(self, from_type, to_type, backend):
        """Test promotion python types."""

        @qjit
        @qp.qnode(qp.device(backend, wires=1))
        def f(x):
            if x.dtype == jnp.dtype(jnp.complex128):
                x = jnp.real(x)
            qp.RX(x.astype(float), wires=0)
            return qp.state()

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
    def test_promote_to_int(self, promote_from, backend):
        """Test promote to int."""

        @qjit
        @qp.qnode(qp.device(backend, wires=1))
        def f(x):
            qp.RX(x.astype(float), wires=0)
            return qp.state()

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
    def test_promote_unsigned(self, promote_from, backend):
        """Test promote to unsigned."""

        @qjit
        @qp.qnode(qp.device(backend, wires=1))
        def f(x):
            qp.RX(x.astype(float), wires=0)
            return qp.state()

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
    def test_promote_complex(self, promote_from, backend):
        """Test promote complex."""

        @qjit
        @qp.qnode(qp.device(backend, wires=1))
        def f(x):
            qp.RX(x.real, wires=0)
            return qp.state()

        res_complex64 = f(jax.lax.complex(jnp.float64(1.0), jnp.float64(2.0)))
        id_64 = id(f.compiled_function)
        res_from = f(promote_from(1))
        id_from = id(f.compiled_function)
        assert id_64 == id_from
        assert jnp.allclose(res_complex64, res_from)


class TestCallsiteCompileVsFunctionDefinitionCompile:
    def test_equivalence(self, backend):
        """Test equivalence."""

        f_jit = f_jit_builder(backend)
        f_aot = f_aot_builder(backend)
        f_jit(0.0)
        assert f_jit.mlir == f_aot.mlir


class TestDecorator:
    def test_function_is_cached(self, backend):
        """Test function is cached with decorator."""

        @qjit
        @qp.qnode(qp.device(backend, wires=1))
        def f_no_parenthesis(x):
            qp.RY(x, wires=0)
            return measure(wires=0)

        @qjit
        @qp.qnode(qp.device(backend, wires=1))
        def f_parenthesis(x):
            qp.RY(x, wires=0)
            return measure(wires=0)

        assert f_no_parenthesis(pi) == f_parenthesis(pi)


class TestQJITTransforms:
    """Test transforms applied directly to captured QJIT workflows."""

    def test_transform_qjit_qnode(self, backend: str) -> None:
        """A transform applied after qjit is scheduled on the wrapped QNode."""

        @qjit(capture=True, collect_decomp_rules=False)
        @qp.qnode(qp.device(backend, wires=1))
        def circuit(theta):
            qp.RX(theta, wires=0)
            qp.Hadamard(0)
            qp.Hadamard(0)
            return qp.expval(qp.Z(0))

        transformed: QJIT = qp.transforms.cancel_inverses(circuit)

        assert isinstance(transformed, QJIT)
        assert transformed is not circuit
        assert qp.math.allclose(transformed(0.4), qp.math.cos(0.4))
        assert qp.math.allclose(circuit(0.4), qp.math.cos(0.4))
        assert transformed.mlir.count('transform.apply_registered_pass "cancel-inverses"') == 1
        assert 'transform.apply_registered_pass "cancel-inverses"' not in circuit.mlir
        assert qp.transforms.cancel_inverses in transformed.original_function.compile_pipeline

        ops = qp.analyze(transformed, level=1)(0.5).resources.quantum_operations
        assert ops.get("Hadamard", 0) == 0

    def test_transform_qjit_classical_processing(self, backend: str) -> None:
        """Transforming a QJIT preserves classical processing around its QNode."""

        @qp.qnode(qp.device(backend, wires=1))
        def circuit(theta):
            qp.RX(theta, wires=0)
            qp.Hadamard(0)
            qp.Hadamard(0)
            return qp.expval(qp.Z(0))

        @qjit(capture=True, collect_decomp_rules=False)
        def workflow(theta):
            return 3 * circuit(2 * theta) + theta

        transformed: QJIT = qp.transforms.cancel_inverses(workflow)
        expected = 3 * qp.math.cos(0.8) + 0.4

        assert isinstance(transformed, QJIT)
        assert qp.math.allclose(transformed(0.4), expected)
        assert qp.math.allclose(workflow(0.4), expected)
        assert transformed.mlir.count('transform.apply_registered_pass "cancel-inverses"') == 1
        assert 'transform.apply_registered_pass "cancel-inverses"' not in workflow.mlir
        assert not circuit.compile_pipeline

    def test_transform_qjit_multiple_qnodes(self, backend: str) -> None:
        """A transform applied to a QJIT reaches every QNode in the workflow."""
        device = qp.device(backend, wires=1)

        @qp.qnode(device)
        def first_circuit(theta):
            qp.RX(theta, wires=0)
            qp.Hadamard(0)
            qp.Hadamard(0)
            return qp.expval(qp.Z(0))

        @qp.qnode(device)
        def second_circuit(theta):
            qp.RY(theta, wires=0)
            qp.X(0)
            qp.X(0)
            return qp.expval(qp.Z(0))

        @qjit(capture=True, collect_decomp_rules=False)
        def workflow(theta):
            return first_circuit(theta), 2 * second_circuit(3 * theta)

        transformed: QJIT = qp.transforms.cancel_inverses(workflow)
        expected = (qp.math.cos(0.4), 2 * qp.math.cos(1.2))

        assert isinstance(transformed, QJIT)
        assert qp.math.allclose(transformed(0.4), expected)
        assert qp.math.allclose(workflow(0.4), expected)
        assert transformed.mlir.count('transform.apply_registered_pass "cancel-inverses"') == 2
        assert 'transform.apply_registered_pass "cancel-inverses"' not in workflow.mlir
        assert not first_circuit.compile_pipeline
        assert not second_circuit.compile_pipeline

    @pytest.mark.parametrize("as_keyword", [False, True], ids=["positional", "keyword"])
    def test_decompose_qjit_arguments(self, backend: str, as_keyword: bool) -> None:
        """Decomposition accepts positional and keyword arguments after qjit."""

        @qjit(capture=True)
        @qp.qnode(qp.device(backend, wires=1))
        def circuit(theta):
            qp.Rot(theta, 2 * theta, 3 * theta, wires=0)
            return qp.expval(qp.Z(0))

        gate_set = [qp.RZ, qp.RY]
        if as_keyword:
            transformed = qp.decompose(circuit, gate_set=gate_set)
        else:
            transformed = qp.decompose(circuit, gate_set)

        assert qp.math.allclose(transformed(0.4), qp.math.cos(0.8))
        assert 'transform.apply_registered_pass "graph-decomposition"' in transformed.mlir
        ops = qp.analyze(transformed, level=1)(0.4).resources.quantum_operations
        assert ops == {"RZ": 2, "RY": 1}
        original_ops = qp.analyze(circuit, level=1)(0.4).resources.quantum_operations
        assert original_ops == {"Rot": 1}

    def test_transform_executed_qjit_does_not_copy_cache(self, backend: str) -> None:
        """Transforming an executed workflow starts with independent compilation state."""

        @qp.qnode(qp.device(backend, wires=1))
        def circuit(theta):
            qp.RX(theta, wires=0)
            qp.Hadamard(0)
            qp.Hadamard(0)
            return qp.expval(qp.Z(0))

        @qjit(capture=True, collect_decomp_rules=False)
        def workflow(theta):
            return 2 * circuit(theta)

        expected = 2 * qp.math.cos(0.4)
        assert qp.math.allclose(workflow(0.4), expected)
        assert workflow.llvmir is not None
        original_artifacts = {
            name: getattr(workflow, name)
            for name in ("jaxpr", "mlir_module", "compiled_function", "workspace")
        }
        for name, artifact in original_artifacts.items():
            assert artifact is not None, name
        original_cache = workflow.fn_cache.cache.copy()
        assert original_cache

        transformed = qp.transforms.cancel_inverses(workflow)

        assert transformed.fn_cache is not workflow.fn_cache
        assert not transformed.fn_cache.cache
        assert transformed.mlir is None
        assert transformed.llvmir is None
        for name, artifact in original_artifacts.items():
            assert getattr(transformed, name) is None
            assert getattr(workflow, name) is artifact

        assert qp.math.allclose(transformed(0.4), expected)
        assert transformed.llvmir is not None
        assert transformed.fn_cache.cache
        assert 'transform.apply_registered_pass "cancel-inverses"' in transformed.mlir
        assert 'transform.apply_registered_pass "cancel-inverses"' not in workflow.mlir
        for name, artifact in original_artifacts.items():
            assert getattr(transformed, name) is not None
            assert getattr(transformed, name) is not artifact
            assert getattr(workflow, name) is artifact
        assert workflow.fn_cache.cache.keys() == original_cache.keys()
        for key, entry in original_cache.items():
            assert workflow.fn_cache.cache[key] is entry
        assert qp.math.allclose(workflow(0.4), expected)

    def test_transform_qjit_copies_compile_options(self, backend: str) -> None:
        """The transformed QJIT owns independent options, including nested mutable values."""

        @qp.qnode(qp.device(backend, wires=1))
        def circuit(theta):
            qp.RX(theta, wires=0)
            return qp.expval(qp.Z(0))

        original = qjit(
            circuit,
            capture=True,
            collect_decomp_rules=False,
            target="mlir",
            pipelines=[("custom", ["canonicalize"])],
            seed=42,
        )
        original_options = original.compile_options
        expected_options = deepcopy(vars(original_options))

        transformed: QJIT = qp.transforms.cancel_inverses(original)

        assert original.compile_options is original_options
        assert vars(original.compile_options) == expected_options
        assert transformed.compile_options is not original_options
        assert transformed.compile_options.capture is True
        assert transformed.compile_options.collect_decomp_rules is False
        assert transformed.compile_options.target == "mlir"
        assert transformed.compile_options.seed == 42
        assert transformed.compile_options.pipelines == expected_options["pipelines"]

        transformed.compile_options.seed = 123
        transformed.compile_options.pipelines[0][1].append("cse")
        transformed.compile_options.pipelines.append(("extra", ["canonicalize"]))

        assert vars(original.compile_options) == expected_options
        assert original.original_function is circuit
        assert not circuit.compile_pipeline
        assert len(transformed.original_function.compile_pipeline) == 1


class TestCaching:
    def test_function_is_cached(self, backend):
        """Test function is cached."""

        @qjit
        @qp.qnode(qp.device(backend, wires=1))
        def f_jit(x):
            qp.RY(x, wires=0)
            return measure(wires=0)

        compile_and_run_start = timer()
        f_jit = f_jit_builder(backend)
        f_jit(0.0)
        compile_and_run_end = timer()
        compile_and_run_time = compile_and_run_end - compile_and_run_start
        run_start = timer()
        f_jit(pi)
        run_end = timer()
        run_time = run_end - run_end
        assert run_time < compile_and_run_time

    def test_subfunction_is_cached(self, backend):
        """
        Test a function called multiple times within an outer jit function is generated only once
        """

        @qp.qnode(qp.device(backend, wires=1))
        def f(x):
            qp.RX(x, wires=0)
            return qp.expval(qp.PauliZ(wires=0))

        @qjit
        def g(x: float):
            return f(x) + f(x)

        assert "func.func public @f(" in g.mlir
        assert g.mlir.count("launch_kernel @module_f::@f") == 2
        # Duplicate function generation results in a "_0" suffix
        assert not "func.func public @f_0(" in g.mlir


class TestShots:
    # Shots influences on the sample instruction
    def test_shots_in_decorator_in_sample(self, backend):
        """Test shots in decorator in sample."""

        max_shots = 500
        max_wires = 9
        for x in range(1, 5):
            shots = random.randint(1, max_shots)
            wires = random.randint(1, max_wires)
            expected_shape = (shots, wires)
            f_aot = fsample_aot_builder(backend, wires=wires, shots=shots)
            observed_val = f_aot(0.0)
            observed_shape = jnp.shape(observed_val)
            assert expected_shape == observed_shape

    def test_shots_in_callsite_in_sample(self, backend):
        """Test shots in callsite in sample."""

        max_shots = 500
        max_wires = 9
        for x in range(1, 5):
            shots = random.randint(1, max_shots)
            wires = random.randint(1, max_wires)
            expected_shape = (shots, wires)
            f_aot = fsample_aot_builder(backend, wires=wires)
            observed_val = f_aot(0.0, shots=shots)  # pylint: disable=unexpected-keyword-arg
            observed_shape = jnp.shape(observed_val)
            # We are failing this test because of the type system.
            # If shots is specified AOT, we would need to recompile
            # since the types are set at compile time. I.e., before the call.
            # assert expected_shape == observed_shape


class TestPromotionRules:
    """Class to test different promotion rules."""

    def test_against_none_target(self):
        """Test type check result when the target is None."""

        retval = typecheck_signatures(None, [1])
        assert TypeCompatibility.NEEDS_COMPILATION == retval

    def test_incompatible_compiled_vs_runtime_different_lengths(self):
        """Test incompatible compiled vs runtime."""

        retval = typecheck_signatures([], [1])
        assert TypeCompatibility.NEEDS_COMPILATION == retval

    def test_incompatible_compiled_vs_runtime_different_types(self):
        """Test incompatible compiled vs runtime with different types."""

        retval = typecheck_signatures(jnp.array([1]), jnp.array([complex(1, 2)]))
        assert TypeCompatibility.NEEDS_COMPILATION == retval

    def test_incompatible_compiled_vs_runtime_different_shapes(self):
        """Test incompatible compiled vs runtime with different shapes."""

        retval = typecheck_signatures(jnp.array([1, 2]), jnp.array([1]))
        assert TypeCompatibility.NEEDS_COMPILATION == retval

    def test_can_skip_promotion(self):
        """Test skipping promotion"""

        retval = typecheck_signatures(jnp.array([1]), jnp.array([1]))
        assert TypeCompatibility.CAN_SKIP_PROMOTION == retval

    def test_needs_promotion(self):
        """Test promotion"""

        retval = typecheck_signatures(jnp.array([1.0]), jnp.array([1]))
        assert TypeCompatibility.NEEDS_PROMOTION == retval


class TestPromotionRulesDictionary:
    """Class for test promotion rules for dictionaries."""

    def test_trivial_no_promotion(self):
        """Test trivial for the same dictionary as input."""
        one = jnp.array(1.0)
        retval = typecheck_signatures({"key1": one}, {"key1": one})
        assert TypeCompatibility.CAN_SKIP_PROMOTION == retval

    def test_trivial_no_promotion_different_values(self):
        """Test trivial for the same dictionary with different values."""
        one = jnp.array(1.0)
        two = jnp.array(2.0)
        retval = typecheck_signatures({"key1": one}, {"key1": two})
        assert TypeCompatibility.CAN_SKIP_PROMOTION == retval

    def test_trivial_promotion_different_values(self):
        """Test promotion where keys have different values."""
        one = jnp.array(1.0)
        one_int = jnp.array(1)
        retval = typecheck_signatures({"key1": one}, {"key1": one_int})
        assert TypeCompatibility.NEEDS_PROMOTION == retval

    def test_recompilation_superset_keys(self):
        """Recompile if the structure is different superset case."""
        one = jnp.array(1.0)
        retval = typecheck_signatures({"key1": one}, {"key2": one, "key1": one})
        assert TypeCompatibility.NEEDS_COMPILATION == retval

    def test_recompilation_subset_keys(self):
        """Recompile if the structure is different subset case."""
        one = jnp.array(1.0)
        retval = typecheck_signatures({"key2": one, "key1": one}, {"key1": one})
        assert TypeCompatibility.NEEDS_COMPILATION == retval


class TestSignatureErrors:
    def test_incompatible_argument(self):
        """Test incompatible argument."""

        string = "hello world"
        with pytest.raises(TypeError, match="<class 'str'> is not a valid JAX type"):
            get_abstract_signature([string])

    def test_incompatible_type_reachable_from_user_code(self):
        """Raise error message for incompatible types"""

        @qjit
        def f(x: str):
            return

        with pytest.raises(TypeError, match="<class 'str'> is not a valid JAX type"):
            f("abc")

    def test_incompatible_abstractify(self):
        """Check error message.

        Note: It is unclear if there's a path that will reach this condition.
        This is because the incompatible argument above would reach it.
        """

        with pytest.raises(TypeError, match="<class 'str'> is not a valid JAX type"):
            _scalar_abstractify(str)


class TestClassicalCompilation:
    @pytest.mark.parametrize("a,b", [(1, 1)])
    def test_pure_classical_function(self, a, b):
        """Test pure classical function."""

        def addi(x: int, y: int):
            return x + y

        addc = qjit(addi)
        assert addc.mlir
        assert addi(a, b) == addc(a, b)


class TestArraysInHamiltonian:
    """Test arrays in ``qp.Hamiltonian``."""

    @pytest.mark.parametrize(
        "coeffs",
        [
            (np.array([0.4, 0.7])),
            (jnp.array([0.4, 0.7])),
        ],
    )
    def test_array_repr_from_context1(self, coeffs, backend):
        """Test array representation from context in Hamiltonian."""

        @qjit(target="mlir")
        @qp.qnode(qp.device(backend, wires=6))
        def f():
            qp.Hadamard(wires=0)
            qp.CNOT(wires=[0, 1])
            obs = [qp.PauliX(0) @ qp.PauliZ(1), qp.Hadamard(0)]
            return qp.expval(qp.Hamiltonian(coeffs, obs))

        f()

    @pytest.mark.parametrize(
        "coeffs",
        [
            (np.array([0.4, 0.7])),
            (jnp.array([0.4, 0.7])),
        ],
    )
    def test_array_repr_as_parameter(self, coeffs, backend):
        """Test array representation as parameter in Hamiltonian."""

        @qjit(target="mlir")
        @qp.qnode(qp.device(backend, wires=6))
        def f(coeffs):
            qp.Hadamard(wires=0)
            qp.CNOT(wires=[0, 1])
            obs = [qp.PauliX(0) @ qp.PauliZ(1), qp.Hadamard(0)]
            return qp.expval(qp.Hamiltonian(coeffs, obs))

        f(coeffs)

    @pytest.mark.parametrize(
        "array",
        [
            (np.array),
            (jnp.array),
        ],
    )
    def test_array_repr_built_in(self, array, backend):
        """Test array representation built-in in Hamiltonian."""

        @qjit(target="mlir")
        @qp.qnode(qp.device(backend, wires=6))
        def f():
            qp.Hadamard(wires=0)
            qp.CNOT(wires=[0, 1])
            obs = [qp.PauliX(0) @ qp.PauliZ(1), qp.Hadamard(0)]
            coeffs = array([0.4, 0.7])
            return qp.expval(qp.Hamiltonian(coeffs, obs))

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
        "array",
        [
            (np.array),
            (jnp.array),
        ],
    )
    def test_array_repr_from_context2(self, matrix, array, backend):
        """Test array representation from context in Hermitian."""

        @qjit(target="mlir")
        @qp.qnode(qp.device(backend, wires=6))
        def f(x: float):
            qp.RX(x, wires=0)
            hermitian = qp.Hermitian(array(matrix), wires=[0, 1])
            return qp.expval(hermitian)

        assert f.mlir

    @pytest.mark.parametrize(
        "array",
        [
            (np.array),
            (jnp.array),
        ],
    )
    def test_array_repr_as_parameter(self, matrix, array, backend):
        """Test array representation as parameter in Hermitian."""

        @qjit(target="mlir")
        @qp.qnode(qp.device(backend, wires=2))
        def f(matrix):
            qp.RX(jnp.pi, wires=0)
            hermitian = qp.Hermitian(matrix, wires=[0, 1])
            return qp.expval(hermitian)

        f(array(matrix))

    @pytest.mark.parametrize(
        "array",
        [
            (np.array),
            (jnp.array),
        ],
    )
    def test_array_repr_built_in(self, array, backend):
        """Test array representation built-in in Hermitian."""

        @qjit(target="mlir")
        @qp.qnode(qp.device(backend, wires=2))
        def f(x: float):
            qp.RX(x, wires=0)
            matrix = array(
                [
                    [complex(2.0, 0.0), complex(1.0, 1.0), complex(9.0, 2.0), complex(0.0, 0.0)],
                    [complex(1.0, -1.0), complex(5.0, 0.0), complex(4.0, 6.0), complex(3.0, -2.0)],
                    [complex(9.0, -2.0), complex(4.0, -6.0), complex(10.0, 0.0), complex(1.0, 7.0)],
                    [complex(0.0, 0.0), complex(3.0, 2.0), complex(1.0, -7.0), complex(4.0, 0.0)],
                ]
            )
            hermitian = qp.Hermitian(matrix, wires=[0, 1])
            return qp.expval(hermitian)

        assert f.mlir


class TestTracingQJITAnnotatedFunctions:
    def test_purely_classical_context(self):
        """Test purely classical context."""

        @qjit
        def f():
            return 1

        assert f.mlir

        @qjit
        def g():
            return f() + 1

        assert g.mlir
        assert g() == 2

    def test_quantum_context(self, backend):
        """Test quantum context."""

        @qp.qnode(qp.device(backend, wires=1))
        def f(x: float):
            qp.RX(x, wires=0)
            return qp.state()

        @qjit
        def g1(x: float):
            return f(x)

        assert g1.mlir

        @qjit
        def g2(x: float):
            return g1(x)

        assert g2.mlir

    @pytest.mark.parametrize("phi", [(0.0), (1.0), (2.0)])
    def test_gradient_of_qjit_equivalence(self, phi, backend):
        """Test gradient of qjit equivalence."""

        # Issue 376
        @qjit
        @qp.qnode(device=qp.device(backend, wires=1))
        def circuit(phi):
            qp.RX(phi, wires=0)
            return qp.expval(qp.PauliZ(0))

        @qjit
        def workflow(phi):
            g = grad(circuit)
            return g(phi)

        assert np.allclose(qjit(grad(qjit(circuit)))(phi), qjit(grad(circuit))(phi))
        assert np.allclose(qjit(grad(circuit))(phi), workflow(phi))
        assert np.allclose(workflow(phi), qjit(grad(circuit))(phi))

    @pytest.mark.parametrize("phi", [(0.0), (1.0), (2.0)])
    def test_gradient_of_qjit_correctness(self, phi, backend):
        """Test gradient of qjit correctness."""

        @qp.qnode(device=qp.device(backend, wires=1))
        def circuit(phi):
            qp.RX(phi, wires=0)
            return qp.expval(qp.PauliZ(0))

        @qjit
        def workflow1(phi):
            g = grad(circuit)
            return g(phi)

        @qjit
        def workflow2(phi):
            g = grad(qjit(circuit))
            return g(phi)

        assert np.allclose(workflow1(phi), workflow2(phi))

    def test_gradient_of_qjit_names(self, backend):
        """Test gradient of qjit names."""

        @qp.qnode(device=qp.device(backend, wires=1))
        def circuit(phi):
            qp.RX(phi, wires=0)
            return qp.expval(qp.PauliZ(0))

        @qjit
        def workflow(phi: float):
            g = grad(circuit)
            return g(phi)

        mlir_v1 = workflow.mlir

        @qjit
        def workflow(phi: float):  # pylint: disable=function-redefined
            g = grad(qjit(circuit))
            return g(phi)

        mlir_v2 = workflow.mlir

        assert mlir_v1 == mlir_v2


class TestDefaultAvailableIR:

    def test_AbstractArray_AbstractWires_AOT(self):
        """Test that AbstractArray and AbstractWires can be used to specify the input
        shapes for AOT compilation."""

        @qp.qjit(capture=True, collect_decomp_rules=False)
        @qp.qnode(qp.device("lightning.qubit", wires=4))
        def c(x: qp.typing.AbstractArray((3,), float), wires: qp.typing.Wire[4]):
            @qp.for_loop(x.shape[0])
            def loop(i):
                qp.RX(x[i], wires[i])

            @qp.for_loop(wires.shape[0])
            def loop2(i):
                qp.X(i)

            loop()
            loop2()
            return qp.expval(qp.Z(0))

        assert c.mlir

    def test_mlir(self):
        """Test mlir."""

        @qjit  # Note that we are using the default qjit
        def f():
            return 1

        assert f.mlir

    def test_llvmir(self, backend):
        """Test llvmir."""

        @qp.qnode(qp.device(backend, wires=1))
        def f(x: float):
            qp.RX(x, wires=0)
            return qp.state()

        @qjit  # Note that we are using the default qjit
        def g(x: float):
            return f(x)

        assert g.llvmir
        assert "__catalyst__qis" in g.llvmir

    def test_mlir_opt(self, backend):
        """Test mlir opt."""

        @qp.qnode(qp.device(backend, wires=1))
        def f(x: float):
            qp.RX(x, wires=0)
            return qp.state()

        @qjit  # Note that we are using the default qjit
        def g(x: float):
            return f(x)

        assert g.mlir_opt
        assert "__catalyst__qis" in g.mlir_opt

    @pytest.mark.xdsl
    def test_mlir_opt_using_xdsl_passes(self, backend):
        """Test mlir opt using xDSL passes."""
        # pylint: disable-next=import-outside-toplevel
        from catalyst.python_interface.transforms import iterative_cancel_inverses_pass

        @qjit(capture=True, collect_decomp_rules=False)
        @iterative_cancel_inverses_pass
        @qp.qnode(qp.device(backend, wires=1))
        def f():
            qp.Hadamard(wires=0)
            qp.Hadamard(wires=0)
            return qp.state()

        mlir_opt = f.mlir_opt
        assert mlir_opt
        assert not "__catalyst__qis__Hadamard" in mlir_opt

    def test_mlir_opt_with_keep_intermediate(self, backend):
        """Test that keep_intermediate with target='mlir' writes the initial IR and
        pipeline output files into the workspace when mlir_opt is accessed."""

        @qjit(target="mlir", keep_intermediate=True)
        @qp.qnode(qp.device(backend, wires=1))
        def circuit():
            qp.H(0)
            return qp.state()

        # Create tmp workspaces for intermediates to avoid CI race conditions
        circuit.use_cwd_for_workspace = False

        _ = circuit.mlir_opt

        workspace_files = os.listdir(str(circuit.workspace))
        assert any(f.startswith("0_") and f.endswith(".mlir") for f in workspace_files)
        assert any("After" in f and f.endswith(".mlir") for f in workspace_files)
        circuit.workspace.cleanup()

    def test_jaxpr_target(self, backend):
        """Test no mlir is generated for jaxpr target."""

        @qjit(target="jaxpr")
        @qp.qnode(qp.device(backend, wires=1))
        def f(x: float):
            qp.RX(x, wires=0)
            return qp.state()

        assert f.mlir_opt is None
        assert f.mlir is None


class TestAvoidVerification:
    def test_no_verification(self, capfd, backend):
        """Test no verification."""

        dev1 = qp.device(backend, wires=1)

        @qp.qnode(device=dev1)
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


class TestTwoQJITsOneName:
    """Test two QJITs with the same name."""

    # pylint: disable=disallowed-name
    # pylint: disable=function-redefined

    def test_two_qjit(self):
        """Test two qjits with the same name"""

        def foo():
            """Returns 1"""
            return 1

        foo_1 = qjit(foo)

        def foo():
            """Returns 2"""
            return 2

        foo_2 = qjit(foo)

        assert foo_1() == 1
        assert foo_2() == 2

    def test_two_qjit_keep_intermediate(self):
        """Test two qjits with the same name but also keep intermediate=True"""

        def foo():
            """Returns 1"""
            return 1

        foo_1 = qjit(keep_intermediate=True)(foo)

        def foo():
            """Returns 2"""
            return 2

        foo_2 = qjit(keep_intermediate=True)(foo)

        assert foo_1() == 1
        assert foo_2() == 2
        foo_1.workspace.cleanup()
        foo_2.workspace.cleanup()


class TestQJITUsagePatterns:
    """Test usage patterns of catalyst.qjit."""

    def test_usage_patterns(self):
        """Test two usage patterns of catalyst.qjit."""

        def fn(x, y):
            return x * y

        res_pattern_fn_as_argument = qjit(fn, autograph=False)(5, 6)
        res_pattern_partial = qjit(autograph=False)(fn)(5, 6)

        expected = 30
        assert res_pattern_fn_as_argument == expected
        assert res_pattern_partial == expected


class TestGradPartial:
    """Test functools.partial with catalyst.qjit and catalyst.qjit."""

    def test_partial_func_grad(self):
        """Test if partial triggers function name error with gjit and grad."""

        def fn(x, y):
            return x * y

        partial_fn = partial(fn, y=1)

        @qjit
        def grad_partial_fn(x):
            return grad(partial_fn)(x)

        expected = jax.grad(partial_fn)(0.3)

        assert np.allclose(grad_partial_fn(0.3), expected)


class TestParamsAnnotations:
    """Test param annotations"""

    def test_params_invalid_annotation(self):
        def foo(hello: "BAD ANNOTATION"): ...

        assert not params_are_annotated(foo)


class TestErrorNestedQNode:
    """Test error is raised with nested qnodes"""

    def test_nested_qnode(self):
        """Test autograph on a QNode raises error."""

        dev = qp.device("lightning.qubit", wires=1)

        @qp.qnode(dev)
        def inner():
            return qp.state()

        @qp.qnode(dev)
        def outer():
            inner()
            return qp.state()

        @qjit
        def fn():
            return outer()

        with pytest.raises(CompileError):

            fn()


class TestAOTFailures:
    """Tests that an AOT failure does not cause a full error."""

    def test_capture_failure(self):
        """Test a failure capturing the jaxpr."""

        @qp.qjit(capture=True, collect_decomp_rules=False)
        @qp.qnode(qp.device("null.qubit", wires=1))
        def c():
            raise ValueError

        assert c.jaxpr is None

    def test_ir_generation_failure(self):
        """Test a failure lowering to mlir."""

        dummy_p = jax.extend.core.Primitive("dummy")
        dummy_p.multiple_results = True

        @dummy_p.def_abstract_eval
        def f():
            return []

        @qp.qjit(capture=True, collect_decomp_rules=False)
        def c():
            dummy_p.bind()
            return 2

        assert c.jaxpr.eqns[0].primitive == dummy_p

    def test_llvm_generation_failure(self):
        """Test a failure lowering to llvmir."""

        @qp.qjit(capture=True, collect_decomp_rules=False)
        @qp.qnode(qp.device("null.qubit", wires=1))
        def c():
            qp.RX(qp.capture.symbolic_array((), float), 0)
            return qp.probs(wires=0)

        assert c.mlir
        assert "RX" in c.mlir


class TestAOTCompileWithArguments:
    """Test ahead-of-time compilation for given arguments."""

    def test_compile_up_to_target(self):
        """Test that compilation for the given arguments stops at the target, and that a later
        call still compiles and runs the function."""

        @qjit(target="mlir")
        def f(x, y=1.0):
            return x + y

        f.aot_compile(0.5, y=2.0)

        assert f.mlir_module is not None
        assert f.compiled_function is None
        assert not f.fn_cache.cache
        assert f(0.5, y=2.0) == 2.5

    def test_call_reuses_compiled_function(self):
        """Test that a call with arguments matching those of the ahead-of-time compilation reuses
        the compiled function."""

        @qjit
        def f(x):
            return x * 2

        f.aot_compile(jax.ShapeDtypeStruct((3,), jnp.float64))
        compiled_function = f.compiled_function

        assert compiled_function is not None
        assert jnp.allclose(f(jnp.array([1.0, 2.0, 3.0])), jnp.array([2.0, 4.0, 6.0]))
        assert f.compiled_function is compiled_function

    def test_call_after_aot_compile_with_other_arguments(self):
        """Test that a function compiled for some arguments can still be called with them after
        compiling it ahead of time, up to MLIR, for arguments with another structure."""

        @qjit(target="mlir")
        def f(x):
            return x

        assert f(1.0) == 1.0
        f.aot_compile((1.0, 2.0))
        assert f(1.0) == 1.0


class TestAOTCompileErrors:
    """Test the handling of errors during ahead-of-time compilation."""

    def test_errors_are_raised(self):
        """Test that the errors of an ahead-of-time compilation requested explicitly are raised."""

        @qjit
        def f(x):
            raise ValueError("Cannot compile")

        with pytest.raises(ValueError, match="Cannot compile"):
            f.aot_compile(0.5)

    def test_errors_on_creation_are_logged(self, caplog):
        """Test that the errors of the ahead-of-time compilation from the type hints, when the QJIT
        object is created, are logged instead of raised, and raised when the function is called."""

        @qjit
        def f(x: float):
            raise ValueError("Cannot compile")

        assert "Cannot compile" in caplog.text
        with pytest.raises(ValueError, match="Cannot compile"):
            f(0.5)


@pytest.mark.parametrize("enable_aot", [True, False])
def test_toggle_aot_compilation(enable_aot):
    @qp.qjit(capture=True, enable_aot_compilation=enable_aot)
    @qp.qnode(qp.device("null.qubit", wires=2))
    def circ():
        qp.RX(0.5, wires=0)
        return qp.expval(qp.PauliZ(0))

    if enable_aot:
        assert circ.mlir is not None
    else:
        assert circ.mlir is None


if __name__ == "__main__":
    pytest.main(["-x", __file__])
