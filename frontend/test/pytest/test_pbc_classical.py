# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Native execution of classical decisions used in logical PBC depth."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import jax
import jax.numpy as jnp
import numpy as np
import pennylane as qp
import pytest
from jax.interpreters import mlir
from jaxlib.mlir import ir

from catalyst.utils import pbc_classical
from catalyst.utils.pbc_classical import (
    _compile_module,
    _evaluate_cached,
    evaluate_pbc_classical,
)


def decode_real(text):
    """Read a noncomplex result using the native MLIR buffer interface."""
    with mlir.make_ir_context():
        return np.asarray(ir.DenseElementsAttr(ir.Attribute.parse(text))).copy()


def test_rounding_and_boolean_results():
    """Round-to-even and packed booleans preserve the actual branch decisions."""
    module = """module {
      func.func @main(%arg: tensor<9xf64>) -> (tensor<9xi64>, tensor<9xi1>) {
        %rounded = stablehlo.round_nearest_even %arg : tensor<9xf64>
        %integer = stablehlo.convert %rounded : (tensor<9xf64>) -> tensor<9xi64>
        %zero = stablehlo.constant dense<0> : tensor<9xi64>
        %positive = stablehlo.compare GT, %integer, %zero, SIGNED :
            (tensor<9xi64>, tensor<9xi64>) -> tensor<9xi1>
        return %integer, %positive : tensor<9xi64>, tensor<9xi1>
      }
    }"""
    outputs = evaluate_pbc_classical(
        module, ["dense<[-2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5]> : tensor<9xf64>"]
    )
    expected = np.array([-2, -2, 0, 0, 2, 2, 4, 4, 6], dtype=np.int64)
    np.testing.assert_array_equal(decode_real(outputs[0]), expected)
    np.testing.assert_array_equal(decode_real(outputs[1]), expected > 0)


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("shape", [(), (2,), (2, 3), (0, 3), (100,)])
def test_complex_identity(dtype, shape):
    """Complex scalars, splats, empty tensors and dense arrays survive transport."""
    if shape in [(), (2, 3)]:
        array = np.full(shape, complex(-0.0, -2.0), dtype=dtype)
    else:
        array = np.arange(np.prod(shape), dtype=np.float64).astype(dtype).reshape(shape)
        array *= 1 + 2j
    with mlir.make_ir_context(), ir.Location.unknown():
        real_type = ir.F32Type.get() if dtype == np.complex64 else ir.F64Type.get()
        tensor_type = ir.RankedTensorType.get(shape, ir.ComplexType.get(real_type))
        type_text = str(tensor_type)
        attribute = str(ir.DenseElementsAttr.get(array, type=tensor_type))
    module = f"""module {{
      func.func @main(%arg: {type_text}) -> {type_text} {{
        return %arg : {type_text}
      }}
    }}"""
    assert evaluate_pbc_classical(module, [attribute]) == [attribute]


def test_complex_hexadecimal_attribute():
    """Opaque hexadecimal input preserves the real/imaginary component bits."""
    array = np.array([complex(-0.0, 2.0), 3.0 - 4.0j], dtype=np.complex128)
    attribute = f'dense<"0x{array.tobytes().hex()}"> : tensor<2xcomplex<f64>>'
    module = """module {
      func.func @main(%arg: tensor<2xcomplex<f64>>) -> tensor<2xcomplex<f64>> {
        return %arg : tensor<2xcomplex<f64>>
      }
    }"""
    with mlir.make_ir_context(), ir.Location.unknown():
        tensor_type = ir.RankedTensorType.get([2], ir.ComplexType.get(ir.F64Type.get()))
        expected = str(ir.DenseElementsAttr.get(array, type=tensor_type))
    assert evaluate_pbc_classical(module, [attribute]) == [expected]


def test_complex_splat_preserves_nan_bits():
    """Expanding a complex splat must not quiet a signalling NaN component."""
    array = np.array([0x7F800001, 0x80000000], dtype=np.uint32).view(np.complex64)
    array = np.broadcast_to(array, (3,)).copy()
    with mlir.make_ir_context(), ir.Location.unknown():
        tensor_type = ir.RankedTensorType.get([3], ir.ComplexType.get(ir.F32Type.get()))
        attribute = str(ir.DenseElementsAttr.get(array, type=tensor_type))
    module = """module {
      func.func @main(%arg: tensor<3xcomplex<f32>>) -> tensor<3xcomplex<f32>> {
        return %arg : tensor<3xcomplex<f32>>
      }
    }"""
    assert evaluate_pbc_classical(module, [attribute]) == [attribute]


def test_complex_magnitude_and_structural_operations():
    """Complex arithmetic followed by f64 reshaping/slicing selects real values."""
    module = """module {
      func.func @main(%arg: tensor<2x2xcomplex<f64>>) -> tensor<2xf64> {
        %abs = stablehlo.abs %arg : (tensor<2x2xcomplex<f64>>) -> tensor<2x2xf64>
        %flat = stablehlo.reshape %abs : (tensor<2x2xf64>) -> tensor<4xf64>
        %slice = stablehlo.slice %flat [1:4:2] : (tensor<4xf64>) -> tensor<2xf64>
        return %slice : tensor<2xf64>
      }
    }"""
    outputs = evaluate_pbc_classical(
        module,
        [
            "dense<[[(3.0, 4.0), (-5.0, 12.0)], [(8.0, 15.0), (7.0, 24.0)]]>"
            " : tensor<2x2xcomplex<f64>>"
        ],
    )
    np.testing.assert_array_equal(decode_real(outputs[0]), [13.0, 25.0])


def test_lapack_determinant():
    """The same CPU lowering evaluates LAPACK calls and their control flow."""
    array = np.array(
        [[1.0, 2.0, 3.0, 4.0], [2.0, -3.0, 5.0, 1.0], [3.0, 1.0, 2.0, -4.0], [1.0, 3.0, -2.0, 5.0]]
    )
    with jax.experimental.enable_x64():
        function = jax.jit(jnp.linalg.det)
        module = str(function.lower(array).compiler_ir(dialect="stablehlo"))
        expected = np.asarray(function(array))
    assert "lapack_dgetrf_ffi" in module
    with mlir.make_ir_context(), ir.Location.unknown():
        attribute = str(ir.DenseElementsAttr.get(array))
    outputs = evaluate_pbc_classical(module, [attribute])
    actual = decode_real(outputs[0])
    assert actual.dtype == expected.dtype
    assert actual.tobytes() == expected.tobytes()


def test_dynamic_while():
    """Known runtime inputs execute a classical while loop before scheduling."""

    def accumulate(value, count):
        return jax.lax.fori_loop(0, count, lambda _, total: total + value, 0.0)

    with jax.experimental.enable_x64():
        module = str(jax.jit(accumulate).lower(0.25, 4).compiler_ir(dialect="stablehlo"))
    assert "stablehlo.while" in module
    outputs = evaluate_pbc_classical(
        module, ["dense<0.25> : tensor<f64>", "dense<4> : tensor<i64>"]
    )
    np.testing.assert_array_equal(decode_real(outputs[0]), 1.0)


def test_compilation_cache_keeps_arguments_dynamic():
    """A reused executable must produce different results for different inputs."""
    module = """module {
      func.func @main(%arg: tensor<ui64>) -> tensor<ui64> {
        %one = stablehlo.constant dense<1> : tensor<ui64>
        %sum = stablehlo.add %arg, %one : tensor<ui64>
        return %sum : tensor<ui64>
      }
    }"""
    _compile_module.cache_clear()
    _evaluate_cached.cache_clear()
    first = evaluate_pbc_classical(module, ["dense<18446744073709551614> : tensor<ui64>"])
    second = evaluate_pbc_classical(module, ["dense<0> : tensor<ui64>"])
    np.testing.assert_array_equal(decode_real(first[0]), np.uint64(18446744073709551615))
    np.testing.assert_array_equal(decode_real(second[0]), np.uint64(1))
    assert _compile_module.cache_info().misses == 1
    assert _compile_module.cache_info().hits >= 1
    assert _compile_module.cache_info().maxsize == 128


def test_result_cache_preserves_inputs_and_isolates_mutation(mocker):
    """Cached values retain signed zero/NaN bits and cannot be changed by callers."""
    module = """module @cached_identity {
      func.func @main(%arg: tensor<f64>) -> tensor<f64> {
        return %arg : tensor<f64>
      }
    }"""
    positive = "dense<0.000000e+00> : tensor<f64>"
    negative = "dense<-0.000000e+00> : tensor<f64>"
    nan = "dense<0x7FF8000000000001> : tensor<f64>"
    arguments = [positive]
    _evaluate_cached.cache_clear()
    execute = mocker.spy(pbc_classical, "_execute_module")
    first = evaluate_pbc_classical(module, arguments)
    first[0] = "changed by caller"
    assert evaluate_pbc_classical(module, arguments) == [positive]
    arguments[0] = negative
    assert evaluate_pbc_classical(module, arguments) == [negative]
    arguments[0] = nan
    assert evaluate_pbc_classical(module, arguments) == [nan]
    assert execute.call_count == 3


def test_large_result_is_not_cached(mocker):
    """A small input cannot fill the result cache with large expanded tensors."""
    module = """module {
      func.func @main(%arg: tensor<i64>) -> tensor<10000xi64> {
        %result = stablehlo.broadcast_in_dim %arg, dims = [] :
            (tensor<i64>) -> tensor<10000xi64>
        return %result : tensor<10000xi64>
      }
    }"""
    _evaluate_cached.cache_clear()
    execute = mocker.spy(pbc_classical, "_execute_module")
    result = evaluate_pbc_classical(module, ["dense<7> : tensor<i64>"])
    np.testing.assert_array_equal(decode_real(result[0]), np.full(10000, 7, dtype=np.int64))
    assert evaluate_pbc_classical(module, ["dense<7> : tensor<i64>"]) == result
    assert execute.call_count == 2


def test_parallel_evaluations_with_context_recycling():
    """Thread-local contexts and their recycling preserve each call's inputs."""
    module = """module {
      func.func @main(%arg: tensor<i64>) -> tensor<i64> {
        %one = stablehlo.constant dense<1> : tensor<i64>
        %sum = stablehlo.add %arg, %one : tensor<i64>
        return %sum : tensor<i64>
      }
    }"""
    evaluate_pbc_classical(module, ["dense<0> : tensor<i64>"])
    barrier = Barrier(4)

    def evaluate_sequence(offset):
        barrier.wait()
        for value in range(offset, offset + 300):
            result = evaluate_pbc_classical(module, [f"dense<{value}> : tensor<i64>"])
            assert result == [f"dense<{value + 1}> : tensor<i64>"]

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(evaluate_sequence, (0, 1000, 2000, 3000)))


@pytest.mark.parametrize(
    "arguments, message",
    [
        ([], "argument count"),
        (["dense<1> : tensor<i32>"], "argument type"),
    ],
)
def test_invalid_signature(arguments, message):
    """A failed evaluation is reported instead of silently coercing input values."""
    module = """module {
      func.func @main(%arg: tensor<i64>) -> tensor<i64> {
        return %arg : tensor<i64>
      }
    }"""
    with pytest.raises(ValueError, match=message):
        evaluate_pbc_classical(module, arguments)


def test_unsupported_buffer_type():
    """Unsupported buffer element types must fail before native conversion."""
    module = """module {
      func.func @main(%arg: tensor<bf16>) -> tensor<bf16> {
        return %arg : tensor<bf16>
      }
    }"""
    with pytest.raises(ValueError, match="Unsupported classical tensor element type"):
        evaluate_pbc_classical(module, ["dense<1.0> : tensor<bf16>"])


@pytest.mark.parametrize("sign, expected_depth", [(1.0, 2), (-1.0, 1)])
def test_captured_determinant_selects_exact_depth(sign, expected_depth, monkeypatch):
    """Analysis uses a captured LAPACK result to choose the actual qubit overlap."""
    # Notebook kernels can use a virtual environment without shell activation.
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    matrix = np.diag([sign, 1.0, 1.0, 1.0])

    @qp.qjit(capture=True, target="mlir")
    @qp.transforms.to_ppr
    @qp.decompose(gate_set={qp.X, qp.Z})
    @qp.qnode(qp.device("null.qubit", wires=3))
    def circuit():
        def positive():
            qp.X(0)

        def negative():
            qp.X(1)

        qp.cond(jnp.linalg.det(matrix) > 0, positive, negative)()
        qp.Z(0)
        qp.X(2)

    resources = qp.analyze(circuit, level=2)().resources
    assert resources.qubit_disjoint_depth == expected_depth
    assert resources.extra["qubit_disjoint_depth_status"] == "exact"
    assert resources.num_wires == 3
    assert resources.quantum_operations["PPR-pi/2-w1"] == 3


def test_long_classical_chain_rebinds_on_loop_iteration(monkeypatch):
    """Computed call results survive long chains and are refreshed in each iteration."""
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)

    @jax.jit
    def increment(value):
        return value + 1

    @qp.qjit(capture=True, target="mlir")
    @qp.transforms.to_ppr
    @qp.decompose(gate_set={qp.X, qp.Z})
    @qp.qnode(qp.device("null.qubit", wires=3))
    def circuit():
        def first_wire():
            qp.X(0)

        def second_wire():
            qp.X(1)

        def loop(index):
            value = index
            for _ in range(300):
                value = increment(value)
            qp.cond(value == 300, first_wire, second_wire)()

        qp.for_loop(0, 2)(loop)()
        qp.Z(0)
        qp.X(2)

    resources = qp.analyze(circuit, level=2)().resources
    assert resources.qubit_disjoint_depth == 2
    assert resources.extra["qubit_disjoint_depth_status"] == "exact"
    assert resources.num_wires == 3
    assert resources.quantum_operations["PPR-pi/2-w1"] == 4
