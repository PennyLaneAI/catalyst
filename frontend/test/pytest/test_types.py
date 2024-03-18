import numpy as np
import pytest
from jax._src.lib.mlir import ir

from catalyst.jax_extras import ShapedArray
from catalyst.utils.types import convert_numpy_dtype_to_mlir

ctx = ir.Context()
f64 = ir.F64Type.get(ctx)
f32 = ir.F32Type.get(ctx)
complex128 = ir.ComplexType.get(f64)
complex64 = ir.ComplexType.get(f32)
i64 = ir.IntegerType.get_signless(64, ctx)
i32 = ir.IntegerType.get_signless(32, ctx)
i16 = ir.IntegerType.get_signless(16, ctx)
i8 = ir.IntegerType.get_signless(8, ctx)
i1 = ir.IntegerType.get_signless(1, ctx)


@pytest.mark.parametrize(
    "inp,exp",
    [
        (np.dtype(np.complex128), complex128),
        (np.dtype(np.complex64), complex64),
        (np.dtype(np.float64), f64),
        (np.dtype(np.float32), f32),
        (np.dtype(np.bool_), i1),
        (np.dtype(np.int8), i8),
        (np.dtype(np.int16), i16),
        (np.dtype(np.int32), i32),
        (np.dtype(np.int64), i64),
    ],
)
def test_convert_numpy_dtype_to_mlir(inp, exp):
    with ctx:
        assert convert_numpy_dtype_to_mlir(inp) == exp


def test_convert_numpy_dtype_to_mlir_error():
    with pytest.raises(ValueError, match="Requested type conversion not available."):
        convert_numpy_dtype_to_mlir(np.dtype(object))
