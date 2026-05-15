import pathlib

import jax
import jax.numpy as jnp
import numpy as np

from catalyst.debug import compile_mlir

from helpers import SHARED_LIB_EXT, build_shared_lib, patch_artifact_path

HERE = pathlib.Path(__file__).resolve().parent
LIB_SRC = HERE / "libxor_ref.c"
LIB_PATH = HERE / f"libxor_ref{SHARED_LIB_EXT}"
MLIR_FILE = HERE / "mock.mlir"


lib_path = build_shared_lib(LIB_SRC, LIB_PATH)
ir_text = patch_artifact_path(MLIR_FILE.read_text(), lib_path)

fn = compile_mlir(
    ir_text,
    func_name="jit_circuit",
    result_types=[
        jax.ShapeDtypeStruct((), np.float64),    # expval
        jax.ShapeDtypeStruct((4,), np.float64),  # probs
        jax.ShapeDtypeStruct((1,), np.int32),    # xor_reduce result
    ],
    keep_intermediate="changed",
    verbose=True,
)

theta = jnp.float64(0.5)
weights = jnp.array([[0.1, 0.2], [0.3, 0.4]], dtype=jnp.float64)
expval, probs, xor_result = fn(theta, weights)
print("expval:     ", expval)
print("probs:      ", probs)
print("xor_result: ", xor_result)
