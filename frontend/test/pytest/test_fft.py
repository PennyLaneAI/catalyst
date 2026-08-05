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

"""Test that jax.numpy.fft functions compile and produce correct results under qjit."""

import jax
import numpy as np
import pytest
from jax import numpy as jnp

from catalyst import qjit

# The direct DFT lowering with exactly reduced twiddle angles keeps f64 errors
# near the 1e-12 scale for these sizes so 1e-8 gives ample slack. f32 twiddles
# are accurate to f64 but the accumulation is f32 hence the looser tolerance.
TOL_F64 = {"atol": 1e-8, "rtol": 1e-8}
TOL_F32 = {"atol": 1e-3, "rtol": 1e-3}

LENGTHS = [1, 2, 3, 6, 8, 13, 16, 100]


def random_complex(rng, shape, dtype=np.complex128):
    """Uniform random complex array in the unit square."""
    return (rng.random(shape) + 1j * rng.random(shape)).astype(dtype)


class TestFFTC2C:
    """Complex-to-complex forward and inverse transforms."""

    @pytest.mark.parametrize("n", LENGTHS)
    def test_fft_1d(self, n):
        """Forward FFT matches NumPy and plain JAX for all length classes."""
        rng = np.random.default_rng(42)
        x = random_complex(rng, (n,))

        def f(x):
            return jnp.fft.fft(x)

        observed = qjit(f)(x)
        assert np.allclose(observed, np.fft.fft(x), **TOL_F64)
        assert np.allclose(observed, f(x), **TOL_F64)

    @pytest.mark.parametrize("n", [1, 6, 8, 13])
    def test_ifft_1d(self, n):
        """Inverse FFT carries the 1/n normalization."""
        rng = np.random.default_rng(7)
        x = random_complex(rng, (n,))

        def f(x):
            return jnp.fft.ifft(x)

        observed = qjit(f)(x)
        assert np.allclose(observed, np.fft.ifft(x), **TOL_F64)

    def test_fft_ifft_roundtrip(self):
        """ifft(fft(x)) recovers x."""
        rng = np.random.default_rng(3)
        x = random_complex(rng, (17,))

        def f(x):
            return jnp.fft.ifft(jnp.fft.fft(x))

        assert np.allclose(qjit(f)(x), x, **TOL_F64)

    def test_fft_complex64(self):
        """Single-precision transform stays in complex64 and is f32-accurate."""
        rng = np.random.default_rng(11)
        x = random_complex(rng, (16,), dtype=np.complex64)

        def f(x):
            return jnp.fft.fft(x)

        observed = qjit(f)(x)
        assert observed.dtype == np.complex64
        assert np.allclose(observed, np.fft.fft(x.astype(np.complex128)), **TOL_F32)

    def test_fft_real_input_promotes(self):
        """A real input to fft is promoted to complex, like NumPy."""
        rng = np.random.default_rng(5)
        x = rng.random(10)

        def f(x):
            return jnp.fft.fft(x)

        assert np.allclose(qjit(f)(x), np.fft.fft(x), **TOL_F64)

    def test_fft_batched(self):
        """Leading dimensions are batch dimensions."""
        rng = np.random.default_rng(13)
        x = random_complex(rng, (4, 5, 6))

        def f(x):
            return jnp.fft.fft(x, axis=-1)

        assert np.allclose(qjit(f)(x), np.fft.fft(x, axis=-1), **TOL_F64)

    def test_fft2(self):
        """2D transform lowered separably with one stage per axis."""
        rng = np.random.default_rng(17)
        x = random_complex(rng, (4, 6))

        def f(x):
            return jnp.fft.fft2(x)

        assert np.allclose(qjit(f)(x), np.fft.fft2(x), **TOL_F64)

    def test_ifft2_batched(self):
        """Batched 2D inverse transform with 1/(n1*n2) normalization."""
        rng = np.random.default_rng(19)
        x = random_complex(rng, (3, 4, 6))

        def f(x):
            return jnp.fft.ifft2(x, axes=(-2, -1))

        assert np.allclose(qjit(f)(x), np.fft.ifft2(x, axes=(-2, -1)), **TOL_F64)


class TestFFTReal:
    """Real-to-complex (rfft) and complex-to-real (irfft) transforms."""

    @pytest.mark.parametrize("n", LENGTHS)
    def test_rfft_1d(self, n):
        """rfft stores floor(n/2)+1 bins and matches NumPy."""
        rng = np.random.default_rng(23)
        x = rng.random(n)

        def f(x):
            return jnp.fft.rfft(x)

        observed = qjit(f)(x)
        assert observed.shape == (n // 2 + 1,)
        assert np.allclose(observed, np.fft.rfft(x), **TOL_F64)

    @pytest.mark.parametrize("n", [1, 2, 6, 8, 9, 13, 16])
    def test_irfft_1d_non_hermitian_input(self, n):
        """irfft on arbitrary complex input matches NumPy exactly. The
        imaginary parts of the DC and Nyquist bins are discarded and interior
        bins contribute via their conjugate pairs."""
        rng = np.random.default_rng(29)
        x = random_complex(rng, (n // 2 + 1,))

        def f(x):
            return jnp.fft.irfft(x, n=n)

        observed = qjit(f)(x)
        assert observed.shape == (n,)
        assert np.allclose(observed, np.fft.irfft(x, n=n), **TOL_F64)

    @pytest.mark.parametrize("n", [8, 9])
    def test_rfft_irfft_roundtrip(self, n):
        """irfft(rfft(x), n) == x for even and odd n."""
        rng = np.random.default_rng(31)
        x = rng.random(n)

        def f(x):
            return jnp.fft.irfft(jnp.fft.rfft(x), n=n)

        assert np.allclose(qjit(f)(x), x, **TOL_F64)

    def test_rfft_batched(self):
        """rfft with leading batch dimensions."""
        rng = np.random.default_rng(37)
        x = rng.random((3, 10))

        def f(x):
            return jnp.fft.rfft(x, axis=-1)

        assert np.allclose(qjit(f)(x), np.fft.rfft(x, axis=-1), **TOL_F64)

    def test_rfft2(self):
        """2D real transform with r2c along the last axis and c2c along the other."""
        rng = np.random.default_rng(41)
        x = rng.random((5, 6))

        def f(x):
            return jnp.fft.rfft2(x)

        assert np.allclose(qjit(f)(x), np.fft.rfft2(x), **TOL_F64)

    def test_irfft2(self):
        """2D complex to real inverse transform."""
        rng = np.random.default_rng(43)
        x = random_complex(rng, (5, 4))

        def f(x):
            return jnp.fft.irfft2(x, s=(5, 6))

        assert np.allclose(qjit(f)(x), np.fft.irfft2(x, s=(5, 6)), **TOL_F64)

    def test_rfft_float32(self):
        """Single-precision rfft."""
        rng = np.random.default_rng(47)
        x = rng.random(12).astype(np.float32)

        def f(x):
            return jnp.fft.rfft(x)

        observed = qjit(f)(x)
        assert observed.dtype == np.complex64
        assert np.allclose(observed, np.fft.rfft(x.astype(np.float64)), **TOL_F32)


class TestFFTIndirectCall:
    """FFT called through a plain function inside a qjit block."""

    def test_fft_indirect_call(self):
        """A wrapped fft on real f64 input compiles and matches NumPy."""

        def jax_fft(x):
            return jnp.fft.fft(x)

        @qjit
        def test(x):
            fft_result = jax_fft(x)
            return fft_result

        data = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=jnp.float64)
        assert np.allclose(test(data), np.fft.fft(np.asarray(data)), **TOL_F64)


class TestFFTGradient:
    """The emitted linalg loops are transparent to Enzyme and the DFT is a
    linear map so reverse mode differentiation through an FFT is exact."""

    def test_grad_through_fft(self):
        """d/dx sum(|fft(x)|^2) == 2*n*x by Parseval's theorem."""
        from catalyst import grad

        rng = np.random.default_rng(53)
        n = 8
        x = rng.random(n)

        def loss(x):
            return jnp.sum(jnp.abs(jnp.fft.fft(x)) ** 2)

        observed = qjit(grad(loss, argnums=0))(x)
        expected = jax.grad(loss)(x)
        assert np.allclose(observed, expected, **TOL_F64)
        # Parseval gives sum |X_k|^2 = n * sum x_j^2 so the gradient is 2*n*x.
        assert np.allclose(observed, 2 * n * x, **TOL_F64)


if __name__ == "__main__":
    pytest.main(["-x", __file__])
