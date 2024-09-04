# Copyright 2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Test that the jax linear algebra functions yield correct results when compiled with qml.qjit"""

import numpy as np
import pytest
from jax import numpy as jnp
from jax import scipy as jsp

from catalyst import qjit


class MatrixGenerator:
    """
    A class for generating random matrices.

    Each static method instantiates its own random number generator with a fixed seed to
    make the generated matrices deterministic and reproducible.
    """

    @staticmethod
    def random_real_matrix(m, n, positive=False, seed=42, dtype=None):
        """
        Generate a random m x n real matrix.

        By default, this method generates a matrix with elements that are real numbers
        on the interval [-1, 1). If the `positive` option is True, then it generates a
        positive matrix with elements on the interval [0, 1).

        This is a wrapper function for numpy.random.Generator.uniform:

        https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.uniform.html

        Parameters
        ----------
        m : int
            Number of rows in the matrix.
        n : int
            Number of columns in the matrix.
        positive : bool, optional
            If true, generate a positive matrix (default is false).
        seed : int, optional
            Seed for the random number generator (default is 42).
        dtype : str or numpy.dtype, optional
            Data type of the returned matrix. If None (the default), no data-type
            casting is performed.

        Returns
        -------
        numpy.ndarray
            An m x n matrix with random real values.
        """
        rng = np.random.default_rng(seed)
        lo = 0 if positive else -1
        hi = 1
        A = rng.uniform(lo, hi, (m, n))

        if dtype is None:
            return A
        else:
            return A.astype(dtype)

    @staticmethod
    def random_integer_matrix(m, n, lo, hi, seed=42, dtype=None):
        """
        Generate a random m x n integer matrix.

        The elements of the generated matrix are on the interval [`lo`, `hi`).

        This is a wrapper function for numpy.random.Generator.integers:

        https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.integers.html

        Parameters
        ----------
        m : int
            Number of rows in the matrix.
        n : int
            Number of columns in the matrix.
        lo : int
            Lowest (signed) integers to be drawn from the distribution.
        hi : int
            One above the largest (signed) integer to be drawn from the distribution.
        seed : int, optional
            Seed for the random number generator (default is 42).
        dtype : str or numpy.dtype, optional
            Data type of the returned matrix. If None (the default), no data-type
            casting is performed.

        Returns
        -------
        numpy.ndarray
            An m x n matrix with random integer values.
        """
        rng = np.random.default_rng(seed)
        A = rng.integers(lo, hi, (m, n))

        if dtype is None:
            return A
        else:
            return A.astype(dtype)

    @staticmethod
    def random_complex_matrix(m, n, seed=42, dtype=None):
        """
        Generate a random m x n complex matrix.

        This method generates two matrices A and B using numpy.random.Generator.uniform
        and returns the sum C = A + iB. The real and imaginary parts of each element of
        the generated matrix are on the interval [-1, 1).

        Parameters
        ----------
        m : int
            Number of rows in the matrix.
        n : int
            Number of columns in the matrix.
        seed : int, optional
            Seed for the random number generator (default is 42).
        dtype : str or numpy.dtype, optional
            Data type of the real and imaginary parts of the returned matrix. If None
            (the default), no data-type casting is performed. For example, if
            `dtype=np.float32`, the returned matrix is of type `np.complex64`.

        Returns
        -------
        numpy.ndarray
            An m x n matrix with random complex values.
        """
        rng = np.random.default_rng(seed)
        A = rng.uniform(-1, 1, (m, n))
        B = rng.uniform(-1, 1, (m, n))

        if dtype is not None:
            A = A.astype(dtype)
            B = B.astype(dtype)

        return A + 1j * B

    def random_real_symmetric_matrix(n, positive=False, seed=42, dtype=None):
        """
        Generate a random n x n real symmetric matrix.

        This method generates a matrix A with elements on the interval [-1, 1). If the
        `positive` option is True, then it generates a positive matrix with elements on
        the interval [0, 1). It then returns a symmetric matrix computed as

            S = (A + A^T) / 2.

        Parameters
        ----------
        n : int
            Number of rows and columns in the matrix.
        positive : bool, optional
            If true, generate a positive matrix (default is false).
        seed : int, optional
            Seed for the random number generator (default is 42).
        dtype : str or numpy.dtype, optional
            Data type of the returned matrix. If None (the default), no data-type
            casting is performed.

        Returns
        -------
        numpy.ndarray
            An n x n symmetric matrix with random real values.
        """
        rng = np.random.default_rng(seed)
        lo = 0 if positive else -1
        hi = 1
        A = rng.uniform(lo, hi, (n, n))

        if dtype is not None:
            A = A.astype(dtype)

        S = (A + A.T) / 2
        assert np.allclose(S, S.T)  # assert that the matrix is symmetric

        return S

    def random_real_symmetric_positive_definite_matrix(n, seed=42, dtype=None):
        """
        Generate a random n x n real symmetric positive-definite matrix.

        This method generates a real lower-triangular positive matrix L and computes a
        symmetric positive-definite matrix S using Cholesky decomposition:

            S = L L†

        Parameters
        ----------
        n : int
            Number of rows and columns in the matrix.
        seed : int, optional
            Seed for the random number generator (default is 42).
        dtype : str or numpy.dtype, optional
            Data type of the returned matrix. If None (the default), no data-type
            casting is performed.

        Returns
        -------
        numpy.ndarray
            An n x n symmetric positive-definite matrix with random real values.
        """
        L = np.tril(
            MatrixGenerator.random_real_symmetric_matrix(n, positive=True, seed=seed, dtype=dtype)
        )
        S = L * L.T

        assert np.allclose(S, S.T)  # assert that the matrix is symmetric
        assert np.all(np.linalg.eigvalsh(S) > 0)  # assert that the matrix is positive-definite

        return S

    def random_complex_hermitian_matrix(n, seed=42, dtype=None):
        """
        Generate a random n x n complex Hermitian matrix.

        This method generates two matrices A and B using numpy.random.Generator.uniform
        and defines a complex matrix C = A + iB. The real and imaginary parts of each
        element of the generated matrix are on the interval [-1, 1). It then returns a
        Hermitian matrix computed as H = (C + C†) / 2.

        Parameters
        ----------
        n : int
            Number of rows and columns in the matrix.
        seed : int, optional
            Seed for the random number generator (default is 42).
        dtype : str or numpy.dtype, optional
            Data type of the real and imaginary parts of the returned matrix. If None
            (the default), no data-type casting is performed. For example, if
            `dtype=np.float32`, the returned matrix is of type `np.complex64`.

        Returns
        -------
        numpy.ndarray
            An n x n Hermitian matrix with random complex values.
        """
        rng = np.random.default_rng(seed)
        A = rng.uniform(-1, 1, (n, n))
        B = rng.uniform(-1, 1, (n, n))

        if dtype is not None:
            A = A.astype(dtype)
            B = B.astype(dtype)

        C = A + 1j * B

        H = (C + C.T.conj()) / 2
        assert np.allclose(H, H.T.conj())  # assert that the matrix is Hermitian

        return H

    def random_complex_hermitian_positive_definite_matrix(n, seed=42, dtype=None):
        """
        Generate a random n x n complex Hermitian positive-definite matrix.

        This method generates a complex lower-triangular matrix L and computes a
        Hermitian positive-definite matrix A using Cholesky decomposition:

            H = L L†

        Parameters
        ----------
        n : int
            Number of rows and columns in the matrix.
        seed : int, optional
            Seed for the random number generator (default is 42).
        dtype : str or numpy.dtype, optional
            Data type of the real and imaginary parts of the returned matrix. If None
            (the default), no data-type casting is performed. For example, if
            `dtype=np.float32`, the returned matrix is of type `np.complex64`.

        Returns
        -------
        numpy.ndarray
            An n x n Hermitian positive-definite matrix with random complex values.
        """
        L = np.tril(MatrixGenerator.random_complex_hermitian_matrix(n, seed, dtype))
        H = L * L.T.conj()

        assert np.allclose(H, H.T.conj())  # assert that the matrix is Hermitian
        assert np.all(np.linalg.eigvalsh(H) > 0)  # assert that the matrix is positive-definite

        return H


class TestCholesky:
    """Test results of jax.scipy.linalg.cholesky are numerically correct when qjit compiled.

    See: https://jax.readthedocs.io/en/latest/_autosummary/jax.scipy.linalg.cholesky.html

    The Cholesky decomposition of a Hermitian positive-definite matrix H is a
    factorization of the form

        H = L L†  or  H = U† U,

    where L is a lower-triangular matrix with real and positive diagonal entries, and
    and similarly where U is an upper-triangular matrix.
    """

    @pytest.mark.parametrize(
        "A",
        [
            # Real matrices
            jnp.array(MatrixGenerator.random_real_symmetric_positive_definite_matrix(2, seed=11)),
            jnp.array(MatrixGenerator.random_real_symmetric_positive_definite_matrix(3, seed=12)),
            jnp.array(MatrixGenerator.random_real_symmetric_positive_definite_matrix(9, seed=13)),
            # Complex matrices
            jnp.array(
                MatrixGenerator.random_complex_hermitian_positive_definite_matrix(2, seed=21)
            ),
            jnp.array(
                MatrixGenerator.random_complex_hermitian_positive_definite_matrix(3, seed=22)
            ),
            jnp.array(
                MatrixGenerator.random_complex_hermitian_positive_definite_matrix(9, seed=23)
            ),
        ],
    )
    def test_cholesky_numerical_lower(self, A):
        """Test basic numerical correctness of jax.scipy.linalg.cholesky with option
        lower=True (giving decomposition in form A = L L†), for various functions and
        matrices of various data types and sizes.
        """

        @qjit
        def f(X):
            return jsp.linalg.cholesky(X, lower=True)

        L_obs = f(A)
        L_exp = jsp.linalg.cholesky(A, lower=True)

        assert np.allclose(L_exp @ L_exp.T.conj(), A)  # Check jax solution is correct
        assert jnp.allclose(L_obs, L_exp)

    @pytest.mark.parametrize(
        "A",
        [
            # Real matrices
            jnp.array(MatrixGenerator.random_real_symmetric_positive_definite_matrix(2, seed=11)),
            jnp.array(MatrixGenerator.random_real_symmetric_positive_definite_matrix(3, seed=12)),
            jnp.array(MatrixGenerator.random_real_symmetric_positive_definite_matrix(9, seed=13)),
            # Complex matrices
            jnp.array(
                MatrixGenerator.random_complex_hermitian_positive_definite_matrix(2, seed=21)
            ),
            jnp.array(
                MatrixGenerator.random_complex_hermitian_positive_definite_matrix(3, seed=22)
            ),
            jnp.array(
                MatrixGenerator.random_complex_hermitian_positive_definite_matrix(9, seed=23)
            ),
        ],
    )
    def test_cholesky_numerical_upper(self, A):
        """Test basic numerical correctness of jax.scipy.linalg.cholesky with option
        lower=False (giving decomposition in form A = U† U), for various functions and
        matrices of various data types and sizes.
        """

        @qjit
        def f(X):
            return jsp.linalg.cholesky(X, lower=False)

        U_obs = f(A)
        U_exp = jsp.linalg.cholesky(A, lower=False)

        assert np.allclose(U_exp.T.conj() @ U_exp, A)  # Check jax solution is correct
        assert jnp.allclose(U_obs, U_exp)


class TestExpm:
    """Test results of jax.scipy.linalg.expm are numerically correct when qjit compiled.

    See: https://jax.readthedocs.io/en/latest/_autosummary/jax.scipy.linalg.expm.html

    The exponential of a real or complex n x n matrix X, denoted exp(X), has the key
    property that if XY = YX, then

        exp(X) exp(Y) = exp(X+Y).
    """

    @pytest.mark.parametrize(
        "A",
        [
            # Real matrices
            jnp.array(MatrixGenerator.random_real_matrix(2, 2, seed=11)),
            jnp.array(MatrixGenerator.random_real_matrix(3, 3, seed=12)),
            jnp.array(MatrixGenerator.random_real_matrix(9, 9, seed=13)),
            jnp.array(MatrixGenerator.random_real_matrix(3, 3, positive=True, seed=14)),
            jnp.array(np.triu(MatrixGenerator.random_real_matrix(3, 3, seed=15))),
            jnp.array(np.tril(MatrixGenerator.random_real_matrix(3, 3, seed=16))),
            # Integer matrices
            jnp.array(MatrixGenerator.random_integer_matrix(3, 3, lo=-9, hi=9, seed=21)),
            jnp.array(MatrixGenerator.random_integer_matrix(3, 3, lo=0, hi=9, seed=22)),
            jnp.array(np.triu(MatrixGenerator.random_integer_matrix(3, 3, lo=-9, hi=9, seed=23))),
            jnp.array(np.tril(MatrixGenerator.random_integer_matrix(3, 3, lo=-9, hi=9, seed=24))),
            # Complex matrices
            jnp.array(MatrixGenerator.random_complex_matrix(3, 3, seed=31)),
            jnp.array(np.triu(MatrixGenerator.random_complex_matrix(3, 3, seed=32))),
            jnp.array(np.tril(MatrixGenerator.random_complex_matrix(3, 3, seed=33))),
        ],
    )
    def test_expm_numerical(self, A):
        """Test basic numerical correctness of jax.scipy.linalg.expm for matrices of
        various data types and sizes.
        """

        @qjit
        def f(X):
            return jsp.linalg.expm(X)

        expmA_obs = f(A)
        expmA_exp = jsp.linalg.expm(A)

        assert jnp.allclose(
            jsp.linalg.expm(A + A), jsp.linalg.expm(A) @ jsp.linalg.expm(A)
        )  # Check jax solution is correct
        assert np.allclose(expmA_obs, expmA_exp)

    @pytest.mark.parametrize(
        "A",
        [
            # Real upper-triangular matrices
            jnp.array(np.triu(MatrixGenerator.random_real_matrix(2, 2, seed=41))),
            jnp.array(np.triu(MatrixGenerator.random_real_matrix(3, 3, seed=42))),
            # Integer upper-triangular matrices
            jnp.array(np.triu(MatrixGenerator.random_integer_matrix(2, 2, lo=-9, hi=9, seed=43))),
            jnp.array(np.triu(MatrixGenerator.random_integer_matrix(3, 3, lo=-9, hi=9, seed=44))),
            # Complex upper-triangular matrices
            jnp.array(np.triu(MatrixGenerator.random_complex_matrix(2, 2, seed=45))),
            jnp.array(np.triu(MatrixGenerator.random_complex_matrix(3, 3, seed=46))),
        ],
    )
    def test_expm_numerical_upper_triangular(self, A):
        """Test basic numerical correctness of jax.scipy.linalg.expm for matrices of
        various data types and sizes when using the `upper_triangular=True` option.
        """

        @qjit
        def f(X):
            return jsp.linalg.expm(X, upper_triangular=True)

        expmA_obs = f(A)
        expmA_exp = jsp.linalg.expm(A, upper_triangular=True)

        assert jnp.allclose(
            jsp.linalg.expm(A + A), jsp.linalg.expm(A) @ jsp.linalg.expm(A)
        )  # Check jax solution is correct
        assert np.allclose(expmA_obs, expmA_exp)


class TestFunmNumerical:
    """Test results of jax.scipy.linalg.funm are numerically correct when qjit compiled.

    See: https://jax.readthedocs.io/en/latest/_autosummary/jax.scipy.linalg.funm.html
    """

    @pytest.mark.parametrize(
        "A",
        [
            # Real matrices
            jnp.array(MatrixGenerator.random_real_matrix(2, 2, seed=11)),
            jnp.array(MatrixGenerator.random_real_matrix(3, 3, seed=12)),
            jnp.array(MatrixGenerator.random_real_matrix(9, 9, seed=13)),
            jnp.array(MatrixGenerator.random_real_matrix(3, 3, positive=True, seed=14)),
            jnp.array(np.triu(MatrixGenerator.random_real_matrix(3, 3, seed=15))),
            jnp.array(np.tril(MatrixGenerator.random_real_matrix(3, 3, seed=16))),
            # Complex matrices
            jnp.array(MatrixGenerator.random_complex_matrix(3, 3, seed=21)),
            jnp.array(np.triu(MatrixGenerator.random_complex_matrix(3, 3, seed=22))),
            jnp.array(np.tril(MatrixGenerator.random_complex_matrix(3, 3, seed=23))),
        ],
    )
    def test_funm_numerical(self, A):
        """Test basic numerical correctness of jax.scipy.linalg.funm for various
        matrices of various data types and sizes.
        """

        @qjit
        def f(X):
            def func(X):
                return jnp.sin(X) + 2 * jnp.cos(X)

            return jsp.linalg.funm(X, func)

        fA_obs = f(A)
        fA_exp = jsp.linalg.funm(A, lambda X: jnp.sin(X) + 2 * jnp.cos(X))

        assert jnp.allclose(fA_obs, fA_exp)


class TestHessenberg:
    """Test results of jax.scipy.linalg.hessenberg are numerically correct when qjit compiled.

    See: https://jax.readthedocs.io/en/latest/_autosummary/jax.scipy.linalg.hessenberg.html

    The Hessenberg form H of an n x n matrix A satisfies

        A = Q H Q†,

    where Q is unitary and H is zero below the first diagonal.
    """

    @pytest.mark.parametrize(
        "A",
        [
            # Real matrices
            jnp.array(MatrixGenerator.random_real_matrix(2, 2, seed=11)),
            jnp.array(MatrixGenerator.random_real_matrix(3, 3, seed=12)),
            jnp.array(MatrixGenerator.random_real_matrix(9, 9, seed=13)),
            jnp.array(MatrixGenerator.random_real_matrix(3, 3, positive=True, seed=14)),
            jnp.array(np.triu(MatrixGenerator.random_real_matrix(3, 3, seed=15))),
            jnp.array(np.tril(MatrixGenerator.random_real_matrix(3, 3, seed=16))),
            # Complex matrices
            jnp.array(MatrixGenerator.random_complex_matrix(3, 3, seed=21)),
            jnp.array(np.triu(MatrixGenerator.random_complex_matrix(3, 3, seed=22))),
            jnp.array(np.tril(MatrixGenerator.random_complex_matrix(3, 3, seed=23))),
        ],
    )
    def test_hessenberg_numerical(self, A):
        """Test basic numerical correctness of jax.scipy.linalg.hessenberg for matrices
        of various data types and sizes.

        Note that jax does not support integer matrices for this function.
        """

        @qjit
        def f(X):
            return jsp.linalg.hessenberg(X, calc_q=True)

        H_obs, Q_obs = f(A)
        H_exp, Q_exp = jsp.linalg.hessenberg(A, calc_q=True)

        assert jnp.allclose(Q_exp @ H_exp @ Q_exp.conj().T, A)  # Check jax solution is correct
        assert jnp.allclose(H_obs, H_exp)
        assert jnp.allclose(Q_obs, Q_exp)


class TestLU:
    """Test results of jax.scipy.linalg.lu are numerically correct when qjit compiled.

    See: https://jax.readthedocs.io/en/latest/_autosummary/jax.scipy.linalg.lu.html

    The LU decomposition with partial pivoting of a real or complex n x n matrix A satisfies

        A = P L U

    where P is a permutation matrix, L lower triangular with unit diagonal elements, and
    U is upper triangular.

    JAX (and SciPy) also support LU decomposition of m x n matrices, in which case L has
    dimension m x k, where k = min(m, n) and U has dimension k x n.
    """

    @pytest.mark.parametrize(
        "A",
        [
            # Real matrices
            jnp.array(MatrixGenerator.random_real_matrix(2, 2, seed=11)),
            jnp.array(MatrixGenerator.random_real_matrix(3, 3, seed=12)),
            jnp.array(MatrixGenerator.random_real_matrix(9, 9, seed=13)),
            jnp.array(MatrixGenerator.random_real_matrix(3, 3, positive=True, seed=14)),
            jnp.array(MatrixGenerator.random_real_matrix(5, 7, seed=15)),
            jnp.array(MatrixGenerator.random_real_matrix(7, 5, seed=16)),
            jnp.array(np.triu(MatrixGenerator.random_real_matrix(3, 3, seed=17))),
            jnp.array(np.tril(MatrixGenerator.random_real_matrix(3, 3, seed=18))),
            # Integer matrices
            jnp.array(MatrixGenerator.random_integer_matrix(3, 3, lo=-9, hi=9, seed=21)),
            jnp.array(MatrixGenerator.random_integer_matrix(3, 3, lo=0, hi=9, seed=22)),
            jnp.array(MatrixGenerator.random_integer_matrix(5, 7, lo=-9, hi=9, seed=23)),
            jnp.array(MatrixGenerator.random_integer_matrix(7, 5, lo=-9, hi=9, seed=24)),
            jnp.array(np.triu(MatrixGenerator.random_integer_matrix(3, 3, lo=-9, hi=9, seed=25))),
            jnp.array(np.tril(MatrixGenerator.random_integer_matrix(3, 3, lo=-9, hi=9, seed=26))),
            # Complex matrices
            jnp.array(MatrixGenerator.random_complex_matrix(3, 3, seed=31)),
            jnp.array(np.triu(MatrixGenerator.random_complex_matrix(3, 3, seed=32))),
            jnp.array(np.tril(MatrixGenerator.random_complex_matrix(3, 3, seed=33))),
        ],
    )
    def test_lu_numerical(self, A):
        """Test basic numerical correctness for jax.scipy.linalg.lu for for matrices of
        various data types and sizes.
        """

        @qjit
        def f(X):
            return jsp.linalg.lu(X)

        P_obs, L_obs, U_obs = f(A)
        P_exp, L_exp, U_exp = jsp.linalg.lu(A)

        assert jnp.allclose(P_exp @ L_exp @ U_exp, A)  # Check jax solution is correct
        assert jnp.allclose(P_obs, P_exp)
        assert jnp.allclose(L_obs, L_exp)
        assert jnp.allclose(U_obs, U_exp)


class TestLUSolve:
    """Test results of jax.scipy.linalg.lu_solve are numerically correct when qjit compiled.

    See: https://jax.readthedocs.io/en/latest/_autosummary/jax.scipy.linalg.lu_solve.html

    This method solves a linear system using an LU factorization (see above).
    """

    @pytest.mark.parametrize(
        "A,b",
        [
            # Real coefficient matrices
            (
                jnp.array(MatrixGenerator.random_real_matrix(2, 2, seed=11)),
                jnp.array(MatrixGenerator.random_real_matrix(2, 1, seed=12)),
            ),
            (
                jnp.array(MatrixGenerator.random_real_matrix(3, 3, seed=13)),
                jnp.array(MatrixGenerator.random_real_matrix(3, 1, seed=14)),
            ),
            (
                jnp.array(MatrixGenerator.random_real_matrix(9, 9, seed=15)),
                jnp.array(MatrixGenerator.random_real_matrix(9, 1, seed=16)),
            ),
            # Complex coefficient matrices
            (
                jnp.array(MatrixGenerator.random_complex_matrix(3, 3, seed=21)),
                jnp.array(MatrixGenerator.random_complex_matrix(3, 1, seed=22)),
            ),
        ],
    )
    def test_lu_solve_numerical(self, A, b):
        """Test basic numerical correctness of jax.scipy.linalg.lu_solve for matrices and
        vectors of various data types and sizes.

        Note that jax does not support integer matrices for this function.
        """

        @qjit
        def f(A, b):
            lu_and_piv = jsp.linalg.lu_factor(A)
            return jsp.linalg.lu_solve(lu_and_piv, b)

        x_obs = f(A, b)
        lu_and_piv = jsp.linalg.lu_factor(A)
        x_exp = jsp.linalg.lu_solve(lu_and_piv, b)

        assert jnp.allclose(A @ x_exp, b)  # Check jax solution is correct
        assert jnp.allclose(x_obs, x_exp)


class TestPolar:
    """Test results of jax.scipy.linalg.polar are numerically correct when qjit compiled.

    See: https://jax.readthedocs.io/en/latest/_autosummary/jax.scipy.linalg.polar.html

    The polar decomposition of a real or complex n x n matrix A is a factorization of
    the form

        A = U P,

    where where U is a unitary matrix and P is a positive semi-definite Hermitian matrix.

    JAX (and SciPy) also support polar decomposition of m x n matrices, in which case U
    has dimension m x n and P has dimension n x n. This is known as "right-side" polar
    decomposition. "Left-side" decomposition, in the A = P U, is also possible, in which
    case P has dimension m x m.
    """

    @pytest.mark.parametrize(
        "A",
        [
            # Real matrices
            jnp.array(MatrixGenerator.random_real_matrix(2, 2, seed=11)),
            jnp.array(MatrixGenerator.random_real_matrix(3, 3, seed=12)),
            jnp.array(MatrixGenerator.random_real_matrix(9, 9, seed=13)),
            jnp.array(MatrixGenerator.random_real_matrix(3, 3, positive=True, seed=14)),
            jnp.array(MatrixGenerator.random_real_matrix(5, 7, seed=15)),
            jnp.array(MatrixGenerator.random_real_matrix(7, 5, seed=16)),
            # Complex matrices
            jnp.array(MatrixGenerator.random_complex_matrix(3, 3, seed=21)),
            jnp.array(MatrixGenerator.random_complex_matrix(5, 7, seed=22)),
            jnp.array(MatrixGenerator.random_complex_matrix(7, 5, seed=23)),
        ],
    )
    def test_polar_numerical_svd(self, A):
        """Test basic numerical correctness of jax.scipy.linalg.polar for matrices
        of various data types and sizes using the 'svd' method.

        Note that jax does not support integer matrices for this function.
        """

        @qjit
        def f(X):
            return jsp.linalg.polar(X, method="svd")

        U_obs, P_obs = f(A)
        U_exp, P_exp = jsp.linalg.polar(A, method="svd")

        assert jnp.allclose(U_exp @ P_exp, A)  # Check jax solution is correct
        assert jnp.allclose(U_obs, U_exp)
        assert jnp.allclose(P_obs, P_exp)

    @pytest.mark.parametrize(
        "A",
        [
            # Real matrices
            jnp.array(MatrixGenerator.random_real_matrix(2, 2, seed=11)),
            jnp.array(MatrixGenerator.random_real_matrix(3, 3, seed=12)),
            jnp.array(MatrixGenerator.random_real_matrix(9, 9, seed=13)),
            jnp.array(MatrixGenerator.random_real_matrix(3, 3, positive=True, seed=14)),
            jnp.array(MatrixGenerator.random_real_matrix(7, 5, seed=15)),
            # Complex matrices
            jnp.array(MatrixGenerator.random_complex_matrix(3, 3, seed=21)),
            jnp.array(MatrixGenerator.random_complex_matrix(7, 5, seed=22)),
        ],
    )
    def test_polar_numerical_qdwh(self, A):
        """Test basic numerical correctness of jax.scipy.linalg.polar for matrices
        of various data types and sizes using the 'qdwh' method.

        Note that jax does not support integer matrices for this function.
        """
        print("Start")

        @qjit
        def f(X):
            return jsp.linalg.polar(X, method="qdwh")

        U_obs, P_obs = f(A)
        U_exp, P_exp = jsp.linalg.polar(A, method="qdwh")

        assert jnp.allclose(U_exp @ P_exp, A)  # Check jax solution is correct
        assert jnp.allclose(U_obs, U_exp)
        assert jnp.allclose(P_obs, P_exp)


class TestQR:
    """Test results of jax.scipy.linalg.qr are numerically correct when qjit compiled.

    See: https://jax.readthedocs.io/en/latest/_autosummary/jax.scipy.linalg.qr.html

    QR decomposition of a real or complex n x n matrix A is a factorization of the form

        A = Q R,

    where Q is a unitary matrix (i.e. Q† Q = Q Q† = I) and R is an upper-triangular
    matrix (also called right-triangular matrix).

    JAX (and SciPy) also support QR decomposition of m x n matrices, in which case Q has
    dimension m x k, where k = min(m, n) and R has dimension k x n.
    """

    @pytest.mark.parametrize(
        "A",
        [
            # Real matrices
            jnp.array(MatrixGenerator.random_real_matrix(2, 2, seed=11)),
            jnp.array(MatrixGenerator.random_real_matrix(3, 3, seed=12)),
            jnp.array(MatrixGenerator.random_real_matrix(9, 9, seed=13)),
            jnp.array(MatrixGenerator.random_real_matrix(3, 3, positive=True, seed=14)),
            jnp.array(np.triu(MatrixGenerator.random_real_matrix(3, 3, seed=15))),
            jnp.array(np.tril(MatrixGenerator.random_real_matrix(3, 3, seed=16))),
            jnp.array(MatrixGenerator.random_real_matrix(5, 7, seed=17)),
            jnp.array(MatrixGenerator.random_real_matrix(7, 5, seed=18)),
            # Complex matrices
            jnp.array(MatrixGenerator.random_complex_matrix(3, 3, seed=21)),
            jnp.array(np.triu(MatrixGenerator.random_complex_matrix(3, 3, seed=22))),
            jnp.array(np.tril(MatrixGenerator.random_complex_matrix(3, 3, seed=23))),
            jnp.array(MatrixGenerator.random_complex_matrix(5, 7, seed=24)),
            jnp.array(MatrixGenerator.random_complex_matrix(7, 5, seed=25)),
        ],
    )
    def test_qr_numerical(self, A):
        """Test basic numerical correctness of jax.scipy.linalg.qr for matrices
        of various data types and sizes.

        Note that jax does not support integer matrices for this function.
        """

        @qjit
        def f(X):
            return jsp.linalg.qr(X)

        Q_obs, R_obs = f(A)
        Q_exp, R_exp = jsp.linalg.qr(A)

        assert jnp.allclose(Q_exp @ R_exp, A)  # Check jax solution is correct
        assert jnp.allclose(Q_obs, Q_exp)
        assert jnp.allclose(R_obs, R_exp)


class TestSchur:
    """Test results of jax.scipy.linalg.schur are numerically correct when qjit compiled.

    See: https://jax.readthedocs.io/en/latest/_autosummary/jax.scipy.linalg.schur.html

    The Schur decomposition of a real or complex n x n matrix A is a factorization of
    the form

        A = Z T Z†

    where Z is a unitary matrix and T is an upper-triangular matrix.
    """

    @pytest.mark.parametrize(
        "A",
        [
            # Real matrices
            jnp.array(MatrixGenerator.random_real_matrix(2, 2, seed=11)),
            jnp.array(MatrixGenerator.random_real_matrix(3, 3, seed=12)),
            jnp.array(MatrixGenerator.random_real_matrix(9, 9, seed=13)),
            jnp.array(MatrixGenerator.random_real_matrix(3, 3, positive=True, seed=14)),
            jnp.array(np.triu(MatrixGenerator.random_real_matrix(3, 3, seed=15))),
            jnp.array(np.tril(MatrixGenerator.random_real_matrix(3, 3, seed=16))),
            # Complex matrices
            jnp.array(MatrixGenerator.random_complex_matrix(3, 3, seed=21)),
            jnp.array(np.triu(MatrixGenerator.random_complex_matrix(3, 3, seed=22))),
            jnp.array(np.tril(MatrixGenerator.random_complex_matrix(3, 3, seed=23))),
        ],
    )
    def test_schur_numerical_real(self, A):
        """Test basic numerical correctness of jax.scipy.linalg.schur with output='real'
        for matrices of various data types and sizes.

        Note that jax does not support integer matrices for this function.
        """

        @qjit
        def f(X):
            return jsp.linalg.schur(X, output="real")

        T_obs, Z_obs = f(A)
        T_exp, Z_exp = jsp.linalg.schur(A, output="real")

        assert jnp.allclose(Z_exp @ T_exp @ Z_exp.conj().T, A)  # Check jax solution is correct
        assert jnp.allclose(T_obs, T_exp)
        assert jnp.allclose(Z_obs, Z_exp)

    @pytest.mark.parametrize(
        "A",
        [
            # Real matrices
            jnp.array(MatrixGenerator.random_real_matrix(2, 2, seed=11)),
            jnp.array(MatrixGenerator.random_real_matrix(3, 3, seed=12)),
            jnp.array(MatrixGenerator.random_real_matrix(9, 9, seed=13)),
            jnp.array(MatrixGenerator.random_real_matrix(3, 3, positive=True, seed=14)),
            jnp.array(np.triu(MatrixGenerator.random_real_matrix(3, 3, seed=15))),
            jnp.array(np.tril(MatrixGenerator.random_real_matrix(3, 3, seed=16))),
            # Complex matrices
            jnp.array(MatrixGenerator.random_complex_matrix(3, 3, seed=21)),
            jnp.array(np.triu(MatrixGenerator.random_complex_matrix(3, 3, seed=22))),
            jnp.array(np.tril(MatrixGenerator.random_complex_matrix(3, 3, seed=23))),
        ],
    )
    def test_schur_numerical_complex(self, A):
        """Test basic numerical correctness of jax.scipy.linalg.schur with output='complex'
        for matrices of various data types and sizes.

        Note that jax does not support integer matrices for this function.
        """

        @qjit
        def f(X):
            return jsp.linalg.schur(X, output="complex")

        T_obs, Z_obs = f(A)
        T_exp, Z_exp = jsp.linalg.schur(A, output="complex")

        assert jnp.allclose(Z_exp @ T_exp @ Z_exp.conj().T, A)  # Check jax solution is correct
        assert jnp.allclose(T_obs, T_exp)
        assert jnp.allclose(Z_obs, Z_exp)


class TestSolve:
    """Test results of jax.scipy.linalg.solve are numerically correct when qjit compiled.

    See: https://jax.readthedocs.io/en/latest/_autosummary/jax.scipy.linalg.solve.html

    Solves a linear system of equations of the form A x = b for x given the n x n matrix
    A and length n vector b.
    """

    @pytest.mark.parametrize(
        "A,b",
        [
            # Real coefficient matrices
            (
                jnp.array(MatrixGenerator.random_real_matrix(2, 2, seed=11)),
                jnp.array(MatrixGenerator.random_real_matrix(2, 1, seed=12)),
            ),
            (
                jnp.array(MatrixGenerator.random_real_matrix(3, 3, seed=13)),
                jnp.array(MatrixGenerator.random_real_matrix(3, 1, seed=14)),
            ),
            (
                jnp.array(MatrixGenerator.random_real_matrix(9, 9, seed=15)),
                jnp.array(MatrixGenerator.random_real_matrix(9, 1, seed=16)),
            ),
            # Integer coefficient matrices
            (
                jnp.array(MatrixGenerator.random_integer_matrix(3, 3, -9, 9, seed=21)),
                jnp.array(MatrixGenerator.random_integer_matrix(3, 1, -9, 9, seed=22)),
            ),
            # Complex coefficient matrices
            (
                jnp.array(MatrixGenerator.random_complex_matrix(3, 3, seed=31)),
                jnp.array(MatrixGenerator.random_complex_matrix(3, 1, seed=22)),
            ),
        ],
    )
    def test_solve_numerical(self, A, b):
        """Test basic numerical correctness of jax.scipy.linalg.solve for matrices and
        vectors of various data types and sizes.
        """

        @qjit
        def f(A, b):
            return jsp.linalg.solve(A, b)

        x_obs = f(A, b)
        x_exp = jsp.linalg.solve(A, b)

        assert jnp.allclose(A @ x_exp, b)  # Check jax solution is correct
        assert jnp.allclose(x_obs, x_exp)

    @pytest.mark.parametrize(
        "A,b",
        [
            # Hermitian coefficient matrices
            (
                jnp.array(MatrixGenerator.random_complex_hermitian_matrix(3, seed=11)),
                jnp.array(MatrixGenerator.random_complex_matrix(3, 1, seed=12)),
            ),
        ],
    )
    def test_solve_numerical_hermitian(self, A, b):
        """Test basic numerical correctness of jax.scipy.linalg.solve for Hermitian
        matrices and vectors of various data types and sizes to test the `assume_a="her"`
        option.
        """

        @qjit
        def f(A, b):
            return jsp.linalg.solve(A, b, assume_a="her")

        x_obs = f(A, b)
        x_exp = jsp.linalg.solve(A, b, assume_a="her")

        assert jnp.allclose(A @ x_exp, b)  # Check jax solution is correct
        assert jnp.allclose(x_obs, x_exp)


class TestSqrtm:
    """Test results of jax.scipy.linalg.sqrtm are numerically correct when qjit compiled.

    See: https://jax.readthedocs.io/en/latest/_autosummary/jax.scipy.linalg.sqrtm.html

    The square root of a real or complex n x n matrix A, denoted sqrt(A), has the key
    property that

        sqrt(A) sqrt(A) = A
    """

    @pytest.mark.parametrize(
        "A",
        [
            # Real matrices
            jnp.array(MatrixGenerator.random_real_matrix(2, 2, seed=11)),
            jnp.array(MatrixGenerator.random_real_matrix(3, 3, seed=12)),
            jnp.array(MatrixGenerator.random_real_matrix(9, 9, seed=13)),
            jnp.array(MatrixGenerator.random_real_matrix(3, 3, positive=True, seed=14)),
            jnp.array(np.triu(MatrixGenerator.random_real_matrix(3, 3, seed=15))),
            jnp.array(np.tril(MatrixGenerator.random_real_matrix(3, 3, seed=16))),
            # Integer matrices
            jnp.array(MatrixGenerator.random_integer_matrix(3, 3, lo=-9, hi=9, seed=21)),
            jnp.array(MatrixGenerator.random_integer_matrix(3, 3, lo=0, hi=9, seed=22)),
            jnp.array(np.triu(MatrixGenerator.random_integer_matrix(3, 3, lo=-9, hi=9, seed=23))),
            jnp.array(np.tril(MatrixGenerator.random_integer_matrix(3, 3, lo=-9, hi=9, seed=24))),
            # Complex matrices
            jnp.array(MatrixGenerator.random_complex_matrix(3, 3, seed=31)),
            jnp.array(np.triu(MatrixGenerator.random_complex_matrix(3, 3, seed=32))),
            jnp.array(np.tril(MatrixGenerator.random_complex_matrix(3, 3, seed=33))),
        ],
    )
    def test_sqrtm_numerical(self, A):
        """Test basic numerical correctness of jax.scipy.linalg.sqrtm for matrices of
        various data types and sizes.
        """

        @qjit
        def f(X):
            return jsp.linalg.sqrtm(X)

        sqrtmA_obs = f(A)
        sqrtmA_exp = jsp.linalg.sqrtm(A)

        assert jnp.allclose(sqrtmA_exp @ sqrtmA_exp, A)  # Check jax solution is correct
        assert jnp.allclose(sqrtmA_obs, sqrtmA_exp)


class TestSVD:
    """Test results of jax.scipy.linalg.svd are numerically correct when qjit compiled.

    See: https://jax.readthedocs.io/en/latest/_autosummary/jax.scipy.linalg.svd.html

    Singular value decomposition (SVD) of a real or complex m x n matrix A is a
    factorization of the form

        A = U Σ V†,

    where U and V are m x n unitary matrices containing the left and right singular
    vectors, respectively, and Σ is an m x n diagonal matrix of singular values.
    """

    @pytest.mark.parametrize(
        "A",
        [
            # Real matrices
            jnp.array(MatrixGenerator.random_real_matrix(2, 2, seed=11)),
            jnp.array(MatrixGenerator.random_real_matrix(3, 3, seed=12)),
            jnp.array(MatrixGenerator.random_real_matrix(9, 9, seed=13)),
            jnp.array(MatrixGenerator.random_real_matrix(3, 3, positive=True, seed=14)),
            jnp.array(np.triu(MatrixGenerator.random_real_matrix(3, 3, seed=15))),
            jnp.array(np.tril(MatrixGenerator.random_real_matrix(3, 3, seed=16))),
            jnp.array(MatrixGenerator.random_real_matrix(5, 7, seed=17)),
            jnp.array(MatrixGenerator.random_real_matrix(7, 5, seed=18)),
            # Integer matrices
            jnp.array(MatrixGenerator.random_integer_matrix(3, 3, lo=-9, hi=9, seed=21)),
            jnp.array(MatrixGenerator.random_integer_matrix(3, 3, lo=0, hi=9, seed=22)),
            # Complex matrices
            jnp.array(MatrixGenerator.random_complex_matrix(3, 3, seed=31)),
            jnp.array(MatrixGenerator.random_complex_matrix(5, 7, seed=32)),
            jnp.array(MatrixGenerator.random_complex_matrix(7, 5, seed=33)),
        ],
    )
    def test_svd_numerical(self, A):
        """Test basic numerical correctness of jax.scipy.linalg.svd for matrices
        of various data types and sizes.
        """

        @qjit
        def f(X):
            return jsp.linalg.svd(X)

        U_obs, S_obs, Vt_obs = f(A)
        U_exp, S_exp, Vt_exp = jsp.linalg.svd(A)

        # Pad S_exp with rows/cols of zeros if input matrix is not square
        S_padded = np.zeros(A.shape)
        for i in range(min(A.shape)):
            S_padded[i, i] = S_exp[i]

        assert jnp.allclose(U_exp @ S_padded @ Vt_exp, A)  # Check jax solution is correct
        assert jnp.allclose(U_obs, U_exp)
        assert jnp.allclose(S_obs, S_exp)
        assert jnp.allclose(Vt_obs, Vt_exp)
