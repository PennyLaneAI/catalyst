// Copyright 2024 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// This file has been modified from its original form in the JAX project at
// https://github.com/google/jax released under the Apache License, Version 2.0,
// with the following copyright notice:

// Copyright 2021 The JAX Authors.

/*
 * This file is a modified version of
 *
 *   https://github.com/google/jax/blob/jaxlib-v0.4.28/jaxlib/cpu/lapack_kernels.cc
 *
 * from jaxlib-v0.4.28.
 *
 * See note in lapack_kernels.h for a high-level explanation of the
 * modifications and the motivation for them. Specifically, the modifications
 * made in this file include:
 *
 *  1. Used the C interfaces to the BLAS and LAPACK routines instead of the
 *     FORTRAN interfaces, and always use row-major matrix layout. This
 *     modification generally involves the following:
 *       - Adding the matrix layout parameter as the first argument to the BLAS/
 *         LAPACK call, either `CblasRowMajor` for BLAS or `LAPACK_ROW_MAJOR`
 *         for LAPACK.
 *       - Specifying the array leading dimensions (e.g. `lda`) such that they
 *         are dependent upon the matrix layout, rather than hard-coding them.
 *         Note that these should always evaluate to the value required for
 *         row-major matrix layout (typically the number of columns n of the
 *         matrix).
 *       - Remove parameters used by the FORTRAN interfaces but not by the C
 *         interfaces, e.g. the workspace array parameters `lwork`, `rwork`,
 *         `iwork`, etc.
 *  2. Guarded the #include of the ABSEIL `dynamic_annotations.h header by the
 *     `USE_ABSEIL_LIB` macro and the uses of `ABSL_ANNOTATE_MEMORY_IS_INITIALIZED`,
 *     since they are not needed for Catalyst.
 *  3. Opportunistically improved const-correctness.
 *  4. Applied Catalyst C++ code formatting.
 */

#include "lapack_kernels.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

#ifdef USE_ABSEIL_LIB
#include "absl/base/dynamic_annotations.h"
#endif

namespace jax {

// Trsm (Triangular System Solver)
// ~~~~

template <typename T> typename RealTrsm<T>::FnType *RealTrsm<T>::fn = nullptr;

template <typename T> void RealTrsm<T>::Kernel(void *out, void **data, XlaCustomCallStatus *)
{
    const uint8_t *diag_tensor = reinterpret_cast<uint8_t *>(data[0]);
    const char diag = static_cast<char>(*diag_tensor);
    const uint8_t *side_tensor = reinterpret_cast<uint8_t *>(data[1]);
    const char side = static_cast<char>(*side_tensor);
    const uint8_t *trans_tensor = reinterpret_cast<uint8_t *>(data[2]);
    const char trans = static_cast<char>(*trans_tensor);
    const uint8_t *uplo_tensor = reinterpret_cast<uint8_t *>(data[3]);
    const char uplo = static_cast<char>(*uplo_tensor);
    const int batch = *reinterpret_cast<int32_t *>(data[7]);
    const int m = *reinterpret_cast<int32_t *>(data[8]);
    const int n = *reinterpret_cast<int32_t *>(data[9]);
    const T *a = reinterpret_cast<T *>(data[10]);
    T *b = reinterpret_cast<T *>(data[11]);
    const T alpha = static_cast<T>(1);

    T *x = reinterpret_cast<T *>(out);
    if (x != b) {
        std::memcpy(x, b,
                    static_cast<int64_t>(batch) * static_cast<int64_t>(m) *
                        static_cast<int64_t>(n) * sizeof(T));
    }

    constexpr CBLAS_ORDER corder = CblasRowMajor;
    const CBLAS_SIDE cside = (side == 'L') ? CblasLeft : CblasRight;
    const CBLAS_UPLO cuplo = (uplo == 'L') ? CblasLower : CblasUpper;
    const CBLAS_TRANSPOSE ctransa = (trans == 'T')   ? CblasTrans
                                    : (trans == 'C') ? CblasConjTrans
                                                     : CblasNoTrans;
    const CBLAS_DIAG cdiag = (diag == 'U') ? CblasUnit : CblasNonUnit;
    const int lda = (side == 'L') ? m : n;
    const int ldb = (corder == CblasColMajor) ? m : n; // Note: m if col-major, n if row-major

    const int64_t x_plus = static_cast<int64_t>(m) * static_cast<int64_t>(n);
    const int64_t a_plus = static_cast<int64_t>(lda) * static_cast<int64_t>(lda);

    for (int i = 0; i < batch; ++i) {
        fn(CblasRowMajor, cside, cuplo, ctransa, cdiag, m, n, alpha, a, lda, x, ldb);
        x += x_plus;
        a += a_plus;
    }
}

template <typename T> typename ComplexTrsm<T>::FnType *ComplexTrsm<T>::fn = nullptr;

template <typename T> void ComplexTrsm<T>::Kernel(void *out, void **data, XlaCustomCallStatus *)
{
    const uint8_t *diag_tensor = reinterpret_cast<uint8_t *>(data[0]);
    const char diag = static_cast<char>(*diag_tensor);
    const uint8_t *side_tensor = reinterpret_cast<uint8_t *>(data[1]);
    const char side = static_cast<char>(*side_tensor);
    const uint8_t *trans_tensor = reinterpret_cast<uint8_t *>(data[2]);
    const char trans = static_cast<char>(*trans_tensor);
    const uint8_t *uplo_tensor = reinterpret_cast<uint8_t *>(data[3]);
    const char uplo = static_cast<char>(*uplo_tensor);
    const int batch = *reinterpret_cast<int32_t *>(data[7]);
    const int m = *reinterpret_cast<int32_t *>(data[8]);
    const int n = *reinterpret_cast<int32_t *>(data[9]);
    const T *a = reinterpret_cast<T *>(data[10]);
    T *b = reinterpret_cast<T *>(data[11]);
    const T alpha = static_cast<T>(1);

    T *x = reinterpret_cast<T *>(out);
    if (x != b) {
        std::memcpy(x, b,
                    static_cast<int64_t>(batch) * static_cast<int64_t>(m) *
                        static_cast<int64_t>(n) * sizeof(T));
    }

    constexpr CBLAS_ORDER corder = CblasRowMajor;
    const CBLAS_SIDE cside = (side == 'L') ? CblasLeft : CblasRight;
    const CBLAS_UPLO cuplo = (uplo == 'L') ? CblasLower : CblasUpper;
    const CBLAS_TRANSPOSE ctransa = (trans == 'T')   ? CblasTrans
                                    : (trans == 'C') ? CblasConjTrans
                                                     : CblasNoTrans;
    const CBLAS_DIAG cdiag = (diag == 'U') ? CblasUnit : CblasNonUnit;
    const int lda = (side == 'L') ? m : n;
    const int ldb = (corder == CblasColMajor) ? m : n; // Note: m if col-major, n if row-major

    const int64_t x_plus = static_cast<int64_t>(m) * static_cast<int64_t>(n);
    const int64_t a_plus = static_cast<int64_t>(lda) * static_cast<int64_t>(lda);

    for (int i = 0; i < batch; ++i) {
        fn(CblasRowMajor, cside, cuplo, ctransa, cdiag, m, n, &alpha, a, lda, x, ldb);
        x += x_plus;
        a += a_plus;
    }
}

template struct RealTrsm<float>;
template struct RealTrsm<double>;
template struct ComplexTrsm<std::complex<float>>;
template struct ComplexTrsm<std::complex<double>>;

// Getrf (LU Decomposition)
// ~~~~~

template <typename T> typename Getrf<T>::FnType *Getrf<T>::fn = nullptr;

template <typename T> void Getrf<T>::Kernel(void *out_tuple, void **data, XlaCustomCallStatus *)
{
    const int b = *(reinterpret_cast<int32_t *>(data[0]));
    const int m = *(reinterpret_cast<int32_t *>(data[1]));
    const int n = *(reinterpret_cast<int32_t *>(data[2]));
    const T *a_in = reinterpret_cast<T *>(data[3]);

    void **out = reinterpret_cast<void **>(out_tuple);
    T *a_out = reinterpret_cast<T *>(out[0]);
    int *ipiv = reinterpret_cast<int *>(out[1]);
    int *info = reinterpret_cast<int *>(out[2]);
    if (a_out != a_in) {
        std::memcpy(a_out, a_in,
                    static_cast<int64_t>(b) * static_cast<int64_t>(m) * static_cast<int64_t>(n) *
                        sizeof(T));
    }

    constexpr int corder = LAPACK_ROW_MAJOR;
    const int lda = (corder == LAPACK_ROW_MAJOR) ? n : m;

    for (int i = 0; i < b; ++i) {
        *info = fn(corder, m, n, a_out, lda, ipiv);
        a_out += static_cast<int64_t>(m) * static_cast<int64_t>(n);
        ipiv += std::min(m, n);
        ++info;
    }
}

template struct Getrf<float>;
template struct Getrf<double>;
template struct Getrf<std::complex<float>>;
template struct Getrf<std::complex<double>>;

// Geqrf (QR Factorization)
// ~~~~~

template <typename T> typename Geqrf<T>::FnType *Geqrf<T>::fn = nullptr;

template <typename T> void Geqrf<T>::Kernel(void *out_tuple, void **data, XlaCustomCallStatus *)
{
    const int b = *(reinterpret_cast<int32_t *>(data[0]));
    const int m = *(reinterpret_cast<int32_t *>(data[1]));
    const int n = *(reinterpret_cast<int32_t *>(data[2]));
    const T *a_in = reinterpret_cast<T *>(data[3]);
    void **out = reinterpret_cast<void **>(out_tuple);
    T *a_out = reinterpret_cast<T *>(out[0]);
    T *tau = reinterpret_cast<T *>(out[1]);

    if (a_out != a_in) {
        std::memcpy(a_out, a_in,
                    static_cast<int64_t>(b) * static_cast<int64_t>(m) * static_cast<int64_t>(n) *
                        sizeof(T));
    }

    constexpr int corder = LAPACK_ROW_MAJOR;
    const int lda = (corder == LAPACK_ROW_MAJOR) ? n : m;

    for (int i = 0; i < b; ++i) {
        fn(LAPACK_ROW_MAJOR, m, n, a_out, lda, tau);
        a_out += static_cast<int64_t>(m) * static_cast<int64_t>(n);
        tau += std::min(m, n);
    }
}

template struct Geqrf<float>;
template struct Geqrf<double>;
template struct Geqrf<std::complex<float>>;
template struct Geqrf<std::complex<double>>;

// Orgqr (Orthogonal Matrix from QR Decomposition)
// ~~~~~

template <typename T> typename Orgqr<T>::FnType *Orgqr<T>::fn = nullptr;

template <typename T> void Orgqr<T>::Kernel(void *out_tuple, void **data, XlaCustomCallStatus *)
{
    const int b = *(reinterpret_cast<int32_t *>(data[0]));
    const int m = *(reinterpret_cast<int32_t *>(data[1]));
    const int n = *(reinterpret_cast<int32_t *>(data[2]));
    const int k = *(reinterpret_cast<int32_t *>(data[3]));
    const T *a_in = reinterpret_cast<T *>(data[4]);
    T *tau = reinterpret_cast<T *>(data[5]);

    T *a_out = reinterpret_cast<T *>(out_tuple);

    if (a_out != a_in) {
        std::memcpy(a_out, a_in,
                    static_cast<int64_t>(b) * static_cast<int64_t>(m) * static_cast<int64_t>(n) *
                        sizeof(T));
    }

    constexpr int corder = LAPACK_ROW_MAJOR;
    const int lda = (corder == LAPACK_ROW_MAJOR) ? n : m;

    for (int i = 0; i < b; ++i) {
        fn(LAPACK_ROW_MAJOR, m, n, k, a_out, lda, tau);
        a_out += static_cast<int64_t>(m) * static_cast<int64_t>(n);
        tau += k;
    }
}

template struct Orgqr<float>;
template struct Orgqr<double>;
template struct Orgqr<std::complex<float>>;
template struct Orgqr<std::complex<double>>;

// Potrf (Cholesky Factorization)
// ~~~~~

template <typename T> typename Potrf<T>::FnType *Potrf<T>::fn = nullptr;

template <typename T> void Potrf<T>::Kernel(void *out_tuple, void **data, XlaCustomCallStatus *)
{
    const uint8_t *uplo_tensor = reinterpret_cast<uint8_t *>(data[0]);
    const char uplo = static_cast<char>(*uplo_tensor);
    const int b = *(reinterpret_cast<int32_t *>(data[1]));
    const int n_row = *(reinterpret_cast<int32_t *>(data[2]));
    const int n_col = *(reinterpret_cast<int32_t *>(data[3]));
    const T *a_in = reinterpret_cast<T *>(data[4]);

    void **out = reinterpret_cast<void **>(out_tuple);
    T *a_out = reinterpret_cast<T *>(out[0]);
    int *info = reinterpret_cast<int *>(out[1]);
    if (a_out != a_in) {
        std::memcpy(a_out, a_in,
                    static_cast<int64_t>(b) * static_cast<int64_t>(n_row) *
                        static_cast<int64_t>(n_col) * sizeof(T));
    }

    constexpr int corder = LAPACK_ROW_MAJOR;

    for (int i = 0; i < b; ++i) {
        *info = fn(corder, uplo, n_col, a_out, n_col);
        a_out += static_cast<int64_t>(n_row) * static_cast<int64_t>(n_col);
        ++info;
    }
}

template struct Potrf<float>;
template struct Potrf<double>;
template struct Potrf<std::complex<float>>;
template struct Potrf<std::complex<double>>;

// Gesdd (Singular Value Decomposition)
// using a divide and conquer method
// ~~~~~

static char GesddJobz(bool job_opt_compute_uv, bool job_opt_full_matrices)
{
    if (!job_opt_compute_uv) {
        return 'N';
    }
    else if (!job_opt_full_matrices) {
        return 'S';
    }
    return 'A';
}

static int Gesdd_ldu(const int order, const char jobz, const int m, const int n)
{
    int ldu = 0;
    if (jobz == 'N') {
        ldu = 1;
    }
    else if (jobz == 'A') {
        ldu = m;
    }
    else if (jobz == 'S') {
        if (m >= n) {
            ldu = (order == LAPACK_ROW_MAJOR) ? n : m;
        }
        else {
            ldu = m;
        }
    }
    return ldu;
}

static int Gesdd_ldvt(const int order, const char jobz, const int m, const int n)
{
    int ldu = 0;
    if (jobz == 'N') {
        ldu = 1;
    }
    else if (jobz == 'A') {
        ldu = n;
    }
    else if (jobz == 'S') {
        if (m >= n) {
            ldu = n;
        }
        else {
            ldu = (order == LAPACK_ROW_MAJOR) ? n : m;
        }
    }
    return ldu;
}

template <typename T> typename RealGesdd<T>::FnType *RealGesdd<T>::fn = nullptr;

template <typename T> void RealGesdd<T>::Kernel(void *out_tuple, void **data, XlaCustomCallStatus *)
{
    const uint8_t *jobz_tensor = reinterpret_cast<uint8_t *>(data[0]);
    const char jobz = static_cast<char>(*jobz_tensor);
    const int b = *(reinterpret_cast<int32_t *>(data[1]));
    const int m = *(reinterpret_cast<int32_t *>(data[2]));
    const int n = *(reinterpret_cast<int32_t *>(data[3]));
    T *a_in = reinterpret_cast<T *>(data[4]);

    void **out = reinterpret_cast<void **>(out_tuple);
    T *a_out = reinterpret_cast<T *>(out[0]);
    T *s = reinterpret_cast<T *>(out[1]);
    T *u = reinterpret_cast<T *>(out[2]);
    T *vt = reinterpret_cast<T *>(out[3]);
    int *info = reinterpret_cast<int *>(out[4]);

    if (a_out != a_in) {
        std::memcpy(a_out, a_in,
                    static_cast<int64_t>(b) * static_cast<int64_t>(m) * static_cast<int64_t>(n) *
                        sizeof(T));
    }

    constexpr int corder = LAPACK_ROW_MAJOR;
    const int lda = (corder == LAPACK_ROW_MAJOR) ? n : m;
    const int ldu = Gesdd_ldu(corder, jobz, m, n);
    const int tdu = ldu;
    const int ldvt = Gesdd_ldvt(corder, jobz, m, n);

    for (int i = 0; i < b; ++i) {
        *info = fn(corder, jobz, m, n, a_out, lda, s, u, ldu, vt, ldvt);
        a_out += static_cast<int64_t>(m) * n;
        s += std::min(m, n);
        u += static_cast<int64_t>(m) * tdu;
        vt += static_cast<int64_t>(ldvt) * n;
        ++info;
    }
}

template <typename T> typename ComplexGesdd<T>::FnType *ComplexGesdd<T>::fn = nullptr;

template <typename T>
void ComplexGesdd<T>::Kernel(void *out_tuple, void **data, XlaCustomCallStatus *)
{
    const uint8_t *jobz_tensor = reinterpret_cast<uint8_t *>(data[0]);
    const char jobz = static_cast<char>(*jobz_tensor);
    const int b = *(reinterpret_cast<int32_t *>(data[1]));
    const int m = *(reinterpret_cast<int32_t *>(data[2]));
    const int n = *(reinterpret_cast<int32_t *>(data[3]));
    T *a_in = reinterpret_cast<T *>(data[4]);

    void **out = reinterpret_cast<void **>(out_tuple);
    T *a_out = reinterpret_cast<T *>(out[0]);
    typename T::value_type *s = reinterpret_cast<typename T::value_type *>(out[1]);
    T *u = reinterpret_cast<T *>(out[2]);
    T *vt = reinterpret_cast<T *>(out[3]);
    int *info = reinterpret_cast<int *>(out[4]);

    if (a_out != a_in) {
        std::memcpy(a_out, a_in,
                    static_cast<int64_t>(b) * static_cast<int64_t>(m) * static_cast<int64_t>(n) *
                        sizeof(T));
    }

    constexpr int corder = LAPACK_ROW_MAJOR;
    const int lda = (corder == LAPACK_ROW_MAJOR) ? n : m;
    const int ldu = Gesdd_ldu(corder, jobz, m, n);
    const int tdu = ldu;
    const int ldvt = Gesdd_ldvt(corder, jobz, m, n);

    for (int i = 0; i < b; ++i) {
        *info = fn(LAPACK_ROW_MAJOR, jobz, m, n, a_out, lda, s, u, ldu, vt, ldvt);
        a_out += static_cast<int64_t>(m) * n;
        s += std::min(m, n);
        u += static_cast<int64_t>(m) * tdu;
        vt += static_cast<int64_t>(ldvt) * n;
        ++info;
    }
}

template struct RealGesdd<float>;
template struct RealGesdd<double>;
template struct ComplexGesdd<std::complex<float>>;
template struct ComplexGesdd<std::complex<double>>;

// Syevd/Heevd (Eigenvalues and eigenvectors for Symmetric Matrices)
// ~~~~~

template <typename T> typename RealSyevd<T>::FnType *RealSyevd<T>::fn = nullptr;

template <typename T> void RealSyevd<T>::Kernel(void *out_tuple, void **data, XlaCustomCallStatus *)
{
    const uint8_t *jobz_tensor = reinterpret_cast<uint8_t *>(data[0]);
    const char jobz = static_cast<char>(*jobz_tensor);
    const uint8_t *uplo_tensor = reinterpret_cast<uint8_t *>(data[1]);
    const char uplo = static_cast<char>(*uplo_tensor);
    const int b = *(reinterpret_cast<int32_t *>(data[2]));
    const int m = *(reinterpret_cast<int32_t *>(data[3]));
    const int n = *(reinterpret_cast<int32_t *>(data[4]));
    const T *a_in = reinterpret_cast<T *>(data[5]);

    void **out = reinterpret_cast<void **>(out_tuple);
    T *a_out = reinterpret_cast<T *>(out[0]);
    T *w_out = reinterpret_cast<T *>(out[1]);
    int *info = reinterpret_cast<int *>(out[2]);

    if (a_out != a_in) {
        std::memcpy(a_out, a_in,
                    static_cast<int64_t>(b) * static_cast<int64_t>(n) * static_cast<int64_t>(n) *
                        sizeof(T));
    }

    constexpr int corder = LAPACK_ROW_MAJOR;

    for (int i = 0; i < b; ++i) {
        *info = fn(corder, jobz, uplo, n, a_out, n, w_out);
        a_out += static_cast<int64_t>(n) * n;
        w_out += n;
        ++info;
    }
}

template <typename T> typename ComplexHeevd<T>::FnType *ComplexHeevd<T>::fn = nullptr;

template <typename T>
void ComplexHeevd<T>::Kernel(void *out_tuple, void **data, XlaCustomCallStatus *)
{
    const uint8_t *jobz_tensor = reinterpret_cast<uint8_t *>(data[0]);
    const char jobz = static_cast<char>(*jobz_tensor);
    const uint8_t *uplo_tensor = reinterpret_cast<uint8_t *>(data[1]);
    const char uplo = static_cast<char>(*uplo_tensor);
    const int b = *(reinterpret_cast<int32_t *>(data[2]));
    const int m = *(reinterpret_cast<int32_t *>(data[3]));
    const int n = *(reinterpret_cast<int32_t *>(data[4]));
    const T *a_in = reinterpret_cast<T *>(data[5]);

    void **out = reinterpret_cast<void **>(out_tuple);
    T *a_out = reinterpret_cast<T *>(out[0]);
    typename T::value_type *w_out = reinterpret_cast<typename T::value_type *>(out[1]);
    int *info = reinterpret_cast<int *>(out[2]);

    if (a_out != a_in) {
        std::memcpy(a_out, a_in,
                    static_cast<int64_t>(b) * static_cast<int64_t>(n) * static_cast<int64_t>(n) *
                        sizeof(T));
    }

    constexpr int corder = LAPACK_ROW_MAJOR;

    for (int i = 0; i < b; ++i) {
        *info = fn(corder, jobz, uplo, n, a_out, n, w_out);
        a_out += static_cast<int64_t>(n) * n;
        w_out += n;
        ++info;
    }
}

template struct RealSyevd<float>;
template struct RealSyevd<double>;
template struct ComplexHeevd<std::complex<float>>;
template struct ComplexHeevd<std::complex<double>>;

// LAPACK uses a packed representation to represent a mixture of real
// eigenvectors and complex conjugate pairs. This helper unpacks the
// representation into regular complex matrices.
template <typename T>
static void UnpackEigenvectors(int n, const T *im_eigenvalues, const T *packed,
                               std::complex<T> *unpacked)
{
    T re, im;
    int j;
    j = 0;
    while (j < n) {
        if (im_eigenvalues[j] == 0. || std::isnan(im_eigenvalues[j])) {
            for (int k = 0; k < n; ++k) {
                unpacked[j * n + k] = {packed[j * n + k], 0.};
            }
            ++j;
        }
        else {
            for (int k = 0; k < n; ++k) {
                re = packed[j * n + k];
                im = packed[(j + 1) * n + k];
                unpacked[j * n + k] = {re, im};
                unpacked[(j + 1) * n + k] = {re, -im};
            }
            j += 2;
        }
    }
}

// Geev (Eigenvalues and eigenvectors for General Matrices)
// ~~~~~

template <typename T> typename RealGeev<T>::FnType *RealGeev<T>::fn = nullptr;

template <typename T> void RealGeev<T>::Kernel(void *out_tuple, void **data, XlaCustomCallStatus *)
{
    const uint8_t *jobvl_tensor = reinterpret_cast<uint8_t *>(data[0]);
    const char jobvl = static_cast<char>(*jobvl_tensor);
    const uint8_t *jobvr_tensor = reinterpret_cast<uint8_t *>(data[1]);
    const char jobvr = static_cast<char>(*jobvr_tensor);
    const int b = *(reinterpret_cast<int32_t *>(data[2]));
    const int n_int = *(reinterpret_cast<int32_t *>(data[4]));
    const int64_t n = n_int;

    const T *a_in = reinterpret_cast<T *>(data[5]);

    void **out = reinterpret_cast<void **>(out_tuple);
    T *wr_out = reinterpret_cast<T *>(out[0]);
    T *wi_out = reinterpret_cast<T *>(out[1]);
    std::complex<T> *vl_out = reinterpret_cast<std::complex<T> *>(out[2]);
    std::complex<T> *vr_out = reinterpret_cast<std::complex<T> *>(out[3]);
    int *info = reinterpret_cast<int *>(out[4]);

    T *a_work = new T[n * n];
    std::memcpy(a_work, a_in, n * n * sizeof(T));
    T *vl_work = new T[n * n];
    T *vr_work = new T[n * n];

    constexpr int corder = LAPACK_ROW_MAJOR;

    // TODO(phawkins): preallocate workspace using XLA.
    *info = fn(corder, jobvl, jobvr, n_int, a_work, n_int, wr_out, wi_out, vl_work, n_int, vr_work,
               n_int);

    auto is_finite = [](T *a_work, int64_t n) {
        for (int64_t j = 0; j < n; ++j) {
            for (int64_t k = 0; k < n; ++k) {
                if (!std::isfinite(a_work[j * n + k])) {
                    return false;
                }
            }
        }
        return true;
    };
    for (int i = 0; i < b; ++i) {
        size_t a_size = n * n * sizeof(T);
        std::memcpy(a_work, a_in, a_size);
        if (is_finite(a_work, n)) {
            *info = fn(corder, jobvl, jobvr, n_int, a_work, n_int, wr_out, wi_out, vl_work, n_int,
                       vr_work, n_int);
#ifdef USE_ABSEIL_LIB
            ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(a_work, a_size);
            ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(wr_out, sizeof(T) * n);
            ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(wi_out, sizeof(T) * n);
            ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(vl_work, sizeof(T) * n * n);
            ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(vr_work, sizeof(T) * n * n);
            ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(info, sizeof(int));
#endif
            if (info[0] == 0) {
                UnpackEigenvectors(n, wi_out, vl_work, vl_out);
                UnpackEigenvectors(n, wi_out, vr_work, vr_out);
            }
        }
        else {
            *info = -4;
        }
        a_in += n * n;
        wr_out += n;
        wi_out += n;
        vl_out += n * n;
        vr_out += n * n;
        ++info;
    }
}

template <typename T> typename ComplexGeev<T>::FnType *ComplexGeev<T>::fn = nullptr;

template <typename T>
void ComplexGeev<T>::Kernel(void *out_tuple, void **data, XlaCustomCallStatus *)
{
    const uint8_t *jobvl_tensor = reinterpret_cast<uint8_t *>(data[0]);
    const char jobvl = static_cast<char>(*jobvl_tensor);
    const uint8_t *jobvr_tensor = reinterpret_cast<uint8_t *>(data[1]);
    const char jobvr = static_cast<char>(*jobvr_tensor);
    const int b = *(reinterpret_cast<int32_t *>(data[2]));
    const int n_int = *(reinterpret_cast<int32_t *>(data[4]));
    const int64_t n = n_int;

    const T *a_in = reinterpret_cast<T *>(data[5]);

    void **out = reinterpret_cast<void **>(out_tuple);
    T *w_out = reinterpret_cast<T *>(out[0]);
    T *vl_out = reinterpret_cast<T *>(out[1]);
    T *vr_out = reinterpret_cast<T *>(out[2]);
    int *info = reinterpret_cast<int *>(out[3]);

    T *a_work = new T[n * n];
    std::memcpy(a_work, a_in, n * n * sizeof(T));

    constexpr int corder = LAPACK_ROW_MAJOR;

    // TODO(phawkins): preallocate workspace using XLA.
    *info = fn(corder, jobvl, jobvr, n_int, a_work, n_int, w_out, vl_out, n_int, vr_out, n_int);

    auto is_finite = [](T *a_work, int64_t n) {
        for (int64_t j = 0; j < n; ++j) {
            for (int64_t k = 0; k < n; ++k) {
                T v = a_work[j * n + k];
                if (!std::isfinite(v.real()) || !std::isfinite(v.imag())) {
                    return false;
                }
            }
        }
        return true;
    };

    for (int i = 0; i < b; ++i) {
        size_t a_size = n * n * sizeof(T);
        std::memcpy(a_work, a_in, a_size);
        if (is_finite(a_work, n)) {
            *info =
                fn(corder, jobvl, jobvr, n_int, a_work, n_int, w_out, vl_out, n_int, vr_out, n_int);
#ifdef USE_ABSEIL_LIB
            ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(a_work, a_size);
            ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(w_out, sizeof(T) * n);
            ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(vl_out, sizeof(T) * n * n);
            ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(vr_out, sizeof(T) * n * n);
            ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(info_out, sizeof(int));
#endif
        }
        else {
            *info = -4;
        }
        a_in += n * n;
        w_out += n;
        vl_out += n * n;
        vr_out += n * n;
        info += 1;
    }
}

template struct RealGeev<float>;
template struct RealGeev<double>;
template struct ComplexGeev<std::complex<float>>;
template struct ComplexGeev<std::complex<double>>;

// Gees (Schur Decomposition)
// ~~~~~
template <typename T> typename RealGees<T>::FnType *RealGees<T>::fn = nullptr;

template <typename T> void RealGees<T>::Kernel(void *out_tuple, void **data, XlaCustomCallStatus *)
{
    const uint8_t *jobvs_tensor = reinterpret_cast<uint8_t *>(data[0]);
    const char jobvs = static_cast<char>(*jobvs_tensor);
    const uint8_t *sort_tensor = reinterpret_cast<uint8_t *>(data[1]);
    const char sort = static_cast<char>(*sort_tensor);
    const int b = *(reinterpret_cast<int32_t *>(data[2]));
    const int n_int = *(reinterpret_cast<int32_t *>(data[4]));
    const int64_t n = n_int;
    const T *a_in = reinterpret_cast<T *>(data[5]);

    bool (*select)(T, T) = nullptr;

    void **out = reinterpret_cast<void **>(out_tuple);
    T *a_out = reinterpret_cast<T *>(out[0]);
    T *vs_out = reinterpret_cast<T *>(out[1]);
    T *wr_out = reinterpret_cast<T *>(out[2]);
    T *wi_out = reinterpret_cast<T *>(out[3]);
    int *sdim_out = reinterpret_cast<int *>(out[4]);
    int *info = reinterpret_cast<int *>(out[5]);

    constexpr int corder = LAPACK_ROW_MAJOR;

    *info = fn(corder, jobvs, sort, select, n_int, a_out, n_int, sdim_out, wr_out, wi_out, vs_out,
               n_int);

    size_t a_size = static_cast<int64_t>(n) * static_cast<int64_t>(n) * sizeof(T);
    if (a_out != a_in) {
        std::memcpy(a_out, a_in, static_cast<int64_t>(b) * a_size);
    }

    for (int i = 0; i < b; ++i) {
        *info = fn(corder, jobvs, sort, select, n_int, a_out, n_int, sdim_out, wr_out, wi_out,
                   vs_out, n_int);
#ifdef USE_ABSEIL_LIB
        ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(a_out, a_size);
        ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(sdim_out, sizeof(int));
        ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(wr_out, sizeof(T) * n);
        ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(wi_out, sizeof(T) * n);
        ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(vs_out, sizeof(T) * n * n);
        ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(info, sizeof(int));
#endif

        a_in += n * n;
        a_out += n * n;
        wr_out += n;
        wi_out += n;
        vs_out += n * n;
        ++info;
        ++sdim_out;
    }
}

template <typename T> typename ComplexGees<T>::FnType *ComplexGees<T>::fn = nullptr;

template <typename T>
void ComplexGees<T>::Kernel(void *out_tuple, void **data, XlaCustomCallStatus *)
{
    const uint8_t *jobvs_tensor = reinterpret_cast<uint8_t *>(data[0]);
    const char jobvs = static_cast<char>(*jobvs_tensor);
    const uint8_t *sort_tensor = reinterpret_cast<uint8_t *>(data[1]);
    const char sort = static_cast<char>(*sort_tensor);
    const int b = *(reinterpret_cast<int32_t *>(data[2]));
    const int n_int = *(reinterpret_cast<int32_t *>(data[4]));
    const int64_t n = n_int;

    const T *a_in = reinterpret_cast<T *>(data[5]);

    // bool* select (T, T) = reinterpret_cast<bool* (T, T)>(data[5]);
    bool (*select)(T) = nullptr;

    void **out = reinterpret_cast<void **>(out_tuple);
    T *a_out = reinterpret_cast<T *>(out[0]);
    T *vs_out = reinterpret_cast<T *>(out[1]);
    T *w_out = reinterpret_cast<T *>(out[2]);
    int *sdim_out = reinterpret_cast<int *>(out[3]);
    int *info = reinterpret_cast<int *>(out[4]);

    constexpr int corder = LAPACK_ROW_MAJOR;

    *info = fn(corder, jobvs, sort, select, n_int, a_out, n_int, sdim_out, w_out, vs_out, n_int);

    if (a_out != a_in) {
        std::memcpy(a_out, a_in,
                    static_cast<int64_t>(b) * static_cast<int64_t>(n) * static_cast<int64_t>(n) *
                        sizeof(T));
    }

    for (int i = 0; i < b; ++i) {
        *info =
            fn(corder, jobvs, sort, select, n_int, a_out, n_int, sdim_out, w_out, vs_out, n_int);
#ifdef USE_ABSEIL_LIB
        ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(w_out, sizeof(T) * n);
        ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(vs_out, sizeof(T) * n * n);
        ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(info_out, sizeof(int));
        ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(sdim_out, sizeof(int));
#endif

        a_in += n * n;
        a_out += n * n;
        w_out += n;
        vs_out += n * n;
        ++info;
        ++sdim_out;
    }
}

template struct RealGees<float>;
template struct RealGees<double>;
template struct ComplexGees<std::complex<float>>;
template struct ComplexGees<std::complex<double>>;

// Gehrd (Hessenberg Decomposition)
// ~~~~~
template <typename T> typename Gehrd<T>::FnType *Gehrd<T>::fn = nullptr;

template <typename T> void Gehrd<T>::Kernel(void *out_tuple, void **data, XlaCustomCallStatus *)
{
    const int32_t ihi = *reinterpret_cast<int32_t *>(data[0]);
    const int32_t ilo = *reinterpret_cast<int32_t *>(data[1]);
    const int32_t batch = *reinterpret_cast<int32_t *>(data[2]);
    const int32_t lda = *reinterpret_cast<int32_t *>(data[3]);
    const int32_t n = *reinterpret_cast<int32_t *>(data[4]);
    T *a = reinterpret_cast<T *>(data[5]);

    void **out = reinterpret_cast<void **>(out_tuple);
    T *a_out = reinterpret_cast<T *>(out[0]);
    T *tau = reinterpret_cast<T *>(out[1]);
    int *info = reinterpret_cast<int *>(out[2]);

    if (a_out != a) {
        std::memcpy(a_out, a,
                    static_cast<int64_t>(batch) * static_cast<int64_t>(n) *
                        static_cast<int64_t>(n) * sizeof(T));
    }

    const int64_t a_plus = static_cast<int64_t>(lda) * static_cast<int64_t>(n);

    constexpr int corder = LAPACK_ROW_MAJOR;

    for (int i = 0; i < batch; ++i) {
        *info = fn(corder, n, ilo, ihi, a_out, lda, tau);
        a_out += a_plus;
        tau += n - 1;
        ++info;
    }
}

template struct Gehrd<float>;
template struct Gehrd<double>;
template struct Gehrd<std::complex<float>>;
template struct Gehrd<std::complex<double>>;

// Sytrd
// ~~~~~

template <typename T> typename Sytrd<T>::FnType *Sytrd<T>::fn = nullptr;

template <typename T> void Sytrd<T>::Kernel(void *out_tuple, void **data, XlaCustomCallStatus *)
{
    const uint8_t *uplo_tensor = reinterpret_cast<uint8_t *>(data[0]);
    const char cuplo = static_cast<char>(*uplo_tensor);
    const int32_t batch = *reinterpret_cast<int32_t *>(data[1]);
    const int32_t lda = *reinterpret_cast<int32_t *>(data[2]);
    const int32_t n = *reinterpret_cast<int32_t *>(data[3]);

    T *a = reinterpret_cast<T *>(data[4]);

    void **out = reinterpret_cast<void **>(out_tuple);
    T *a_out = reinterpret_cast<T *>(out[0]);
    typedef typename real_type<T>::type Real;
    Real *d = reinterpret_cast<Real *>(out[1]);
    Real *e = reinterpret_cast<Real *>(out[2]);
    T *tau = reinterpret_cast<T *>(out[3]);
    int *info = reinterpret_cast<int *>(out[4]);

    if (a_out != a) {
        std::memcpy(a_out, a,
                    static_cast<int64_t>(batch) * static_cast<int64_t>(n) *
                        static_cast<int64_t>(n) * sizeof(T));
    }

    constexpr int corder = LAPACK_ROW_MAJOR;

    const int64_t a_plus = static_cast<int64_t>(lda) * static_cast<int64_t>(n);

    for (int i = 0; i < batch; ++i) {
        *info = fn(corder, cuplo, n, a_out, lda, d, e, tau);
        a_out += a_plus;
        d += n;
        e += n - 1;
        tau += n - 1;
        ++info;
    }
}

template struct Sytrd<float>;
template struct Sytrd<double>;
template struct Sytrd<std::complex<float>>;
template struct Sytrd<std::complex<double>>;

} // namespace jax
