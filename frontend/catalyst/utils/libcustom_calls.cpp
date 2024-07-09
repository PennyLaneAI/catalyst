// Copyright 2023 Xanadu Quantum Technologies Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace {

typedef int lapack_int;
typedef std::complex<double> dComplex;

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

} // namespace

extern "C" {

// MemRef type
struct EncodedMemref {
    int64_t rank;
    void *data_aligned;
    int8_t dtype;
};

void dgesdd_(char *jobz, lapack_int *m, lapack_int *n, double *a, lapack_int *lda, double *s,
             double *u, lapack_int *ldu, double *vt, lapack_int *ldvt, double *work,
             lapack_int *lwork, lapack_int *iwork, lapack_int *info);

void dsyevd_(char *jobz, char *uplo, lapack_int *n, double *a, int *lda, double *w, double *work,
             lapack_int *lwork, lapack_int *iwork, lapack_int *liwork, lapack_int *info);

void dtrsm_(char *side, char *uplo, char *transa, char *diag, lapack_int *m, lapack_int *n,
            double *alpha, double *a, lapack_int *lda, double *b, lapack_int *ldb);

void ztrsm_(char *side, char *uplo, char *transa, char *diag, lapack_int *m, lapack_int *n,
            dComplex *alpha, dComplex *a, lapack_int *lda, dComplex *b, lapack_int *ldb);

void dgetrf_(lapack_int *m, lapack_int *n, double *a, lapack_int *lda, lapack_int *ipiv,
             lapack_int *info);

void zgetrf_(lapack_int *m, lapack_int *n, dComplex *a, lapack_int *lda, lapack_int *ipiv,
             lapack_int *info);

// Wrapper to call various blas core routine. Currently includes:
// - the SVD solver `dgesdd_`
// - the eigen vectors/values computation `dsyevd_`
// - the double (complex) triangular matrix equation solver `dtrsm_` (`ztrsm_`)
// - the double (complex) LU factorization `dgetrf_` (`zgetrf_`)
// from Lapack:
// https://github.com/google/jax/blob/main/jaxlib/cpu/lapack_kernels.cc released under the Apache
// License, Version 2.0, with the following copyright notice:

// Copyright 2021 The JAX Authors.
void lapack_dgesdd(void **dataEncoded, void **resultsEncoded)
{
    std::vector<void *> data;
    for (size_t i = 0; i < 7; ++i) {
        auto encodedMemref = *(reinterpret_cast<EncodedMemref *>(dataEncoded[i]));
        data.push_back(encodedMemref.data_aligned);
    }

    std::vector<void *> out;
    for (size_t i = 0; i < 7; ++i) {
        auto encodedMemref = *(reinterpret_cast<EncodedMemref *>(resultsEncoded[i]));
        out.push_back(encodedMemref.data_aligned);
    }

    int32_t job_opt_full_matrices = *(reinterpret_cast<int32_t *>(data[0]));
    int32_t job_opt_compute_uv = *(reinterpret_cast<int32_t *>(data[1]));
    int b = *(reinterpret_cast<int32_t *>(data[2]));
    int m = *(reinterpret_cast<int32_t *>(data[3]));
    int n = *(reinterpret_cast<int32_t *>(data[4]));
    int lwork = *(reinterpret_cast<int32_t *>(data[5]));
    double *a_in = reinterpret_cast<double *>(data[6]);

    double *a_out = reinterpret_cast<double *>(out[0]);
    double *s = reinterpret_cast<double *>(out[1]);
    // U and vt are switched to produce the right results...
    double *vt = reinterpret_cast<double *>(out[2]);
    double *u = reinterpret_cast<double *>(out[3]);

    int *info = reinterpret_cast<int *>(out[4]);
    int *iwork = reinterpret_cast<int *>(out[5]);
    double *work = reinterpret_cast<double *>(out[6]);

    if (a_out != a_in) {
        std::memcpy(a_out, a_in,
                    static_cast<int64_t>(b) * static_cast<int64_t>(m) * static_cast<int64_t>(n) *
                        sizeof(double));
    }

    char jobz = GesddJobz(job_opt_compute_uv, job_opt_full_matrices);

    int lda = m;
    int ldu = m;
    int tdu = job_opt_full_matrices ? m : std::min(m, n);
    int ldvt = job_opt_full_matrices ? n : std::min(m, n);

    for (int i = 0; i < b; ++i) {
        dgesdd_(&jobz, &m, &n, a_out, &lda, s, u, &ldu, vt, &ldvt, work, &lwork, iwork, info);
        a_out += static_cast<int64_t>(m) * n;
        s += std::min(m, n);
        u += static_cast<int64_t>(m) * tdu;
        vt += static_cast<int64_t>(ldvt) * n;
        ++info;
    }
}

// Copyright 2021 The JAX Authors.
void lapack_dsyevd(void **dataEncoded, void **resultsEncoded)
{
    std::vector<void *> data;
    for (size_t i = 0; i < 4; ++i) {
        auto encodedMemref = *(reinterpret_cast<EncodedMemref *>(dataEncoded[i]));
        data.push_back(encodedMemref.data_aligned);
    }

    std::vector<void *> out;
    for (size_t i = 0; i < 5; ++i) {
        auto encodedMemref = *(reinterpret_cast<EncodedMemref *>(resultsEncoded[i]));
        out.push_back(encodedMemref.data_aligned);
    }

    int32_t lower = *(reinterpret_cast<int32_t *>(data[0]));
    int b = *(reinterpret_cast<int32_t *>(data[1]));
    int n = *(reinterpret_cast<int32_t *>(data[2]));
    const double *a_in = reinterpret_cast<double *>(data[3]);

    double *a_out = reinterpret_cast<double *>(out[0]);
    double *w_out = reinterpret_cast<double *>(out[1]);
    int *info_out = reinterpret_cast<int *>(out[2]);
    double *work = reinterpret_cast<double *>(out[3]);
    int *iwork = reinterpret_cast<int *>(out[4]);
    if (a_out != a_in) {
        std::memcpy(a_out, a_in,
                    static_cast<int64_t>(b) * static_cast<int64_t>(n) * static_cast<int64_t>(n) *
                        sizeof(double));
    }

    char jobz = 'V';
    char uplo = lower ? 'L' : 'U';

    lapack_int lwork =
        std::min<int64_t>(std::numeric_limits<lapack_int>::max(), 1 + 6 * n + 2 * n * n);
    lapack_int liwork = std::min<int64_t>(std::numeric_limits<lapack_int>::max(), 3 + 5 * n);
    for (int i = 0; i < b; ++i) {
        dsyevd_(&jobz, &uplo, &n, a_out, &n, w_out, work, &lwork, iwork, &liwork, info_out);
        a_out += static_cast<int64_t>(n) * n;
        w_out += n;
        ++info_out;
    }
}

// Copyright 2021 The JAX Authors.
void blas_dtrsm(void **dataEncoded, void **resultsEncoded)
{
    std::vector<void *> data;
    for (size_t i = 0; i < 10; ++i) {
        auto encodedMemref = *(reinterpret_cast<EncodedMemref *>(dataEncoded[i]));
        data.push_back(encodedMemref.data_aligned);
    }

    std::vector<void *> out;
    for (size_t i = 0; i < 1; ++i) {
        auto encodedMemref = *(reinterpret_cast<EncodedMemref *>(resultsEncoded[i]));
        out.push_back(encodedMemref.data_aligned);
    }

    int32_t left_side = *reinterpret_cast<int32_t *>(data[0]);
    int32_t lower = *reinterpret_cast<int32_t *>(data[1]);
    int32_t trans_a = *reinterpret_cast<int32_t *>(data[2]);
    int32_t diag = *reinterpret_cast<int32_t *>(data[3]);
    int m = *reinterpret_cast<int32_t *>(data[4]);
    int n = *reinterpret_cast<int32_t *>(data[5]);
    int batch = *reinterpret_cast<int32_t *>(data[6]);
    double *alpha = reinterpret_cast<double *>(data[7]);
    double *a = reinterpret_cast<double *>(data[8]);
    double *b = reinterpret_cast<double *>(data[9]);

    double *x = reinterpret_cast<double *>(out[0]);
    if (x != b) {
        std::memcpy(x, b,
                    static_cast<int64_t>(batch) * static_cast<int64_t>(m) *
                        static_cast<int64_t>(n) * sizeof(double));
    }

    char cside = left_side ? 'L' : 'R';
    char cuplo = lower ? 'L' : 'U';
    char ctransa = 'N';
    if (trans_a == 1) {
        ctransa = 'T';
    }
    else if (trans_a == 2) {
        ctransa = 'C';
    }
    char cdiag = diag ? 'U' : 'N';
    int lda = left_side ? m : n;
    int ldb = m;

    int64_t x_plus = static_cast<int64_t>(m) * static_cast<int64_t>(n);
    int64_t a_plus = static_cast<int64_t>(lda) * static_cast<int64_t>(lda);

    for (int i = 0; i < batch; ++i) {
        dtrsm_(&cside, &cuplo, &ctransa, &cdiag, &m, &n, alpha, a, &lda, x, &ldb);
        x += x_plus;
        a += a_plus;
    }
}

// Copyright 2021 The JAX Authors.
void blas_ztrsm(void **dataEncoded, void **resultsEncoded)
{
    std::vector<void *> data;
    for (size_t i = 0; i < 10; ++i) {
        auto encodedMemref = *(reinterpret_cast<EncodedMemref *>(dataEncoded[i]));
        data.push_back(encodedMemref.data_aligned);
    }

    std::vector<void *> out;
    for (size_t i = 0; i < 1; ++i) {
        auto encodedMemref = *(reinterpret_cast<EncodedMemref *>(resultsEncoded[i]));
        out.push_back(encodedMemref.data_aligned);
    }

    int32_t left_side = *reinterpret_cast<int32_t *>(data[0]);
    int32_t lower = *reinterpret_cast<int32_t *>(data[1]);
    int32_t trans_a = *reinterpret_cast<int32_t *>(data[2]);
    int32_t diag = *reinterpret_cast<int32_t *>(data[3]);
    int m = *reinterpret_cast<int32_t *>(data[4]);
    int n = *reinterpret_cast<int32_t *>(data[5]);
    int batch = *reinterpret_cast<int32_t *>(data[6]);
    dComplex *alpha = reinterpret_cast<dComplex *>(data[7]);
    dComplex *a = reinterpret_cast<dComplex *>(data[8]);
    dComplex *b = reinterpret_cast<dComplex *>(data[9]);

    dComplex *x = reinterpret_cast<dComplex *>(out[0]);
    if (x != b) {
        std::memcpy(x, b,
                    static_cast<int64_t>(batch) * static_cast<int64_t>(m) *
                        static_cast<int64_t>(n) * sizeof(dComplex));
    }

    char cside = left_side ? 'L' : 'R';
    char cuplo = lower ? 'L' : 'U';
    char ctransa = 'N';
    if (trans_a == 1) {
        ctransa = 'T';
    }
    else if (trans_a == 2) {
        ctransa = 'C';
    }
    char cdiag = diag ? 'U' : 'N';
    int lda = left_side ? m : n;
    int ldb = m;

    int64_t x_plus = static_cast<int64_t>(m) * static_cast<int64_t>(n);
    int64_t a_plus = static_cast<int64_t>(lda) * static_cast<int64_t>(lda);

    for (int i = 0; i < batch; ++i) {
        ztrsm_(&cside, &cuplo, &ctransa, &cdiag, &m, &n, alpha, a, &lda, x, &ldb);
        x += x_plus;
        a += a_plus;
    }
}

// Copyright 2021 The JAX Authors.
void lapack_dgetrf(void **dataEncoded, void **resultsEncoded)
{
    std::vector<void *> data;
    for (size_t i = 0; i < 4; ++i) {
        auto encodedMemref = *(reinterpret_cast<EncodedMemref *>(dataEncoded[i]));
        data.push_back(encodedMemref.data_aligned);
    }

    std::vector<void *> out;
    for (size_t i = 0; i < 3; ++i) {
        auto encodedMemref = *(reinterpret_cast<EncodedMemref *>(resultsEncoded[i]));
        out.push_back(encodedMemref.data_aligned);
    }

    int b = *(reinterpret_cast<int32_t *>(data[0]));
    int m = *(reinterpret_cast<int32_t *>(data[1]));
    int n = *(reinterpret_cast<int32_t *>(data[2]));
    const double *a_in = reinterpret_cast<double *>(data[3]);

    double *a_out = reinterpret_cast<double *>(out[0]);
    int *ipiv = reinterpret_cast<int *>(out[1]);
    int *info = reinterpret_cast<int *>(out[2]);
    if (a_out != a_in) {
        std::memcpy(a_out, a_in,
                    static_cast<int64_t>(b) * static_cast<int64_t>(m) * static_cast<int64_t>(n) *
                        sizeof(double));
    }
    for (int i = 0; i < b; ++i) {
        dgetrf_(&m, &n, a_out, &m, ipiv, info);
        a_out += static_cast<int64_t>(m) * static_cast<int64_t>(n);
        ipiv += std::min(m, n);
        ++info;
    }
}

// Copyright 2021 The JAX Authors.
void lapack_zgetrf(void **dataEncoded, void **resultsEncoded)
{
    std::vector<void *> data;
    for (size_t i = 0; i < 4; ++i) {
        auto encodedMemref = *(reinterpret_cast<EncodedMemref *>(dataEncoded[i]));
        data.push_back(encodedMemref.data_aligned);
    }

    std::vector<void *> out;
    for (size_t i = 0; i < 3; ++i) {
        auto encodedMemref = *(reinterpret_cast<EncodedMemref *>(resultsEncoded[i]));
        out.push_back(encodedMemref.data_aligned);
    }

    int b = *(reinterpret_cast<int32_t *>(data[0]));
    int m = *(reinterpret_cast<int32_t *>(data[1]));
    int n = *(reinterpret_cast<int32_t *>(data[2]));
    const dComplex *a_in = reinterpret_cast<dComplex *>(data[3]);

    dComplex *a_out = reinterpret_cast<dComplex *>(out[0]);
    int *ipiv = reinterpret_cast<int *>(out[1]);
    int *info = reinterpret_cast<int *>(out[2]);
    if (a_out != a_in) {
        std::memcpy(a_out, a_in,
                    static_cast<int64_t>(b) * static_cast<int64_t>(m) * static_cast<int64_t>(n) *
                        sizeof(dComplex));
    }
    for (int i = 0; i < b; ++i) {
        zgetrf_(&m, &n, a_out, &m, ipiv, info);
        a_out += static_cast<int64_t>(m) * static_cast<int64_t>(n);
        ipiv += std::min(m, n);
        ++info;
    }
}
}
