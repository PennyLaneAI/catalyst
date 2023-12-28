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
#include <cstdint>
#include <cstring>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

namespace {

typedef int lapack_int;

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

// Wrapper to call the SVD solver dgesdd_ from Lapack:
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
}
