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

#include <complex>

#include "jax_cpu_lapack_kernels/lapack_kernels.h"

#ifdef DEBUG
#include <iostream>
#define DEBUG_MSG(str) std::cout << "DEBUG: " << str << std::endl;
#else
#define DEBUG_MSG(str) // No operation
#endif

// MemRef type
struct EncodedMemref {
    int64_t rank;
    void *data_aligned;
    int8_t dtype;
};

#define DEFINE_LAPACK_FUNC(FUNC_NAME, DATA_SIZE, OUT_SIZE, KERNEL)                                 \
    extern "C" {                                                                                   \
    void FUNC_NAME(void **dataEncoded, void **resultsEncoded)                                      \
    {                                                                                              \
        DEBUG_MSG(#FUNC_NAME);                                                                     \
        void *data[DATA_SIZE];                                                                     \
        for (size_t i = 0; i < DATA_SIZE; ++i) {                                                   \
            auto encodedMemref = *(reinterpret_cast<EncodedMemref *>(dataEncoded[i]));             \
            data[i] = encodedMemref.data_aligned;                                                  \
        }                                                                                          \
                                                                                                   \
        if (OUT_SIZE > 1) {                                                                        \
            void *out[OUT_SIZE];                                                                   \
            for (size_t i = 0; i < OUT_SIZE; ++i) {                                                \
                auto encodedMemref = *(reinterpret_cast<EncodedMemref *>(resultsEncoded[i]));      \
                out[i] = encodedMemref.data_aligned;                                               \
            }                                                                                      \
            KERNEL::Kernel(out, data, nullptr);                                                    \
        }                                                                                          \
        else {                                                                                     \
            auto encodedMemref = *(reinterpret_cast<EncodedMemref *>(resultsEncoded[0]));          \
            KERNEL::Kernel(encodedMemref.data_aligned, data, nullptr);                             \
        }                                                                                          \
    }                                                                                              \
    }

DEFINE_LAPACK_FUNC(blas_strsm, 10, 1, jax::RealTrsm<float>)
DEFINE_LAPACK_FUNC(blas_dtrsm, 10, 1, jax::RealTrsm<double>)
DEFINE_LAPACK_FUNC(blas_ctrsm, 10, 1, jax::ComplexTrsm<std::complex<float>>)
DEFINE_LAPACK_FUNC(blas_ztrsm, 10, 1, jax::ComplexTrsm<std::complex<double>>)

DEFINE_LAPACK_FUNC(lapack_sgetrf, 4, 3, jax::Getrf<float>)
DEFINE_LAPACK_FUNC(lapack_dgetrf, 4, 3, jax::Getrf<double>)
DEFINE_LAPACK_FUNC(lapack_cgetrf, 4, 3, jax::Getrf<std::complex<float>>)
DEFINE_LAPACK_FUNC(lapack_zgetrf, 4, 3, jax::Getrf<std::complex<double>>)

DEFINE_LAPACK_FUNC(lapack_sgeqrf, 5, 4, jax::Geqrf<float>)
DEFINE_LAPACK_FUNC(lapack_dgeqrf, 5, 4, jax::Geqrf<double>)
DEFINE_LAPACK_FUNC(lapack_cgeqrf, 5, 4, jax::Geqrf<std::complex<float>>)
DEFINE_LAPACK_FUNC(lapack_zgeqrf, 5, 4, jax::Geqrf<std::complex<double>>)

DEFINE_LAPACK_FUNC(lapack_sorgqr, 7, 3, jax::Orgqr<float>)
DEFINE_LAPACK_FUNC(lapack_dorgqr, 7, 3, jax::Orgqr<double>)
DEFINE_LAPACK_FUNC(lapack_cungqr, 7, 3, jax::Orgqr<std::complex<float>>)
DEFINE_LAPACK_FUNC(lapack_zungqr, 7, 3, jax::Orgqr<std::complex<double>>)

DEFINE_LAPACK_FUNC(lapack_spotrf, 4, 2, jax::Potrf<float>)
DEFINE_LAPACK_FUNC(lapack_dpotrf, 4, 2, jax::Potrf<double>)
DEFINE_LAPACK_FUNC(lapack_cpotrf, 4, 2, jax::Potrf<std::complex<float>>)
DEFINE_LAPACK_FUNC(lapack_zpotrf, 4, 2, jax::Potrf<std::complex<double>>)

DEFINE_LAPACK_FUNC(lapack_sgesdd, 7, 7, jax::RealGesdd<float>)
DEFINE_LAPACK_FUNC(lapack_dgesdd, 7, 7, jax::RealGesdd<double>)
DEFINE_LAPACK_FUNC(lapack_cgesdd, 7, 8, jax::ComplexGesdd<std::complex<float>>)
DEFINE_LAPACK_FUNC(lapack_zgesdd, 7, 8, jax::ComplexGesdd<std::complex<double>>)

DEFINE_LAPACK_FUNC(lapack_ssyevd, 4, 5, jax::RealSyevd<float>)
DEFINE_LAPACK_FUNC(lapack_dsyevd, 4, 5, jax::RealSyevd<double>)
DEFINE_LAPACK_FUNC(lapack_cheevd, 4, 6, jax::ComplexHeevd<std::complex<float>>)
DEFINE_LAPACK_FUNC(lapack_zheevd, 4, 6, jax::ComplexHeevd<std::complex<double>>)

DEFINE_LAPACK_FUNC(lapack_sgeev, 5, 6, jax::RealGeev<float>)
DEFINE_LAPACK_FUNC(lapack_dgeev, 5, 6, jax::RealGeev<double>)
DEFINE_LAPACK_FUNC(lapack_cgeev, 5, 6, jax::ComplexGeev<std::complex<float>>)
DEFINE_LAPACK_FUNC(lapack_zgeev, 5, 6, jax::ComplexGeev<std::complex<double>>)

DEFINE_LAPACK_FUNC(lapack_sgees, 5, 6, jax::RealGees<float>)
DEFINE_LAPACK_FUNC(lapack_dgees, 5, 6, jax::RealGees<double>)
DEFINE_LAPACK_FUNC(lapack_cgees, 5, 6, jax::ComplexGees<std::complex<float>>)
DEFINE_LAPACK_FUNC(lapack_zgees, 5, 6, jax::ComplexGees<std::complex<double>>)

DEFINE_LAPACK_FUNC(lapack_sgehrd, 7, 4, jax::Gehrd<float>)
DEFINE_LAPACK_FUNC(lapack_dgehrd, 7, 4, jax::Gehrd<double>)
DEFINE_LAPACK_FUNC(lapack_cgehrd, 7, 4, jax::Gehrd<std::complex<float>>)
DEFINE_LAPACK_FUNC(lapack_zgehrd, 7, 4, jax::Gehrd<std::complex<double>>)

DEFINE_LAPACK_FUNC(lapack_ssytrd, 6, 6, jax::Sytrd<float>)
DEFINE_LAPACK_FUNC(lapack_dsytrd, 6, 6, jax::Sytrd<double>)
DEFINE_LAPACK_FUNC(lapack_chetrd, 6, 6, jax::Sytrd<std::complex<float>>)
DEFINE_LAPACK_FUNC(lapack_zhetrd, 6, 6, jax::Sytrd<std::complex<double>>)
