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

#include "jax_cpu_lapack_kernels/lapack_kernels.hpp"

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

DEFINE_LAPACK_FUNC(lapack_strsm_ffi, 12, 1, jax::RealTrsm<float>)
DEFINE_LAPACK_FUNC(lapack_dtrsm_ffi, 12, 1, jax::RealTrsm<double>)
DEFINE_LAPACK_FUNC(lapack_ctrsm_ffi, 12, 1, jax::ComplexTrsm<std::complex<float>>)
DEFINE_LAPACK_FUNC(lapack_ztrsm_ffi, 12, 1, jax::ComplexTrsm<std::complex<double>>)

DEFINE_LAPACK_FUNC(lapack_sgetrf_ffi, 4, 3, jax::Getrf<float>)
DEFINE_LAPACK_FUNC(lapack_dgetrf_ffi, 4, 3, jax::Getrf<double>)
DEFINE_LAPACK_FUNC(lapack_cgetrf_ffi, 4, 3, jax::Getrf<std::complex<float>>)
DEFINE_LAPACK_FUNC(lapack_zgetrf_ffi, 4, 3, jax::Getrf<std::complex<double>>)

DEFINE_LAPACK_FUNC(lapack_sgeqrf_ffi, 4, 2, jax::Geqrf<float>)
DEFINE_LAPACK_FUNC(lapack_dgeqrf_ffi, 4, 2, jax::Geqrf<double>)
DEFINE_LAPACK_FUNC(lapack_cgeqrf_ffi, 4, 2, jax::Geqrf<std::complex<float>>)
DEFINE_LAPACK_FUNC(lapack_zgeqrf_ffi, 4, 2, jax::Geqrf<std::complex<double>>)

DEFINE_LAPACK_FUNC(lapack_sorgqr_ffi, 6, 1, jax::Orgqr<float>)
DEFINE_LAPACK_FUNC(lapack_dorgqr_ffi, 6, 1, jax::Orgqr<double>)
DEFINE_LAPACK_FUNC(lapack_cungqr_ffi, 6, 1, jax::Orgqr<std::complex<float>>)
DEFINE_LAPACK_FUNC(lapack_zungqr_ffi, 6, 1, jax::Orgqr<std::complex<double>>)

DEFINE_LAPACK_FUNC(lapack_spotrf_ffi, 5, 2, jax::Potrf<float>)
DEFINE_LAPACK_FUNC(lapack_dpotrf_ffi, 5, 2, jax::Potrf<double>)
DEFINE_LAPACK_FUNC(lapack_cpotrf_ffi, 5, 2, jax::Potrf<std::complex<float>>)
DEFINE_LAPACK_FUNC(lapack_zpotrf_ffi, 5, 2, jax::Potrf<std::complex<double>>)

DEFINE_LAPACK_FUNC(lapack_sgesdd_ffi, 5, 5, jax::RealGesdd<float>)
DEFINE_LAPACK_FUNC(lapack_dgesdd_ffi, 5, 5, jax::RealGesdd<double>)
DEFINE_LAPACK_FUNC(lapack_cgesdd_ffi, 5, 5, jax::ComplexGesdd<std::complex<float>>)
DEFINE_LAPACK_FUNC(lapack_zgesdd_ffi, 5, 5, jax::ComplexGesdd<std::complex<double>>)

DEFINE_LAPACK_FUNC(lapack_ssyevd_ffi, 6, 3, jax::RealSyevd<float>)
DEFINE_LAPACK_FUNC(lapack_dsyevd_ffi, 6, 3, jax::RealSyevd<double>)
DEFINE_LAPACK_FUNC(lapack_cheevd_ffi, 6, 3, jax::ComplexHeevd<std::complex<float>>)
DEFINE_LAPACK_FUNC(lapack_zheevd_ffi, 6, 3, jax::ComplexHeevd<std::complex<double>>)

DEFINE_LAPACK_FUNC(lapack_sgeev_ffi, 6, 5, jax::RealGeev<float>)
DEFINE_LAPACK_FUNC(lapack_dgeev_ffi, 6, 5, jax::RealGeev<double>)
DEFINE_LAPACK_FUNC(lapack_cgeev_ffi, 6, 4, jax::ComplexGeev<std::complex<float>>)
DEFINE_LAPACK_FUNC(lapack_zgeev_ffi, 6, 4, jax::ComplexGeev<std::complex<double>>)

DEFINE_LAPACK_FUNC(lapack_sgees_ffi, 6, 6, jax::RealGees<float>)
DEFINE_LAPACK_FUNC(lapack_dgees_ffi, 6, 6, jax::RealGees<double>)
DEFINE_LAPACK_FUNC(lapack_cgees_ffi, 6, 5, jax::ComplexGees<std::complex<float>>)
DEFINE_LAPACK_FUNC(lapack_zgees_ffi, 6, 5, jax::ComplexGees<std::complex<double>>)

DEFINE_LAPACK_FUNC(lapack_sgehrd_ffi, 6, 3, jax::Gehrd<float>)
DEFINE_LAPACK_FUNC(lapack_dgehrd_ffi, 6, 3, jax::Gehrd<double>)
DEFINE_LAPACK_FUNC(lapack_cgehrd_ffi, 6, 3, jax::Gehrd<std::complex<float>>)
DEFINE_LAPACK_FUNC(lapack_zgehrd_ffi, 6, 3, jax::Gehrd<std::complex<double>>)

DEFINE_LAPACK_FUNC(lapack_ssytrd_ffi, 5, 5, jax::Sytrd<float>)
DEFINE_LAPACK_FUNC(lapack_dsytrd_ffi, 5, 5, jax::Sytrd<double>)
DEFINE_LAPACK_FUNC(lapack_chetrd_ffi, 5, 5, jax::Sytrd<std::complex<float>>)
DEFINE_LAPACK_FUNC(lapack_zhetrd_ffi, 5, 5, jax::Sytrd<std::complex<double>>)
