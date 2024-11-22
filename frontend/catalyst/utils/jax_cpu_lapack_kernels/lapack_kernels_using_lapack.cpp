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
 *   https://github.com/google/jax/blob/jaxlib-v0.4.28/jaxlib/cpu/lapack_kernels_using_lapack.cc
 *
 * from jaxlib-v0.4.28.
 *
 * See note in lapack_kernels.h for a high-level explanation of the
 * modifications and the motivation for them. Specifically, the names of the
 * BLAS and LAPACK routine symbols have been changed from the FORTRAN interfaces
 * to the equivalent C interfaces. For example, the `dtrsm` BLAS routine has
 * been changed from `dtrsm_` to `cblas_dtrsm`, and the `dgetrf` LAPACK routine
 * has been changed from `dgetrf_` to `LAPACKE_dgetrf`.
 */

#include "lapack_kernels.hpp"

// From a Python binary, JAX obtains its LAPACK/BLAS kernels from Scipy, but
// a C++ user should link against LAPACK directly. This is needed when using
// JAX-generated HLO from C++.

extern "C" {

// jax::RealTrsm<float>::FnType cblas_strsm;
jax::RealTrsm<double>::FnType cblas_dtrsm;
// jax::ComplexTrsm<std::complex<float>>::FnType cblas_ctrsm;
// jax::ComplexTrsm<std::complex<double>>::FnType cblas_ztrsm;

// jax::Getrf<float>::FnType sgetrf;
jax::Getrf<double>::FnType dgetrf$NEWLAPACK;
// jax::Getrf<std::complex<float>>::FnType cgetrf;
// jax::Getrf<std::complex<double>>::FnType zgetrf;

// jax::Geqrf<float>::FnType scipy_LAPACKE_sgeqrf;
// jax::Geqrf<double>::FnType scipy_LAPACKE_dgeqrf;
// jax::Geqrf<std::complex<float>>::FnType scipy_LAPACKE_cgeqrf;
// jax::Geqrf<std::complex<double>>::FnType scipy_LAPACKE_zgeqrf;

// jax::Orgqr<float>::FnType scipy_LAPACKE_sorgqr;
// jax::Orgqr<double>::FnType scipy_LAPACKE_dorgqr;
// jax::Orgqr<std::complex<float>>::FnType scipy_LAPACKE_cungqr;
// jax::Orgqr<std::complex<double>>::FnType scipy_LAPACKE_zungqr;

// jax::Potrf<float>::FnType scipy_LAPACKE_spotrf;
// jax::Potrf<double>::FnType scipy_LAPACKE_dpotrf;
// jax::Potrf<std::complex<float>>::FnType scipy_LAPACKE_cpotrf;
// jax::Potrf<std::complex<double>>::FnType scipy_LAPACKE_zpotrf;

// jax::RealGesdd<float>::FnType scipy_LAPACKE_sgesdd;
// jax::RealGesdd<double>::FnType scipy_LAPACKE_dgesdd;
// jax::ComplexGesdd<std::complex<float>>::FnType scipy_LAPACKE_cgesdd;
// jax::ComplexGesdd<std::complex<double>>::FnType scipy_LAPACKE_zgesdd;

// jax::RealSyevd<float>::FnType scipy_LAPACKE_ssyevd;
// jax::RealSyevd<double>::FnType scipy_LAPACKE_dsyevd;
// jax::ComplexHeevd<std::complex<float>>::FnType scipy_LAPACKE_cheevd;
// jax::ComplexHeevd<std::complex<double>>::FnType scipy_LAPACKE_zheevd;

// jax::RealGeev<float>::FnType scipy_LAPACKE_sgeev;
// jax::RealGeev<double>::FnType scipy_LAPACKE_dgeev;
// jax::ComplexGeev<std::complex<float>>::FnType scipy_LAPACKE_cgeev;
// jax::ComplexGeev<std::complex<double>>::FnType scipy_LAPACKE_zgeev;

// jax::RealGees<float>::FnType scipy_LAPACKE_sgees;
// jax::RealGees<double>::FnType scipy_LAPACKE_dgees;
// jax::ComplexGees<std::complex<float>>::FnType scipy_LAPACKE_cgees;
// jax::ComplexGees<std::complex<double>>::FnType scipy_LAPACKE_zgees;

// jax::Gehrd<float>::FnType scipy_LAPACKE_sgehrd;
// jax::Gehrd<double>::FnType scipy_LAPACKE_dgehrd;
// jax::Gehrd<std::complex<float>>::FnType scipy_LAPACKE_cgehrd;
// jax::Gehrd<std::complex<double>>::FnType scipy_LAPACKE_zgehrd;

// jax::Sytrd<float>::FnType scipy_LAPACKE_ssytrd;
// jax::Sytrd<double>::FnType scipy_LAPACKE_dsytrd;
// jax::Sytrd<std::complex<float>>::FnType scipy_LAPACKE_chetrd;
// jax::Sytrd<std::complex<double>>::FnType scipy_LAPACKE_zhetrd;

} // extern "C"

namespace jax {

static auto init = []() -> int {
    // RealTrsm<float>::fn = cblas_strsm;
    RealTrsm<double>::fn = cblas_dtrsm;
    // ComplexTrsm<std::complex<float>>::fn = cblas_ctrsm;
    // ComplexTrsm<std::complex<double>>::fn = cblas_ztrsm;

    // Getrf<float>::fn = sgetrf;
    Getrf<double>::fn = dgetrf$NEWLAPACK;
    // Getrf<std::complex<float>>::fn = cgetrf;
    // Getrf<std::complex<double>>::fn = zgetrf;

    // Geqrf<float>::fn = scipy_LAPACKE_sgeqrf;
    // Geqrf<double>::fn = scipy_LAPACKE_dgeqrf;
    // Geqrf<std::complex<float>>::fn = scipy_LAPACKE_cgeqrf;
    // Geqrf<std::complex<double>>::fn = scipy_LAPACKE_zgeqrf;

    // Orgqr<float>::fn = scipy_LAPACKE_sorgqr;
    // Orgqr<double>::fn = scipy_LAPACKE_dorgqr;
    // Orgqr<std::complex<float>>::fn = scipy_LAPACKE_cungqr;
    // Orgqr<std::complex<double>>::fn = scipy_LAPACKE_zungqr;

    // Potrf<float>::fn = scipy_LAPACKE_spotrf;
    // Potrf<double>::fn = scipy_LAPACKE_dpotrf;
    // Potrf<std::complex<float>>::fn = scipy_LAPACKE_cpotrf;
    // Potrf<std::complex<double>>::fn = scipy_LAPACKE_zpotrf;

    // RealGesdd<float>::fn = scipy_LAPACKE_sgesdd;
    // RealGesdd<double>::fn = scipy_LAPACKE_dgesdd;
    // ComplexGesdd<std::complex<float>>::fn = scipy_LAPACKE_cgesdd;
    // ComplexGesdd<std::complex<double>>::fn = scipy_LAPACKE_zgesdd;

    // RealSyevd<float>::fn = scipy_LAPACKE_ssyevd;
    // RealSyevd<double>::fn = scipy_LAPACKE_dsyevd;
    // ComplexHeevd<std::complex<float>>::fn = scipy_LAPACKE_cheevd;
    // ComplexHeevd<std::complex<double>>::fn = scipy_LAPACKE_zheevd;

    // RealGeev<float>::fn = scipy_LAPACKE_sgeev;
    // RealGeev<double>::fn = scipy_LAPACKE_dgeev;
    // ComplexGeev<std::complex<float>>::fn = scipy_LAPACKE_cgeev;
    // ComplexGeev<std::complex<double>>::fn = scipy_LAPACKE_zgeev;

    // RealGees<float>::fn = scipy_LAPACKE_sgees;
    // RealGees<double>::fn = scipy_LAPACKE_dgees;
    // ComplexGees<std::complex<float>>::fn = scipy_LAPACKE_cgees;
    // ComplexGees<std::complex<double>>::fn = scipy_LAPACKE_zgees;

    // Gehrd<float>::fn = scipy_LAPACKE_sgehrd;
    // Gehrd<double>::fn = scipy_LAPACKE_dgehrd;
    // Gehrd<std::complex<float>>::fn = scipy_LAPACKE_cgehrd;
    // Gehrd<std::complex<double>>::fn = scipy_LAPACKE_zgehrd;

    // Sytrd<float>::fn = scipy_LAPACKE_ssytrd;
    // Sytrd<double>::fn = scipy_LAPACKE_dsytrd;
    // Sytrd<std::complex<float>>::fn = scipy_LAPACKE_chetrd;
    // Sytrd<std::complex<double>>::fn = scipy_LAPACKE_zhetrd;

    return 0;
}();

} // namespace jax
