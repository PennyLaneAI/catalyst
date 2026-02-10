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

// With SciPy OpenBLAS, symbols are now prefixed with scipy_ when using scipy>=1.14 or
// scipy-openblas32, whereas this is not the case for the reference implementation used on M1 mac.
#if defined(__APPLE__) && defined(__arm64__)
#define SYM_PREFIX
#else
#define SYM_PREFIX scipy_
#endif

// The CONCAT macro and its helper CONCAT_ are required here since macro arguments are not expanded
// around a ## preprocessing token. See http://port70.net/%7Ensz/c/c11/n1570.html#6.10.3.1.
#define CONCAT_(X, Y) X##Y
#define CONCAT(X, Y) CONCAT_(X, Y)
#define GET_SYMBOL(X) CONCAT(SYM_PREFIX, X)

// From a Python binary, JAX obtains its LAPACK/BLAS kernels from Scipy, but
// a C++ user should link against LAPACK directly. This is needed when using
// JAX-generated HLO from C++.

extern "C" {

jax::RealTrsm<float>::FnType GET_SYMBOL(cblas_strsm);
jax::RealTrsm<double>::FnType GET_SYMBOL(cblas_dtrsm);
jax::ComplexTrsm<std::complex<float>>::FnType GET_SYMBOL(cblas_ctrsm);
jax::ComplexTrsm<std::complex<double>>::FnType GET_SYMBOL(cblas_ztrsm);

jax::Getrf<float>::FnType GET_SYMBOL(LAPACKE_sgetrf);
jax::Getrf<double>::FnType GET_SYMBOL(LAPACKE_dgetrf);
jax::Getrf<std::complex<float>>::FnType GET_SYMBOL(LAPACKE_cgetrf);
jax::Getrf<std::complex<double>>::FnType GET_SYMBOL(LAPACKE_zgetrf);

jax::Geqrf<float>::FnType GET_SYMBOL(LAPACKE_sgeqrf);
jax::Geqrf<double>::FnType GET_SYMBOL(LAPACKE_dgeqrf);
jax::Geqrf<std::complex<float>>::FnType GET_SYMBOL(LAPACKE_cgeqrf);
jax::Geqrf<std::complex<double>>::FnType GET_SYMBOL(LAPACKE_zgeqrf);

jax::Orgqr<float>::FnType GET_SYMBOL(LAPACKE_sorgqr);
jax::Orgqr<double>::FnType GET_SYMBOL(LAPACKE_dorgqr);
jax::Orgqr<std::complex<float>>::FnType GET_SYMBOL(LAPACKE_cungqr);
jax::Orgqr<std::complex<double>>::FnType GET_SYMBOL(LAPACKE_zungqr);

jax::Potrf<float>::FnType GET_SYMBOL(LAPACKE_spotrf);
jax::Potrf<double>::FnType GET_SYMBOL(LAPACKE_dpotrf);
jax::Potrf<std::complex<float>>::FnType GET_SYMBOL(LAPACKE_cpotrf);
jax::Potrf<std::complex<double>>::FnType GET_SYMBOL(LAPACKE_zpotrf);

jax::RealGesdd<float>::FnType GET_SYMBOL(LAPACKE_sgesdd);
jax::RealGesdd<double>::FnType GET_SYMBOL(LAPACKE_dgesdd);
jax::ComplexGesdd<std::complex<float>>::FnType GET_SYMBOL(LAPACKE_cgesdd);
jax::ComplexGesdd<std::complex<double>>::FnType GET_SYMBOL(LAPACKE_zgesdd);

jax::RealSyevd<float>::FnType GET_SYMBOL(LAPACKE_ssyevd);
jax::RealSyevd<double>::FnType GET_SYMBOL(LAPACKE_dsyevd);
jax::ComplexHeevd<std::complex<float>>::FnType GET_SYMBOL(LAPACKE_cheevd);
jax::ComplexHeevd<std::complex<double>>::FnType GET_SYMBOL(LAPACKE_zheevd);

jax::RealGeev<float>::FnType GET_SYMBOL(LAPACKE_sgeev);
jax::RealGeev<double>::FnType GET_SYMBOL(LAPACKE_dgeev);
jax::ComplexGeev<std::complex<float>>::FnType GET_SYMBOL(LAPACKE_cgeev);
jax::ComplexGeev<std::complex<double>>::FnType GET_SYMBOL(LAPACKE_zgeev);

jax::RealGees<float>::FnType GET_SYMBOL(LAPACKE_sgees);
jax::RealGees<double>::FnType GET_SYMBOL(LAPACKE_dgees);
jax::ComplexGees<std::complex<float>>::FnType GET_SYMBOL(LAPACKE_cgees);
jax::ComplexGees<std::complex<double>>::FnType GET_SYMBOL(LAPACKE_zgees);

jax::Gehrd<float>::FnType GET_SYMBOL(LAPACKE_sgehrd);
jax::Gehrd<double>::FnType GET_SYMBOL(LAPACKE_dgehrd);
jax::Gehrd<std::complex<float>>::FnType GET_SYMBOL(LAPACKE_cgehrd);
jax::Gehrd<std::complex<double>>::FnType GET_SYMBOL(LAPACKE_zgehrd);

jax::Sytrd<float>::FnType GET_SYMBOL(LAPACKE_ssytrd);
jax::Sytrd<double>::FnType GET_SYMBOL(LAPACKE_dsytrd);
jax::Sytrd<std::complex<float>>::FnType GET_SYMBOL(LAPACKE_chetrd);
jax::Sytrd<std::complex<double>>::FnType GET_SYMBOL(LAPACKE_zhetrd);

} // extern "C"

namespace jax {

static auto init = []() -> int {
    AssignKernelFn<RealTrsm<float>>(GET_SYMBOL(cblas_strsm));
    AssignKernelFn<RealTrsm<double>>(GET_SYMBOL(cblas_dtrsm));
    AssignKernelFn<ComplexTrsm<std::complex<float>>>(GET_SYMBOL(cblas_ctrsm));
    AssignKernelFn<ComplexTrsm<std::complex<double>>>(GET_SYMBOL(cblas_ztrsm));

    AssignKernelFn<Getrf<float>>(GET_SYMBOL(LAPACKE_sgetrf));
    AssignKernelFn<Getrf<double>>(GET_SYMBOL(LAPACKE_dgetrf));
    AssignKernelFn<Getrf<std::complex<float>>>(GET_SYMBOL(LAPACKE_cgetrf));
    AssignKernelFn<Getrf<std::complex<double>>>(GET_SYMBOL(LAPACKE_zgetrf));

    AssignKernelFn<Geqrf<float>>(GET_SYMBOL(LAPACKE_sgeqrf));
    AssignKernelFn<Geqrf<double>>(GET_SYMBOL(LAPACKE_dgeqrf));
    AssignKernelFn<Geqrf<std::complex<float>>>(GET_SYMBOL(LAPACKE_cgeqrf));
    AssignKernelFn<Geqrf<std::complex<double>>>(GET_SYMBOL(LAPACKE_zgeqrf));

    AssignKernelFn<Orgqr<float>>(GET_SYMBOL(LAPACKE_sorgqr));
    AssignKernelFn<Orgqr<double>>(GET_SYMBOL(LAPACKE_dorgqr));
    AssignKernelFn<Orgqr<std::complex<float>>>(GET_SYMBOL(LAPACKE_cungqr));
    AssignKernelFn<Orgqr<std::complex<double>>>(GET_SYMBOL(LAPACKE_zungqr));

    AssignKernelFn<Potrf<float>>(GET_SYMBOL(LAPACKE_spotrf));
    AssignKernelFn<Potrf<double>>(GET_SYMBOL(LAPACKE_dpotrf));
    AssignKernelFn<Potrf<std::complex<float>>>(GET_SYMBOL(LAPACKE_cpotrf));
    AssignKernelFn<Potrf<std::complex<double>>>(GET_SYMBOL(LAPACKE_zpotrf));
    AssignKernelFn<RealGesdd<float>>(GET_SYMBOL(LAPACKE_sgesdd));
    AssignKernelFn<RealGesdd<double>>(GET_SYMBOL(LAPACKE_dgesdd));
    AssignKernelFn<ComplexGesdd<std::complex<float>>>(GET_SYMBOL(LAPACKE_cgesdd));
    AssignKernelFn<ComplexGesdd<std::complex<double>>>(GET_SYMBOL(LAPACKE_zgesdd));

    AssignKernelFn<RealSyevd<float>>(GET_SYMBOL(LAPACKE_ssyevd));
    AssignKernelFn<RealSyevd<double>>(GET_SYMBOL(LAPACKE_dsyevd));
    AssignKernelFn<ComplexHeevd<std::complex<float>>>(GET_SYMBOL(LAPACKE_cheevd));
    AssignKernelFn<ComplexHeevd<std::complex<double>>>(GET_SYMBOL(LAPACKE_zheevd));

    AssignKernelFn<RealGeev<float>>(GET_SYMBOL(LAPACKE_sgeev));
    AssignKernelFn<RealGeev<double>>(GET_SYMBOL(LAPACKE_dgeev));
    AssignKernelFn<ComplexGeev<std::complex<float>>>(GET_SYMBOL(LAPACKE_cgeev));
    AssignKernelFn<ComplexGeev<std::complex<double>>>(GET_SYMBOL(LAPACKE_zgeev));

    AssignKernelFn<RealGees<float>>(GET_SYMBOL(LAPACKE_sgees));
    AssignKernelFn<RealGees<double>>(GET_SYMBOL(LAPACKE_dgees));
    AssignKernelFn<ComplexGees<std::complex<float>>>(GET_SYMBOL(LAPACKE_cgees));
    AssignKernelFn<ComplexGees<std::complex<double>>>(GET_SYMBOL(LAPACKE_zgees));

    AssignKernelFn<Gehrd<float>>(GET_SYMBOL(LAPACKE_sgehrd));
    AssignKernelFn<Gehrd<double>>(GET_SYMBOL(LAPACKE_dgehrd));
    AssignKernelFn<Gehrd<std::complex<float>>>(GET_SYMBOL(LAPACKE_cgehrd));
    AssignKernelFn<Gehrd<std::complex<double>>>(GET_SYMBOL(LAPACKE_zgehrd));

    AssignKernelFn<Sytrd<float>>(GET_SYMBOL(LAPACKE_ssytrd));
    AssignKernelFn<Sytrd<double>>(GET_SYMBOL(LAPACKE_dsytrd));
    AssignKernelFn<Sytrd<std::complex<float>>>(GET_SYMBOL(LAPACKE_chetrd));
    AssignKernelFn<Sytrd<std::complex<double>>>(GET_SYMBOL(LAPACKE_zhetrd));

    return 0;
}();

} // namespace jax
