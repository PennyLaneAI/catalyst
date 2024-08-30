/* Copyright 2021 The JAX Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

/*
 * This file is a modified version of
 *
 *   https://github.com/google/jax/blob/jaxlib-v0.4.28/jaxlib/cpu/lapack_kernels.h
 *
 * from jaxlib-v0.4.28.
 *
 * The LAPACK kernels below have been modified from their original form in JAX
 * to use the respective C interfaces to the underlying BLAS and LAPACK
 * routines, rather than the FORTRAN interfaces that JAX uses, for compatibility
 * with Catalyst. Recall that the FORTRAN interfaces require arrays and matrices
 * in column-major order, while the C interfaces allow row-major order, which is
 * required for Catalyst.
 */

#ifndef JAXLIB_CPU_LAPACK_KERNELS_H_
#define JAXLIB_CPU_LAPACK_KERNELS_H_

#include <complex>
#include <cstdint>

#ifdef USE_XLA_LIB
#include "xla/service/custom_call_status.h"
#else
typedef struct XlaCustomCallStatus_ XlaCustomCallStatus;
#endif


// Underlying function pointers (e.g., Trsm<double>::Fn) are initialized either
// by the pybind wrapper that links them to an existing SciPy lapack instance,
// or using the lapack_kernels_strong.cc static initialization to link them
// directly to lapack for use in a pure C++ context.

namespace jax {

// Copied from cblas.h
typedef enum CBLAS_ORDER     {CblasRowMajor=101, CblasColMajor=102} CBLAS_ORDER;
typedef enum CBLAS_TRANSPOSE {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113, CblasConjNoTrans=114} CBLAS_TRANSPOSE;
typedef enum CBLAS_UPLO      {CblasUpper=121, CblasLower=122} CBLAS_UPLO;
typedef enum CBLAS_DIAG      {CblasNonUnit=131, CblasUnit=132} CBLAS_DIAG;
typedef enum CBLAS_SIDE      {CblasLeft=141, CblasRight=142} CBLAS_SIDE;
typedef CBLAS_ORDER CBLAS_LAYOUT;

typedef int lapack_int;

// Copied from lapacke.h
#define LAPACK_ROW_MAJOR  101
#define LAPACK_COL_MAJOR  102

// #ifndef lapack_logical
// #define lapack_logical lapack_int
// #endif

// typedef lapack_logical (*LAPACK_S_SELECT2) ( const float*, const float* );

// trsm: Solves a triangular matrix equation.
template <typename T>
struct RealTrsm {
  using FnType = void(const CBLAS_LAYOUT layout, const CBLAS_SIDE Side,
                      const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA,
                      const CBLAS_DIAG Diag, const int M, const int N,
                      const T alpha, const T *A, const int lda,
                      T *B, const int ldb);
  static FnType* fn;
  static void Kernel(void* out, void** data, XlaCustomCallStatus*);
};

template <typename T>
struct ComplexTrsm {
  using FnType = void(const CBLAS_LAYOUT layout, const CBLAS_SIDE Side,
                      const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA,
                      const CBLAS_DIAG Diag, const int M, const int N,
                      const void *alpha, const void *A, const int lda,
                      void *B, const int ldb);
  static FnType* fn;
  static void Kernel(void* out, void** data, XlaCustomCallStatus*);
};

// getrf: Computes the LU factorization of a general m-by-n matrix
template <typename T>
struct Getrf {
  using FnType = lapack_int(int matrix_layout, lapack_int m, lapack_int n, T* a,
                            lapack_int lda, lapack_int* ipiv);
  static FnType* fn;
  static void Kernel(void* out, void** data, XlaCustomCallStatus*);
};

// geqrf: Computes the QR factorization of a general m-by-n matrix.
template <typename T>
struct Geqrf {
  using FnType = lapack_int(int matrix_layout, lapack_int m, lapack_int n,
                            T* a, lapack_int lda, T* tau);
  static FnType* fn;
  static void Kernel(void* out, void** data, XlaCustomCallStatus*);
};

// orgqr: Generates the real orthogonal matrix Q of the QR factorization formed by geqrf
template <typename T>
struct Orgqr {
  using FnType = lapack_int(int matrix_layout, lapack_int m, lapack_int n,
                            lapack_int k, T* a, lapack_int lda, const T* tau);
  static FnType* fn;
  static void Kernel(void* out, void** data, XlaCustomCallStatus*);
};

// potrf: Computes the Cholesky factorization of a symmetric (Hermitian) positive-definite matrix
template <typename T>
struct Potrf {
  using FnType = lapack_int(int matrix_layout, char uplo, lapack_int n, T* a,
                            lapack_int lda);
  static FnType* fn;
  static void Kernel(void* out, void** data, XlaCustomCallStatus*);
};

// gesdd: computes the singular value decomposition (SVD) of an m-by-n matrix
template <typename T>
struct RealGesdd {
  using FnType = lapack_int(int matrix_layout, char jobz, lapack_int m,
                            lapack_int n, T* a, lapack_int lda, T* s,
                            T* u, lapack_int ldu, T* vt, lapack_int ldvt);
  static FnType* fn;
  static void Kernel(void* out, void** data, XlaCustomCallStatus*);
};

template <typename T>
struct ComplexGesdd {
  using FnType = lapack_int(int matrix_layout, char jobz, lapack_int m, lapack_int n,
                            T* a, lapack_int lda, typename T::value_type* s,
                            T* u, lapack_int ldu, T* vt, lapack_int ldvt);
  static FnType* fn;
  static void Kernel(void* out, void** data, XlaCustomCallStatus*);
};

// syevd: Computes all eigenvalues and, optionally, all eigenvectors of a real symmetric matrix
template <typename T>
struct RealSyevd {
  using FnType = lapack_int(int matrix_layout, char jobz, char uplo, lapack_int n,
                            T* a, lapack_int lda, T* w);
  static FnType* fn;
  static void Kernel(void* out, void** data, XlaCustomCallStatus*);
};

// heevd: Computes all eigenvalues and, optionally, all eigenvectors of a complex Hermitian matrix
template <typename T>
struct ComplexHeevd {
  using FnType = lapack_int(int matrix_layout, char jobz, char uplo, lapack_int n,
                            T* a, lapack_int lda, typename T::value_type* w);
  static FnType* fn;
  static void Kernel(void* out, void** data, XlaCustomCallStatus*);
};

// geev: Computes the eigenvalues and left and right eigenvectors of a general matrix
template <typename T>
struct RealGeev {
  using FnType = lapack_int(int matrix_layout, char jobvl, char jobvr,
                            lapack_int n, T* a, lapack_int lda, T* wr,
                            T* wi, T* vl, lapack_int ldvl, T* vr,
                            lapack_int ldvr);
  static FnType* fn;
  static void Kernel(void* out, void** data, XlaCustomCallStatus*);
};

template <typename T>
struct ComplexGeev {
  using FnType = lapack_int(int matrix_layout, char jobvl, char jobvr,
                            lapack_int n, T* a, lapack_int lda, T* w, T* vl,
                            lapack_int ldvl, T* vr, lapack_int ldvr);
  static FnType* fn;
  static void Kernel(void* out, void** data, XlaCustomCallStatus*);
};

// gees: Computes the eigenvalues and Schur factorization of a general matrix
template <typename T>
struct RealGees {
  using FnType = lapack_int(int matrix_layout, char jobvs, char sort,
                            bool (*select)(T, T), lapack_int n, T* a,
                            lapack_int lda, lapack_int* sdim, T* wr, T* wi,
                            T* vs, lapack_int ldvs);
  static FnType* fn;
  static void Kernel(void* out, void** data, XlaCustomCallStatus*);
};

template <typename T>
struct ComplexGees {
  using FnType = lapack_int(int matrix_layout, char jobvs, char sort,
                            bool (*select)(T), lapack_int n, T* a,
                            lapack_int lda, lapack_int* sdim, T* w, T* vs,
                            lapack_int ldvs);
  static FnType* fn;
  static void Kernel(void* out, void** data, XlaCustomCallStatus*);
};

// Gehrd: Reduces a non-symmetric square matrix to upper Hessenberg form
template <typename T>
struct Gehrd {
  using FnType = lapack_int(int matrix_layout, lapack_int n, lapack_int ilo,
                            lapack_int ihi, T* a, lapack_int lda, T* tau);
  static FnType* fn;
  static void Kernel(void* out, void** data, XlaCustomCallStatus*);
};

template <typename T>
struct real_type {
  typedef T type;
};
template <typename T>
struct real_type<std::complex<T>> {
  typedef T type;
};

// Sytrd/Hetrd: Reduces a symmetric (Hermitian) square matrix to tridiagonal form
template <typename T>
struct Sytrd {
  using FnType = lapack_int(int matrix_layout, char uplo, lapack_int n, T* a,
                            lapack_int lda, typename real_type<T>::type* d,
                            typename real_type<T>::type* e, T* tau);
  static FnType* fn;
  static void Kernel(void* out, void** data, XlaCustomCallStatus*);
};

}  // namespace jax

#endif  // JAXLIB_CPU_LAPACK_KERNELS_H_
