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
 *
 * In addition, the following modifications have been made:
 *
 *   1. Guarded the #include of the XLA `custom_call_status.h` header by the
 *      `USE_XLA_LIB` macro; simply declared the `XlaCustomCallStatus` type
 *      instead, since it is not explicitly used.
 *   2. Copied the BLAS and LAPACK enums and option codes (e.g. `CBLAS_ORDER`
 *      and `LAPACK_ROW_MAJOR`) needed for the C interfaces.
 *   3. Applied Catalyst C++ code formatting.
 */

#ifndef JAXLIB_CPU_LAPACK_KERNELS_H_
#define JAXLIB_CPU_LAPACK_KERNELS_H_

#include <complex>
#include <cstdint>
#include <numeric>

#ifdef USE_XLA_LIB
#include "xla/service/custom_call_status.h"
#else
typedef struct XlaCustomCallStatus_ XlaCustomCallStatus;
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/api.h"
#include "xla/ffi/api/ffi.h"
#endif

namespace ffi = xla::ffi;

namespace jax {
struct MatrixParams {
  enum class Side : char { kLeft = 'L', kRight = 'R' };
  enum class UpLo : char { kLower = 'L', kUpper = 'U' };
  enum class Diag : char { kNonUnit = 'N', kUnit = 'U' };
  enum class Transpose : char {
    kNoTrans = 'N',
    kTrans = 'T',
    kConjTrans = 'C'
  };
};

}



#define DEFINE_CHAR_ENUM_ATTR_DECODING(ATTR)                             \
  template <>                                                            \
  struct xla::ffi::AttrDecoding<ATTR> {                                  \
    using Type = ATTR;                                                   \
    static std::optional<Type> Decode(XLA_FFI_AttrType type, void* attr, \
                                      DiagnosticEngine& diagnostic);     \
  }

// XLA needs attributes to have deserialization method specified
DEFINE_CHAR_ENUM_ATTR_DECODING(jax::MatrixParams::Side);
DEFINE_CHAR_ENUM_ATTR_DECODING(jax::MatrixParams::UpLo);
DEFINE_CHAR_ENUM_ATTR_DECODING(jax::MatrixParams::Transpose);
DEFINE_CHAR_ENUM_ATTR_DECODING(jax::MatrixParams::Diag);
#undef DEFINE_CHAR_ENUM_ATTR_DECODING

namespace {

template <typename T>
inline T CastNoOverflow(int64_t value, const std::string& source = __FILE__) {
  if constexpr (sizeof(T) == sizeof(int64_t)) {
    return value;
  } else {
    if (value > std::numeric_limits<T>::max()) [[unlikely]] {
#ifdef USE_ABSEIL_LIB
      throw std::overflow_error{
          absl::StrFormat("%s: Value (=%d) exceeds the maximum representable "
                          "value of the desired type",
                          source, value)};
#else
      throw std::overflow_error{"Value exceeds the maximum representable "
	                 "value of the desired type"};
#endif
    }
    return static_cast<T>(value);
  }
}

template <typename T>
std::tuple<int64_t, int64_t, int64_t> SplitBatch2D(ffi::Span<T> dims) {
  if (dims.size() < 2) {
    throw std::invalid_argument("Matrix must have at least 2 dimensions");
  }
  auto matrix_dims = dims.last(2);
#ifdef USE_ABSEIL_LIB
  return std::make_tuple(absl::c_accumulate(dims.first(dims.size() - 2), 1,
                                            std::multiplies<int64_t>()),
                         matrix_dims.front(), matrix_dims.back());
#else
  auto sequence = dims.first(dims.size() - 2);
  return std::make_tuple(std::accumulate(std::begin(sequence), std::end(sequence), 1,
                                            std::multiplies<int64_t>()),
                         matrix_dims.front(), matrix_dims.back());
#endif
}

template <ffi::DataType dtype>
void CopyIfDiffBuffer(ffi::Buffer<dtype> x, ffi::ResultBuffer<dtype> x_out) {
  auto [batch_count, x_rows, x_cols] = SplitBatch2D(x.dimensions);
  if (x.data != x_out->data) {
    const auto x_size = batch_count * x_rows * x_cols;
    std::copy_n(x.data, x_size, x_out->data);
  }
}

} // namespace



// Underlying function pointers (e.g., Trsm<double>::Fn) are initialized either
// by the pybind wrapper that links them to an existing SciPy lapack instance,
// or using the lapack_kernels_strong.cc static initialization to link them
// directly to lapack for use in a pure C++ context.

namespace jax {

// Copied from cblas.h
typedef enum CBLAS_ORDER { CblasRowMajor = 101, CblasColMajor = 102 } CBLAS_ORDER;
typedef enum CBLAS_TRANSPOSE {
    CblasNoTrans = 111,
    CblasTrans = 112,
    CblasConjTrans = 113,
    CblasConjNoTrans = 114
} CBLAS_TRANSPOSE;
typedef enum CBLAS_UPLO { CblasUpper = 121, CblasLower = 122 } CBLAS_UPLO;
typedef enum CBLAS_DIAG { CblasNonUnit = 131, CblasUnit = 132 } CBLAS_DIAG;
typedef enum CBLAS_SIDE { CblasLeft = 141, CblasRight = 142 } CBLAS_SIDE;
typedef CBLAS_ORDER CBLAS_LAYOUT;

typedef int lapack_int;
inline constexpr auto LapackIntDtype = ::xla::ffi::DataType::S32;
template <typename KernelType>
void AssignKernelFn(void* func) {
  KernelType::fn = reinterpret_cast<typename KernelType::FnType*>(func);
}

template <typename KernelType>
void AssignKernelFn(typename KernelType::FnType* func) {
  KernelType::fn = func;
}

} // namespace jax



namespace jax {

// Copied from lapacke.h
#define LAPACK_ROW_MAJOR 101
#define LAPACK_COL_MAJOR 102

// trsm: Solves a triangular matrix equation.
template <typename T> struct RealTrsm {
    using FnType = void(const CBLAS_LAYOUT layout, const CBLAS_SIDE Side, const CBLAS_UPLO Uplo,
                        const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag, const int M,
                        const int N, const T alpha, const T *A, const int lda, T *B, const int ldb);
    static FnType *fn;
    static void Kernel(void *out, void **data, XlaCustomCallStatus *);
};

template <typename T> struct ComplexTrsm {
    using FnType = void(const CBLAS_LAYOUT layout, const CBLAS_SIDE Side, const CBLAS_UPLO Uplo,
                        const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag, const int M,
                        const int N, const void *alpha, const void *A, const int lda, void *B,

                        const int ldb);
    static FnType *fn;
    static void Kernel(void *out, void **data, XlaCustomCallStatus *);
};

// getrf: Computes the LU factorization of a general m-by-n matrix
template <typename T> struct Getrf {
    using FnType = lapack_int(int matrix_layout, lapack_int m, lapack_int n, T *a, lapack_int lda,
                              lapack_int *ipiv);
    static FnType *fn;
    static void Kernel(void *out, void **data, XlaCustomCallStatus *);
};

// geqrf: Computes the QR factorization of a general m-by-n matrix.
template <typename T> struct Geqrf {
    using FnType = lapack_int(int matrix_layout, lapack_int m, lapack_int n, T *a, lapack_int lda,
                              T *tau);
    static FnType *fn;
    static void Kernel(void *out, void **data, XlaCustomCallStatus *);
};

// orgqr: Generates the real orthogonal matrix Q of the QR factorization formed by geqrf
template <typename T> struct Orgqr {
    using FnType = lapack_int(int matrix_layout, lapack_int m, lapack_int n, lapack_int k, T *a,
                              lapack_int lda, const T *tau);
    static FnType *fn;
    static void Kernel(void *out, void **data, XlaCustomCallStatus *);
};

// potrf: Computes the Cholesky factorization of a symmetric (Hermitian) positive-definite matrix
template <typename T> struct Potrf {
    using FnType = lapack_int(int matrix_layout, char uplo, lapack_int n, T *a, lapack_int lda);
    static FnType *fn;
    static void Kernel(void *out, void **data, XlaCustomCallStatus *);
};

// FFI Kernel

template <::xla::ffi::DataType dtype>
struct CholeskyFactorization {
  using ValueType = ::xla::ffi::NativeType<dtype>;
  using FnType = void(char* uplo, lapack_int* n, ValueType* a, lapack_int* lda,
                      lapack_int* info);
  inline static FnType* fn = nullptr;
  static ::xla::ffi::Error Kernel(
      ::xla::ffi::Buffer<dtype> x, MatrixParams::UpLo uplo,
      ::xla::ffi::ResultBuffer<dtype> x_out,
      ::xla::ffi::ResultBuffer<LapackIntDtype> info);
};

// gesdd: computes the singular value decomposition (SVD) of an m-by-n matrix
template <typename T> struct RealGesdd {
    using FnType = lapack_int(int matrix_layout, char jobz, lapack_int m, lapack_int n, T *a,
                              lapack_int lda, T *s, T *u, lapack_int ldu, T *vt, lapack_int ldvt);
    static FnType *fn;
    static void Kernel(void *out, void **data, XlaCustomCallStatus *);
};

template <typename T> struct ComplexGesdd {
    using FnType = lapack_int(int matrix_layout, char jobz, lapack_int m, lapack_int n, T *a,
                              lapack_int lda, typename T::value_type *s, T *u, lapack_int ldu,
                              T *vt, lapack_int ldvt);
    static FnType *fn;
    static void Kernel(void *out, void **data, XlaCustomCallStatus *);
};

// syevd: Computes all eigenvalues and, optionally, all eigenvectors of a real symmetric matrix
template <typename T> struct RealSyevd {
    using FnType = lapack_int(int matrix_layout, char jobz, char uplo, lapack_int n, T *a,
                              lapack_int lda, T *w);
    static FnType *fn;
    static void Kernel(void *out, void **data, XlaCustomCallStatus *);
};

// heevd: Computes all eigenvalues and, optionally, all eigenvectors of a complex Hermitian matrix
template <typename T> struct ComplexHeevd {
    using FnType = lapack_int(int matrix_layout, char jobz, char uplo, lapack_int n, T *a,
                              lapack_int lda, typename T::value_type *w);
    static FnType *fn;
    static void Kernel(void *out, void **data, XlaCustomCallStatus *);
};

// geev: Computes the eigenvalues and left and right eigenvectors of a general matrix
template <typename T> struct RealGeev {
    using FnType = lapack_int(int matrix_layout, char jobvl, char jobvr, lapack_int n, T *a,
                              lapack_int lda, T *wr, T *wi, T *vl, lapack_int ldvl, T *vr,
                              lapack_int ldvr);
    static FnType *fn;
    static void Kernel(void *out, void **data, XlaCustomCallStatus *);
};

template <typename T> struct ComplexGeev {
    using FnType = lapack_int(int matrix_layout, char jobvl, char jobvr, lapack_int n, T *a,
                              lapack_int lda, T *w, T *vl, lapack_int ldvl, T *vr, lapack_int ldvr);
    static FnType *fn;
    static void Kernel(void *out, void **data, XlaCustomCallStatus *);
};

// gees: Computes the eigenvalues and Schur factorization of a general matrix
template <typename T> struct RealGees {
    using FnType = lapack_int(int matrix_layout, char jobvs, char sort, bool (*select)(T, T),
                              lapack_int n, T *a, lapack_int lda, lapack_int *sdim, T *wr, T *wi,
                              T *vs, lapack_int ldvs);
    static FnType *fn;
    static void Kernel(void *out, void **data, XlaCustomCallStatus *);
};

template <typename T> struct ComplexGees {
    using FnType = lapack_int(int matrix_layout, char jobvs, char sort, bool (*select)(T),
                              lapack_int n, T *a, lapack_int lda, lapack_int *sdim, T *w, T *vs,
                              lapack_int ldvs);
    static FnType *fn;
    static void Kernel(void *out, void **data, XlaCustomCallStatus *);
};

// Gehrd: Reduces a non-symmetric square matrix to upper Hessenberg form
template <typename T> struct Gehrd {
    using FnType = lapack_int(int matrix_layout, lapack_int n, lapack_int ilo, lapack_int ihi, T *a,
                              lapack_int lda, T *tau);
    static FnType *fn;
    static void Kernel(void *out, void **data, XlaCustomCallStatus *);
};

template <typename T> struct real_type {
    typedef T type;
};
template <typename T> struct real_type<std::complex<T>> {
    typedef T type;
};

// Sytrd/Hetrd: Reduces a symmetric (Hermitian) square matrix to tridiagonal form
template <typename T> struct Sytrd {
    using FnType = lapack_int(int matrix_layout, char uplo, lapack_int n, T *a, lapack_int lda,
                              typename real_type<T>::type *d, typename real_type<T>::type *e,
                              T *tau);
    static FnType *fn;
    static void Kernel(void *out, void **data, XlaCustomCallStatus *);
};

} // namespace jax

#endif // JAXLIB_CPU_LAPACK_KERNELS_H_
