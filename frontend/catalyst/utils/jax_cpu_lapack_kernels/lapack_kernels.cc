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
 *   https://github.com/google/jax/blob/jaxlib-v0.4.28/jaxlib/cpu/lapack_kernels.cc
 *
 * from jaxlib-v0.4.28.
 *
 * See note in lapack_kernels.h for explanation of modifications.
 */

#include "lapack_kernels.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <iostream>

#ifdef USE_ABSEIL_LIB
#include "absl/base/dynamic_annotations.h"
#endif

namespace {

inline int64_t catch_lapack_int_overflow(const std::string& source, int64_t value) {
  if constexpr (sizeof(jax::lapack_int) == sizeof(int64_t)) {
    return value;
  } else {
    if (value > std::numeric_limits<jax::lapack_int>::max()) {
      throw std::overflow_error(source + "(=" + std::to_string(value) + ") exceeds maximum value of jax::lapack_int");
    }
    return value;
  }
}

}

namespace jax {

static_assert(sizeof(lapack_int) == sizeof(int32_t),
              "Expected LAPACK integers to be 32-bit");

// Trsm
// ~~~~

template <typename T>
typename RealTrsm<T>::FnType* RealTrsm<T>::fn = nullptr;

template <typename T>
void RealTrsm<T>::Kernel(void* out, void** data, XlaCustomCallStatus*) {
  const int32_t left_side = *reinterpret_cast<int32_t*>(data[0]);
  const int32_t lower = *reinterpret_cast<int32_t*>(data[1]);
  const int32_t trans_a = *reinterpret_cast<int32_t*>(data[2]);
  const int32_t diag = *reinterpret_cast<int32_t*>(data[3]);
  const int m = *reinterpret_cast<int32_t*>(data[4]);
  const int n = *reinterpret_cast<int32_t*>(data[5]);
  const int batch = *reinterpret_cast<int32_t*>(data[6]);
  const T alpha = *reinterpret_cast<T*>(data[7]);
  const T* a = reinterpret_cast<T*>(data[8]);
  T* b = reinterpret_cast<T*>(data[9]);

  T* x = reinterpret_cast<T*>(out);
  if (x != b) {
    std::memcpy(x, b,
                static_cast<int64_t>(batch) * static_cast<int64_t>(m) *
                    static_cast<int64_t>(n) * sizeof(T));
  }

  constexpr CBLAS_ORDER corder = CblasRowMajor;
  const CBLAS_SIDE cside = left_side ? CblasLeft : CblasRight;
  const CBLAS_UPLO cuplo = lower ? CblasLower : CblasUpper;
  const CBLAS_TRANSPOSE ctransa = (trans_a == 1) ? CblasTrans : (trans_a == 2) ? CblasConjTrans : CblasNoTrans;
  const CBLAS_DIAG cdiag = diag ? CblasUnit : CblasNonUnit;
  const int lda = left_side ? m : n;
  const int ldb = (corder == CblasColMajor) ? m : n;  // Note: m if col-major, n if row-major

  const int64_t x_plus = static_cast<int64_t>(m) * static_cast<int64_t>(n);
  const int64_t a_plus = static_cast<int64_t>(lda) * static_cast<int64_t>(lda);

  for (int i = 0; i < batch; ++i) {
    fn(CblasRowMajor, cside, cuplo, ctransa, cdiag, m, n, alpha, a, lda, x, ldb);
    x += x_plus;
    a += a_plus;
  }
}

template <typename T>
typename ComplexTrsm<T>::FnType* ComplexTrsm<T>::fn = nullptr;

template <typename T>
void ComplexTrsm<T>::Kernel(void* out, void** data, XlaCustomCallStatus*) {
  const int32_t left_side = *reinterpret_cast<int32_t*>(data[0]);
  const int32_t lower = *reinterpret_cast<int32_t*>(data[1]);
  const int32_t trans_a = *reinterpret_cast<int32_t*>(data[2]);
  const int32_t diag = *reinterpret_cast<int32_t*>(data[3]);
  const int m = *reinterpret_cast<int32_t*>(data[4]);
  const int n = *reinterpret_cast<int32_t*>(data[5]);
  const int batch = *reinterpret_cast<int32_t*>(data[6]);
  const T* alpha = reinterpret_cast<T*>(data[7]);
  const T* a = reinterpret_cast<T*>(data[8]);
  T* b = reinterpret_cast<T*>(data[9]);

  T* x = reinterpret_cast<T*>(out);
  if (x != b) {
    std::memcpy(x, b,
                static_cast<int64_t>(batch) * static_cast<int64_t>(m) *
                    static_cast<int64_t>(n) * sizeof(T));
  }

  constexpr CBLAS_ORDER corder = CblasRowMajor;
  const CBLAS_SIDE cside = left_side ? CblasLeft : CblasRight;
  const CBLAS_UPLO cuplo = lower ? CblasLower : CblasUpper;
  const CBLAS_TRANSPOSE ctransa = (trans_a == 1) ? CblasTrans : (trans_a == 2) ? CblasConjTrans : CblasNoTrans;
  const CBLAS_DIAG cdiag = diag ? CblasUnit : CblasNonUnit;
  const int lda = left_side ? m : n;
  const int ldb = (corder == CblasColMajor) ? m : n;  // Note: m if col-major, n if row-major

  const int64_t x_plus = static_cast<int64_t>(m) * static_cast<int64_t>(n);
  const int64_t a_plus = static_cast<int64_t>(lda) * static_cast<int64_t>(lda);

  for (int i = 0; i < batch; ++i) {
    fn(CblasRowMajor, cside, cuplo, ctransa, cdiag, m, n, alpha, a, lda, x, ldb);
    x += x_plus;
    a += a_plus;
  }
}

template struct RealTrsm<float>;
template struct RealTrsm<double>;
template struct ComplexTrsm<std::complex<float>>;
template struct ComplexTrsm<std::complex<double>>;

// Getrf
// ~~~~~

template <typename T>
typename Getrf<T>::FnType* Getrf<T>::fn = nullptr;

template <typename T>
void Getrf<T>::Kernel(void* out_tuple, void** data, XlaCustomCallStatus*) {
  const int b = *(reinterpret_cast<int32_t*>(data[0]));
  const int m = *(reinterpret_cast<int32_t*>(data[1]));
  const int n = *(reinterpret_cast<int32_t*>(data[2]));
  const T* a_in = reinterpret_cast<T*>(data[3]);

  void** out = reinterpret_cast<void**>(out_tuple);
  T* a_out = reinterpret_cast<T*>(out[0]);
  int* ipiv = reinterpret_cast<int*>(out[1]);
  int* info = reinterpret_cast<int*>(out[2]);
  if (a_out != a_in) {
    std::memcpy(a_out, a_in,
                static_cast<int64_t>(b) * static_cast<int64_t>(m) *
                    static_cast<int64_t>(n) * sizeof(T));
  }
  for (int i = 0; i < b; ++i) {
    *info = fn(LAPACK_ROW_MAJOR, m, n, a_out, m, ipiv);
    a_out += static_cast<int64_t>(m) * static_cast<int64_t>(n);
    ipiv += std::min(m, n);
    ++info;
  }
}

template struct Getrf<float>;
template struct Getrf<double>;
template struct Getrf<std::complex<float>>;
template struct Getrf<std::complex<double>>;

// Geqrf
// ~~~~~

template <typename T>
typename Geqrf<T>::FnType* Geqrf<T>::fn = nullptr;

template <typename T>
void Geqrf<T>::Kernel(void* out_tuple, void** data, XlaCustomCallStatus*) {
  const int b = *(reinterpret_cast<int32_t*>(data[0]));
  const int m = *(reinterpret_cast<int32_t*>(data[1]));
  const int n = *(reinterpret_cast<int32_t*>(data[2]));
  const int lwork = *(reinterpret_cast<int32_t*>(data[3]));
  const T* a_in = reinterpret_cast<T*>(data[4]);

  void** out = reinterpret_cast<void**>(out_tuple);
  T* a_out = reinterpret_cast<T*>(out[0]);
  T* tau = reinterpret_cast<T*>(out[1]);
  int* info = reinterpret_cast<int*>(out[2]);
  T* work = reinterpret_cast<T*>(out[3]);

  if (a_out != a_in) {
    std::memcpy(a_out, a_in,
                static_cast<int64_t>(b) * static_cast<int64_t>(m) *
                    static_cast<int64_t>(n) * sizeof(T));
  }

  for (int i = 0; i < b; ++i) {
    *info = fn(LAPACK_ROW_MAJOR, m, n, a_out, m, tau);
    a_out += static_cast<int64_t>(m) * static_cast<int64_t>(n);
    tau += std::min(m, n);
    ++info;
  }
}

template struct Geqrf<float>;
template struct Geqrf<double>;
template struct Geqrf<std::complex<float>>;
template struct Geqrf<std::complex<double>>;

// Orgqr
// ~~~~~

template <typename T>
typename Orgqr<T>::FnType* Orgqr<T>::fn = nullptr;

template <typename T>
void Orgqr<T>::Kernel(void* out_tuple, void** data, XlaCustomCallStatus*) {
  const int b = *(reinterpret_cast<int32_t*>(data[0]));
  const int m = *(reinterpret_cast<int32_t*>(data[1]));
  const int n = *(reinterpret_cast<int32_t*>(data[2]));
  const int k = *(reinterpret_cast<int32_t*>(data[3]));
  const int lwork = *(reinterpret_cast<int32_t*>(data[4]));
  const T* a_in = reinterpret_cast<T*>(data[5]);
  T* tau = reinterpret_cast<T*>(data[6]);

  void** out = reinterpret_cast<void**>(out_tuple);
  T* a_out = reinterpret_cast<T*>(out[0]);
  int* info = reinterpret_cast<int*>(out[1]);
  T* work = reinterpret_cast<T*>(out[2]);

  if (a_out != a_in) {
    std::memcpy(a_out, a_in,
                static_cast<int64_t>(b) * static_cast<int64_t>(m) *
                    static_cast<int64_t>(n) * sizeof(T));
  }

  for (int i = 0; i < b; ++i) {
    *info = fn(LAPACK_ROW_MAJOR, m, n, k, a_out, m, tau);
    a_out += static_cast<int64_t>(m) * static_cast<int64_t>(n);
    tau += k;
    ++info;
  }
}

template struct Orgqr<float>;
template struct Orgqr<double>;
template struct Orgqr<std::complex<float>>;
template struct Orgqr<std::complex<double>>;

// Potrf
// ~~~~~

template <typename T>
typename Potrf<T>::FnType* Potrf<T>::fn = nullptr;

template <typename T>
void Potrf<T>::Kernel(void* out_tuple, void** data, XlaCustomCallStatus*) {
  const int32_t lower = *(reinterpret_cast<int32_t*>(data[0]));
  const int b = *(reinterpret_cast<int32_t*>(data[1]));
  const int n = *(reinterpret_cast<int32_t*>(data[2]));
  const T* a_in = reinterpret_cast<T*>(data[3]);
  const char uplo = lower ? 'L' : 'U';

  void** out = reinterpret_cast<void**>(out_tuple);
  T* a_out = reinterpret_cast<T*>(out[0]);
  int* info = reinterpret_cast<int*>(out[1]);
  if (a_out != a_in) {
    std::memcpy(a_out, a_in,
                static_cast<int64_t>(b) * static_cast<int64_t>(n) *
                    static_cast<int64_t>(n) * sizeof(T));
  }

  for (int i = 0; i < b; ++i) {
    *info = fn(LAPACK_ROW_MAJOR, uplo, n, a_out, n);
    a_out += static_cast<int64_t>(n) * static_cast<int64_t>(n);
    ++info;
  }
}

template struct Potrf<float>;
template struct Potrf<double>;
template struct Potrf<std::complex<float>>;
template struct Potrf<std::complex<double>>;

// Gesdd

static char GesddJobz(bool job_opt_compute_uv, bool job_opt_full_matrices) {
  if (!job_opt_compute_uv) {
    return 'N';
  } else if (!job_opt_full_matrices) {
    return 'S';
  }
  return 'A';
}

template <typename T>
typename RealGesdd<T>::FnType* RealGesdd<T>::fn = nullptr;

template <typename T>
void RealGesdd<T>::Kernel(void* out_tuple, void** data, XlaCustomCallStatus*) {
  const int32_t job_opt_full_matrices = *(reinterpret_cast<int32_t*>(data[0]));
  const int32_t job_opt_compute_uv = *(reinterpret_cast<int32_t*>(data[1]));
  const int b = *(reinterpret_cast<int32_t*>(data[2]));
  const int m = *(reinterpret_cast<int32_t*>(data[3]));
  const int n = *(reinterpret_cast<int32_t*>(data[4]));
  const int lwork = *(reinterpret_cast<int32_t*>(data[5]));
  T* a_in = reinterpret_cast<T*>(data[6]);

  void** out = reinterpret_cast<void**>(out_tuple);
  T* a_out = reinterpret_cast<T*>(out[0]);
  T* s = reinterpret_cast<T*>(out[1]);
  T* u = reinterpret_cast<T*>(out[2]);
  T* vt = reinterpret_cast<T*>(out[3]);
  int* info = reinterpret_cast<int*>(out[4]);
  int* iwork = reinterpret_cast<int*>(out[5]);
  T* work = reinterpret_cast<T*>(out[6]);

  if (a_out != a_in) {
    std::memcpy(a_out, a_in,
                static_cast<int64_t>(b) * static_cast<int64_t>(m) *
                    static_cast<int64_t>(n) * sizeof(T));
  }

  const char jobz = GesddJobz(job_opt_compute_uv, job_opt_full_matrices);

  const int lda = m;
  const int ldu = m;
  const int tdu = job_opt_full_matrices ? m : std::min(m, n);
  const int ldvt = job_opt_full_matrices ? n : std::min(m, n);

  for (int i = 0; i < b; ++i) {
    *info = fn(LAPACK_ROW_MAJOR, jobz, m, n, a_out, lda, s, u, ldu, vt, ldvt);
    a_out += static_cast<int64_t>(m) * n;
    s += std::min(m, n);
    u += static_cast<int64_t>(m) * tdu;
    vt += static_cast<int64_t>(ldvt) * n;
    ++info;
  }
}

template <typename T>
typename ComplexGesdd<T>::FnType* ComplexGesdd<T>::fn = nullptr;

template <typename T>
void ComplexGesdd<T>::Kernel(void* out_tuple, void** data,
                             XlaCustomCallStatus*) {
  const int32_t job_opt_full_matrices = *(reinterpret_cast<int32_t*>(data[0]));
  const int32_t job_opt_compute_uv = *(reinterpret_cast<int32_t*>(data[1]));
  const int b = *(reinterpret_cast<int32_t*>(data[2]));
  const int m = *(reinterpret_cast<int32_t*>(data[3]));
  const int n = *(reinterpret_cast<int32_t*>(data[4]));
  const int lwork = *(reinterpret_cast<int32_t*>(data[5]));
  T* a_in = reinterpret_cast<T*>(data[6]);

  void** out = reinterpret_cast<void**>(out_tuple);
  T* a_out = reinterpret_cast<T*>(out[0]);
  typename T::value_type* s = reinterpret_cast<typename T::value_type*>(out[1]);
  T* u = reinterpret_cast<T*>(out[2]);
  T* vt = reinterpret_cast<T*>(out[3]);
  int* info = reinterpret_cast<int*>(out[4]);
  int* iwork = reinterpret_cast<int*>(out[5]);
  typename T::value_type* rwork =
      reinterpret_cast<typename T::value_type*>(out[6]);
  T* work = reinterpret_cast<T*>(out[7]);

  if (a_out != a_in) {
    std::memcpy(a_out, a_in,
                static_cast<int64_t>(b) * static_cast<int64_t>(m) *
                    static_cast<int64_t>(n) * sizeof(T));
  }

  const char jobz = GesddJobz(job_opt_compute_uv, job_opt_full_matrices);

  const int lda = m;
  const int ldu = m;
  const int tdu = job_opt_full_matrices ? m : std::min(m, n);
  const int ldvt = job_opt_full_matrices ? n : std::min(m, n);

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

// Syevd/Heevd
// ~~~~~~~~~~~

template <typename T>
typename RealSyevd<T>::FnType* RealSyevd<T>::fn = nullptr;

template <typename T>
void RealSyevd<T>::Kernel(void* out_tuple, void** data, XlaCustomCallStatus*) {
  const int32_t lower = *(reinterpret_cast<int32_t*>(data[0]));
  const int b = *(reinterpret_cast<int32_t*>(data[1]));
  const int n = *(reinterpret_cast<int32_t*>(data[2]));
  const T* a_in = reinterpret_cast<T*>(data[3]);

  void** out = reinterpret_cast<void**>(out_tuple);
  T* a_out = reinterpret_cast<T*>(out[0]);
  T* w_out = reinterpret_cast<T*>(out[1]);
  int* info = reinterpret_cast<int*>(out[2]);
  T* work = reinterpret_cast<T*>(out[3]);
  int* iwork = reinterpret_cast<int*>(out[4]);
  if (a_out != a_in) {
    std::memcpy(a_out, a_in,
                static_cast<int64_t>(b) * static_cast<int64_t>(n) *
                    static_cast<int64_t>(n) * sizeof(T));
  }

  const char jobz = 'V';
  const char uplo = lower ? 'L' : 'U';

  for (int i = 0; i < b; ++i) {
    *info = fn(LAPACK_ROW_MAJOR, jobz, uplo, n, a_out, n, w_out);
    a_out += static_cast<int64_t>(n) * n;
    w_out += n;
    ++info;
  }
}

template <typename T>
typename ComplexHeevd<T>::FnType* ComplexHeevd<T>::fn = nullptr;

template <typename T>
void ComplexHeevd<T>::Kernel(void* out_tuple, void** data,
                             XlaCustomCallStatus*) {
  const int32_t lower = *(reinterpret_cast<int32_t*>(data[0]));
  const int b = *(reinterpret_cast<int32_t*>(data[1]));
  const int n = *(reinterpret_cast<int32_t*>(data[2]));
  const T* a_in = reinterpret_cast<T*>(data[3]);

  void** out = reinterpret_cast<void**>(out_tuple);
  T* a_out = reinterpret_cast<T*>(out[0]);
  typename T::value_type* w_out = reinterpret_cast<typename T::value_type*>(out[1]);
  int* info = reinterpret_cast<int*>(out[2]);
  T* work = reinterpret_cast<T*>(out[3]);
  typename T::value_type* rwork = reinterpret_cast<typename T::value_type*>(out[4]);
  int* iwork = reinterpret_cast<int*>(out[5]);
  if (a_out != a_in) {
    std::memcpy(a_out, a_in,
                static_cast<int64_t>(b) * static_cast<int64_t>(n) *
                    static_cast<int64_t>(n) * sizeof(T));
  }

  const char jobz = 'V';
  const char uplo = lower ? 'L' : 'U';

  for (int i = 0; i < b; ++i) {
    *info = fn(LAPACK_ROW_MAJOR, jobz, uplo, n, a_out, n, w_out);
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
static void UnpackEigenvectors(int n, const T* im_eigenvalues, const T* packed,
                               std::complex<T>* unpacked) {
  T re, im;
  int j;
  j = 0;
  while (j < n) {
    if (im_eigenvalues[j] == 0. || std::isnan(im_eigenvalues[j])) {
      for (int k = 0; k < n; ++k) {
        unpacked[j * n + k] = {packed[j * n + k], 0.};
      }
      ++j;
    } else {
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

// Geev
// ~~~~

template <typename T>
typename RealGeev<T>::FnType* RealGeev<T>::fn = nullptr;

template <typename T>
void RealGeev<T>::Kernel(void* out_tuple, void** data, XlaCustomCallStatus*) {
  const int b = *(reinterpret_cast<int32_t*>(data[0]));
  const int n_int = *(reinterpret_cast<int32_t*>(data[1]));
  const int64_t n = n_int;
  const char jobvl = *(reinterpret_cast<uint8_t*>(data[2]));
  const char jobvr = *(reinterpret_cast<uint8_t*>(data[3]));

  const T* a_in = reinterpret_cast<T*>(data[4]);

  void** out = reinterpret_cast<void**>(out_tuple);
  T* a_work = reinterpret_cast<T*>(out[0]);
  T* vl_work = reinterpret_cast<T*>(out[1]);
  T* vr_work = reinterpret_cast<T*>(out[2]);

  T* wr_out = reinterpret_cast<T*>(out[3]);
  T* wi_out = reinterpret_cast<T*>(out[4]);
  std::complex<T>* vl_out = reinterpret_cast<std::complex<T>*>(out[5]);
  std::complex<T>* vr_out = reinterpret_cast<std::complex<T>*>(out[6]);
  int* info = reinterpret_cast<int*>(out[7]);

  // TODO(phawkins): preallocate workspace using XLA.
  *info = fn(LAPACK_ROW_MAJOR, jobvl, jobvr, n_int, a_work, n_int, wr_out, wi_out,
             vl_work, n_int, vr_work, n_int);

  auto is_finite = [](T* a_work, int64_t n) {
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
      *info = fn(LAPACK_ROW_MAJOR, jobvl, jobvr, n_int, a_work, n_int, wr_out, wi_out,
                 vl_work, n_int, vr_work, n_int);
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
    } else {
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

template <typename T>
typename ComplexGeev<T>::FnType* ComplexGeev<T>::fn = nullptr;

template <typename T>
void ComplexGeev<T>::Kernel(void* out_tuple, void** data,
                            XlaCustomCallStatus*) {
  const int b = *(reinterpret_cast<int32_t*>(data[0]));
  const int n_int = *(reinterpret_cast<int32_t*>(data[1]));
  const int64_t n = n_int;
  const char jobvl = *(reinterpret_cast<uint8_t*>(data[2]));
  const char jobvr = *(reinterpret_cast<uint8_t*>(data[3]));

  const T* a_in = reinterpret_cast<T*>(data[4]);

  void** out = reinterpret_cast<void**>(out_tuple);
  T* a_work = reinterpret_cast<T*>(out[0]);
  typename T::value_type* r_work =
      reinterpret_cast<typename T::value_type*>(out[1]);

  T* w_out = reinterpret_cast<T*>(out[2]);
  T* vl_out = reinterpret_cast<T*>(out[3]);
  T* vr_out = reinterpret_cast<T*>(out[4]);
  int* info = reinterpret_cast<int*>(out[5]);

  // TODO(phawkins): preallocate workspace using XLA.
  *info = fn(LAPACK_ROW_MAJOR, jobvl, jobvr, n_int, a_work, n_int, w_out, vl_out,
             n_int, vr_out, n_int);

  auto is_finite = [](T* a_work, int64_t n) {
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
      *info = fn(LAPACK_ROW_MAJOR, jobvl, jobvr, n_int, a_work, n_int, w_out, vl_out, n_int, vr_out, n_int);
#ifdef USE_ABSEIL_LIB
      ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(a_work, a_size);
      ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(w_out, sizeof(T) * n);
      ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(vl_out, sizeof(T) * n * n);
      ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(vr_out, sizeof(T) * n * n);
      ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(info_out, sizeof(int));
#endif
    } else {
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

// Gees
// ~~~~

template <typename T>
typename RealGees<T>::FnType* RealGees<T>::fn = nullptr;

template <typename T>
void RealGees<T>::Kernel(void* out_tuple, void** data, XlaCustomCallStatus*) {
  const int b = *(reinterpret_cast<int32_t*>(data[0]));
  const int n_int = *(reinterpret_cast<int32_t*>(data[1]));
  const int64_t n = n_int;
  const char jobvs = *(reinterpret_cast<uint8_t*>(data[2]));
  const char sort = *(reinterpret_cast<uint8_t*>(data[3]));

  const T* a_in = reinterpret_cast<T*>(data[4]);

  // bool* select (T, T) = reinterpret_cast<bool* (T, T)>(data[5]);
  bool (*select)(T, T) = nullptr;

  void** out = reinterpret_cast<void**>(out_tuple);
  T* a_out = reinterpret_cast<T*>(out[0]);

  T* wr_out = reinterpret_cast<T*>(out[1]);
  T* wi_out = reinterpret_cast<T*>(out[2]);
  T* vs_out = reinterpret_cast<T*>(out[3]);
  int* sdim_out = reinterpret_cast<int*>(out[4]);
  int* info = reinterpret_cast<int*>(out[5]);

  *info = fn(LAPACK_ROW_MAJOR, jobvs, sort, select, n_int, a_out, n_int, sdim_out,
             wr_out, wi_out, vs_out, n_int);

  size_t a_size = static_cast<int64_t>(n) * static_cast<int64_t>(n) * sizeof(T);
  if (a_out != a_in) {
    std::memcpy(a_out, a_in, static_cast<int64_t>(b) * a_size);
  }

  for (int i = 0; i < b; ++i) {
    *info = fn(LAPACK_ROW_MAJOR, jobvs, sort, select, n_int, a_out, n_int, sdim_out,
               wr_out, wi_out, vs_out, n_int);
#ifdef USE_ABSEIL_LIB
    ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(a_out, a_size);
    ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(sdim_out, sizeof(int));
    ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(wr_out, sizeof(T) * n);
    ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(wi_out, sizeof(T) * n);
    ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(vs_out, sizeof(T) * n * n);
    ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(info_out, sizeof(int));
#endif

    a_in += n * n;
    a_out += n * n;
    wr_out += n;
    wi_out += n;
    vs_out += n * n;
    ++sdim_out;
    ++info;
  }
}

template <typename T>
typename ComplexGees<T>::FnType* ComplexGees<T>::fn = nullptr;

template <typename T>
void ComplexGees<T>::Kernel(void* out_tuple, void** data,
                            XlaCustomCallStatus*) {
  const int b = *(reinterpret_cast<int32_t*>(data[0]));
  const int n_int = *(reinterpret_cast<int32_t*>(data[1]));
  const int64_t n = n_int;
  const char jobvs = *(reinterpret_cast<uint8_t*>(data[2]));
  const char sort = *(reinterpret_cast<uint8_t*>(data[3]));

  const T* a_in = reinterpret_cast<T*>(data[4]);

  // bool* select (T, T) = reinterpret_cast<bool* (T, T)>(data[5]);
  bool (*select)(T) = nullptr;

  void** out = reinterpret_cast<void**>(out_tuple);
  T* a_out = reinterpret_cast<T*>(out[0]);
  typename T::value_type* r_work =
      reinterpret_cast<typename T::value_type*>(out[1]);
  T* w_out = reinterpret_cast<T*>(out[2]);
  T* vs_out = reinterpret_cast<T*>(out[3]);
  int* sdim_out = reinterpret_cast<int*>(out[4]);
  int* info = reinterpret_cast<int*>(out[5]);

  *info = fn(LAPACK_ROW_MAJOR, jobvs, sort, select, n_int, a_out, n_int, sdim_out,
             w_out, vs_out, n_int);

  if (a_out != a_in) {
    std::memcpy(a_out, a_in,
                static_cast<int64_t>(b) * static_cast<int64_t>(n) *
                    static_cast<int64_t>(n) * sizeof(T));
  }

  for (int i = 0; i < b; ++i) {
    *info = fn(LAPACK_ROW_MAJOR, jobvs, sort, select, n_int, a_out, n_int, sdim_out,
               w_out, vs_out, n_int);
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

// Gehrd

template <typename T>
typename Gehrd<T>::FnType* Gehrd<T>::fn = nullptr;

template <typename T>
void Gehrd<T>::Kernel(void* out_tuple, void** data, XlaCustomCallStatus*) {
  const int32_t n = *reinterpret_cast<int32_t*>(data[0]);
  const int32_t ilo = *reinterpret_cast<int32_t*>(data[1]);
  const int32_t ihi = *reinterpret_cast<int32_t*>(data[2]);
  const int32_t lda = *reinterpret_cast<int32_t*>(data[3]);
  const int32_t batch = *reinterpret_cast<int32_t*>(data[4]);
  const int32_t lwork = *reinterpret_cast<int32_t*>(data[5]);
  T* a = reinterpret_cast<T*>(data[6]);

  void** out = reinterpret_cast<void**>(out_tuple);
  T* a_out = reinterpret_cast<T*>(out[0]);
  T* tau = reinterpret_cast<T*>(out[1]);
  int* info = reinterpret_cast<int*>(out[2]);
  T* work = reinterpret_cast<T*>(out[3]);

  if (a_out != a) {
    std::memcpy(a_out, a,
                static_cast<int64_t>(batch) * static_cast<int64_t>(n) *
                    static_cast<int64_t>(n) * sizeof(T));
  }

  const int64_t a_plus = static_cast<int64_t>(lda) * static_cast<int64_t>(n);

  for (int i = 0; i < batch; ++i) {
    *info = fn(LAPACK_ROW_MAJOR, n, ilo, ihi, a_out, lda, tau);
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

template <typename T>
typename Sytrd<T>::FnType* Sytrd<T>::fn = nullptr;

template <typename T>
void Sytrd<T>::Kernel(void* out_tuple, void** data, XlaCustomCallStatus*) {
  const int32_t n = *reinterpret_cast<int32_t*>(data[0]);
  const int32_t lower = *reinterpret_cast<int32_t*>(data[1]);
  const int32_t lda = *reinterpret_cast<int32_t*>(data[2]);
  const int32_t batch = *reinterpret_cast<int32_t*>(data[3]);
  const int32_t lwork = *reinterpret_cast<int32_t*>(data[4]);
  T* a = reinterpret_cast<T*>(data[5]);

  void** out = reinterpret_cast<void**>(out_tuple);
  T* a_out = reinterpret_cast<T*>(out[0]);
  typedef typename real_type<T>::type Real;
  Real* d = reinterpret_cast<Real*>(out[1]);
  Real* e = reinterpret_cast<Real*>(out[2]);
  T* tau = reinterpret_cast<T*>(out[3]);
  int* info = reinterpret_cast<int*>(out[4]);
  T* work = reinterpret_cast<T*>(out[5]);

  if (a_out != a) {
    std::memcpy(a_out, a,
                static_cast<int64_t>(batch) * static_cast<int64_t>(n) *
                    static_cast<int64_t>(n) * sizeof(T));
  }

  const char cuplo = lower ? 'L' : 'U';

  const int64_t a_plus = static_cast<int64_t>(lda) * static_cast<int64_t>(n);

  for (int i = 0; i < batch; ++i) {
    *info = fn(LAPACK_ROW_MAJOR, cuplo, n, a_out, lda, d, e, tau);
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

}  // namespace jax
