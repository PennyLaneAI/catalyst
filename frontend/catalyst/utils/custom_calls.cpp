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

#include <optional>
#include <string>
#include <cmath>
#include <cstdint>

extern "C"{

struct XlaCustomCallStatus_ {
    std::optional<std::string> message;
};
typedef struct XlaCustomCallStatus_ XlaCustomCallStatus;

}
namespace jax {

typedef int lapack_int;
template <typename T> struct RealGesdd {
    using FnType = void(char *jobz, lapack_int *m, lapack_int *n, T *a, lapack_int *lda, T *s, T *u,
                        lapack_int *ldu, T *vt, lapack_int *ldvt, T *work, lapack_int *lwork,
                        lapack_int *iwork, lapack_int *info);
    static FnType *fn;
    static void Kernel(void *out, void **data, XlaCustomCallStatus *);

    static int64_t Workspace(lapack_int m, lapack_int n, bool job_opt_compute_uv,
                             bool job_opt_full_matrices);
};


} // namespace jax
extern "C" {

enum NumericType : int8_t {
    idx = 0,
    i1,
    i8,
    i16,
    i32,
    i64,
    f32,
    f64,
    c64,
    c128,
};

//   MemRef encoded as:
//   { i8: dtype, i8: rank, ptr<i8>: data,
//     array<2*rank x i64>: sizes_and_strides }
struct EncodedMemref {
  uint8_t dtype;
  uint8_t rank;
  void* data;
  int64_t dims[];
};

void lapack_dgesdd(EncodedMemref args, EncodedMemref results)
{
    void **data;
    void *res;
    XlaCustomCallStatus status = XlaCustomCallStatus();
    XlaCustomCallStatus *statusPointer = &status;
    jax::RealGesdd<float>::Kernel(res, data, statusPointer);
}
}