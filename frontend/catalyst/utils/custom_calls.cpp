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

// #include<lapack>

// extern "C" {

// enum NumericType : int8_t {
//     idx = 0,
//     i1,
//     i8,
//     i16,
//     i32,
//     i64,
//     f32,
//     f64,
//     c64,
//     c128,
// };

// struct OpaqueMemRef {
//     int64_t rank;
//     void *data;
//     NumericType datatype;
// };

// void *lapack_dgesdd(OpaqueMemRef args, OpaqueMemRef res) {
//     void** args = get_bufffer(args);
//     void* results = get_buffer(res);

// }
// }