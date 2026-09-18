// Copyright 2026 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// RUN: quantum-opt --split-input-file --verify-diagnostics %s

func.func @push_block_element_type(%list: !catalyst.arraylist<f64>, %values: tensor<2xf32>) {
    // expected-error @below {{expects the element type of 'elements' (f32) to match the element type of the list (f64)}}
    catalyst.list_push_block %values, %list : tensor<2xf32>, !catalyst.arraylist<f64>
    return
}

// -----

// A rank-0 tensor holds a single element and is cached with catalyst.list_push instead: it has no
// rank-1 buffer for the list to copy from.

func.func @push_block_rank_zero(%list: !catalyst.arraylist<f64>, %value: tensor<f64>) {
    // expected-error @below {{expects 'elements' to have rank at least 1}}
    catalyst.list_push_block %value, %list : tensor<f64>, !catalyst.arraylist<f64>
    return
}

// -----

// A strided buffer is not a contiguous run of elements, so it cannot be the list's storage.

func.func @pop_block_strided(%list: !catalyst.arraylist<f64>,
                             %destination: memref<4xf64, strided<[2]>>) {
    // expected-error @below {{expects 'destination' to have an identity layout}}
    catalyst.list_pop_block %list, %destination
      : !catalyst.arraylist<f64>, memref<4xf64, strided<[2]>>
    return
}

// -----

// Before bufferization the popped elements are returned as a tensor; after it they are written into
// the destination buffer in place.

func.func @pop_block_missing_result(%list: !catalyst.arraylist<f64>, %destination: tensor<4xf64>) {
    // expected-error @below {{expects a result if and only if 'destination' is a tensor}}
    catalyst.list_pop_block %list, %destination : !catalyst.arraylist<f64>, tensor<4xf64>
    return
}
