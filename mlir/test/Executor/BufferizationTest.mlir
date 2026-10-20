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

// RUN: quantum-opt --split-input-file --verify-diagnostics \
// RUN:   --pass-pipeline="builtin.module( \
// RUN:     one-shot-bufferize{unknown-type-conversion=identity-layout-map} \
// RUN:   )" %s | FileCheck %s

// Allocate output buffer for the reply.

// CHECK-LABEL: func.func @call_with_reply
// CHECK: [[IN:%.+]] = bufferization.to_buffer %arg1
// CHECK: [[OUT:%.+]] = memref.alloc() {{.*}}: memref<2xf64>
// CHECK: executor.call %arg0("remote_run") ([[IN]], [[OUT]]) {num_input_args = 1 : i32}
// CHECK-SAME: (memref<1xi64>, memref<2xf64>) -> ()
// CHECK: [[TENSOR:%.+]] = bufferization.to_tensor [[OUT]]
// CHECK: return [[TENSOR]]
func.func @call_with_reply(%session: !executor.session, %arg0: tensor<1xi64>) -> tensor<2xf64> {
    %0 = executor.call %session("remote_run") (%arg0) : !executor.session, (tensor<1xi64>) -> tensor<2xf64>
    return %0 : tensor<2xf64>
}

// -----

// Bufferize input.

// CHECK-LABEL: func.func @call_without_reply
// CHECK: [[IN:%.+]] = bufferization.to_buffer %arg1
// CHECK: executor.call %arg0("remote_log") ([[IN]]) {num_input_args = 1 : i32}
// CHECK-SAME: (memref<1xi64>) -> ()
func.func @call_without_reply(%session: !executor.session, %arg0: tensor<1xi64>) {
    executor.call %session("remote_log") (%arg0) : !executor.session, (tensor<1xi64>) -> ()
    return
}

// -----

// Make a strided input contiguous.

// CHECK-LABEL: func.func @strided_input
// CHECK: [[VIEW:%.+]] = memref.subview
// CHECK: [[COPY:%.+]] = memref.alloc() {{.*}}: memref<4xi64>
// CHECK: memref.copy [[VIEW]], [[COPY]]
// CHECK: executor.call %arg0("remote_log") ([[COPY]]) {num_input_args = 1 : i32}
// CHECK-SAME: (memref<4xi64>) -> ()
func.func @strided_input(%session: !executor.session, %arg0: tensor<8xi64>) {
    %slice = tensor.extract_slice %arg0[0] [4] [2] : tensor<8xi64> to tensor<4xi64>
    executor.call %session("remote_log") (%slice) : !executor.session, (tensor<4xi64>) -> ()
    return
}

// -----

// Already bufferized test.

// CHECK-LABEL: func.func @already_bufferized
// CHECK: executor.call %arg0("remote_run") (%arg1, %arg2) {num_input_args = 1 : i32}
// CHECK-SAME: (memref<1xi64>, memref<2xf64>) -> ()
func.func @already_bufferized(%session: !executor.session, %in: memref<1xi64>, %out: memref<2xf64>) {
    executor.call %session("remote_run") (%in, %out) {num_input_args = 1 : i32} : !executor.session, (memref<1xi64>, memref<2xf64>) -> ()
    return
}
