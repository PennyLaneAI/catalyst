// Copyright 2023 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// RUN: quantum-opt %s --tensor-init-lowering --split-input-file --verify-diagnostics | FileCheck %s

// CHECK-LABEL: tinit
func.func public @tinit(%arg: tensor<3xi32>) -> tensor<?x?x?xi32> attributes {llvm.emit_c_interface} {
  // CHECK: [[c:%.*]] = arith.constant 0 : i32
  // CHECK: [[a:%.*]] = tensor.empty{{[^:]*}} : tensor<?x?x?xi32>
  // CHECK: [[r:%.*]] = linalg.fill
  // CHECK-SAME:                    ins([[c]] : i32)
  // CHECK-SAME:                    outs([[a]] : tensor<?x?x?xi32>)
  // CHECK: return [[r]]
  %a = catalyst.tensor_init %arg {initializer = 0: i32} : tensor<3xi32> -> tensor<?x?x?xi32>
  return %a : tensor<?x?x?xi32>
}


// CHECK-LABEL: tinit2
func.func public @tinit2(%arg: tensor<2xi32>) -> tensor<?x?xf16> attributes {llvm.emit_c_interface} {
  // CHECK: [[c:%.*]] = arith.constant {{1\.0+e\+00}} : f16
  // CHECK: [[a:%.*]] = tensor.empty{{[^:]*}} : tensor<?x?xf16>
  // CHECK: [[r:%.*]] = linalg.fill
  // CHECK-SAME:                    ins([[c]] : f16)
  // CHECK-SAME:                    outs([[a]] : tensor<?x?xf16>)
  // CHECK: return [[r]]
  %a = catalyst.tensor_init %arg {initializer = 1.0: f16} : tensor<2xi32> -> tensor<?x?xf16>
  return %a : tensor<?x?xf16>
}


// CHECK-LABEL: tinit3
func.func public @tinit3(%arg: tensor<2xi32>) -> tensor<?x?xf16> attributes {llvm.emit_c_interface} {
  // CHECK: [[r:%.*]] = tensor.empty{{[^:]*}} : tensor<?x?xf16>
  // CHECK: return [[r]]
  %a = catalyst.tensor_init %arg {empty, initializer = 1.0: f16} : tensor<2xi32> -> tensor<?x?xf16>
  return %a : tensor<?x?xf16>
}

