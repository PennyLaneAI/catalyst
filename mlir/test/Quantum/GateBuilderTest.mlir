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

// RUN: quantum-opt --test-gate-builder --split-input-file -verify-diagnostics %s | FileCheck %s


%true = llvm.mlir.constant (1 : i1) :i1
%angle = arith.constant 37.420000e+00 : f64
%_ = quantum.alloc( 2) : !quantum.reg
%IN = quantum.extract %_[ 0] : !quantum.reg -> !quantum.bit

// CHECK: [[TRUE:%.+]] = llvm.mlir.constant(
// CHECK: [[ANGLE:%.+]] = arith.constant {{.+}} : f64
// CHECK: [[IN:%.+]] = quantum.extract {{%.+}}[ 0] : !quantum.reg -> !quantum.bit
// CHECK: [[PZ:%.+]] = quantum.custom "PauliZ"() [[IN]] : !quantum.bit
// CHECK:  {{%.+}} = quantum.custom "PauliY"() [[IN]] {adjoint} : !quantum.bit
// CHECK: [[RX:%.+]] = quantum.custom "RX"([[ANGLE]]) [[IN]] : !quantum.bit
// CHECK:  {{%.+}}:2 = quantum.custom "SWAP"() [[IN]], [[PZ]] : !quantum.bit
// CHECK:  {{%.+}}:2, {{%.+}} = quantum.custom "Rot"([[ANGLE]]) [[IN]], [[PZ]] ctrls([[RX]]) ctrlvals([[TRUE]]) : !quantum.bit, !quantum.bit ctrls !quantum.bit
// CHECK:  {{%.+}}:2, {{%.+}} = quantum.custom "my_controlled_U"() [[IN]], [[PZ]] ctrls([[RX]]) ctrlvals([[TRUE]]) : !quantum.bit, !quantum.bit ctrls !quantum.bit
