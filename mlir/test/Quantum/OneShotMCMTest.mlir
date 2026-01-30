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

// RUN: quantum-opt --one-shot-mcm --split-input-file -verify-diagnostics %s | FileCheck %s

func.func public @test_expval(%arg0: f64) -> tensor<f64> {
  %1000 = arith.constant 1000 : i64
  quantum.device shots(%1000) ["", "", ""]
  %0 = quantum.alloc( 2) : !quantum.reg
  %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
  %out_qubits = quantum.custom "RX"(%arg0) %1 : !quantum.bit
  %mres, %out_qubit = quantum.measure %out_qubits : i1, !quantum.bit
  %2 = quantum.namedobs %out_qubit[ PauliZ] : !quantum.obs
  %3 = quantum.expval %2 : f64
  %from_elements = tensor.from_elements %3 : tensor<f64>
  %4 = quantum.insert %0[ 0], %out_qubit : !quantum.reg, !quantum.bit
  quantum.dealloc %4 : !quantum.reg
  quantum.device_release
  return %from_elements : tensor<f64>
}


// CHECK: func.func public @test_expval.quantum_kernel(%arg0: f64) -> tensor<f64>
// CHECK:  [[one:%.+]] = arith.constant 1 : i64
// CHECK:  quantum.device shots([[one]]) ["", "", ""]

// CHECK: func.func public @test_expval(%arg0: f64) -> tensor<f64> {
// CHECK:   [[shots:%.+]] = arith.constant 1000 : i64
// CHECK:   [[loopIterSum:%.+]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK:   [[lb:%.+]] = arith.constant 0 : index
// CHECK:   [[step:%.+]] = arith.constant 1 : index
// CHECK:   [[ub:%.+]] = index.casts [[shots]] : i64 to index
// CHECK:   [[totalSum:%.+]] = scf.for %arg1 = [[lb]] to [[ub]] step [[step]] iter_args(%arg2 = [[loopIterSum]]) -> (tensor<f64>) {
// CHECK:     [[call:%.+]] = func.call @test_expval.quantum_kernel(%arg0) : (f64) -> tensor<f64>
// CHECK:     [[add:%.+]] = stablehlo.add [[call]], %arg2 : tensor<f64>
// CHECK:     scf.yield [[add]] : tensor<f64>
// CHECK:   [[castShots:%.+]] = arith.sitofp [[shots]] : i64 to f64
// CHECK:   [[fromElem:%.+]] = tensor.from_elements [[castShots]] : tensor<f64>
// CHECK:   [[broadcastShots:%.+]] = stablehlo.broadcast_in_dim [[fromElem]], dims = [] : (tensor<f64>) -> tensor<f64>
// CHECK:   [[div:%.+]] = stablehlo.divide [[totalSum]], [[broadcastShots]] : tensor<f64>
// CHECK:   return [[div]] : tensor<f64>


// -----


func.func public @test_probs(%arg0: f64) -> tensor<4xf64> {
  %1000 = arith.constant 1000 : i64
  quantum.device shots(%1000) ["", "", ""]
  %0 = quantum.alloc( 2) : !quantum.reg
  %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
  %out_qubits = quantum.custom "RX"(%arg0) %1 : !quantum.bit
  %mres, %out_qubit = quantum.measure %out_qubits : i1, !quantum.bit
  %2 = quantum.insert %0[ 0], %out_qubit : !quantum.reg, !quantum.bit
  %3 = quantum.compbasis qreg %2 : !quantum.obs
  %4 = quantum.probs %3 : tensor<4xf64>
  quantum.dealloc %2 : !quantum.reg
  quantum.device_release
  return %4 : tensor<4xf64>
}

// CHECK: func.func public @test_probs.quantum_kernel(%arg0: f64) -> tensor<4xf64>
// CHECK:   [[one:%.+]] = arith.constant 1 : i64
// CHECK:   quantum.device shots([[one]]) ["", "", ""]

// CHECK: func.func public @test_probs(%arg0: f64) -> tensor<4xf64> {
// CHECK:   [[shots:%.+]] = arith.constant 1000 : i64
// CHECK:   [[loopIterSum:%.+]] = stablehlo.constant dense<0.000000e+00> : tensor<4xf64>
// CHECK:   [[lb:%.+]] = arith.constant 0 : index
// CHECK:   [[step:%.+]] = arith.constant 1 : index
// CHECK:   [[ub:%.+]] = index.casts [[shots]] : i64 to index
// CHECK:   [[totalSum:%.+]] = scf.for %arg1 = [[lb]] to [[ub]] step [[step]] iter_args(%arg2 = [[loopIterSum]]) -> (tensor<4xf64>) {
// CHECK:     [[call:%.+]] = func.call @test_probs.quantum_kernel(%arg0) : (f64) -> tensor<4xf64>
// CHECK:     [[add:%.+]] = stablehlo.add [[call]], %arg2 : tensor<4xf64>
// CHECK:     scf.yield [[add]] : tensor<4xf64>
// CHECK:   [[castShots:%.+]] = arith.sitofp [[shots]] : i64 to f64
// CHECK:   [[fromElem:%.+]] = tensor.from_elements [[castShots]] : tensor<f64>
// CHECK:   [[broadcastShots:%.+]] = stablehlo.broadcast_in_dim [[fromElem]], dims = [] : (tensor<f64>) -> tensor<4xf64>
// CHECK:   [[div:%.+]] = stablehlo.divide [[totalSum]], [[broadcastShots]] : tensor<4xf64>
// CHECK:   return [[div]] : tensor<4xf64>


// -----


func.func public @test_sample(%arg0: f64) -> tensor<1000x2xi64> {
  %1000 = arith.constant 1000 : i64
  quantum.device shots(%1000) ["", "", ""]
  %0 = quantum.alloc( 2) : !quantum.reg
  %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
  %out_qubits = quantum.custom "RX"(%arg0) %1 : !quantum.bit
  %mres, %out_qubit = quantum.measure %out_qubits : i1, !quantum.bit
  %2 = quantum.insert %0[ 0], %out_qubit : !quantum.reg, !quantum.bit
  %3 = quantum.compbasis qreg %2 : !quantum.obs
  %4 = quantum.sample %3 : tensor<1000x2xf64>
  %5 = stablehlo.convert %4 : (tensor<1000x2xf64>) -> tensor<1000x2xi64>
  quantum.dealloc %2 : !quantum.reg
  quantum.device_release
  return %5 : tensor<1000x2xi64>
}


// CHECK: func.func public @test_sample.quantum_kernel(%arg0: f64) -> tensor<1x2xi64>
// CHECK:   [[one:%.+]] = arith.constant 1 : i64
// CHECK:   quantum.device shots([[one]]) ["", "", ""]
// CHECK:   [[sample:%.+]] = quantum.sample {{%.+}} : tensor<1x2xf64>
// CHECK:   [[cast:%.+]] = stablehlo.convert [[sample]] : (tensor<1x2xf64>) -> tensor<1x2xi64>
// CHECK:   return [[cast]] : tensor<1x2xi64>

// CHECK: func.func public @test_sample(%arg0: f64) -> tensor<1000x2xi64> {
// CHECK:   [[shots:%.+]] = arith.constant 1000 : i64
// CHECK:   [[empty:%.+]] = tensor.empty() : tensor<1000x2xi64>
// CHECK:   [[lb:%.+]] = arith.constant 0 : index
// CHECK:   [[step:%.+]] = arith.constant 1 : index
// CHECK:   [[ub:%.+]] = index.casts [[shots]] : i64 to index
// CHECK:   [[fullSamples:%.+]] = scf.for %arg1 = [[lb]] to [[ub]] step [[step]] iter_args(%arg2 = [[empty]]) -> (tensor<1000x2xi64>) {
// CHECK:      [[call:%.+]] = func.call @test_sample.quantum_kernel(%arg0) : (f64) -> tensor<1x2xi64>
// CHECK:      [[insert:%.+]] = tensor.insert_slice [[call]] into %arg2[%arg1, 0] [1, 2] [1, 1] : tensor<1x2xi64> into tensor<1000x2xi64>
// CHECK:      scf.yield [[insert]] : tensor<1000x2xi64>
// CHECK:    return [[fullSamples]] : tensor<1000x2xi64>
