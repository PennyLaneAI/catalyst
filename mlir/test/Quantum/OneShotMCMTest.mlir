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
  %out_qubit = quantum.custom "RX"(%arg0) %1 : !quantum.bit
  %2 = quantum.namedobs %out_qubit[ PauliZ] : !quantum.obs
  %3 = quantum.expval %2 : f64
  %from_elements = tensor.from_elements %3 : tensor<f64>
  %4 = quantum.insert %0[ 0], %out_qubit : !quantum.reg, !quantum.bit
  quantum.dealloc %4 : !quantum.reg
  quantum.device_release
  return %from_elements : tensor<f64>
}


// CHECK: func.func public @test_expval.one_shot_kernel(%arg0: f64) -> tensor<f64>
// CHECK:  [[one:%.+]] = arith.constant 1 : i64
// CHECK:  quantum.device shots([[one]]) ["", "", ""]

// CHECK: func.func public @test_expval(%arg0: f64) -> tensor<f64> {
// CHECK:   [[shots:%.+]] = arith.constant 1000 : i64
// CHECK:   [[loopIterSum:%.+]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK:   [[lb:%.+]] = arith.constant 0 : index
// CHECK:   [[step:%.+]] = arith.constant 1 : index
// CHECK:   [[ub:%.+]] = index.casts [[shots]] : i64 to index
// CHECK:   [[totalSum:%.+]] = scf.for %arg1 = [[lb]] to [[ub]] step [[step]] iter_args(%arg2 = [[loopIterSum]]) -> (tensor<f64>) {
// CHECK:     [[call:%.+]] = func.call @test_expval.one_shot_kernel(%arg0) : (f64) -> tensor<f64>
// CHECK:     [[add:%.+]] = stablehlo.add [[call]], %arg2 : tensor<f64>
// CHECK:     scf.yield [[add]] : tensor<f64>
// CHECK:   [[castShots:%.+]] = arith.sitofp [[shots]] : i64 to f64
// CHECK:   [[fromElem:%.+]] = tensor.from_elements [[castShots]] : tensor<f64>
// CHECK:   [[div:%.+]] = stablehlo.divide [[totalSum]], [[fromElem]] : tensor<f64>
// CHECK:   return [[div]] : tensor<f64>


// -----


func.func public @test_expval_mcm(%arg0: f64) -> tensor<f64> {
  %1000 = arith.constant 1000 : i64
  quantum.device shots(%1000) ["", "", ""]
  %0 = quantum.alloc( 2) : !quantum.reg
  %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
  %out_qubits = quantum.custom "RX"(%arg0) %1 : !quantum.bit
  %mres, %out_qubit = quantum.measure %out_qubits : i1, !quantum.bit
  %2 = quantum.mcmobs %mres : !quantum.obs
  %3 = quantum.expval %2 : f64
  %from_elements = tensor.from_elements %3 : tensor<f64>
  %4 = quantum.insert %0[ 0], %out_qubit : !quantum.reg, !quantum.bit
  quantum.dealloc %4 : !quantum.reg
  quantum.device_release
  return %from_elements : tensor<f64>
}


// CHECK:   func.func public @test_expval_mcm.one_shot_kernel(%arg0: f64) -> tensor<f64>
// CHECK:     [[one:%.+]] = arith.constant 1 : i64
// CHECK:     quantum.device shots([[one]]) ["", "", ""]
// CHECK:     [[mres:%.+]], {{%.+}} = quantum.measure {{%.+}} : i1, !quantum.bit
// CHECK-NOT:    quantum.mcmobs
// CHECK-NOT:    quantum.expval
// CHECK:     quantum.dealloc %2 : !quantum.reg
// CHECK:     quantum.device_release
// CHECK:     [[extend:%.+]] = arith.extui [[mres]] : i1 to i64
// CHECK:     [[cast:%.+]] = arith.sitofp [[extend]] : i64 to f64
// CHECK:     [[fromElements:%.+]] = tensor.from_elements [[cast]] : tensor<f64>
// CHECK:     return [[fromElements]] : tensor<f64>
//
// CHECK:   func.func public @test_expval_mcm(%arg0: f64) -> tensor<f64>
// CHECK:      scf.for
// CHECK:      func.call @test_expval_mcm.one_shot_kernel(%arg0)
// CHECK:      stablehlo.add
// CHECK:      stablehlo.divide


// -----


func.func public @test_var_mcm(%arg0: f64) -> tensor<f64> {
  %1000 = arith.constant 1000 : i64
  quantum.device shots(%1000) ["", "", ""]
  %0 = quantum.alloc( 2) : !quantum.reg
  %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
  %out_qubits = quantum.custom "RX"(%arg0) %1 : !quantum.bit
  %mres, %out_qubit = quantum.measure %out_qubits : i1, !quantum.bit
  %2 = quantum.mcmobs %mres : !quantum.obs
  %3 = quantum.var %2 : f64
  %from_elements = tensor.from_elements %3 : tensor<f64>
  %4 = quantum.insert %0[ 0], %out_qubit : !quantum.reg, !quantum.bit
  quantum.dealloc %4 : !quantum.reg
  quantum.device_release
  return %from_elements : tensor<f64>
}

// CHECK:   func.func public @test_var_mcm.one_shot_kernel(%arg0: f64) -> tensor<f64>
// CHECK:     [[one:%.+]] = arith.constant 1 : i64
// CHECK:     quantum.device shots([[one]]) ["", "", ""]
// CHECK:     [[mres:%.+]], {{%.+}} = quantum.measure {{%.+}} : i1, !quantum.bit
// CHECK-NOT:    quantum.mcmobs
// CHECK-NOT:    quantum.expval
// CHECK:     quantum.dealloc %2 : !quantum.reg
// CHECK:     quantum.device_release
// CHECK:     [[extend:%.+]] = arith.extui [[mres]] : i1 to i64
// CHECK:     [[cast:%.+]] = arith.sitofp [[extend]] : i64 to f64
// CHECK:     [[fromElements:%.+]] = tensor.from_elements [[cast]] : tensor<f64>
// CHECK:     return [[fromElements]] : tensor<f64>
//
// CHECK:   func.func public @test_var_mcm(%arg0: f64) -> tensor<f64>
// CHECK:      scf.for
// CHECK:      func.call @test_var_mcm.one_shot_kernel(%arg0)
// CHECK:      stablehlo.add
// CHECK:      [[expval:%.+]] = stablehlo.divide
// CHECK:      [[square:%.+]] = stablehlo.multiply [[expval]], [[expval]]
// CHECK:      [[variance:%.+]] = stablehlo.subtract [[expval]], [[square]]
// CHECK:      return [[variance]]


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

// CHECK: func.func public @test_probs.one_shot_kernel(%arg0: f64) -> tensor<4xf64>
// CHECK:   [[one:%.+]] = arith.constant 1 : i64
// CHECK:   quantum.device shots([[one]]) ["", "", ""]

// CHECK: func.func public @test_probs(%arg0: f64) -> tensor<4xf64> {
// CHECK:   [[shots:%.+]] = arith.constant 1000 : i64
// CHECK:   [[loopIterSum:%.+]] = stablehlo.constant dense<0.000000e+00> : tensor<4xf64>
// CHECK:   [[lb:%.+]] = arith.constant 0 : index
// CHECK:   [[step:%.+]] = arith.constant 1 : index
// CHECK:   [[ub:%.+]] = index.casts [[shots]] : i64 to index
// CHECK:   [[totalSum:%.+]] = scf.for %arg1 = [[lb]] to [[ub]] step [[step]] iter_args(%arg2 = [[loopIterSum]]) -> (tensor<4xf64>) {
// CHECK:     [[call:%.+]] = func.call @test_probs.one_shot_kernel(%arg0) : (f64) -> tensor<4xf64>
// CHECK:     [[add:%.+]] = stablehlo.add [[call]], %arg2 : tensor<4xf64>
// CHECK:     scf.yield [[add]] : tensor<4xf64>
// CHECK:   [[castShots:%.+]] = arith.sitofp [[shots]] : i64 to f64
// CHECK:   [[fromElem:%.+]] = tensor.from_elements [[castShots]] : tensor<f64>
// CHECK:   [[broadcastShots:%.+]] = stablehlo.broadcast_in_dim [[fromElem]], dims = [] : (tensor<f64>) -> tensor<4xf64>
// CHECK:   [[div:%.+]] = stablehlo.divide [[totalSum]], [[broadcastShots]] : tensor<4xf64>
// CHECK:   return [[div]] : tensor<4xf64>


// -----


func.func public @test_probs_mcm(%arg0: f64) -> tensor<4xf64> {
  %1000 = arith.constant 1000 : i64
  quantum.device shots(%1000) ["", "", ""]
  %0 = quantum.alloc( 2) : !quantum.reg
  %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
  %out_qubits = quantum.custom "RX"(%arg0) %1 : !quantum.bit
  %mres_0, %out_qubit = quantum.measure %out_qubits : i1, !quantum.bit
  %2 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit
  %mres_1, %out_qubit_1 = quantum.measure %2 : i1, !quantum.bit
  %3 = quantum.mcmobs %mres_0, %mres_1 : !quantum.obs
  %4 = quantum.probs %3 : tensor<4xf64>
  %5 = quantum.insert %0[ 0], %out_qubit : !quantum.reg, !quantum.bit
  %6 = quantum.insert %0[ 1], %out_qubit_1 : !quantum.reg, !quantum.bit
  quantum.dealloc %6 : !quantum.reg
  quantum.device_release
  return %4 : tensor<4xf64>
}


// CHECK:   func.func public @test_probs_mcm.one_shot_kernel(%arg0: f64) -> tensor<4xf64>
// CHECK:     [[one:%.+]] = arith.constant 1 : i64
// CHECK:     quantum.device shots([[one]]) ["", "", ""]
// CHECK:     [[mres0:%.+]], {{%.+}} = quantum.measure {{%.+}} : i1, !quantum.bit
// CHECK:     [[mres1:%.+]], {{%.+}} = quantum.measure {{%.+}} : i1, !quantum.bit
// CHECK-NOT:    quantum.mcmobs
// CHECK-NOT:    quantum.probs
//
// CHECK:     [[zero:%.+]] = arith.constant 0
// CHECK:     [[zeroTensor:%.+]] = tensor.from_elements [[zero]], [[zero]], [[zero]], [[zero]] : tensor<4xf64>
// CHECK:     [[totalIndexBase:%.+]] = arith.constant 0 : i64
//
// CHECK:     [[mcm_bit0_extend:%.+]] = arith.extui [[mres1]] : i1 to i64
// CHECK:     [[bit0ShiftSize:%.+]] = arith.constant 0 : i64
// CHECK:     [[mcm_bit0_shifted:%.+]] = arith.shli [[mcm_bit0_extend]], [[bit0ShiftSize]] : i64
// CHECK:     [[mcm_bit0_ormask:%.+]] = arith.ori [[totalIndexBase]], [[mcm_bit0_shifted]] : i64
//
// CHECK:     [[mcm_bit1_extend:%.+]] = arith.extui [[mres0]] : i1 to i64
// CHECK:     [[bit1ShiftSize:%.+]] = arith.constant 1 : i64
// CHECK:     [[mcm_bit1_shifted:%.+]] = arith.shli [[mcm_bit1_extend]], [[bit1ShiftSize]] : i64
// CHECK:     [[mcm_bit1_ormask:%.+]] = arith.ori [[mcm_bit0_ormask]], [[mcm_bit1_shifted]] : i64
//
// CHECK:     [[indexCast:%.+]] = arith.index_cast [[mcm_bit1_ormask]] : i64 to index
// CHECK:     [[one:%.+]] = arith.constant 1.000000e+00 : f64
// CHECK:     [[inserted:%.+]] = tensor.insert [[one]] into [[zeroTensor]][[[indexCast]]] : tensor<4xf64>
// CHECK:     return [[inserted]] : tensor<4xf64>
//
// CHECK:   func.func public @test_probs_mcm(%arg0: f64) -> tensor<4xf64>
// CHECK:      scf.for
// CHECK:      func.call @test_probs_mcm.one_shot_kernel(%arg0)
// CHECK:      stablehlo.add
// CHECK:      stablehlo.divide


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


// CHECK: func.func public @test_sample.one_shot_kernel(%arg0: f64) -> tensor<1x2xi64>
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
// CHECK:      [[call:%.+]] = func.call @test_sample.one_shot_kernel(%arg0) : (f64) -> tensor<1x2xi64>
// CHECK:      [[insert:%.+]] = tensor.insert_slice [[call]] into %arg2[%arg1, 0] [1, 2] [1, 1] : tensor<1x2xi64> into tensor<1000x2xi64>
// CHECK:      scf.yield [[insert]] : tensor<1000x2xi64>
// CHECK:    return [[fullSamples]] : tensor<1000x2xi64>


// -----


func.func public @test_sample_mcm(%arg0: f64) -> tensor<1000x1xi64> {
  %1000 = arith.constant 1000 : i64
  quantum.device shots(%1000) ["", "", ""]
  %0 = quantum.alloc( 2) : !quantum.reg
  %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
  %out_qubits = quantum.custom "RX"(%arg0) %1 : !quantum.bit
  %mres, %out_qubit = quantum.measure %out_qubits : i1, !quantum.bit
  %2 = quantum.insert %0[ 0], %out_qubit : !quantum.reg, !quantum.bit
  %3 = quantum.mcmobs %mres : !quantum.obs
  %4 = quantum.sample %3 : tensor<1000x1xf64>
  %5 = stablehlo.convert %4 : (tensor<1000x1xf64>) -> tensor<1000x1xi64>
  quantum.dealloc %2 : !quantum.reg
  quantum.device_release
  return %5 : tensor<1000x1xi64>
}

// CHECK:   func.func public @test_sample_mcm.one_shot_kernel(%arg0: f64) -> tensor<1x1xi64>
// CHECK:     [[one:%.+]] = arith.constant 1 : i64
// CHECK:     quantum.device shots([[one]]) ["", "", ""]
// CHECK:     [[mres:%.+]], {{%.+}} = quantum.measure {{%.+}} : i1, !quantum.bit
// CHECK-NOT:    quantum.mcmobs
// CHECK-NOT:    quantum.sample
// CHECK:     [[extend:%.+]] = arith.extui [[mres]] : i1 to i64
// CHECK:     [[fromElements:%.+]] = tensor.from_elements [[extend]] : tensor<1x1xi64>
// CHECK:     return [[fromElements]] : tensor<1x1xi64>
//
// CHECK:   func.func public @test_sample_mcm(%arg0: f64) -> tensor<1000x1xi64>
// CHECK:      scf.for
// CHECK:      func.call @test_sample_mcm.one_shot_kernel(%arg0)
// CHECK:      tensor.insert_slice


// -----


func.func public @test_counts(%arg0: f64) -> (tensor<4xi64>, tensor<4xi64>) {
  %1000 = arith.constant 1000 : i64
  quantum.device shots(%1000) ["", "", ""]
  %0 = quantum.alloc( 2) : !quantum.reg
  %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
  %out_qubits = quantum.custom "RX"(%arg0) %1 : !quantum.bit
  %mres, %out_qubit = quantum.measure %out_qubits : i1, !quantum.bit
  %2 = quantum.insert %0[ 0], %out_qubit : !quantum.reg, !quantum.bit
  %3 = quantum.compbasis qreg %2 : !quantum.obs
  %eigvals, %counts = quantum.counts %3 : tensor<4xf64>, tensor<4xi64>
  %4 = stablehlo.convert %eigvals : (tensor<4xf64>) -> tensor<4xi64>
  quantum.dealloc %2 : !quantum.reg
  quantum.device_release
  return %4, %counts : tensor<4xi64>, tensor<4xi64>
}


// CHECK: func.func public @test_counts.one_shot_kernel(%arg0: f64) -> (tensor<4xi64>, tensor<4xi64>)
// CHECK:   [[one:%.+]] = arith.constant 1 : i64
// CHECK:   quantum.device shots([[one]]) ["", "", ""]

// CHECK: func.func public @test_counts(%arg0: f64) -> (tensor<4xi64>, tensor<4xi64>) {
// CHECK:   [[shots:%.+]] = arith.constant 1000 : i64
// CHECK:   [[countsSum:%.+]] = stablehlo.constant dense<0> : tensor<4xi64>
// CHECK:   [[eigens:%.+]] = tensor.empty() : tensor<4xi64>
// CHECK:   [[lb:%.+]] = arith.constant 0 : index
// CHECK:   [[step:%.+]] = arith.constant 1 : index
// CHECK:   [[ub:%.+]] = index.casts [[shots]] : i64 to index
// CHECK:   [[forOut:%.+]]:2 = scf.for %arg1 = [[lb]] to [[ub]] step [[step]]
// CHECK-SAME:   (%arg2 = [[eigens]], %arg3 = [[countsSum]]) -> (tensor<4xi64>, tensor<4xi64>)
// CHECK:     [[call:%.+]]:2 = func.call @test_counts.one_shot_kernel(%arg0) : (f64) -> (tensor<4xi64>, tensor<4xi64>)
// CHECK:     [[add:%.+]] = stablehlo.add [[call]]#1, %arg3 : tensor<4xi64>
// CHECK:     scf.yield [[call]]#0, [[add]] : tensor<4xi64>, tensor<4xi64>
// CHECK:   return [[forOut]]#0, [[forOut]]#1 : tensor<4xi64>, tensor<4xi64>


// -----


func.func public @test_counts_mcm(%arg0: f64) -> (tensor<4xi64>, tensor<4xi64>) {
  %1000 = arith.constant 1000 : i64
  quantum.device shots(%1000) ["", "", ""]
  %0 = quantum.alloc( 2) : !quantum.reg
  %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
  %out_qubits = quantum.custom "RX"(%arg0) %1 : !quantum.bit
  %mres0, %out_qubit = quantum.measure %out_qubits : i1, !quantum.bit
  %2 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit
  %mres1, %out_qubit_1 = quantum.measure %2 : i1, !quantum.bit
  %3 = quantum.mcmobs %mres0, %mres1 : !quantum.obs
  %eigvals, %counts = quantum.counts %3 : tensor<4xf64>, tensor<4xi64>
  %4 = stablehlo.convert %eigvals : (tensor<4xf64>) -> tensor<4xi64>
  %5 = quantum.insert %0[ 0], %out_qubit : !quantum.reg, !quantum.bit
  %6 = quantum.insert %5[ 1], %out_qubit_1 : !quantum.reg, !quantum.bit
  quantum.dealloc %6 : !quantum.reg
  quantum.device_release
  return %4, %counts : tensor<4xi64>, tensor<4xi64>
}


// CHECK:   func.func public @test_counts_mcm.one_shot_kernel(%arg0: f64) -> (tensor<4xi64>, tensor<4xi64>)
// CHECK:     [[one:%.+]] = arith.constant 1 : i64
// CHECK:     quantum.device shots([[one]]) ["", "", ""]
// CHECK:     [[mres0:%.+]], {{%.+}} = quantum.measure {{%.+}} : i1, !quantum.bit
// CHECK:     [[mres1:%.+]], {{%.+}} = quantum.measure {{%.+}} : i1, !quantum.bit
// CHECK-NOT:    quantum.mcmobs
// CHECK-NOT:    quantum.counts
//
// CHECK:     [[iota:%.+]] = stablehlo.iota dim = 0 : tensor<4xi64>
// CHECK:     [[zero:%.+]] = arith.constant 0
// CHECK:     [[zeroTensor:%.+]] = tensor.from_elements [[zero]], [[zero]], [[zero]], [[zero]] : tensor<4xi64>
// CHECK:     [[totalIndexBase:%.+]] = arith.constant 0 : i64
//
// CHECK:     [[mcm_bit0_extend:%.+]] = arith.extui [[mres1]] : i1 to i64
// CHECK:     [[bit0ShiftSize:%.+]] = arith.constant 0 : i64
// CHECK:     [[mcm_bit0_shifted:%.+]] = arith.shli [[mcm_bit0_extend]], [[bit0ShiftSize]] : i64
// CHECK:     [[mcm_bit0_ormask:%.+]] = arith.ori [[totalIndexBase]], [[mcm_bit0_shifted]] : i64
//
// CHECK:     [[mcm_bit1_extend:%.+]] = arith.extui [[mres0]] : i1 to i64
// CHECK:     [[bit1ShiftSize:%.+]] = arith.constant 1 : i64
// CHECK:     [[mcm_bit1_shifted:%.+]] = arith.shli [[mcm_bit1_extend]], [[bit1ShiftSize]] : i64
// CHECK:     [[mcm_bit1_ormask:%.+]] = arith.ori [[mcm_bit0_ormask]], [[mcm_bit1_shifted]] : i64
//
// CHECK:     [[indexCast:%.+]] = arith.index_cast [[mcm_bit1_ormask]] : i64 to index
// CHECK:     [[one:%.+]] = arith.constant 1 : i64
// CHECK:     [[inserted:%.+]] = tensor.insert [[one]] into [[zeroTensor]][[[indexCast]]] : tensor<4xi64>
// CHECK:     return [[iota]], [[inserted]] : tensor<4xi64>, tensor<4xi64>
//
// CHECK:   func.func public @test_counts_mcm(%arg0: f64) -> (tensor<4xi64>, tensor<4xi64>)
// CHECK:      scf.for
// CHECK:      func.call @test_counts_mcm.one_shot_kernel(%arg0)
// CHECK:      stablehlo.add


// -----


func.func public @test_many_MPs(%arg0: f64) -> (tensor<1000x2xi64>, tensor<4xi64>, tensor<4xi64>, tensor<f64>, tensor<4xf64>) {
  %1000 = arith.constant 1000 : i64
  quantum.device shots(%1000) ["", "", ""]

  // circuit
  %0 = quantum.alloc( 2) : !quantum.reg
  %1 = quantum.extract %0[0] : !quantum.reg -> !quantum.bit
  %out_qubits = quantum.custom "RX"(%arg0) %1 : !quantum.bit
  %2 = quantum.insert %0[0], %out_qubits : !quantum.reg, !quantum.bit

  // sample
  %3 = quantum.compbasis qreg %2 : !quantum.obs
  %4 = quantum.sample %3 : tensor<1000x2xf64>
  %sample = stablehlo.convert %4 : (tensor<1000x2xf64>) -> tensor<1000x2xi64>

  // counts
  %6 = quantum.compbasis qreg %2 : !quantum.obs
  %eigvals, %counts = quantum.counts %6 : tensor<4xf64>, tensor<4xi64>
  %7 = stablehlo.convert %eigvals : (tensor<4xf64>) -> tensor<4xi64>

  // expval
  %8 = quantum.extract %2[0] : !quantum.reg -> !quantum.bit
  %9 = quantum.namedobs %8[ PauliX] : !quantum.obs
  %10 = quantum.expval %9 : f64
  %expval = tensor.from_elements %10 : tensor<f64>

  // probs
  %11 = quantum.insert %2[0], %8 : !quantum.reg, !quantum.bit
  %12 = quantum.compbasis qreg %11 : !quantum.obs
  %probs = quantum.probs %12 : tensor<4xf64>

  quantum.dealloc %11 : !quantum.reg
  quantum.device_release

  return %sample, %7, %counts, %expval, %probs : tensor<1000x2xi64>, tensor<4xi64>, tensor<4xi64>, tensor<f64>, tensor<4xf64>
}


// CHECK: func.func public @test_many_MPs.one_shot_kernel(%arg0: f64) -> (tensor<1x2xi64>, tensor<4xi64>, tensor<4xi64>, tensor<f64>, tensor<4xf64>)
// CHECK:   [[one:%.+]] = arith.constant 1 : i64
// CHECK:   quantum.device shots([[one]]) ["", "", ""]

// CHECK: func.func public @test_many_MPs(%arg0: f64) -> (tensor<1000x2xi64>, tensor<4xi64>, tensor<4xi64>, tensor<f64>, tensor<4xf64>) {
// CHECK:   [[shots:%.+]] = arith.constant 1000 : i64
// CHECK:   [[sampleFull:%.+]] = tensor.empty() : tensor<1000x2xi64>
// CHECK:   [[countsSum:%.+]] = stablehlo.constant dense<0> : tensor<4xi64>
// CHECK:   [[eigens:%.+]] = tensor.empty() : tensor<4xi64>
// CHECK:   [[expvalSum:%.+]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK:   [[probsSum:%.+]] = stablehlo.constant dense<0.000000e+00> : tensor<4xf64>
//
// CHECK:   [[lb:%.+]] = arith.constant 0 : index
// CHECK:   [[step:%.+]] = arith.constant 1 : index
// CHECK:   [[ub:%.+]] = index.casts [[shots]] : i64 to index
//
// CHECK:   [[forOut:%.+]]:5 = scf.for %arg1 = [[lb]] to [[ub]] step [[step]] iter_args
// CHECK-SAME:   (%arg2 = [[sampleFull]], %arg3 = [[eigens]], %arg4 = [[countsSum]], %arg5 = [[expvalSum]], %arg6 = [[probsSum]])
// CHECK-SAME:   -> (tensor<1000x2xi64>, tensor<4xi64>, tensor<4xi64>, tensor<f64>, tensor<4xf64>)
// CHECK:     [[call:%.+]]:5 = func.call @test_many_MPs.one_shot_kernel(%arg0) :
// CHECK-SAME:   (f64) -> (tensor<1x2xi64>, tensor<4xi64>, tensor<4xi64>, tensor<f64>, tensor<4xf64>)
// CHECK:     [[sample_insert_slice:%.+]] = tensor.insert_slice [[call]]#0 into %arg2[%arg1, 0] [1, 2] [1, 1] : tensor<1x2xi64> into tensor<1000x2xi64>
// CHECK:     [[countsAdd:%.+]] = stablehlo.add [[call]]#2, %arg4 : tensor<4xi64>
// CHECK:     [[expvalAdd:%.+]] = stablehlo.add [[call]]#3, %arg5 : tensor<f64>
// CHECK:     [[probsAdd:%.+]] = stablehlo.add [[call]]#4, %arg6 : tensor<4xf64>
// CHECK:     scf.yield [[sample_insert_slice]], [[call]]#1, [[countsAdd]], [[expvalAdd]], [[probsAdd]]
// CHECK-SAME:   tensor<1000x2xi64>, tensor<4xi64>, tensor<4xi64>, tensor<f64>, tensor<4xf64>
//
// CHECK:    [[shotsCast:%.+]] = arith.sitofp [[shots]] : i64 to f64
// CHECK:    [[shotsTensor:%.+]] = tensor.from_elements [[shotsCast]] : tensor<f64>
// CHECK:    [[expvalDivide:%.+]] = stablehlo.divide [[forOut]]#3, [[shotsTensor]] : tensor<f64>
// CHECK:    [[shotsCast:%.+]] = arith.sitofp [[shots]] : i64 to f64
// CHECK:    [[shotsTensor:%.+]] = tensor.from_elements [[shotsCast]] : tensor<f64>
// CHECK:    [[shotsBroadcast:%.+]] = stablehlo.broadcast_in_dim [[shotsTensor]], dims = [] : (tensor<f64>) -> tensor<4xf64>
// CHECK:    [[probsDivide:%.+]] = stablehlo.divide [[forOut]]#4, [[shotsBroadcast]] : tensor<4xf64>
// CHECK:    return [[forOut]]#0, [[forOut]]#1, [[forOut]]#2, [[expvalDivide]], %8 : tensor<1000x2xi64>, tensor<4xi64>, tensor<4xi64>, tensor<f64>, tensor<4xf64>
