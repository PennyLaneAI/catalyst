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

// RUN: quantum-opt --adjoint-lowering --split-input-file -verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: @qubit_unitary_test
func.func @qubit_unitary_test() -> tensor<4xcomplex<f64>> {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index

  %0 = quantum.alloc( 2) : !quantum.reg

  // CHECK-NOT: quantum.adjoint
  %1 = quantum.adjoint(%0) : !quantum.reg {
  ^bb0(%arg1: !quantum.reg):

    // COM: 4x4xcomplex<f64> is 4*4*128/8 = 256 bytes
    // CHECK-DAG: [[_256:%.+]] = index.constant 256
    // CHECK-DAG: [[zero:%.+]] = index.constant 0

    // CHECK: [[data_vector:%.+]] = memref.alloc() : memref<2048xi8>
    // CHECK: [[cur_offset:%.+]] = memref.alloc() : memref<index>
    // CHECK: memref.store [[zero]], [[cur_offset]][] : memref<index>
    // CHECK: [[offset_vector:%.+]] = catalyst.list_init : <index>

    // CHECK: scf.for
    // CHECK:   [[param:%.+]] = "test.op"() : () -> tensor<4x4xcomplex<f64>>
    // CHECK:   [[offset:%.+]] = memref.load [[cur_offset]][] : memref<index>
    // CHECK:   [[param_memref:%.+]] = bufferization.to_buffer [[param]]
    // CHECK-SAME:    tensor<4x4xcomplex<f64>> to memref<4x4xcomplex<f64>>
    // CHECK:   [[view:%.+]] = memref.view [[data_vector]][[[offset]]][]
    // CHECK-SAME:    memref<2048xi8> to memref<4x4xcomplex<f64>>
    // CHECK:   memref.copy [[param_memref]], [[view]]
    // CHECK-SAME:    memref<4x4xcomplex<f64>> to memref<4x4xcomplex<f64>>
    // CHECK:   catalyst.list_push [[offset]], [[offset_vector]] : <index>
    // CHECK:   [[new_offset:%.+]] = index.add [[offset]], [[_256]]
    // CHECK:   memref.store [[new_offset]], [[cur_offset]][] : memref<index>

    // CHECK: scf.for
    // CHECK-SAME:  -> (!quantum.reg) {
    %for_reg = scf.for %i = %c0 to %c4 step %c1 iter_args(%reg = %arg1) -> (!quantum.reg) {
      %u = "test.op"() : () -> tensor<4x4xcomplex<f64>>
      %6 = quantum.extract %reg[ 0] : !quantum.reg -> !quantum.bit
      %7 = quantum.extract %reg[ 1] : !quantum.reg -> !quantum.bit

      // CHECK: [[offset:%.+]] = catalyst.list_pop [[offset_vector]] : <index>
      // CHECK: [[view:%.+]] = memref.view [[data_vector]][[[offset]]][]
      // CHECK-SAME:    memref<2048xi8> to memref<4x4xcomplex<f64>>
      // CHECK: [[param_tensor:%.+]] = bufferization.to_tensor [[view]] restrict
      // CHECK-SAME:    memref<4x4xcomplex<f64>> to tensor<4x4xcomplex<f64>>
      //
      // CHECK: quantum.unitary([[param_tensor]] : tensor<4x4xcomplex<f64>>) {{%.+}} {{%.+}} adj
      %8:2 = quantum.unitary(%u : tensor<4x4xcomplex<f64>>) %6, %7 : !quantum.bit, !quantum.bit
      %9 = quantum.insert %reg[ 0], %8#0 : !quantum.reg, !quantum.bit
      %10 = quantum.insert %9[ 1], %8#1 : !quantum.reg, !quantum.bit
      scf.yield %10 : !quantum.reg
    }
    // CHECK: memref.dealloc [[data_vector]] : memref<2048xi8>
    // CHECK: memref.dealloc [[cur_offset]] : memref<index>
    // CHECK: catalyst.list_dealloc [[offset_vector]] : <index>

    quantum.yield %for_reg : !quantum.reg
  }

  %2 = quantum.extract %1[ 0] : !quantum.reg -> !quantum.bit
  %3 = quantum.extract %1[ 1] : !quantum.reg -> !quantum.bit
  %4 = quantum.compbasis qubits %2, %3 : !quantum.obs
  %5 = quantum.state %4 : tensor<4xcomplex<f64>>
  quantum.dealloc %0 : !quantum.reg
  return %5 : tensor<4xcomplex<f64>>
}

// -----

// CHECK-LABEL: @adjoint_real_matrix_param
func.func @adjoint_real_matrix_param(%arg0: !quantum.reg) -> !quantum.reg {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index

  %out = quantum.adjoint(%arg0) : !quantum.reg {
  ^bb0(%r: !quantum.reg):

    // COM: tensor<2x2xf64> is 2*2*64/8 = 32 bytes
    // CHECK-DAG: [[_32:%.+]] = index.constant 32
    // CHECK-DAG: [[zero:%.+]] = index.constant 0

    // CHECK: [[data_vector:%.+]] = memref.alloc() : memref<2048xi8>
    // CHECK: [[cur_offset:%.+]] = memref.alloc() : memref<index>
    // CHECK: memref.store [[zero]], [[cur_offset]][] : memref<index>
    // CHECK: [[offset_vector:%.+]] = catalyst.list_init : <index>

    // CHECK: scf.for
    // CHECK:   [[param:%.+]] = "test.op"() : () -> tensor<2x2xf64>
    // CHECK:   [[offset:%.+]] = memref.load [[cur_offset]][] : memref<index>
    // CHECK:   [[param_memref:%.+]] = bufferization.to_buffer [[param]] : tensor<2x2xf64> to memref<2x2xf64>
    // CHECK:   [[view:%.+]] = memref.view [[data_vector]][[[offset]]][] : memref<2048xi8> to memref<2x2xf64>
    // CHECK:   memref.copy [[param_memref]], [[view]] : memref<2x2xf64> to memref<2x2xf64>
    // CHECK:   catalyst.list_push [[offset]], [[offset_vector]] : <index>
    // CHECK:   [[new_offset:%.+]] = index.add [[offset]], [[_32]]
    // CHECK:   memref.store [[new_offset]], [[cur_offset]][] : memref<index>

    // CHECK: scf.for
    // CHECK-SAME:  -> (!quantum.reg) {
    %for_reg = scf.for %i = %c0 to %c4 step %c1 iter_args(%reg = %r) -> (!quantum.reg) {
      %matrix = "test.op"() : () -> tensor<2x2xf64>
      %q0 = quantum.extract %reg[ 0] : !quantum.reg -> !quantum.bit
      %q1 = quantum.extract %reg[ 1] : !quantum.reg -> !quantum.bit

      // CHECK: [[offset:%.+]] = catalyst.list_pop [[offset_vector]] : <index>
      // CHECK: [[view:%.+]] = memref.view [[data_vector]][[[offset]]][]
      // CHECK-SAME:    memref<2048xi8> to memref<2x2xf64>
      // CHECK: [[param_tensor:%.+]] = bufferization.to_tensor [[view]] restrict
      // CHECK-SAME:    memref<2x2xf64> to tensor<2x2xf64>
      //
      // CHECK: quantum.operator "BasisRotation"([[param_tensor]]: tensor<2x2xf64>) adj
      %op:2 = quantum.operator "BasisRotation"(%matrix: tensor<2x2xf64>) qubits(%q0, %q1)
        static_data = {}
        param_map = {unitary_matrix = [0]} qubit_map = {wires = [0, 1]}
      %r0 = quantum.insert %reg[ 0], %op#0 : !quantum.reg, !quantum.bit
      %r1 = quantum.insert %r0[ 1], %op#1 : !quantum.reg, !quantum.bit
      scf.yield %r1 : !quantum.reg
    }
    // CHECK: memref.dealloc [[data_vector]] : memref<2048xi8>
    // CHECK: memref.dealloc [[cur_offset]] : memref<index>
    // CHECK: catalyst.list_dealloc [[offset_vector]] : <index>

    quantum.yield %for_reg : !quantum.reg
  }
  return %out : !quantum.reg
}

// -----

// CHECK-LABEL: @mixed_param_types
func.func @mixed_param_types(%0: !quantum.reg) -> !quantum.reg {
  %i0 = arith.constant 0 : index
  %i4 = arith.constant 4 : index
  %i1 = arith.constant 1 : index

  // CHECK-NOT: quantum.adjoint
  %1 = quantum.adjoint(%0) : !quantum.reg {
  ^bb0(%r0: !quantum.reg):

    // COM: f64: 8 bytes
    // COM: i1: 1 byte
    // COM: complex<i32>: 8 bytes
    // COM: tensor<6xi1>: 6 bytes
    // CHECK-DAG: [[_1:%.+]] = index.constant 1
    // CHECK-DAG: [[_6:%.+]] = index.constant 6
    // CHECK-DAG: [[_8:%.+]] = index.constant 8
    // CHECK-DAG: [[zero:%.+]] = index.constant 0

    // CHECK: [[data_vector:%.+]] = memref.alloc() : memref<2048xi8>
    // CHECK: [[cur_offset:%.+]] = memref.alloc() : memref<index>
    // CHECK: memref.store [[zero]], [[cur_offset]][] : memref<index>
    // CHECK: [[offset_vector:%.+]] = catalyst.list_init : <index>

    // CHECK: scf.for [[i:%.+]] =
    // CHECK:   [[c1:%.+]] = "test.op"([[i]]) : (index) -> f64
    // CHECK:   [[c2:%.+]] = "test.op"([[i]]) : (index) -> i1
    // CHECK:   [[c3:%.+]] = "test.op"([[i]]) : (index) -> complex<i32>
    // CHECK:   [[c4:%.+]] = "test.op"([[i]]) : (index) -> tensor<6xi1>
    //
    // CHECK: [[offset:%.+]] = memref.load [[cur_offset]][] : memref<index>
    // CHECK: [[view:%.+]] = memref.view [[data_vector]][[[offset]]][] : memref<2048xi8> to memref<1xf64>
    // CHECK: memref.store [[c1]], [[view]][[[zero]]] : memref<1xf64>
    // CHECK: catalyst.list_push [[offset]], [[offset_vector]] : <index>
    // CHECK: [[new_offset:%.+]] = index.add [[offset]], [[_8]]
    // CHECK: memref.store [[new_offset]], [[cur_offset]][] : memref<index>
    //
    // CHECK: [[offset:%.+]] = memref.load [[cur_offset]][] : memref<index>
    // CHECK: [[view:%.+]] = memref.view [[data_vector]][[[offset]]][] : memref<2048xi8> to memref<1xi1>
    // CHECK: memref.store [[c2]], [[view]][[[zero]]] : memref<1xi1>
    // CHECK: catalyst.list_push [[offset]], [[offset_vector]] : <index>
    // CHECK: [[new_offset:%.+]] = index.add [[offset]], [[_1]]
    // CHECK: memref.store [[new_offset]], [[cur_offset]][] : memref<index>
    //
    // CHECK: [[offset:%.+]] = memref.load [[cur_offset]][] : memref<index>
    // CHECK: [[view:%.+]] = memref.view [[data_vector]][[[offset]]][] : memref<2048xi8> to memref<1xcomplex<i32>>
    // CHECK: memref.store [[c3]], [[view]][[[zero]]] : memref<1xcomplex<i32>>
    // CHECK: catalyst.list_push [[offset]], [[offset_vector]] : <index>
    // CHECK: [[new_offset:%.+]] = index.add [[offset]], [[_8]]
    // CHECK: memref.store [[new_offset]], [[cur_offset]][] : memref<index>
    //
    // CHECK:   [[offset:%.+]] = memref.load [[cur_offset]][] : memref<index>
    // CHECK:   [[c4_memref:%.+]] = bufferization.to_buffer [[c4]] : tensor<6xi1> to memref<6xi1>
    // CHECK:   [[view:%.+]] = memref.view [[data_vector]][[[offset]]][] : memref<2048xi8> to memref<6xi1>
    // CHECK:   memref.copy [[c4_memref]], [[view]] : memref<6xi1> to memref<6xi1>
    // CHECK:   catalyst.list_push [[offset]], [[offset_vector]] : <index>
    // CHECK:   [[new_offset:%.+]] = index.add [[offset]], [[_6]]
    // CHECK:   memref.store [[new_offset]], [[cur_offset]][] : memref<index>

    // CHECK: scf.for
    // CHECK-SAME:  -> (!quantum.reg) {
    %for_reg = scf.for %i = %i0 to %i4 step %i1 iter_args(%reg = %r0) -> (!quantum.reg) {
      %c1 = "test.op" (%i) : (index) -> (f64)
      %c2 = "test.op" (%i) : (index) -> (i1)
      %c3 = "test.op" (%i) : (index) -> (complex<i32>)
      %c4 = "test.op" (%i) : (index) -> (tensor<6xi1>)
      %q0 = quantum.extract %reg[ 0] : !quantum.reg -> !quantum.bit

      // CHECK: [[offset:%.+]] = catalyst.list_pop [[offset_vector]] : <index>
      // CHECK: [[view:%.+]] = memref.view [[data_vector]][[[offset]]][] : memref<2048xi8> to memref<6xi1>
      // CHECK: [[c4:%.+]] = bufferization.to_tensor [[view]] restrict : memref<6xi1> to tensor<6xi1>
      //
      // CHECK: [[offset:%.+]] = catalyst.list_pop [[offset_vector]] : <index>
      // CHECK: [[view:%.+]] = memref.view [[data_vector]][[[offset]]][] : memref<2048xi8> to memref<1xcomplex<i32>>
      // CHECK: [[c3:%.+]] = memref.load [[view]][[[zero]]] : memref<1xcomplex<i32>>
      //
      // CHECK: [[offset:%.+]] = catalyst.list_pop [[offset_vector]] : <index>
      // CHECK: [[view:%.+]] = memref.view [[data_vector]][[[offset]]][] : memref<2048xi8> to memref<1xi1>
      // CHECK: [[c2:%.+]] = memref.load [[view]][[[zero]]] : memref<1xi1>
      //
      // CHECK: [[offset:%.+]] = catalyst.list_pop [[offset_vector]] : <index>
      // CHECK: [[view:%.+]] = memref.view [[data_vector]][[[offset]]][] : memref<2048xi8> to memref<1xf64>
      // CHECK: [[c1:%.+]] = memref.load [[view]][[[zero]]] : memref<1xf64>
      //
      // CHECK: quantum.operator "gate"([[c1]]: f64, [[c2]]: i1, [[c3]]: complex<i32>, [[c4]]: tensor<6xi1>) adj
      %q1 = quantum.operator "gate"(%c1: f64, %c2: i1, %c3: complex<i32>, %c4: tensor<6xi1>) qubits(%q0)

      %r1 = quantum.insert %reg[ 0], %q1 : !quantum.reg, !quantum.bit
      scf.yield %r1 : !quantum.reg
    }
    // CHECK: memref.dealloc [[data_vector]] : memref<2048xi8>
    // CHECK: memref.dealloc [[cur_offset]] : memref<index>
    // CHECK: catalyst.list_dealloc [[offset_vector]] : <index>

    quantum.yield %for_reg : !quantum.reg
  }

  return %1 : !quantum.reg
}
