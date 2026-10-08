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

// COM: All four cache helpers are inserted at the start of the module body, so they appear in the
// COM: reverse of the order in which they are created.

// COM: The cache buffers are allocated and freed inside helpers rather than at the point of the
// COM: quantum.adjoint op, so that no memref.alloc/memref.dealloc for them appears in the function
// COM: being lowered. For an adjoint nested inside a loop, -buffer-loop-hoisting would otherwise
// COM: hoist the allocation out of the loop and leave the deallocation inside it.

// CHECK:  func.func private @__adjoint_lowering_dealloc_param_vector(
// CHECK-SAME:   %arg0: memref<memref<?xi8>>, %arg1: memref<index>, %arg2: memref<index>) {
// CHECK:    [[dealloc_data:%.+]] = memref.load %arg0[] : memref<memref<?xi8>>
// CHECK:    memref.dealloc [[dealloc_data]] : memref<?xi8>
// CHECK:    memref.dealloc %arg0 : memref<memref<?xi8>>
// CHECK:    memref.dealloc %arg1 : memref<index>
// CHECK:    memref.dealloc %arg2 : memref<index>
// CHECK:    return
// CHECK:  }

// CHECK:  func.func private @__adjoint_lowering_init_param_vector()
// CHECK-SAME:   -> (memref<memref<?xi8>>, memref<index>, memref<index>) {
// CHECK-DAG:    [[init_capacity:%.+]] = index.constant 2048
// CHECK-DAG:    [[init_zero:%.+]] = index.constant 0
// CHECK:    [[init_data:%.+]] = memref.alloc([[init_capacity]]) {alignment = 64 : i64}
// CHECK-SAME:   : memref<?xi8>
// CHECK:    [[init_data_vector:%.+]] = memref.alloc() : memref<memref<?xi8>>
// CHECK:    memref.store [[init_data]], [[init_data_vector]][] : memref<memref<?xi8>>
// CHECK:    [[init_capacity_field:%.+]] = memref.alloc() : memref<index>
// CHECK:    memref.store [[init_capacity]], [[init_capacity_field]][] : memref<index>
// CHECK:    [[init_offset_field:%.+]] = memref.alloc() : memref<index>
// CHECK:    memref.store [[init_zero]], [[init_offset_field]][] : memref<index>
// CHECK:    return [[init_data_vector]], [[init_capacity_field]], [[init_offset_field]]
// CHECK-SAME:   : memref<memref<?xi8>>, memref<index>, memref<index>
// CHECK:  }

// COM: The byte buffer holding the params is dynamically sized and kept behind a rank-0 memref,
// COM: so that it can be reallocated in place when a param does not fit. The new capacity is
// COM: max(2 * old_capacity, required), since a single param can be larger than the whole buffer.

// CHECK:  func.func private @__adjoint_lowering_ensure_param_vector_capacity(
// CHECK-SAME:   %arg0: memref<memref<?xi8>>, %arg1: memref<index>, %arg2: index) {
// CHECK:    [[two:%.+]] = index.constant 2
// CHECK:    [[capacity:%.+]] = memref.load %arg1[] : memref<index>
// CHECK:    [[needs_growth:%.+]] = index.cmp ult([[capacity]], %arg2)
// CHECK:    scf.if [[needs_growth]] {
// CHECK:      [[doubled:%.+]] = index.mul [[capacity]], [[two]]
// CHECK:      [[new_capacity:%.+]] = index.maxu [[doubled]], %arg2
// CHECK:      [[old_data:%.+]] = memref.load %arg0[] : memref<memref<?xi8>>
// CHECK:      [[new_data:%.+]] = memref.realloc [[old_data]]([[new_capacity]])
// CHECK-SAME:   {alignment = 64 : i64} : memref<?xi8> to memref<?xi8>
// CHECK:      memref.store [[new_data]], %arg0[] : memref<memref<?xi8>>
// CHECK:      memref.store [[new_capacity]], %arg1[] : memref<index>
// CHECK:    }
// CHECK:    return
// CHECK:  }

// COM: The formula to round up current offset (O) to intended alignment (A) is
// COM: O_aligned = (O + A - 1) & ~(A - 1)
// COM: given that A is a power of 2

// CHECK:  func.func private @__adjoint_lowering_roundup_offset_to_alignment(%arg0: index, %arg1: index) -> index {
// CHECK:    [[minus_one:%.+]] = index.constant -1
// CHECK:    [[one:%.+]] = index.constant 1
// CHECK:    [[A_minus_one:%.+]] = index.sub %arg1, [[one]]
// CHECK:    [[O_plus_A_minus_one:%.+]] = index.add %arg0, [[A_minus_one]]
// CHECK:    [[not_A_minus_one:%.+]] = index.xor [[A_minus_one]], [[minus_one]]
// CHECK:    [[out:%.+]] = index.and [[O_plus_A_minus_one]], [[not_A_minus_one]]
// CHECK:    return [[out]] : index
// CHECK:  }

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
    // CHECK-DAG: [[_64:%.+]] = index.constant 64

    // COM: The cache buffers are allocated by a call rather than in place, so that no memref.alloc
    // COM: for them appears in the function being lowered. See getOrInsertInitParamVectorFunc.
    // CHECK: [[cache:%.+]]:3 = call @__adjoint_lowering_init_param_vector()
    // CHECK-SAME:   : () -> (memref<memref<?xi8>>, memref<index>, memref<index>)
    // CHECK: [[offset_vector:%.+]] = catalyst.list_init : <index>

    // CHECK: scf.for
    // CHECK:   [[param:%.+]] = "test.op"() : () -> tensor<4x4xcomplex<f64>>
    // CHECK:   [[raw_offset:%.+]] = memref.load [[cache]]#2[] : memref<index>
    // CHECK:   [[offset:%.+]] = func.call @__adjoint_lowering_roundup_offset_to_alignment
    // CHECK-SAME:   ([[raw_offset]], [[_64]]) : (index, index) -> index
    // CHECK:   [[new_offset:%.+]] = index.add [[offset]], [[_256]]
    // CHECK:   func.call @__adjoint_lowering_ensure_param_vector_capacity
    // CHECK-SAME:   ([[cache]]#0, [[cache]]#1, [[new_offset]])
    // CHECK-SAME:   : (memref<memref<?xi8>>, memref<index>, index) -> ()
    // CHECK:   [[loaded_data:%.+]] = memref.load [[cache]]#0[] : memref<memref<?xi8>>
    // CHECK:   [[param_memref:%.+]] = bufferization.to_buffer [[param]]
    // CHECK-SAME:    tensor<4x4xcomplex<f64>> to memref<4x4xcomplex<f64>>
    // CHECK:   [[view:%.+]] = memref.view [[loaded_data]][[[offset]]][]
    // CHECK-SAME:    memref<?xi8> to memref<4x4xcomplex<f64>>
    // CHECK:   memref.copy [[param_memref]], [[view]]
    // CHECK-SAME:    memref<4x4xcomplex<f64>> to memref<4x4xcomplex<f64>>
    // CHECK:   catalyst.list_push [[offset]], [[offset_vector]] : <index>
    // CHECK:   memref.store [[new_offset]], [[cache]]#2[] : memref<index>

    // CHECK: scf.for
    // CHECK-SAME:  -> (!quantum.reg) {
    %for_reg = scf.for %i = %c0 to %c4 step %c1 iter_args(%reg = %arg1) -> (!quantum.reg) {
      %u = "test.op"() : () -> tensor<4x4xcomplex<f64>>
      %6 = quantum.extract %reg[ 0] : !quantum.reg -> !quantum.bit
      %7 = quantum.extract %reg[ 1] : !quantum.reg -> !quantum.bit

      // CHECK: [[offset:%.+]] = catalyst.list_pop [[offset_vector]] : <index>
      // CHECK: [[loaded_data:%.+]] = memref.load [[cache]]#0[] : memref<memref<?xi8>>
      // CHECK: [[view:%.+]] = memref.view [[loaded_data]][[[offset]]][]
      // CHECK-SAME:    memref<?xi8> to memref<4x4xcomplex<f64>>
      // CHECK: [[param_tensor:%.+]] = bufferization.to_tensor [[view]] restrict
      // CHECK-SAME:    memref<4x4xcomplex<f64>> to tensor<4x4xcomplex<f64>>
      //
      // CHECK: quantum.unitary([[param_tensor]] : tensor<4x4xcomplex<f64>>) {{%.+}} {{%.+}} adj
      %8:2 = quantum.unitary(%u : tensor<4x4xcomplex<f64>>) %6, %7 : !quantum.bit, !quantum.bit
      %9 = quantum.insert %reg[ 0], %8#0 : !quantum.reg, !quantum.bit
      %10 = quantum.insert %9[ 1], %8#1 : !quantum.reg, !quantum.bit
      scf.yield %10 : !quantum.reg
    }
    // CHECK: call @__adjoint_lowering_dealloc_param_vector
    // CHECK-SAME:   ([[cache]]#0, [[cache]]#1, [[cache]]#2)
    // CHECK-SAME:   : (memref<memref<?xi8>>, memref<index>, memref<index>) -> ()
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
    // CHECK-DAG: [[_64:%.+]] = index.constant 64

    // COM: The cache buffers are allocated by a call rather than in place, so that no memref.alloc
    // COM: for them appears in the function being lowered. See getOrInsertInitParamVectorFunc.
    // CHECK: [[cache:%.+]]:3 = call @__adjoint_lowering_init_param_vector()
    // CHECK-SAME:   : () -> (memref<memref<?xi8>>, memref<index>, memref<index>)
    // CHECK: [[offset_vector:%.+]] = catalyst.list_init : <index>

    // CHECK: scf.for
    // CHECK:   [[param:%.+]] = "test.op"() : () -> tensor<2x2xf64>
    // CHECK:   [[raw_offset:%.+]] = memref.load [[cache]]#2[] : memref<index>
    // CHECK:   [[offset:%.+]] = func.call @__adjoint_lowering_roundup_offset_to_alignment
    // CHECK-SAME:  ([[raw_offset]], [[_64]]) : (index, index) -> index
    // CHECK:   [[new_offset:%.+]] = index.add [[offset]], [[_32]]
    // CHECK:   func.call @__adjoint_lowering_ensure_param_vector_capacity
    // CHECK-SAME:  ([[cache]]#0, [[cache]]#1, [[new_offset]])
    // CHECK-SAME:  : (memref<memref<?xi8>>, memref<index>, index) -> ()
    // CHECK:   [[loaded_data:%.+]] = memref.load [[cache]]#0[] : memref<memref<?xi8>>
    // CHECK:   [[param_memref:%.+]] = bufferization.to_buffer [[param]] : tensor<2x2xf64> to memref<2x2xf64>
    // CHECK:   [[view:%.+]] = memref.view [[loaded_data]][[[offset]]][] : memref<?xi8> to memref<2x2xf64>
    // CHECK:   memref.copy [[param_memref]], [[view]] : memref<2x2xf64> to memref<2x2xf64>
    // CHECK:   catalyst.list_push [[offset]], [[offset_vector]] : <index>
    // CHECK:   memref.store [[new_offset]], [[cache]]#2[] : memref<index>

    // CHECK: scf.for
    // CHECK-SAME:  -> (!quantum.reg) {
    %for_reg = scf.for %i = %c0 to %c4 step %c1 iter_args(%reg = %r) -> (!quantum.reg) {
      %matrix = "test.op"() : () -> tensor<2x2xf64>
      %q0 = quantum.extract %reg[ 0] : !quantum.reg -> !quantum.bit
      %q1 = quantum.extract %reg[ 1] : !quantum.reg -> !quantum.bit

      // CHECK: [[offset:%.+]] = catalyst.list_pop [[offset_vector]] : <index>
      // CHECK: [[loaded_data:%.+]] = memref.load [[cache]]#0[] : memref<memref<?xi8>>
      // CHECK: [[view:%.+]] = memref.view [[loaded_data]][[[offset]]][]
      // CHECK-SAME:    memref<?xi8> to memref<2x2xf64>
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
    // CHECK: call @__adjoint_lowering_dealloc_param_vector
    // CHECK-SAME:   ([[cache]]#0, [[cache]]#1, [[cache]]#2)
    // CHECK-SAME:   : (memref<memref<?xi8>>, memref<index>, memref<index>) -> ()
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
    // CHECK-DAG: [[_64:%.+]] = index.constant 64

    // COM: The cache buffers are allocated by a call rather than in place, so that no memref.alloc
    // COM: for them appears in the function being lowered. See getOrInsertInitParamVectorFunc.
    // CHECK: [[cache:%.+]]:3 = call @__adjoint_lowering_init_param_vector()
    // CHECK-SAME:   : () -> (memref<memref<?xi8>>, memref<index>, memref<index>)
    // CHECK: [[offset_vector:%.+]] = catalyst.list_init : <index>

    // CHECK: scf.for [[i:%.+]] =
    // CHECK:   [[c1:%.+]] = "test.op"([[i]]) : (index) -> f64
    // CHECK:   [[c2:%.+]] = "test.op"([[i]]) : (index) -> i1
    // CHECK:   [[c3:%.+]] = "test.op"([[i]]) : (index) -> complex<i32>
    // CHECK:   [[c4:%.+]] = "test.op"([[i]]) : (index) -> tensor<6xi1>
    //
    // CHECK: [[raw_offset:%.+]] = memref.load [[cache]]#2[] : memref<index>
    // CHECK: [[offset:%.+]] = func.call @__adjoint_lowering_roundup_offset_to_alignment
    // CHECK-SAME:  ([[raw_offset]], [[_8]]) : (index, index) -> index
    // CHECK: [[new_offset:%.+]] = index.add [[offset]], [[_8]]
    // CHECK: func.call @__adjoint_lowering_ensure_param_vector_capacity
    // CHECK-SAME:  ([[cache]]#0, [[cache]]#1, [[new_offset]])
    // CHECK: [[loaded_data:%.+]] = memref.load [[cache]]#0[] : memref<memref<?xi8>>
    // CHECK: [[view:%.+]] = memref.view [[loaded_data]][[[offset]]][] : memref<?xi8> to memref<f64>
    // CHECK: memref.store [[c1]], [[view]][] : memref<f64>
    // CHECK: catalyst.list_push [[offset]], [[offset_vector]] : <index>
    // CHECK: memref.store [[new_offset]], [[cache]]#2[] : memref<index>
    //
    // CHECK: [[raw_offset:%.+]] = memref.load [[cache]]#2[] : memref<index>
    // CHECK: [[offset:%.+]] = func.call @__adjoint_lowering_roundup_offset_to_alignment
    // CHECK-SAME:  ([[raw_offset]], [[_1]]) : (index, index) -> index
    // CHECK: [[new_offset:%.+]] = index.add [[offset]], [[_1]]
    // CHECK: func.call @__adjoint_lowering_ensure_param_vector_capacity
    // CHECK-SAME:  ([[cache]]#0, [[cache]]#1, [[new_offset]])
    // CHECK: [[loaded_data:%.+]] = memref.load [[cache]]#0[] : memref<memref<?xi8>>
    // CHECK: [[view:%.+]] = memref.view [[loaded_data]][[[offset]]][] : memref<?xi8> to memref<i1>
    // CHECK: memref.store [[c2]], [[view]][] : memref<i1>
    // CHECK: catalyst.list_push [[offset]], [[offset_vector]] : <index>
    // CHECK: memref.store [[new_offset]], [[cache]]#2[] : memref<index>
    //
    // CHECK: [[raw_offset:%.+]] = memref.load [[cache]]#2[] : memref<index>
    // CHECK: [[offset:%.+]] = func.call @__adjoint_lowering_roundup_offset_to_alignment
    // CHECK-SAME:  ([[raw_offset]], [[_8]]) : (index, index) -> index
    // CHECK: [[new_offset:%.+]] = index.add [[offset]], [[_8]]
    // CHECK: func.call @__adjoint_lowering_ensure_param_vector_capacity
    // CHECK-SAME:  ([[cache]]#0, [[cache]]#1, [[new_offset]])
    // CHECK: [[loaded_data:%.+]] = memref.load [[cache]]#0[] : memref<memref<?xi8>>
    // CHECK: [[view:%.+]] = memref.view [[loaded_data]][[[offset]]][] : memref<?xi8> to memref<complex<i32>>
    // CHECK: memref.store [[c3]], [[view]][] : memref<complex<i32>>
    // CHECK: catalyst.list_push [[offset]], [[offset_vector]] : <index>
    // CHECK: memref.store [[new_offset]], [[cache]]#2[] : memref<index>
    //
    // CHECK:   [[raw_offset:%.+]] = memref.load [[cache]]#2[] : memref<index>
    // CHECK:   [[offset:%.+]] = func.call @__adjoint_lowering_roundup_offset_to_alignment
    // CHECK-SAME:  ([[raw_offset]], [[_64]]) : (index, index) -> index
    // CHECK:   [[new_offset:%.+]] = index.add [[offset]], [[_6]]
    // CHECK:   func.call @__adjoint_lowering_ensure_param_vector_capacity
    // CHECK-SAME:  ([[cache]]#0, [[cache]]#1, [[new_offset]])
    // CHECK:   [[loaded_data:%.+]] = memref.load [[cache]]#0[] : memref<memref<?xi8>>
    // CHECK:   [[c4_memref:%.+]] = bufferization.to_buffer [[c4]] : tensor<6xi1> to memref<6xi1>
    // CHECK:   [[view:%.+]] = memref.view [[loaded_data]][[[offset]]][] : memref<?xi8> to memref<6xi1>
    // CHECK:   memref.copy [[c4_memref]], [[view]] : memref<6xi1> to memref<6xi1>
    // CHECK:   catalyst.list_push [[offset]], [[offset_vector]] : <index>
    // CHECK:   memref.store [[new_offset]], [[cache]]#2[] : memref<index>

    // CHECK: scf.for
    // CHECK-SAME:  -> (!quantum.reg) {
    %for_reg = scf.for %i = %i0 to %i4 step %i1 iter_args(%reg = %r0) -> (!quantum.reg) {
      %c1 = "test.op" (%i) : (index) -> (f64)
      %c2 = "test.op" (%i) : (index) -> (i1)
      %c3 = "test.op" (%i) : (index) -> (complex<i32>)
      %c4 = "test.op" (%i) : (index) -> (tensor<6xi1>)
      %q0 = quantum.extract %reg[ 0] : !quantum.reg -> !quantum.bit

      // CHECK: [[offset:%.+]] = catalyst.list_pop [[offset_vector]] : <index>
      // CHECK: [[loaded_data:%.+]] = memref.load [[cache]]#0[] : memref<memref<?xi8>>
      // CHECK: [[view:%.+]] = memref.view [[loaded_data]][[[offset]]][] : memref<?xi8> to memref<6xi1>
      // CHECK: [[c4:%.+]] = bufferization.to_tensor [[view]] restrict : memref<6xi1> to tensor<6xi1>
      //
      // CHECK: [[offset:%.+]] = catalyst.list_pop [[offset_vector]] : <index>
      // CHECK: [[loaded_data:%.+]] = memref.load [[cache]]#0[] : memref<memref<?xi8>>
      // CHECK: [[view:%.+]] = memref.view [[loaded_data]][[[offset]]][] : memref<?xi8> to memref<complex<i32>>
      // CHECK: [[c3:%.+]] = memref.load [[view]][] : memref<complex<i32>>
      //
      // CHECK: [[offset:%.+]] = catalyst.list_pop [[offset_vector]] : <index>
      // CHECK: [[loaded_data:%.+]] = memref.load [[cache]]#0[] : memref<memref<?xi8>>
      // CHECK: [[view:%.+]] = memref.view [[loaded_data]][[[offset]]][] : memref<?xi8> to memref<i1>
      // CHECK: [[c2:%.+]] = memref.load [[view]][] : memref<i1>
      //
      // CHECK: [[offset:%.+]] = catalyst.list_pop [[offset_vector]] : <index>
      // CHECK: [[loaded_data:%.+]] = memref.load [[cache]]#0[] : memref<memref<?xi8>>
      // CHECK: [[view:%.+]] = memref.view [[loaded_data]][[[offset]]][] : memref<?xi8> to memref<f64>
      // CHECK: [[c1:%.+]] = memref.load [[view]][] : memref<f64>
      //
      // CHECK: quantum.operator "gate"([[c1]]: f64, [[c2]]: i1, [[c3]]: complex<i32>, [[c4]]: tensor<6xi1>) adj
      %q1 = quantum.operator "gate"(%c1: f64, %c2: i1, %c3: complex<i32>, %c4: tensor<6xi1>) qubits(%q0)

      %r1 = quantum.insert %reg[ 0], %q1 : !quantum.reg, !quantum.bit
      scf.yield %r1 : !quantum.reg
    }
    // CHECK: call @__adjoint_lowering_dealloc_param_vector
    // CHECK-SAME:   ([[cache]]#0, [[cache]]#1, [[cache]]#2)
    // CHECK-SAME:   : (memref<memref<?xi8>>, memref<index>, memref<index>) -> ()
    // CHECK: catalyst.list_dealloc [[offset_vector]] : <index>

    quantum.yield %for_reg : !quantum.reg
  }

  return %1 : !quantum.reg
}
