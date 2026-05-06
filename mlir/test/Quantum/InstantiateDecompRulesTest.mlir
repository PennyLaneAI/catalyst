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

// RUN: catalyst --split-input-file --tool=opt --instantiate-decomp-rules %s | FileCheck %s

func.func @test_paulirot(%q0: !quantum.bit, %q1: !quantum.bit, %q2: !quantum.bit) -> !quantum.bit {
    %pi = arith.constant 3.14 : f64
    %out:3 = quantum.paulirot ["X", "Z", "Y"](%pi) %q0, %q1, %q2: !quantum.bit, !quantum.bit, !quantum.bit
    return %out#1 : !quantum.bit
    // CHECK: @_pauli_rot_decomposition_XZY([[angle_tensor:%.+]]: tensor<f64>, [[q0_tensor:%.+]]: tensor<i64>, [[q1_tensor:%.+]]: tensor<i64>, [[q2_tensor:%.+]]: tensor<i64>)
    
    // CHECK-DAG: [[pi_by_2:%.+]] = {{.*}} 1.57
    // CHECK-DAG: [[m_pi_by_2:%.+]] = {{.*}} -1.57

    // CHECK: [[reg:%.+]] = quantum.alloc( 3)
    // CHECK: [[h_in_index:%.+]] = tensor.extract [[q0_tensor:%.+]]
    // CHECK: [[h_in:%.+]] = quantum.extract [[reg]][[[h_in_index]]]
    // CHECK: [[h_out:%.+]] = quantum.custom "Hadamard"() [[h_in]]
    // CHECK: [[h_out_index:%.+]] = tensor.extract [[q0_tensor]]
    // CHECK: [[reg2:%.+]] = quantum.insert [[reg]][[[h_out_index]]], [[h_out]]
    // CHECK: [[rx_in_index:%.+]] = tensor.extract [[q2_tensor]]
    // CHECK: [[rx_in:%.+]] = quantum.extract [[reg2]][[[rx_in_index]]]
    // CHECK: [[rx_out:%.+]] = quantum.custom "RX"([[pi_by_2]]) [[rx_in]]
    // CHECK: [[rx_out_index:%.+]] = tensor.extract [[q2_tensor]]
    // CHECK: [[reg3:%.+]] = quantum.insert [[reg2]][[[rx_out_index]]], [[h_out]]
    // CHECK-DAG: [[mrz_q0_index:%.+]] = tensor.extract [[q0_tensor]]
    // CHECK-DAG: [[mrz_q0:%.+]] = quantum.extract [[reg3]][[[mrz_q0_index]]]
    // CHECK-DAG: [[mrz_q1_index:%.+]] = tensor.extract [[q1_tensor]]
    // CHECK-DAG: [[mrz_q1:%.+]] = quantum.extract [[reg3]][[[mrz_q1_index]]]
    // CHECK-DAG: [[mrz_q2_index:%.+]] = tensor.extract [[q2_tensor]]
    // CHECK-DAG: [[mrz_q2:%.+]] = quantum.extract [[reg3]][[[mrz_q2_index]]]
    // CHECK-DAG: [[mrz_angle:%.+]] = tensor.extract [[angle_tensor]]
    // CHECK: [[mrz_out:%.+]]:3 = quantum.multirz([[mrz_angle]]) [[mrz_q0]], [[mrz_q1]], [[mrz_q2]]
    // CHECK-DAG: [[mrz_out_q0_index:%.+]] = tensor.extract [[q0_tensor]]
    // CHECK-DAG: [[reg4:%.+]] = quantum.insert [[reg3]][[[mrz_out_q0_index]]], [[mrz_out]]#0
    // CHECK-DAG: [[mrz_out_q1_index:%.+]] = tensor.extract [[q1_tensor]]
    // CHECK-DAG: [[reg5:%.+]] = quantum.insert [[reg4]][[[mrz_out_q1_index]]], [[mrz_out]]#1
    // CHECK-DAG: [[mrz_out_q2_index:%.+]] = tensor.extract [[q2_tensor]]
    // CHECK-DAG: [[reg6:%.+]] = quantum.insert [[reg5]][[[mrz_out_q2_index]]], [[mrz_out]]#2
    // CHECK: [[h2_in_index:%.+]] = tensor.extract [[q0_tensor]]
    // CHECK: [[h2_in:%.+]] = quantum.extract [[reg6]][[[h2_in_index]]]
    // CHECK: [[h2_out:%.+]] = quantum.custom "Hadamard"() [[h2_in]]
    // CHECK: [[h2_out_index:%.+]] = tensor.extract [[q0_tensor]]
    // CHECK: [[reg7:%.+]] = quantum.insert [[reg6]][[[h2_out_index]]], [[h2_out]]
    // CHECK: [[rx2_in_index:%.+]] = tensor.extract [[q2_tensor]]
    // CHECK: [[rx2_in:%.+]] = quantum.extract [[reg7]][[[rx2_in_index]]]
    // CHECK: [[rx2_out:%.+]] = quantum.custom "RX"([[m_pi_by_2]]) [[rx2_in]]
    // CHECK: [[rx2_out_index:%.+]] = tensor.extract [[q2_tensor]]
    // CHECK: [[reg8:%.+]] = quantum.insert [[reg7]][[[rx2_out_index]]], [[rx2_out]]
}
