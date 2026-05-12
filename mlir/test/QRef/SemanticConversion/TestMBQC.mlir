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

// Test conversion to value semantics quantum dialect for MBQC circuits.

// RUN: quantum-opt --convert-to-value-semantics --canonicalize --split-input-file --verify-diagnostics %s | FileCheck %s


// CHECK-LABEL: test_mbqc_measure_in_basis_op
func.func @test_mbqc_measure_in_basis_op(%angle: f64) -> (i1, i1, i1) attributes {quantum.node} {

    // CHECK: [[qreg:%.+]] = quantum.alloc( 1) : !quantum.reg
    // CHECK: [[qb:%.+]] = quantum.alloc_qb : !quantum.bit
    %a = qref.alloc(1) : !qref.reg<1>
    %q0 = qref.get %a[0] : !qref.reg<1> -> !qref.bit
    %qb = qref.alloc_qb : !qref.bit

    // CHECK: [[mres0:%.+]], [[qb_0:%.+]] = mbqc.measure_in_basis[ XY, %arg0] [[qb]] : i1, !quantum.bit
    // CHECK: [[q0:%.+]] = quantum.extract [[qreg]][ 0] : !quantum.reg -> !quantum.bit
    // CHECK: [[mres1:%.+]], [[q0_0:%.+]] = mbqc.measure_in_basis[ YZ, %arg0] [[q0]] : i1, !quantum.bit
    // CHECK: [[mres2:%.+]], [[q0_1:%.+]] = mbqc.measure_in_basis[ ZX, %arg0] [[q0_0]] : i1, !quantum.bit
    %mres0 = mbqc.ref.measure_in_basis [XY, %angle] %qb : i1
    %mres1 = mbqc.ref.measure_in_basis [YZ, %angle] %q0 : i1
    %mres2 = mbqc.ref.measure_in_basis [ZX, %angle] %q0 : i1

    // CHECK: [[insert:%.+]] = quantum.insert [[qreg]][ 0], [[q0_1]] : !quantum.reg, !quantum.bit
    // CHECK: quantum.dealloc [[insert]] : !quantum.reg
    // CHECK: quantum.dealloc_qb [[qb_0]] : !quantum.bit
    qref.dealloc %a : !qref.reg<1>
    qref.dealloc_qb %qb : !qref.bit

    // CHECK: return [[mres0]], [[mres1]], [[mres2]] : i1, i1, i1
    return %mres0, %mres1, %mres2 : i1, i1, i1
}


// -----


// CHECK-LABEL: test_mbqc_graph_state_prep
func.func @test_mbqc_graph_state_prep(%arg0: tensor<6xi1>) attributes {quantum.node} {

    // CHECK: [[qreg:%.+]] = mbqc.graph_state_prep(%arg0 : tensor<6xi1>) [init "Hadamard", entangle "CZ"] : !quantum.reg
    %graph_reg = mbqc.ref.graph_state_prep (%arg0 : tensor<6xi1>) [init "Hadamard", entangle "CZ"] : !qref.reg<4>
    return
}


// -----


// CHECK: func.func @test_mbqc_circuit(%arg0: !quantum.bit) -> !quantum.bit
func.func @test_mbqc_circuit(%reg: !qref.reg<1>) {
    %adj_matrix = arith.constant dense<[1, 0, 1, 0, 0, 1]> : tensor<6xi1>
    %angle = arith.constant 37.42 : f64

    // CHECK: [[graph_reg:%.+]] = mbqc.graph_state_prep({{%.+}} : tensor<6xi1>) [init "Hadamard", entangle "CZ"] : !quantum.reg
    %graph_reg = mbqc.ref.graph_state_prep (%adj_matrix : tensor<6xi1>) [init "Hadamard", entangle "CZ"] : !qref.reg<4>
    %q0 = qref.get %graph_reg[0] : !qref.reg<4> -> !qref.bit

    // CHECK: [[q0:%.+]] = quantum.extract [[graph_reg]][ 0] : !quantum.reg -> !quantum.bit
    // CHECK: [[mres:%.+]], [[m_XY:%.+]] = mbqc.measure_in_basis[ XY, {{%.+}}] [[q0]] : i1, !quantum.bit
    %mres0 = mbqc.ref.measure_in_basis [XY, %angle] %q0 : i1

    // CHECK: [[ifOut:%.+]]:2 = scf.if [[mres]] -> (!quantum.bit, !quantum.bit) {
    // CHECK:   [[CZ:%.+]]:2 = quantum.custom "CZ"() [[m_XY]], %arg0 : !quantum.bit, !quantum.bit
    // CHECK:   scf.yield [[CZ]]#0, [[CZ]]#1 : !quantum.bit, !quantum.bit
    // CHECK: } else {
    // CHECK:   scf.yield [[m_XY]], %arg0 : !quantum.bit, !quantum.bit
    // CHECK: }
    scf.if %mres0 {
        %q0_alias = qref.get %graph_reg[0] : !qref.reg<4> -> !qref.bit
        %main_qubit = qref.get %reg[0] : !qref.reg<1> -> !qref.bit
        qref.custom "CZ"() %q0_alias, %main_qubit : !qref.bit, !qref.bit
    }
    // CHECK: [[insert:%.+]] = quantum.insert [[graph_reg]][ 0], [[ifOut]]#0 : !quantum.reg, !quantum.bit

    // CHECK: quantum.dealloc [[insert]] : !quantum.reg
    qref.dealloc %graph_reg : !qref.reg<4>

    // CHECK: return [[ifOut]]#1 : !quantum.bit
    return
}


// CHECK: func.func @main() attributes {quantum.node}
func.func @main() attributes {quantum.node} {
    // CHECK: [[r:%.+]] = quantum.alloc( 1) : !quantum.reg
    %r = qref.alloc(1) : !qref.reg<1>

    // CHECK: [[extract:%.+]] = quantum.extract [[r]][ 0] : !quantum.reg -> !quantum.bit
    // CHECK: [[call:%.+]] = call @test_mbqc_circuit([[extract]]) : (!quantum.bit) -> !quantum.bit
    func.call @test_mbqc_circuit(%r) : (!qref.reg<1>) -> ()
    // CHECK: [[insert:%.+]] = quantum.insert [[r]][ 0], [[call]] : !quantum.reg, !quantum.bit

    // CHECK: quantum.dealloc [[insert]] : !quantum.reg
    qref.dealloc %r : !qref.reg<1>
    return
}
