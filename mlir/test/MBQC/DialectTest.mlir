// Copyright 2025 Xanadu Quantum Technologies Inc.

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

func.func @test_measure_in_basis_XY(%q1 : !quantum.bit) {
    %angle = arith.constant 3.141592653589793 : f64
    %res, %new_q = mbqc.measure_in_basis [XY, %angle] %q1 : i1, !quantum.bit
    func.return
}

// -----

func.func @test_measure_in_basis_YZ(%q1 : !quantum.bit) {
    %angle = arith.constant 3.141592653589793 : f64
    %res, %new_q = mbqc.measure_in_basis [YZ, %angle] %q1 : i1, !quantum.bit
    func.return
}

// -----

func.func @test_measure_in_basis_ZX(%q1 : !quantum.bit) {
    %angle = arith.constant 3.141592653589793 : f64
    %res, %new_q = mbqc.measure_in_basis [ZX, %angle] %q1 : i1, !quantum.bit
    func.return
}

// -----

func.func @test_measure_in_basis_dynamic_angle(%q1 : !quantum.bit, %arg0: tensor<f64>) {
    %1 = stablehlo.convert %arg0 : tensor<f64>
    %angle = tensor.extract %1[] : tensor<f64>
    %res, %new_q = mbqc.measure_in_basis [XY, %angle] %q1 : i1, !quantum.bit
    func.return
}

// -----

func.func @test_measure_in_basis_invalid_plane(%q1 : !quantum.bit) {
    %angle = arith.constant 3.141592653589793 : f64
    // expected-error@below {{expected catalyst::mbqc::MeasurementPlane to be one of: XY, YZ, ZX}}
    // expected-error@below {{failed to parse MeasurementPlaneAttr parameter}}
    %res, %new_q = mbqc.measure_in_basis [YX, %angle] %q1 : i1, !quantum.bit
    func.return
}

// -----

func.func @test_measure_in_basis_postselect_0(%q1 : !quantum.bit) {
    %angle = arith.constant 3.141592653589793 : f64
    %res, %new_q = mbqc.measure_in_basis [XY, %angle] %q1 postselect 0 : i1, !quantum.bit
    func.return
}

// -----

func.func @test_measure_in_basis_postselect_1(%q1 : !quantum.bit) {
    %angle = arith.constant 3.141592653589793 : f64
    %res, %new_q = mbqc.measure_in_basis [XY, %angle] %q1 postselect 1 : i1, !quantum.bit
    func.return
}

// -----

func.func @test_measure_in_basis_postselect_invalid(%q1 : !quantum.bit) {
    %angle = arith.constant 3.141592653589793 : f64
    // expected-error@below {{op attribute 'postselect' failed to satisfy constraint}}
    %res, %new_q = mbqc.measure_in_basis [XY, %angle] %q1 postselect -1 : i1, !quantum.bit
    func.return
}

// -----

func.func @test_measure_in_basis_postselect_invalid(%q1 : !quantum.bit) {
    %angle = arith.constant 3.141592653589793 : f64
    // expected-error@below {{op attribute 'postselect' failed to satisfy constraint}}
    %res, %new_q = mbqc.measure_in_basis [XY, %angle] %q1 postselect 2 : i1, !quantum.bit
    func.return
}

// -----

func.func @test_graph_state_prep() {
    %adj_matrix = arith.constant dense<[1, 0, 1, 0, 0, 1]> : tensor<6xi1>
    %graph_reg = mbqc.graph_state_prep (%adj_matrix : tensor<6xi1>) [init "Hadamard", entangle "CZ"] : !quantum.reg
    func.return
}
