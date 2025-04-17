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

func.func @foo(%q1 : !quantum.bit) {
    %angle = arith.constant 3.141592653589793 : f64
    %res, %new_q = mbqc.measure_in_basis [XY, %angle] %q1 : i1, !quantum.bit
    func.return
}

// -----

func.func @foo(%q1 : !quantum.bit) {
    %angle = arith.constant 3.141592653589793 : f64
    %res, %new_q = mbqc.measure_in_basis [YZ, %angle] %q1 : i1, !quantum.bit
    func.return
}

// -----

func.func @foo(%q1 : !quantum.bit) {
    %angle = arith.constant 3.141592653589793 : f64
    %res, %new_q = mbqc.measure_in_basis [ZX, %angle] %q1 : i1, !quantum.bit
    func.return
}

// -----

func.func @foo(%q1 : !quantum.bit) {
    %angle = arith.constant 3.141592653589793 : f64
    // expected-error@below {{expected catalyst::mbqc::BlochPlane to be one of: XY, ZX, YZ}}
    // expected-error@below {{failed to parse BlochPlaneAttr parameter}}
    %res, %new_q = mbqc.measure_in_basis [YX, %angle] %q1 : i1, !quantum.bit
    func.return
}
