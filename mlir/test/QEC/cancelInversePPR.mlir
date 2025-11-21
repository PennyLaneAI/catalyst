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

// RUN: quantum-opt --cancel-inverse-ppr --split-input-file -verify-diagnostics %s | FileCheck %s
// RUN: quantum-opt --cancel-inverse-ppr="max-pauli-size=3" --split-input-file -verify-diagnostics %s | FileCheck %s --check-prefixes=CHECK-MPS

# CHECK-LABEL: test_cancel_Z
func.func @test_cancel_Z(%q1 : !quantum.bit){
    
    // Z(-pi/8) * Z(pi/8)
    
    // CHECK-NOT: qec.ppr
    %4 = qec.ppr ["Z"](-8) %3 : !quantum.bit
    %5 = qec.ppr ["Z"](8) %4 : !quantum.bit
    func.return
}


# CHECK-LABEL: test_cancel_X
func.func @test_cancel_X(%q1 : !quantum.bit){
    
    // X(-pi/8) * X(pi/8)
    
    // CHECK-NOT: qec.ppr
    %4 = qec.ppr ["Z"](-8) %3 : !quantum.bit
    %5 = qec.ppr ["Z"](8) %4 : !quantum.bit
    func.return
}
