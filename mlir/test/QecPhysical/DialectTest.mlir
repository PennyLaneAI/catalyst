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

// RUN: quantum-opt --split-input-file --verify-diagnostics %s

func.func @test_qubit_data(%arg0 : !qecp.qubit<data>) {
    func.return
}

// -----

func.func @test_qubit_aux(%arg0 : !qecp.qubit<aux>) {
    func.return
}

// -----

func.func @test_codeblock(%arg0 : !qecp.codeblock<1 x 7>) {
    func.return
}

// -----

// TODO: Currently we must write parameters as '<3 x 1 x 7>' (with spaces)
func.func @test_hyperreg(%arg0 : !qecp.hyperreg<3 x 1 x 7>) {
    func.return
}
