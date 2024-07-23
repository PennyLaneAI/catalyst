// Copyright 2023 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// RUN: quantum-opt --disable-assertion --split-input-file %s | FileCheck %s

//////////////////////////
// Catalyst AssertionOp //
//////////////////////////

func.func @assert_constant(%arg0: i1, %arg1: !llvm.ptr) {
    // CHECK-NOT: @__catalyst__rt__assert_bool
    "catalyst.assert"(%arg0) <{error = "Test Message"}> : (i1) -> ()
    
    return
}
