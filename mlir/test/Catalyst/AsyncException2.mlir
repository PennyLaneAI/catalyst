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

// RUN: quantum-opt --add-exception-handling=stop-after-step=2 --verify-diagnostics --split-input-file %s | FileCheck %s


// Check that personality does not get added.


module {

  // CHECK-NOT: @__gxx_personality_v0

  llvm.func @callee()

  llvm.func @caller() {
    llvm.call @callee() : () -> ()
    llvm.return
  }
}

