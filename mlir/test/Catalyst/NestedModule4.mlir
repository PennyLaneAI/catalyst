// Copyright 2024 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Step 4 is replacing calls to inlined functions
// RUN: quantum-opt --inline-nested-module=stop-after-step=4 --split-input-file --verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: @outer
module @outer {
  module @inner {
    func.func @f() {
      return
    }
  }

  // CHECK: func.call @f_0
  catalyst.call_function_in_module @inner::@f() : () -> ()
}


