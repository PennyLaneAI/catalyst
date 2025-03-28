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

// Step 2 is renaming
// RUN: quantum-opt --inline-nested-module=stop-after-step=2 --split-input-file --verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: @outer
module @outer {
  module @inner {
    // CHECK: func.func @f_0() 
    func.func @f() {
      return
    }
  }
}

// -----

// Test collision
// CHECK-LABEL: @outer
module @outer {
  module @inner0 {
    // CHECK: func.func @f_1() 
    func.func @f() {
      return
    }
  }

  func.func @f_0() {
    return
  }
}

// -----

// Test collision nested
// CHECK-LABEL: @outer
module @outer {
  module @inner0 {
    // CHECK-DAG: func.func @f_0
    func.func @f() {
      return
    }
  }

  module @inner1 {
    // CHECK-DAG: func.func @f_1
    func.func @f() {
      return
    }
  }
}

