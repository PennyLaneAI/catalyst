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

// Step 1 is to generate the fully qualified name as an attribute.
// RUN: quantum-opt --inline-nested-module=stop-after-step=1 --split-input-file --verify-diagnostics %s | FileCheck %s


// CHECK-LABEL: @outer
module @outer {
  module @inner {
    // CHECK: func.func @f() attributes {catalyst.fully_qualified_name = @inner::@f}
    func.func @f() {
      func.return
    }
  }
}

// -----

// Test that when the root module is not a symbol
// (i.e., it doesn't have a name)
// then we still succeed.
module {
  // CHECK-LABEL: @inner_test2
  module @inner_test2 {
    // CHECK: func.func @f() attributes {catalyst.fully_qualified_name = @inner_test2::@f}
    func.func @f() {
      func.return
    }
  }
}

// -----

module {
  // Test that when the inner module is not a symbol
  // (i.e., it doesn't have a name)
  // we still succeed
  module {
    // CHECK: func.func @f() attributes {catalyst.fully_qualified_name = @f}
    func.func @f() {
      func.return
    }
  }
}

// -----

// Test that if the fully_qualified_name is already present, we do not
// annotate the function with the fully qualified name.
module {
  module {
    // CHECK: func.func @f() attributes {catalyst.fully_qualified_name = @hello}
    func.func @f() attributes { catalyst.fully_qualified_name = @hello } {
      func.return
    }
  }
}
