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

module @outer {
  module @inner {
    // Default visibility is public
    func.func @f() attributes {quantum.kernel_entry_point} {
      return
    }
  }

  catalyst.launch_kernel @inner::@f() : () -> ()
}

// -----

module @outer {
  module @inner {
    func.func private @f() attributes {quantum.kernel_entry_point} {
      return
    }
  }

  // expected-error @below {{needs to have public visibility}}
  catalyst.launch_kernel @inner::@f() : () -> ()
}

// -----

module @outer {
  module @inner {
    func.func nested @f() attributes {quantum.kernel_entry_point} {
      return
    }
  }

  // expected-error @below {{needs to have public visibility}}
  catalyst.launch_kernel @inner::@f() : () -> ()
}

// -----

module @outer {
  module @inner {
  }

  // expected-error @below {{could not find kernel function}}
  catalyst.launch_kernel @inner::@f() : () -> ()
}

// -----

module @outer {
  module @inner {
    func.func public @f() {
      return
    }
  }

  // expected-error @below {{requires entry point attribute}}
  catalyst.launch_kernel @inner::@f() : () -> ()
}
