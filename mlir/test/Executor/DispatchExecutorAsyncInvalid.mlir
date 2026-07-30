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

// RUN: quantum-opt %s --dispatch-executor-targets --split-input-file --verify-diagnostics

// A nonblocking call must be `()->()`
module @jit_async_with_operand {
  func.func @bad(%arg0: memref<4xf64>) attributes {llvm.emit_c_interface} {
    // expected-error @below {{nonblocking dispatch requires a '()->()' callee}}
    catalyst.launch_kernel @module_bad::@takes_arg(%arg0) {catalyst.nonblocking} : (memref<4xf64>) -> ()
    return
  }

  module @module_bad attributes {catalyst.object_file = "/tmp/bad.o", catalyst.dispatch = {address = "127.0.0.1:9001"}} {
    func.func public @takes_arg(%arg0: memref<4xf64>) { return }
  }
}

// -----

// The await goes before the entry block's terminator, so a launch nested in a region is rejected:
// its token is not in scope there.

module @jit_async_nested {
  // expected-error @below {{async launch must be in the function's entry block}}
  func.func @bad(%cond: i1) attributes {llvm.emit_c_interface} {
    scf.if %cond {
      catalyst.launch_kernel @module_bad::@serve() {catalyst.nonblocking} : () -> ()
    }
    return
  }

  module @module_bad attributes {catalyst.object_file = "/tmp/bad.o", catalyst.dispatch = {address = "127.0.0.1:9001"}} {
    func.func public @serve() { return }
  }
}
