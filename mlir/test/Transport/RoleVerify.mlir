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

// Role safety: controller-only ops reject a coprocessor session and vice versa.

func.func @stage_payload_requires_controller() {
  %c = transport.create {backend_lib = "x", config = "c"} -> !transport.session<coprocessor>
  %buf = memref.alloc() : memref<1xi8>
  // expected-error @+1 {{operand #0 must be}}
  transport.stage_payload %c, %buf : !transport.session<coprocessor>, memref<1xi8>
  return
}

// -----

func.func @commit_requires_controller() {
  %c = transport.create {backend_lib = "x", config = "c"} -> !transport.session<coprocessor>
  // expected-error @+1 {{operand #0 must be}}
  transport.set_message_sizes %c {in_bytes = 8 : i64, out_bytes = 8 : i64} : !transport.session<coprocessor>
  return
}

// -----

func.func @set_coprocessor_fn_requires_coprocessor() {
  %c = transport.create {backend_lib = "x", config = "c"} -> !transport.session<controller>
  // expected-error @+1 {{operand #0 must be}}
  transport.set_coprocessor_fn %c {symbol = "decode"} : !transport.session<controller>
  return
}
