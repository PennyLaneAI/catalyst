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


// RUN: quantum-opt %s --convert-transport-to-llvm --split-input-file --verify-diagnostics

// A reply slot is one contiguous span from the ring slot's base, so there is no descriptor the
// lowering could build for a dynamically shaped result: it has no size to take from the type, and
// the runtime call reports none. lower-decode-to-transport only ever asks for a slot it has proven
// static, so this is hand-written IR being refused rather than a reachable pipeline state.

func.func @reply_slot_dynamic() {
  %s = transport.get_session {key = "cop0"} : !transport.session<controller>
  // expected-error @below {{failed to legalize operation 'transport.reply_slot'}}
  %slot = transport.reply_slot %s : !transport.session<controller> -> memref<?xindex>
  return
}

// -----

// Nor for a non-identity layout: the slot is contiguous from its base, so a strided view of it is
// not something the ring can hand back.

func.func @reply_slot_strided() {
  %s = transport.get_session {key = "cop0"} : !transport.session<controller>
  // expected-error @below {{failed to legalize operation 'transport.reply_slot'}}
  %slot = transport.reply_slot %s : !transport.session<controller> -> memref<4xindex, strided<[2]>>
  return
}
