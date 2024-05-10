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

// RUN: quantum-opt %s --specialize-active-callback-pass --split-input-file --verify-diagnostics | FileCheck %s

// This test just makes sure that we can
// run the compiler with the option
//
//   --specialize-active-callback-pass

// CHECK-LABEL: @foo
func.func @foo() {
  return
}

