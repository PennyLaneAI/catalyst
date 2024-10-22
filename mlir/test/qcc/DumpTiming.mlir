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

// RUN: catalyst-cli %s --mlir-timing --verify-diagnostics 2>&1 | FileCheck %s

func.func @foo() {
    return
}

// CHECK: Execution time report
// CHECK: Total Execution Time
// CHECK: Parser
// CHECK: Optimization
// CHECK: Translate
// CHECK: llc