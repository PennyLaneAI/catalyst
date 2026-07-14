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

#ifndef EXECUTOR_DIALECT
#define EXECUTOR_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/IR/DialectBase.td"

//===----------------------------------------------------------------------===//
// Executor Dialect Definition
//===----------------------------------------------------------------------===//

def Executor_Dialect : Dialect {
    let summary = "Operations for dispatching kernels and calling symbols on an executor.";
    let description = [{
        The `executor` dialect models a small set of operations that describe how a host program
        interacts with an executor service:

        * Opening a session with an executor endpoint
        * Shipping a cross-compiled kernel object file or assets to that endpoint
        * Dispatching a previously-shipped qnode kernel
        * Invoking an arbitrary symbol in a shared library
    }];

    let name = "executor";
    let cppNamespace = "::catalyst::executor";
}

//===----------------------------------------------------------------------===//
// Executor Operation Base
//===----------------------------------------------------------------------===//

class Executor_Op<string mnemonic, list<Trait> traits = []> :
    Op<Executor_Dialect, mnemonic, traits>;

#endif // EXECUTOR_DIALECT
