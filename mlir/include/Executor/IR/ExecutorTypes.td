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

#ifndef EXECUTOR_TYPES
#define EXECUTOR_TYPES

include "mlir/IR/AttrTypeBase.td"

include "Executor/IR/ExecutorDialect.td"

//===----------------------------------------------------------------------===//
// Executor dialect types.
//===----------------------------------------------------------------------===//

class Executor_Type<string name, string typeMnemonic, list<Trait> traits = []>
        : TypeDef<Executor_Dialect, name, traits> {
    let mnemonic = typeMnemonic;
}

def Executor_SessionType : Executor_Type<"Session", "session"> {
    let summary = "A handle to an open executor session.";

    let description = [{
        A value-semantic handle to a session with an executor endpoint, produced by
        `executor.open` and consumed by `executor.send_binary`, `executor.launch`,
        `executor.call`, and `executor.close`.

        #### Example:

        ```mlir
        %s = executor.open("127.0.0.1:9000") : !executor.session
        executor.close %s : !executor.session
        ```
    }];
}

#endif // EXECUTOR_TYPES
