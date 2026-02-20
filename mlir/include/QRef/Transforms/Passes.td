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

#ifndef QREF_PASSES
#define QREF_PASSES

include "mlir/Pass/PassBase.td"

def ValueSemanticsConversionPass : Pass<"convert-to-value-semantics"> {
    let summary = "Convert to value semantics quantum dialect";

    let dependentDialects = [
       "catalyst::quantum::QuantumDialect"
    ];
}

#endif // QREF_PASSES
