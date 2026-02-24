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

#ifndef QEC_LOGICAL_DIALECT
#define QEC_LOGICAL_DIALECT

include "mlir/IR/DialectBase.td"

//===----------------------------------------------------------------------===//
// QecLogical dialect definition.
//===----------------------------------------------------------------------===//

def QecLogical_Dialect : Dialect {
    let summary = "A value-semantics dialect for logical QEC program representation.";
    let description = [{
        TODO
    }];

    /// This is the namespace of the dialect in MLIR, which is used as a prefix for types and ops.
    let name = "qecl";

    /// This is the C++ namespace in which the dialect and all of its sub-components are placed.
    let cppNamespace = "::catalyst::qecl";

    /// This dialect depends on the following list of other dialects.
    // let dependentDialects = [];

    /// Use the default type printing/parsing hooks, otherwise we would have to explicitly define them.
    let useDefaultTypePrinterParser = 1;
}


#endif // QEC_LOGICAL_DIALECT
