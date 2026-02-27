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

#ifndef QEC_LOGICAL_OPS
#define QEC_LOGICAL_OPS

include "mlir/IR/OpBase.td"

include "QecLogical/IR/QecLogicalDialect.td"
include "QecLogical/IR/QecLogicalTypes.td"


//===----------------------------------------------------------------------===//
// QecLogical dialect operations.
//===----------------------------------------------------------------------===//

// QecLogical dialect base operation
class QecLogical_Op<string mnemonic, list<Trait> traits = []> :
        Op<QecLogical_Dialect, mnemonic, traits>;


#endif // QEC_LOGICAL_OPS
