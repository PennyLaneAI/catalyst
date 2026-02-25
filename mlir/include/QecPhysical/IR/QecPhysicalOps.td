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

#ifndef QEC_PHYSICAL_OPS
#define QEC_PHYSICAL_OPS

include "mlir/IR/OpBase.td"

include "QecPhysical/IR/QecPhysicalDialect.td"
include "QecPhysical/IR/QecPhysicalTypes.td"


//===----------------------------------------------------------------------===//
// QecPhysical dialect operations.
//===----------------------------------------------------------------------===//

// QecPhysical dialect base operation
class QecPhysical_Op<string mnemonic, list<Trait> traits = []> :
        Op<QecPhysical_Dialect, mnemonic, traits>;


#endif // QEC_PHYSICAL_OPS