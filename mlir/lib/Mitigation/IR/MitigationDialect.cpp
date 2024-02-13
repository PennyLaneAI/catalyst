// Copyright 2023 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mlir/Transforms/InliningUtils.h"

#include "Mitigation/IR/MitigationDialect.h"
#include "Mitigation/IR/MitigationOps.h"

using namespace mlir;
using namespace catalyst::mitigation;

#include "Mitigation/IR/MitigationOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Mitigation Dialect Interfaces
//===----------------------------------------------------------------------===//

namespace {
struct MitigationInlinerInterface : public DialectInlinerInterface {
    using DialectInlinerInterface::DialectInlinerInterface;

    /// Operations in Mitigation dialect are always legal to inline.
    bool isLegalToInline(Operation *, Region *, bool, IRMapping &valueMapping) const final
    {
        return true;
    }
};
} // namespace

//===----------------------------------------------------------------------===//
// Mitigation dialect.
//===----------------------------------------------------------------------===//

void MitigationDialect::initialize()
{
    addOperations<
#define GET_OP_LIST
#include "Mitigation/IR/MitigationOps.cpp.inc"
        >();
    addInterface<MitigationInlinerInterface>();
}
