// Copyright 2022-2023 Xanadu Quantum Technologies Inc.

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

#include "Gradient/IR/GradientDialect.h"
#include "Gradient/IR/GradientOps.h"

using namespace mlir;
using namespace catalyst::gradient;

#include "Gradient/IR/GradientOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Gradient Dialect Interfaces
//===----------------------------------------------------------------------===//

namespace {
struct GradientInlinerInterface : public DialectInlinerInterface {
    using DialectInlinerInterface::DialectInlinerInterface;

    /// Operations in Gradient dialect are always legal to inline.
    bool isLegalToInline(Operation *, Region *, bool,
                         BlockAndValueMapping &valueMapping) const final
    {
        return true;
    }
};
} // namespace

//===----------------------------------------------------------------------===//
// Gradient dialect.
//===----------------------------------------------------------------------===//

void GradientDialect::initialize()
{
    addOperations<
#define GET_OP_LIST
#include "Gradient/IR/GradientOps.cpp.inc"
        >();
    addInterface<GradientInlinerInterface>();
}
