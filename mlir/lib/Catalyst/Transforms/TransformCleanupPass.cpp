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

// This pass removes the transform.named_sequence operation and the
// transform.with_named_sequence attribute from the IR after the
// -transform-interpreter is run during the quantum peephole optimizations.

#define DEBUG_TYPE "transformcleanup"

#include "Catalyst/IR/CatalystDialect.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"

using namespace llvm;
using namespace mlir;
using namespace catalyst;

namespace catalyst {
#define GEN_PASS_DEF_TRANSFORMCLEANUPPASS
#include "Catalyst/Transforms/Passes.h.inc"

struct TransformCleanupPass : public impl::TransformCleanupPassBase<TransformCleanupPass> {
    using impl::TransformCleanupPassBase<TransformCleanupPass>::TransformCleanupPassBase;

    void runOnOperation() override
    {
        Operation *module = getOperation();

        auto TransformNamedSequenceEraser = [&](Operation *op) {
            if (op->getName().getStringRef() == "transform.named_sequence") {
                op->erase();
                // Note: because of the design of the transform dialect, transform.named_sequence
                // does not have any users
            }
        };

        module->walk(TransformNamedSequenceEraser);

        module->removeAttr("transform.with_named_sequence");
    }
};

std::unique_ptr<Pass> createTransformCleanupPass()
{
    return std::make_unique<TransformCleanupPass>();
}

} // namespace catalyst