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

// This pass applies the passes scheduled with the transform dialect,
// and then removes the transformer module from the payload.

#define DEBUG_TYPE "applytransformsequence"

#include "Catalyst/IR/CatalystDialect.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"

using namespace llvm;
using namespace mlir;
using namespace catalyst;

namespace catalyst {
#define GEN_PASS_DEF_APPLYTRANSFORMSEQUENCEPASS
#include "Catalyst/Transforms/Passes.h.inc"

struct ApplyTransformSequencePass
    : public impl::ApplyTransformSequencePassBase<ApplyTransformSequencePass> {
    using impl::ApplyTransformSequencePassBase<
        ApplyTransformSequencePass>::ApplyTransformSequencePassBase;

    void runOnOperation() override
    {
        // The top-level module is the payload
        // We need to remove the transformer module from the payload,
        // without deleting the transformer module in memory.
        // Then, apply the transformer module to the payload.

        Operation *payload = getOperation();
        Operation *transformer;

        // Find the transformer module and remove it from payload
        payload->walk([&](Operation *op) {
            if (isa<ModuleOp>(op)) {
                if (op->hasAttr("transform.with_named_sequence")) {
                    transformer = op;
                }
            }
        });

        // Note: operation.remove() will remove the operation from parent,
        // but keep it in memory
        // As opposed to operation.erase(), which removes from parent and
        // deletes it completely.
        // This nuanced detail enables our maneuvering here.
        transformer->remove();

        // The transformer module itself is a builtin.module, not
        // a valid transform with the transform dialect
        // We need to extract the transform.named_sequence in the
        // transformer module.
        Operation *transformer_main_sequence;
        transformer->walk([&](Operation *op) {
            if (op->getName().getStringRef() == "transform.named_sequence") {
                transformer_main_sequence = op;
            }
        });

        // Perform the transform
        if (failed(mlir::transform::applyTransforms(
                payload, cast<mlir::transform::TransformOpInterface>(transformer_main_sequence), {},
                mlir::transform::TransformOptions(), false))) {
            return signalPassFailure();
        };

        // All done, erase the transformer module
        transformer->erase();
    }
};

std::unique_ptr<Pass> createApplyTransformSequencePass()
{
    return std::make_unique<ApplyTransformSequencePass>();
}

} // namespace catalyst
