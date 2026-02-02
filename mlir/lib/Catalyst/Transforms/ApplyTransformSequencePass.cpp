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

#include <cassert>

#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/Dialect/Transform/Transforms/TransformInterpreterUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/AnalysisManager.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassInstrumentation.h"
#include "mlir/Pass/PassManager.h"

using namespace llvm;
using namespace mlir;

namespace catalyst {

/// Generate a meaningful name for a transform operation for pass instrumentation
static std::string getTransformOpName(transform::TransformOpInterface transformOp)
{
    std::string baseName = transformOp->getName().getStringRef().str();

    if (auto applyPassOp = dyn_cast<transform::ApplyRegisteredPassOp>(transformOp.getOperation())) {
        if (auto passName = applyPassOp.getPassName(); !passName.empty()) {
            return "transform_" + passName.str();
        }
    }

    // convert "." to "_"
    std::replace(baseName.begin(), baseName.end(), '.', '_');
    return baseName;
}

/// A fake pass wrapper that represents a single transform operation. Allowing it to be tracked by
/// pass instrumentation.
class TransformOpSubPass : public OperationPass<> {
  private:
    transform::TransformOpInterface transformOp;
    std::string opNameStr;

  public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TransformOpSubPass)

    TransformOpSubPass(transform::TransformOpInterface op)
        : OperationPass(TypeID::get<TransformOpSubPass>()), transformOp(op),
          opNameStr(getTransformOpName(op))
    {
    }

    void runOnOperation() override
    {
        llvm_unreachable("TransformOpSubPass should not be executed");
    }

    StringRef getName() const override { return opNameStr; }
    StringRef getArgument() const override { return opNameStr; }
    StringRef getDescription() const override { return "Transform dialect operation"; }

    std::unique_ptr<Pass> clonePass() const override
    {
        return std::make_unique<TransformOpSubPass>(transformOp);
    }

    transform::TransformOpInterface getTransformOp() const { return transformOp; }
};

/// Apply transforms with individual subpass tracking by executing each transform operation
/// individually with instrumentation hooks. This implements a custom sequence
/// execution that mirrors the logic in NamedSequenceOp::apply but with instrumentation.
/// Reference to transform::NamedSequenceOp::apply in
/// https://github.com/llvm/llvm-project/blob/334e9bf2dd01fbbfe785624c0de477b725cde6f2/mlir/lib/
/// Dialect/Transform/IR/TransformOps.cpp#L2378
LogicalResult applyTransformsWithSubpassTracking(Operation *payload,
                                                 transform::NamedSequenceOp namedSequence,
                                                 PassInstrumentor *passInstrumentor)
{
    // TODO: We currently only expect to have a single block in the sequence. It may change in the
    // future.
    assert(namedSequence.getBody().hasOneBlock() &&
           "Expected exactly one transform op in the sequence block");

    Block &sequenceBlock = namedSequence.getBody().front();
    if (sequenceBlock.without_terminator().empty()) {
        return success();
    }

    transform::TransformState state =
        transform::detail::makeTransformStateForTesting(namedSequence->getParentRegion(), payload);

    // Map the entry block argument to the list of operations.
    // Note: this is the same implementation as PossibleTopLevelTransformOp but
    // without attaching the interface / trait since that is tailored to a
    // dangling top-level op that does not get "called".
    auto scope = state.make_region_scope(namedSequence.getBody());
    if (failed(transform::detail::mapPossibleTopLevelTransformOpBlockArguments(
            state, namedSequence, namedSequence.getBody()))) {
        return failure();
    }

    for (Operation &transformOp : sequenceBlock.without_terminator()) {
        if (auto transformInterface = dyn_cast<transform::TransformOpInterface>(transformOp)) {
            auto subPass = std::make_unique<TransformOpSubPass>(transformInterface);

            // hook before pass
            passInstrumentor->runBeforePass(subPass.get(), payload);

            DiagnosedSilenceableFailure result = state.applyTransform(transformInterface);

            if (result.isDefiniteFailure()) {
                // hook after pass failed
                passInstrumentor->runAfterPassFailed(subPass.get(), payload);
                return failure();
            }

            if (result.isSilenceableFailure()) {
                (void)result.silence();
            }

            // hook after pass
            passInstrumentor->runAfterPass(subPass.get(), payload);
        }
    }

    return success();
}

#define GEN_PASS_DEF_APPLYTRANSFORMSEQUENCEPASS
#include "Catalyst/Transforms/Passes.h.inc"

struct ApplyTransformSequencePass
    : public impl::ApplyTransformSequencePassBase<ApplyTransformSequencePass> {
    using impl::ApplyTransformSequencePassBase<
        ApplyTransformSequencePass>::ApplyTransformSequencePassBase;

    void runOnOperation() override
    {
        // We need to remove the transformer module from the payload,
        // then apply the transformer module to the payload.
        // This is because we should not modify a module that contains
        // the transformer.

        // The top-level module is the payload.
        Operation *payload = getOperation();
        ModuleOp transformer;

        // Find the transformer module and remove it from payload
        // Keep the transformer module in a deep copy clone
        WalkResult result = payload->walk([&](ModuleOp op) {
            if (op->hasAttr("transform.with_named_sequence")) {
                transformer = op.clone();
                op.erase();
                return WalkResult::interrupt();
            }
            return WalkResult::advance();
        });

        // Check that a transformer exists
        if (!result.wasInterrupted()) {
            return;
        }

        // The transformer module itself is a builtin.module, not
        // a valid transform with the transform dialect
        // We need to extract the transform.named_sequence in the
        // transformer module.
        transform::NamedSequenceOp transformer_main_sequence;
        transformer->walk([&](transform::NamedSequenceOp op) {
            assert(!transformer_main_sequence &&
                   "expected only one transform sequence in the transform module");
            transformer_main_sequence = op;
        });

        if (PassInstrumentor *passInstrumentor = getAnalysisManager().getPassInstrumentor()) {
            // Manually execute the transform sequence with individual subpass tracking
            if (failed(applyTransformsWithSubpassTracking(payload, transformer_main_sequence,
                                                          passInstrumentor))) {
                return signalPassFailure();
            }
        }
        else {
            if (failed(transform::applyTransforms(payload, transformer_main_sequence, {},
                                                  transform::TransformOptions(), false))) {
                return signalPassFailure();
            }
        }

        transformer.erase();
    }
};

} // namespace catalyst
