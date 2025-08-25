// Copyright 2025 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#define DEBUG_TYPE "user-defined-decomposition"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/Support/Debug.h"

#include "Quantum/IR/QuantumOps.h"
#include "Quantum/Transforms/Patterns.h"

using namespace mlir;
using namespace catalyst::quantum;

namespace catalyst {
namespace quantum {

bool validateDecompFuncTypeSignature(CustomOp op, func::FuncOp decompFunc)
{
    auto decompFuncType = decompFunc.getFunctionType();

    if (decompFuncType.getNumInputs() != op.getNumOperands()) {
        LLVM_DEBUG(llvm::dbgs() << "  [FAIL] Parameter count mismatch: function expects "
                                << decompFuncType.getNumInputs() << ", op has "
                                << op.getNumOperands() << "\n";);
        return false;
    }

    if (decompFuncType.getNumResults() != op.getNumResults()) {
        LLVM_DEBUG(llvm::dbgs() << "  [FAIL] Result count mismatch: function returns "
                                << decompFuncType.getNumResults() << ", op returns "
                                << op.getNumResults() << "\n";);
        return false;
    }

    if (llvm::any_of(llvm::zip(decompFuncType.getInputs(), op.getOperands()),
                     [](auto pair) { return std::get<0>(pair) != std::get<1>(pair).getType(); })) {
        LLVM_DEBUG(llvm::dbgs() << "  [FAIL] Parameter type mismatch\n";);
        return false;
    }

    if (llvm::any_of(llvm::zip(decompFuncType.getResults(), op.getResults()),
                     [](auto pair) { return std::get<0>(pair) != std::get<1>(pair).getType(); })) {
        LLVM_DEBUG(llvm::dbgs() << "  [FAIL] Result type mismatch\n";);
        return false;
    }

    return true;
}

struct UserDefinedDecompositionRewritePattern : public OpRewritePattern<CustomOp> {
  private:
    const llvm::StringMap<func::FuncOp> &decompositionRegistry;

  public:
    UserDefinedDecompositionRewritePattern(MLIRContext *context,
                                           const llvm::StringMap<func::FuncOp> &registry)
        : OpRewritePattern(context), decompositionRegistry(registry)
    {
    }

    LogicalResult matchAndRewrite(CustomOp op, PatternRewriter &rewriter) const override
    {
        StringRef gateName = op.getGateName();

        auto it = decompositionRegistry.find(gateName);
        if (it == decompositionRegistry.end()) {
            return failure();
        }

        func::FuncOp decompFunc = it->second;
        LLVM_DEBUG(llvm::dbgs() << "  [DEBUG] Found decomposition function: '"
                                << decompFunc.getSymName() << "'\n"
                                << "  [DEBUG] Function type: " << decompFunc.getFunctionType()
                                << "\n"
                                << "  [DEBUG] Verifying type signature\n";);

        if (!validateDecompFuncTypeSignature(op, decompFunc)) {
            return failure();
        }

        LLVM_DEBUG(
            llvm::dbgs() << "  [DEBUG] Generating function call to decomposition function...\n"
                         << "    Function name: " << decompFunc.getSymName() << "\n"
                         << "    Operands: " << op.getNumOperands() << "\n"
                         << "    Results: " << op.getNumResults() << "\n";);

        auto callOp =
            rewriter.create<func::CallOp>(op.getLoc(), decompFunc.getFunctionType().getResults(),
                                          decompFunc.getSymName(), op.getOperands());

        LLVM_DEBUG(llvm::dbgs() << "  [DEBUG] Replacing original operation with function call...\n"
                                << "    Original op: " << *op << "\n"
                                << "    Generated call: " << *callOp << "\n";);

        rewriter.replaceOp(op, callOp.getResults());

        return success();
    }
};

void populateUserDefinedDecompositionPatterns(
    RewritePatternSet &patterns, const llvm::StringMap<func::FuncOp> &decompositionRegistry)
{
    patterns.add<UserDefinedDecompositionRewritePattern>(patterns.getContext(),
                                                         decompositionRegistry);
}

} // namespace quantum
} // namespace catalyst
