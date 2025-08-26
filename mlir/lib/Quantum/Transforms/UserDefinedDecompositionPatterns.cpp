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
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/Support/Debug.h"

#include "Quantum/IR/QuantumOps.h"
#include "Quantum/Transforms/Patterns.h"

using namespace mlir;
using namespace catalyst::quantum;

namespace catalyst {
namespace quantum {

struct ValidationResult {
    bool isValid = false;
    // Maps parameter index to whether it needs scalar->tensor<scalar> conversion
    llvm::SmallVector<bool, 4> needsConversion;
};

ValidationResult validateDecompFuncTypeSignature(CustomOp op, func::FuncOp decompFunc)
{
    ValidationResult result;
    auto decompFuncType = decompFunc.getFunctionType();

    if (decompFuncType.getNumInputs() != op.getNumOperands()) {
        LLVM_DEBUG(llvm::dbgs() << "  [FAIL] Parameter count mismatch: function expects "
                                << decompFuncType.getNumInputs() << ", op has "
                                << op.getNumOperands() << "\n";);
        return result; // isValid = false by default
    }

    if (decompFuncType.getNumResults() != op.getNumResults()) {
        LLVM_DEBUG(llvm::dbgs() << "  [FAIL] Result count mismatch: function returns "
                                << decompFuncType.getNumResults() << ", op returns "
                                << op.getNumResults() << "\n";);
        return result; // isValid = false by default
    }

    result.needsConversion.resize(op.getNumOperands(), false);

    // Check parameter types with flexible matching
    for (auto [idx, pair] :
         llvm::enumerate(llvm::zip(decompFuncType.getInputs(), op.getOperands()))) {
        Type funcParamType = std::get<0>(pair);
        Type opOperandType = std::get<1>(pair).getType();

        if (funcParamType == opOperandType) {
            continue;
        }

        // Check for scalar -> tensor<scalar> conversion
        if (auto tensorType = dyn_cast<RankedTensorType>(funcParamType)) {
            if (tensorType.getRank() == 0 && // scalar tensor
                tensorType.getElementType() == opOperandType) {
                result.needsConversion[idx] = true;
                LLVM_DEBUG(llvm::dbgs()
                               << "  [DEBUG] Parameter " << idx << " will be converted from "
                               << opOperandType << " to tensor<" << opOperandType << ">\n";);
                continue;
            }
        }

        // No valid conversion found
        LLVM_DEBUG(llvm::dbgs() << "  [FAIL] Parameter " << idx << " type mismatch: "
                                << "function expects " << funcParamType << ", op provides "
                                << opOperandType << "\n";);
        return result; // isValid = false by default
    }

    if (llvm::any_of(llvm::zip(decompFuncType.getResults(), op.getResults()),
                     [](auto pair) { return std::get<0>(pair) != std::get<1>(pair).getType(); })) {
        LLVM_DEBUG(llvm::dbgs() << "  [FAIL] Result type mismatch\n";);
        return result; // isValid = false by default
    }

    result.isValid = true;
    return result;
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

        ValidationResult validation = validateDecompFuncTypeSignature(op, decompFunc);
        if (!validation.isValid) {
            return failure();
        }

        LLVM_DEBUG(
            llvm::dbgs() << "  [DEBUG] Generating function call to decomposition function...\n"
                         << "    Function name: " << decompFunc.getSymName() << "\n"
                         << "    Operands: " << op.getNumOperands() << "\n"
                         << "    Results: " << op.getNumResults() << "\n";);

        llvm::SmallVector<Value> processedOperands;
        processedOperands.reserve(op.getNumOperands());

        for (auto [idx, operand] : llvm::enumerate(op.getOperands())) {
            if (validation.needsConversion[idx]) {
                LLVM_DEBUG(llvm::dbgs() << "  [DEBUG] Converting operand " << idx << " from "
                                        << operand.getType() << " to tensor<" << operand.getType()
                                        << ">\n";);

                Type scalarType = operand.getType();
                Type tensorType = RankedTensorType::get({}, scalarType);

                auto convertedTensor = rewriter.create<tensor::FromElementsOp>(
                    op.getLoc(), tensorType, ValueRange{operand});

                processedOperands.push_back(convertedTensor.getResult());
            }
            else {
                processedOperands.push_back(operand);
            }
        }

        auto callOp =
            rewriter.create<func::CallOp>(op.getLoc(), decompFunc.getFunctionType().getResults(),
                                          decompFunc.getSymName(), processedOperands);

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
