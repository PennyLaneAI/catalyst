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

#define DEBUG_TYPE "scatter"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"

#include "Catalyst/IR/CatalystOps.h"
#include "mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/Support/Debug.h"

using namespace mlir;

namespace catalyst {

struct HloCustomCallOpRewritePattern : public mlir::OpRewritePattern<mhlo::CustomCallOp> {
    using mlir::OpRewritePattern<mhlo::CustomCallOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(mhlo::CustomCallOp op,
                                        mlir::PatternRewriter &rewriter) const override
    {
        StringRef calleeName = op.getCallTargetName();
        auto operands = op.getOperands();
        TypeRange resultsType = op.getResultTypes();
        Location loc = op.getLoc();

        SmallVector<Value> newOperands;

        // Check if this is a FFI-style LAPACK function
        bool isFFILapackFunction = calleeName.contains("lapack_") && calleeName.contains("_ffi");
        if (!isFFILapackFunction) {
            op.emitError("Unsupported custom call: ") << calleeName;
            return failure();
        }

        auto makeConst = [&](int64_t val) -> Value {
            auto type = RankedTensorType::get({}, rewriter.getI32Type());
            auto attr = DenseElementsAttr::get(type, APInt(32, static_cast<uint64_t>(val)));
            return rewriter.create<arith::ConstantOp>(loc, attr);
        };

        if (operands.empty()) {
            LLVM_DEBUG(llvm::dbgs()
                       << "DEBUG: No operands for FFI LAPACK function " << calleeName << "\n");
            return failure();
        }
        LLVM_DEBUG(llvm::dbgs() << "DEBUG: Handling FFI LAPACK function " << calleeName << "\n");

        // Lower backend_config dictionary attributes to constants
        if (auto configAttr = llvm::dyn_cast<DictionaryAttr>(op->getAttr("backend_config"))) {
            LLVM_DEBUG(llvm::dbgs() << "DEBUG: Processing backend_config attributes\n");

            // Process and add all attributes from configAttr as operands
            for (auto attr : configAttr) {
                Attribute attrValue = attr.getValue();
                Value constVal;
                LLVM_DEBUG(llvm::dbgs() << "Adding attribute: " << attr.getName().strref() << "\n");

                if (auto intAttr = dyn_cast<IntegerAttr>(attrValue)) {
                    auto type = RankedTensorType::get({}, intAttr.getType());
                    constVal = rewriter.create<arith::ConstantOp>(
                        loc, DenseElementsAttr::get(type, intAttr.getValue()));
                }
                else if (auto floatAttr = llvm::dyn_cast<FloatAttr>(attrValue)) {
                    auto type = RankedTensorType::get({}, floatAttr.getType());
                    constVal = rewriter.create<arith::ConstantOp>(
                        loc, DenseElementsAttr::get(type, floatAttr.getValue()));
                }
                else if (auto boolAttr = llvm::dyn_cast<BoolAttr>(attrValue)) {
                    auto type = RankedTensorType::get({}, rewriter.getI1Type());
                    constVal = rewriter.create<arith::ConstantOp>(
                        loc, DenseElementsAttr::get(type, boolAttr.getValue()));
                }
                else {
                    LLVM_DEBUG(llvm::dbgs() << "Unsupported attribute type for: "
                                            << attr.getName().strref() << "\n");
                    return failure();
                }
                newOperands.push_back(constVal);
            }
        }

        // Extract sizes of operands
        for (auto operand : operands) {
            if (auto tensorType = dyn_cast<RankedTensorType>(operand.getType())) {
                auto shape = tensorType.getShape();
                auto rank = shape.size();

                // Add batch count and matrix dimensions as new operands
                if (llvm::any_of(shape, [](int64_t d) { return d == ShapedType::kDynamic; })) {
                    return failure(); // Bail out on dynamic shapes
                }

                if (rank == 1) {
                    int64_t size = shape[0];
                    newOperands.push_back(makeConst(size));
                    LLVM_DEBUG(llvm::dbgs() << "DEBUG: Appended vector size=" << size << "\n");
                }
                else if (rank >= 2) {
                    int64_t batch = 1;
                    for (size_t i = 0; i < rank - 2; ++i) {
                        batch *= shape[i];
                    }

                    int64_t rows = shape[rank - 2];
                    int64_t cols = shape[rank - 1];

                    newOperands.push_back(makeConst(batch));
                    newOperands.push_back(makeConst(rows));
                    newOperands.push_back(makeConst(cols));

                    LLVM_DEBUG(llvm::dbgs() << "DEBUG: Appended batch=" << batch
                                            << ", rows=" << rows << ", cols=" << cols << "\n");
                }
            }
        }

        newOperands.append(operands.begin(), operands.end());
        auto callTargetAttr = rewriter.getStringAttr(calleeName);
        rewriter.replaceOpWithNewOp<CustomCallOp>(op, resultsType, newOperands, callTargetAttr,
                                                  nullptr);

        return success();
    }
};

void populateHloCustomCallPatterns(RewritePatternSet &patterns)
{
    patterns.add<HloCustomCallOpRewritePattern>(patterns.getContext());
}

} // namespace catalyst
