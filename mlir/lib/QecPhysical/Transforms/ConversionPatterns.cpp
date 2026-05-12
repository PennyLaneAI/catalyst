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

#include <cstdint>

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

#include "QecPhysical/IR/QecPhysicalOps.h"
#include "QecPhysical/Transforms/Patterns.h"

using namespace mlir;

namespace {

using namespace catalyst::qecp;

struct AssembleTannerGraphOpPattern : public OpConversionPattern<AssembleTannerGraphOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(AssembleTannerGraphOp op, AssembleTannerGraphOpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        Location loc = op.getLoc();
        const TypeConverter *conv = getTypeConverter();

        // Both row_idx and col_ptr should be bufferized first.
        if (!op.isBufferized()) {
            return op.emitOpError("op must be bufferized before lowering to LLVM");
        }

        auto mlirTannerType = op.getTannerGraph().getType();
        auto tannerStructType = conv->convertType(mlirTannerType);

        Value tannerStructValue = LLVM::UndefOp::create(rewriter, loc, tannerStructType);

        tannerStructValue = LLVM::InsertValueOp::create(rewriter, loc, tannerStructValue,
                                                        adaptor.getRowIdx(), ArrayRef<int64_t>{0});
        tannerStructValue = LLVM::InsertValueOp::create(rewriter, loc, tannerStructValue,
                                                        adaptor.getColPtr(), ArrayRef<int64_t>{1});

        rewriter.replaceOp(op, tannerStructValue);

        return success();
    }
};

} // namespace

namespace catalyst {
namespace qecp {

void populateQecPhysicalConversionPatterns(LLVMTypeConverter &typeConverter,
                                           RewritePatternSet &patterns)
{
    patterns.add<AssembleTannerGraphOpPattern>(typeConverter, patterns.getContext());
}

} // namespace qecp
} // namespace catalyst
