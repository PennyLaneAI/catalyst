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

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Catalyst/IR/CatalystDialect.h"
#include "Catalyst/IR/CatalystOps.h"

using namespace mlir;
using namespace catalyst;

namespace {

struct BufferizePrintOp : public OpConversionPattern<PrintOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(PrintOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        rewriter.replaceOpWithNewOp<PrintOp>(op, adaptor.getVal(), adaptor.getConstValAttr());

        return success();
    }
};

} // namespace

namespace catalyst {

void populateBufferizationPatterns(TypeConverter &typeConverter, RewritePatternSet &patterns)
{
    patterns.add<BufferizePrintOp>(typeConverter, patterns.getContext());
}

} // namespace catalyst
