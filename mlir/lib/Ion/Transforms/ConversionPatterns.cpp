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

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "Ion/IR/IonOps.h"
#include "Ion/Transforms/Patterns.h"

using namespace mlir;
using namespace catalyst::ion;

namespace {

struct IonOpPattern : public OpConversionPattern<catalyst::ion::IonOp> {
    using OpConversionPattern<catalyst::ion::IonOp>::OpConversionPattern;
    
    LogicalResult matchAndRewrite(catalyst::ion::IonOp op, catalyst::ion::IonOpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        return success();
    }
};

struct ParallelProtocolOpPattern : public OpConversionPattern<catalyst::ion::ParallelProtocolOp> {
    using OpConversionPattern<catalyst::ion::ParallelProtocolOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(catalyst::ion::ParallelProtocolOp op,
                                  catalyst::ion::ParallelProtocolOpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        return success();
    }
};

struct PulseOpPattern : public OpConversionPattern<catalyst::ion::PulseOp> {
    using OpConversionPattern<catalyst::ion::PulseOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(catalyst::ion::PulseOp op, catalyst::ion::PulseOpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        return success();
    }
};

} // namespace

namespace catalyst {
namespace ion {

void populateConversionPatterns(LLVMTypeConverter &typeConverter, RewritePatternSet &patterns)
{
    patterns.add<IonOpPattern>(typeConverter, patterns.getContext());
    patterns.add<ParallelProtocolOpPattern>(typeConverter, patterns.getContext());
    patterns.add<PulseOpPattern>(typeConverter, patterns.getContext());
}

} // namespace ion
} // namespace catalyst
