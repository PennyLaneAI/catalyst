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
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Catalyst/Utils/EnsureFunctionDeclaration.h"
#include "QecPhysical/IR/QecPhysicalOps.h"
#include "QecPhysical/Transforms/Patterns.h"

using namespace mlir;

namespace {

using namespace catalyst::qecp;

constexpr int32_t NO_POSTSELECT = -1;

struct DecodeEsmCssOpPattern : public OpConversionPattern<DecodeEsmCssOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(DecodeEsmCssOp op, DecodeEsmCssOpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        // Location loc = op.getLoc();
        // MLIRContext *ctx = getContext();
        // auto i64 = IntegerType::get(ctx, 64); // row_idx, col_ptr, err_idx
        // auto i8 = IntegerType::get(ctx, 8); // syndrome

        // // Add types to the function signature
        // SmallVector<Type> argSignatures = {i64, i64, i8, i64};

        // // Define function signature
        // StringRef fnName = "__catalyst__qecp__lut_decoder";
        // Type fnSignature = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx), argSignatures);

        // LLVM::LLVMFuncOp fnDecl = catalyst::ensureFunctionDeclaration<LLVM::LLVMFuncOp>(
        //     rewriter, op, fnName, fnSignature);

        // // Add values as arguments of the CallOp

        // SmallVector<Value> args = {adaptor.getTannerGraph(), adaptor.getTannerGraph(), adaptor.getEsm(), adaptor.getErrIdx()};

        return success();
    }
};

} // namespace

namespace catalyst {
namespace qecp {

void populateConversionPatterns(LLVMTypeConverter &typeConverter, RewritePatternSet &patterns)
{
    patterns.add<MeasureInBasisOpPattern>(typeConverter, patterns.getContext());
}

} // namespace qecp
} // namespace catalyst
