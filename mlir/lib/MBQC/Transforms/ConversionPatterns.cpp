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

#include <cstdint>

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Catalyst/Utils/EnsureFunctionDeclaration.h"
#include "MBQC/IR/MBQCOps.h"
#include "MBQC/Transforms/Patterns.h"

using namespace mlir;

namespace {

using namespace catalyst::mbqc;
using namespace catalyst::quantum;

constexpr int32_t NO_POSTSELECT = -1;

struct MeasureInBasisOpPattern : public OpConversionPattern<MeasureInBasisOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(MeasureInBasisOp op, MeasureInBasisOpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        Location loc = op.getLoc();
        MLIRContext *ctx = getContext();
        const TypeConverter *conv = getTypeConverter();

        // Add types to the function signature
        Type qubitTy = conv->convertType(QubitType::get(ctx));
        Type planeTy = IntegerType::get(ctx, 32);
        Type angleTy = Float64Type::get(ctx);
        Type postselectTy = IntegerType::get(ctx, 32);
        SmallVector<Type> argSignatures = {qubitTy, planeTy, angleTy, postselectTy};

        // Define function signature
        StringRef fnName = "__catalyst__mbqc__measure_in_basis";
        Type fnSignature =
            LLVM::LLVMFunctionType::get(conv->convertType(ResultType::get(ctx)), argSignatures);

        LLVM::LLVMFuncOp fnDecl =
            catalyst::ensureFunctionDeclaration(rewriter, op, fnName, fnSignature);

        // Extract the integer value for the plane attribute from its enum
        const auto planeValueInt = static_cast<uint32_t>(op.getPlane());
        Value planeValue =
            rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI32IntegerAttr(planeValueInt));

        // Create the postselect value. If not given, it defaults to NO_POSTSELECT
        LLVM::ConstantOp postselect = rewriter.create<LLVM::ConstantOp>(
            loc, op.getPostselect() ? op.getPostselectAttr()
                                    : rewriter.getI32IntegerAttr(NO_POSTSELECT));

        // Add values as arguments of the CallOp
        SmallVector<Value> args = {adaptor.getInQubit(), planeValue, op.getAngle(), postselect};

        Value resultPtr = rewriter.create<LLVM::CallOp>(loc, fnDecl, args).getResult();
        Value mres = rewriter.create<LLVM::LoadOp>(loc, IntegerType::get(ctx, 1), resultPtr);
        rewriter.replaceOp(op, {mres, adaptor.getInQubit()});

        return success();
    }
};

} // namespace

namespace catalyst {
namespace mbqc {

void populateConversionPatterns(LLVMTypeConverter &typeConverter, RewritePatternSet &patterns)
{
    patterns.add<MeasureInBasisOpPattern>(typeConverter, patterns.getContext());
}

} // namespace mbqc
} // namespace catalyst
