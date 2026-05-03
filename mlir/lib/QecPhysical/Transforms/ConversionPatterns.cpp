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
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Catalyst/Utils/EnsureFunctionDeclaration.h"
#include "Catalyst/Utils/StaticAllocas.h"
#include "QecPhysical/IR/QecPhysicalOps.h"
#include "QecPhysical/Transforms/Patterns.h"

using namespace mlir;

namespace {

using namespace catalyst::qecp;

constexpr int32_t UNKNOWN = ShapedType::kDynamic;

struct AssembleTannerGraphOpPattern : public OpConversionPattern<AssembleTannerGraphOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(AssembleTannerGraphOp op, AssembleTannerGraphOpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        Location loc = op.getLoc();
        MLIRContext *ctx = getContext();
        auto i32 = IntegerType::get(ctx, 32);
        auto voidTy = LLVM::LLVMVoidType::get(ctx);
        auto ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
        const TypeConverter *conv = getTypeConverter();

        // Both row_idx and col_ptr should be bufferized first.
        if (!op.isBufferized())
            return op.emitOpError("op must be bufferized before lowering to LLVM");

        Type vectorType;
        // Define function signature
        StringRef fnName = "__catalyst__qecp__assemble_tanner_graph_int32";

        vectorType = conv->convertType(MemRefType::get({UNKNOWN}, i32));

        // Define Tanner Graph struct type
        auto tannerGraphType = LLVM::LLVMStructType::getLiteral(ctx, {ptrTy, ptrTy});

        Type fnSignature =
            LLVM::LLVMFunctionType::get(voidTy, {ptrTy, ptrTy, tannerGraphType}, false);
        LLVM::LLVMFuncOp fnDecl = catalyst::ensureFunctionDeclaration<LLVM::LLVMFuncOp>(
            rewriter, op, fnName, fnSignature);

        //  Add values as arguments of the CallOp
        Value rowIdxPtr = catalyst::getStaticAlloca(loc, rewriter, vectorType, 1);
        Value colPtrPtr = catalyst::getStaticAlloca(loc, rewriter, vectorType, 1);

        auto tannerGraphAllcoaOp = catalyst::getStaticAlloca(loc, rewriter, tannerGraphType, 1);

        LLVM::StoreOp::create(rewriter, loc, adaptor.getRowIdx(), rowIdxPtr);
        LLVM::StoreOp::create(rewriter, loc, adaptor.getColPtr(), colPtrPtr);
        LLVM::StoreOp::create(rewriter, loc, tannerGraphAllcoaOp.getResult(), tannerGraphAllcoaOp);

        SmallVector<Value> args = {rowIdxPtr, colPtrPtr, tannerGraphAllcoaOp.getResult()};

        LLVM::CallOp::create(rewriter, loc, fnDecl, args);

        rewriter.replaceOp(op, args[2]);

        return success();
    }
};

} // namespace

namespace catalyst {
namespace qecp {

void populateConversionPatterns(LLVMTypeConverter &typeConverter, RewritePatternSet &patterns)
{
    patterns.add<AssembleTannerGraphOpPattern>(typeConverter, patterns.getContext());
}

} // namespace qecp
} // namespace catalyst