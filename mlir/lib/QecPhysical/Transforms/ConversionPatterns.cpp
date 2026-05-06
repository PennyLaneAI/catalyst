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

        tannerStructValue = LLVM::InsertValueOp::create(
            rewriter, loc, tannerStructValue, adaptor.getRowIdx(), SmallVector<int64_t>{0});
        tannerStructValue = LLVM::InsertValueOp::create(
            rewriter, loc, tannerStructValue, adaptor.getColPtr(), SmallVector<int64_t>{1});

        rewriter.replaceOp(op, tannerStructValue);

        return success();
    }
};

struct DecodeEsmCssOpPattern : public OpConversionPattern<DecodeEsmCssOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(DecodeEsmCssOp op, DecodeEsmCssOpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        Location loc = op.getLoc();
        MLIRContext *ctx = getContext();
        auto i1 = IntegerType::get(ctx, 1);
        auto i64 = IntegerType::get(ctx, 64);
        auto voidTy = LLVM::LLVMVoidType::get(ctx);
        auto ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext());

        const TypeConverter *conv = getTypeConverter();

        // Both row_idx and col_ptr should be bufferized first.
        if (!op.isBufferized())
            return op.emitOpError("op must be bufferized before lowering to LLVM");

        Type esmVectorType, errIdxVectorType;
        esmVectorType = conv->convertType(MemRefType::get(
            {llvm::dyn_cast<mlir::ShapedType>(op.getEsm().getType()).getDimSize(0)}, i1));

        errIdxVectorType = conv->convertType(MemRefType::get(
            {llvm::dyn_cast<mlir::ShapedType>(op.getErrIdxIn().getType()).getDimSize(0)}, i64));

        // Define Tanner Graph struct type
        auto tannerGraphStruct = LLVM::LLVMStructType::getLiteral(ctx, {ptrTy, ptrTy});

        // Convert qecp.tanner_graph type to tannerGraphStruct
        auto convertedTannerGraphStruct = UnrealizedConversionCastOp::create(
                                              rewriter, loc, tannerGraphStruct, op.getTannerGraph())
                                              .getResult(0);

        Value esmAlloca = catalyst::getStaticAlloca(loc, rewriter, esmVectorType, 1);
        Value errIdxAlloca = catalyst::getStaticAlloca(loc, rewriter, errIdxVectorType, 1);
        Value tannerGraphAlloca = catalyst::getStaticAlloca(loc, rewriter, tannerGraphStruct, 1);

        LLVM::StoreOp::create(rewriter, loc, adaptor.getEsm(), esmAlloca);
        LLVM::StoreOp::create(rewriter, loc, adaptor.getErrIdxIn(), errIdxAlloca);
        LLVM::StoreOp::create(rewriter, loc, convertedTannerGraphStruct, tannerGraphAlloca);

        // Define function signature
        StringRef fnName = "__catalyst__qecp__lut_decoder";

        Type fnSignature = LLVM::LLVMFunctionType::get(voidTy, {ptrTy, ptrTy, ptrTy}, false);
        LLVM::LLVMFuncOp fnDecl = catalyst::ensureFunctionDeclaration<LLVM::LLVMFuncOp>(
            rewriter, op, fnName, fnSignature);

        SmallVector<Value> args = {tannerGraphAlloca, esmAlloca, errIdxAlloca};

        LLVM::CallOp::create(rewriter, loc, fnDecl, args);

        rewriter.eraseOp(op);

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
    patterns.add<DecodeEsmCssOpPattern>(typeConverter, patterns.getContext());
}

} // namespace qecp
} // namespace catalyst
