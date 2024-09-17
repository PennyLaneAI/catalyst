// Copyright 2024 Xanadu Quantum Technologies Inc.

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
#include "mlir/IR/PatternMatch.h"

#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "Catalyst/Transforms/TBAAUtils.h"

using namespace mlir;

namespace catalyst {
void setTag(mlir::Type baseType, catalyst::TBAATree *tree, mlir::MLIRContext *ctx,
            mlir::LLVM::AliasAnalysisOpInterface newOp)
{
    mlir::LLVM::TBAATagAttr tag;
    if (isa<IndexType>(baseType) || isa<IntegerType>(baseType)) {
        tag = tree->getTag("int");
        newOp.setTBAATags(ArrayAttr::get(ctx, tag));
    }
    else if (isa<FloatType>(baseType)) {
        if (baseType.isF32()) {
            tag = tree->getTag("float");
        }
        else if (baseType.isF64()) {
            tag = tree->getTag("double");
        }

        newOp.setTBAATags(ArrayAttr::get(ctx, tag));
    }
    else if (isa<MemRefType>(baseType)) {
        tag = tree->getTag("any pointer");

        newOp.setTBAATags(ArrayAttr::get(ctx, tag));
    }
}

template <typename... Types> bool isAnyOf(Type baseType) { return (isa<Types>(baseType) || ...); }

struct MemrefLoadTBAARewritePattern : public ConvertOpToLLVMPattern<memref::LoadOp> {
    using ConvertOpToLLVMPattern<memref::LoadOp>::ConvertOpToLLVMPattern;

    template <typename... Args>
    MemrefLoadTBAARewritePattern(catalyst::TBAATree &tree, Args &&...args)
        : ConvertOpToLLVMPattern(std::forward<Args>(args)...), tree(&tree){};

    LogicalResult matchAndRewrite(memref::LoadOp loadOp, memref::LoadOpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        auto type = loadOp.getMemRefType();
        auto baseType = type.getElementType();
        Value dataPtr = getStridedElementPtr(loadOp.getLoc(), type, adaptor.getMemref(),
                                             adaptor.getIndices(), rewriter);
        auto op = rewriter.replaceOpWithNewOp<LLVM::LoadOp>(
            loadOp, typeConverter->convertType(type.getElementType()), dataPtr, 0, false,
            loadOp.getNontemporal());

        if (isAnyOf<IndexType, IntegerType, FloatType, MemRefType>(baseType)) {
            setTag(baseType, tree, loadOp.getContext(), op);
        }
        else {
            return failure();
        }
        return success();
    }

  private:
    catalyst::TBAATree *tree = nullptr;
};

struct MemrefStoreTBAARewritePattern : public ConvertOpToLLVMPattern<memref::StoreOp> {
    using ConvertOpToLLVMPattern<memref::StoreOp>::ConvertOpToLLVMPattern;

    template <typename... Args>
    MemrefStoreTBAARewritePattern(catalyst::TBAATree &tree, Args &&...args)
        : ConvertOpToLLVMPattern(std::forward<Args>(args)...), tree(&tree){};

    LogicalResult matchAndRewrite(memref::StoreOp storeOp, memref::StoreOpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        auto type = storeOp.getMemRefType();
        auto baseType = type.getElementType();

        Value dataPtr = getStridedElementPtr(storeOp.getLoc(), type, adaptor.getMemref(),
                                             adaptor.getIndices(), rewriter);
        auto op = rewriter.replaceOpWithNewOp<LLVM::StoreOp>(storeOp, adaptor.getValue(), dataPtr,
                                                             0, false, storeOp.getNontemporal());

        if (isAnyOf<IndexType, IntegerType, FloatType, MemRefType>(baseType)) {
            setTag(baseType, tree, storeOp.getContext(), op);
        }
        else {
            return failure();
        }
        return success();
    }

  private:
    catalyst::TBAATree *tree = nullptr;
};

void populateTBAATagsPatterns(TBAATree &tree, LLVMTypeConverter &typeConverter,
                              RewritePatternSet &patterns)
{
    patterns.add<catalyst::MemrefLoadTBAARewritePattern>(tree, typeConverter);
    patterns.add<catalyst::MemrefStoreTBAARewritePattern>(tree, typeConverter);
}

} // namespace catalyst
