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

struct MemrefLoadTBAARewritePattern : public ConvertOpToLLVMPattern<memref::LoadOp> {
    using ConvertOpToLLVMPattern<memref::LoadOp>::ConvertOpToLLVMPattern;

    template <typename... Args>
    MemrefLoadTBAARewritePattern(catalyst::TBAATree &tree, Args &&...args)
        : ConvertOpToLLVMPattern(std::forward<Args>(args)...), tree(&tree){};

    LogicalResult matchAndRewrite(memref::LoadOp loadOp, memref::LoadOpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        auto type = loadOp.getMemRefType();
        Value dataPtr = getStridedElementPtr(loadOp.getLoc(), type, adaptor.getMemref(),
                                             adaptor.getIndices(), rewriter);
        auto op = rewriter.replaceOpWithNewOp<LLVM::LoadOp>(
            loadOp, typeConverter->convertType(type.getElementType()), dataPtr, 0, false,
            loadOp.getNontemporal());
        auto baseType = loadOp.getMemRefType().getElementType();

        mlir::LLVM::TBAATagAttr tag;

        if (isa<IndexType>(baseType) || dyn_cast<IntegerType>(baseType)) {
            tag = tree->getTag("int");
            op.setTBAATags(ArrayAttr::get(loadOp.getContext(), tag));
        }
        else if (dyn_cast<FloatType>(baseType)) {
            if (baseType.isF32()) {
                tag = tree->getTag("float");
            }
            else if (baseType.isF64()) {
                tag = tree->getTag("double");
            }

            op.setTBAATags(ArrayAttr::get(loadOp.getContext(), tag));
        }
        else if (dyn_cast<MemRefType>(baseType)) {
            tag = tree->getTag("any pointer");

            op.setTBAATags(ArrayAttr::get(loadOp.getContext(), tag));
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

        mlir::LLVM::TBAATagAttr tag;

        if (isa<IndexType>(baseType) || dyn_cast<IntegerType>(baseType)) {
            tag = tree->getTag("int");
            op.setTBAATags(ArrayAttr::get(storeOp.getContext(), tag));
        }
        else if (dyn_cast<FloatType>(baseType)) {
            if (baseType.isF32()) {
                tag = tree->getTag("float");
            }
            else if (baseType.isF64()) {
                tag = tree->getTag("double");
            }

            op.setTBAATags(ArrayAttr::get(storeOp.getContext(), tag));
        }
        else if (dyn_cast<MemRefType>(baseType)) {
            tag = tree->getTag("any pointer");

            op.setTBAATags(ArrayAttr::get(storeOp.getContext(), tag));
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
