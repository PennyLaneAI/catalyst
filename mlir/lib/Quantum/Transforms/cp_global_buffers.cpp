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

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/FormatVariadic.h"

#include "Quantum/Transforms/Passes.h"
#include "Quantum/Transforms/Patterns.h"

using namespace mlir;
using namespace catalyst::quantum;

namespace {

bool hasCWrapperAttribute(func::FuncOp op)
{
    return (bool)(op->getAttrOfType<UnitAttr>(LLVM::LLVMDialect::getEmitCWrapperAttrName()));
}

bool hasCopyWrapperAttribute(func::FuncOp op)
{
    return (bool)(op->getAttrOfType<UnitAttr>("llvm.copy_memref"));
}

bool hasCWrapperButNoCopyWrapperAttribute(func::FuncOp op)
{
    return hasCWrapperAttribute(op) && !hasCopyWrapperAttribute(op);
}

bool hasMemRefReturnTypes(func::FuncOp op)
{
    auto types = op.getResultTypes();
    if (types.empty())
        return false;

    bool isMemRefType = false;
    for (auto type : types) {
        isMemRefType |= type.isa<MemRefType>();
    }

    return isMemRefType;
}

void setCopyWrapperAttribute(func::FuncOp op, PatternRewriter &rewriter)
{
    return op->setAttr("llvm.copy_memref", rewriter.getUnitAttr());
}

std::vector<func::ReturnOp> getReturnOps(func::FuncOp op)
{
    std::vector<func::ReturnOp> returnOps;
    op.walk([&](func::ReturnOp returnOp) { returnOps.push_back(returnOp); });
    return returnOps;
}

std::vector<Value> getReturnMemRefs(func::ReturnOp op)
{
    auto values = op.getOperands();
    std::vector<Value> memrefs;
    for (auto value : values) {
        Type ty = value.getType();
        if (!ty.isa<MemRefType>())
            continue;

        memrefs.push_back(value);
    }
    return memrefs;
}

void applyCopyGlobalMemRefToReturnOp(func::ReturnOp op, PatternRewriter &rewriter)
{
    std::vector<Value> memrefs = getReturnMemRefs(op);
    if (memrefs.empty())
        return;

    std::vector<Value> newMemRefs;

    LLVMTypeConverter typeConverter(rewriter.getContext());
    Type mlirIndex = rewriter.getIndexType();
    Type llvmIndex = typeConverter.convertType(mlirIndex);
    auto deadbeefAttr = rewriter.getIntegerAttr(mlirIndex, 0xdeadbeef);
    Value deadbeef = rewriter.create<LLVM::ConstantOp>(op->getLoc(), llvmIndex, deadbeefAttr);

    for (Value memref : memrefs) {
        Type ty = memref.getType();
        Type llvmTy = typeConverter.convertType(ty);
        Value llvmMemRef =
            rewriter.create<UnrealizedConversionCastOp>(op->getLoc(), llvmTy, memref).getResult(0);

        Value allocatedPtr = rewriter.create<LLVM::ExtractValueOp>(op->getLoc(), llvmMemRef, 0);
        Value allocatedPtrToInt =
            rewriter.create<LLVM::PtrToIntOp>(op->getLoc(), llvmIndex, allocatedPtr);
        Value comparison = rewriter.create<LLVM::ICmpOp>(op->getLoc(), LLVM::ICmpPredicate::eq,
                                                         deadbeef, allocatedPtrToInt);

        scf::IfOp ifOp = rewriter.create<scf::IfOp>(
            op->getLoc(), ty, comparison,
            [&](OpBuilder &builder, Location loc) { // then
                Value newMemRef =
                    rewriter.create<memref::AllocOp>(op->getLoc(), ty.cast<MemRefType>());
                rewriter.create<memref::CopyOp>(op->getLoc(), memref, newMemRef);
                builder.create<scf::YieldOp>(loc, newMemRef);
            },
            [&](OpBuilder &builder, Location loc) { // else
                builder.create<scf::YieldOp>(loc, memref);
            });

        newMemRefs.push_back(ifOp.getResult(0));
    }

    rewriter.replaceOpWithNewOp<func::ReturnOp>(op, newMemRefs);
}

void applyCopyGlobalMemRefTransform(func::FuncOp op, PatternRewriter &rewriter)
{
    std::vector<func::ReturnOp> returnOps = getReturnOps(op);
    for (func::ReturnOp returnOp : returnOps) {
        // The insertion point will be just right before
        // the return op.
        rewriter.setInsertionPoint(returnOp);
        applyCopyGlobalMemRefToReturnOp(returnOp, rewriter);
    }
}

struct CopyGlobalMemRefTransform : public OpRewritePattern<func::FuncOp> {
    using OpRewritePattern<func::FuncOp>::OpRewritePattern;

    LogicalResult match(func::FuncOp op) const override;
    void rewrite(func::FuncOp op, PatternRewriter &rewriter) const override;
};

LogicalResult CopyGlobalMemRefTransform::match(func::FuncOp op) const
{
    bool isCandidate = hasCWrapperButNoCopyWrapperAttribute(op);
    if (!isCandidate)
        return failure();

    return hasMemRefReturnTypes(op) ? success() : failure();
}

void CopyGlobalMemRefTransform::rewrite(func::FuncOp op, PatternRewriter &rewriter) const
{
    setCopyWrapperAttribute(op, rewriter);
    applyCopyGlobalMemRefTransform(op, rewriter);
}

} // namespace

namespace catalyst {

struct CopyGlobalMemRefPass : public PassWrapper<CopyGlobalMemRefPass, OperationPass<ModuleOp>> {
    CopyGlobalMemRefPass() {}

    StringRef getArgument() const override { return "cp-global-memref"; }

    StringRef getDescription() const override
    {
        return "Copy global memrefs before returning from C interface.";
    }

    void getDependentDialects(DialectRegistry &registry) const override
    {
        registry.insert<memref::MemRefDialect>();
        registry.insert<func::FuncDialect>();
        registry.insert<scf::SCFDialect>();
    }

    void runOnOperation() final
    {
        MLIRContext *context = &getContext();
        RewritePatternSet patterns(context);
        patterns.add<CopyGlobalMemRefTransform>(context);

        if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
            signalPassFailure();
        }
    }
};

std::unique_ptr<Pass> createCopyGlobalMemRefPass()
{
    return std::make_unique<CopyGlobalMemRefPass>();
}

} // namespace catalyst
