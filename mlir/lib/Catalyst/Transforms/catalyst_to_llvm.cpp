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

#include <unordered_map>

#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Catalyst/IR/CatalystOps.h"
#include "Catalyst/Transforms/Passes.h"
#include "Catalyst/Transforms/Patterns.h"

using namespace mlir;
using namespace catalyst;

namespace {

LLVM::LLVMFuncOp ensureFunctionDeclaration(PatternRewriter &rewriter, Operation *op,
                                           StringRef fnSymbol, Type fnType)
{
    Operation *fnDecl = SymbolTable::lookupNearestSymbolFrom(op, rewriter.getStringAttr(fnSymbol));

    if (!fnDecl) {
        PatternRewriter::InsertionGuard insertGuard(rewriter);
        ModuleOp mod = op->getParentOfType<ModuleOp>();
        rewriter.setInsertionPointToStart(mod.getBody());

        fnDecl = rewriter.create<LLVM::LLVMFuncOp>(op->getLoc(), fnSymbol, fnType);
    }
    else {
        assert(isa<LLVM::LLVMFuncOp>(fnDecl) && "QIR function declaration is not a LLVMFuncOp");
    }

    return cast<LLVM::LLVMFuncOp>(fnDecl);
}

Value getGlobalString(Location loc, OpBuilder &rewriter, StringRef key, StringRef value,
                      ModuleOp mod)
{
    LLVM::GlobalOp glb = mod.lookupSymbol<LLVM::GlobalOp>(key);
    if (!glb) {
        OpBuilder::InsertionGuard guard(rewriter); // to reset the insertion point
        rewriter.setInsertionPointToStart(mod.getBody());
        glb = rewriter.create<LLVM::GlobalOp>(
            loc, LLVM::LLVMArrayType::get(IntegerType::get(rewriter.getContext(), 8), value.size()),
            true, LLVM::Linkage::Internal, key, rewriter.getStringAttr(value));
    }

    auto idx =
        rewriter.create<LLVM::ConstantOp>(loc, IntegerType::get(rewriter.getContext(), 64),
                                          rewriter.getIntegerAttr(rewriter.getIndexType(), 0));
    return rewriter.create<LLVM::GEPOp>(
        loc, LLVM::LLVMPointerType::get(IntegerType::get(rewriter.getContext(), 8)),
        rewriter.create<LLVM::AddressOfOp>(loc, glb), ArrayRef<Value>({idx, idx}));
}

struct PrintOpPattern : public OpConversionPattern<PrintOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(PrintOp op, PrintOpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {

        Location loc = op.getLoc();
        MLIRContext *ctx = this->getContext();

        Type voidType = LLVM::LLVMVoidType::get(ctx);

        if (op.getConstVal().has_value()) {
            ModuleOp mod = op->getParentOfType<ModuleOp>();

            StringRef qirName = "__quantum__rt__print_string";

            Type charPtrType = LLVM::LLVMPointerType::get(IntegerType::get(ctx, 8));
            Type qirSignature = LLVM::LLVMFunctionType::get(voidType, charPtrType);
            LLVM::LLVMFuncOp fnDecl =
                ensureFunctionDeclaration(rewriter, op, qirName, qirSignature);

            StringRef stringValue = op.getConstVal().value();
            std::string symbolName = std::to_string(std::hash<std::string>()(stringValue.str()));
            Value global = getGlobalString(loc, rewriter, symbolName, stringValue, mod);
            rewriter.create<LLVM::CallOp>(loc, fnDecl, global);
            rewriter.eraseOp(op);
        }
        else {
            constexpr int64_t UNKNOWN = ShapedType::kDynamic;
            TypeConverter *conv = getTypeConverter();
            StringRef qirName = "_catalyst_memref_print";
            Type vectorType =
                conv->convertType(MemRefType::get({UNKNOWN}, IntegerType::get(ctx, 64)));

            Type qirSignature =
                LLVM::LLVMFunctionType::get(voidType, LLVM::LLVMPointerType::get(vectorType));
            LLVM::LLVMFuncOp fnDecl =
                ensureFunctionDeclaration(rewriter, op, qirName, qirSignature);

            Value c1 = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64IntegerAttr(1));
            Value structPtr =
                rewriter.create<LLVM::AllocaOp>(loc, LLVM::LLVMPointerType::get(vectorType), c1);
            rewriter.create<LLVM::StoreOp>(loc, adaptor.getVal(), structPtr);
            rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, fnDecl, structPtr);
        }

        return success();
    }
};

} // namespace

namespace catalyst {

#define GEN_PASS_DEF_CATALYSTCONVERSIONPASS
#include "Catalyst/Transforms/Passes.h.inc"

struct CatalystConversionPass : impl::CatalystConversionPassBase<CatalystConversionPass> {
    using CatalystConversionPassBase::CatalystConversionPassBase;

    void runOnOperation() final
    {
        MLIRContext *context = &getContext();
        LLVMTypeConverter typeConverter(context);

        RewritePatternSet patterns(context);
        patterns.add<PrintOpPattern>(typeConverter, context);

        LLVMConversionTarget target(*context);
        target.addIllegalDialect<CatalystDialect>();

        if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
            signalPassFailure();
        }
    }
};

std::unique_ptr<Pass> createCatalystConversionPass()
{
    return std::make_unique<catalyst::CatalystConversionPass>();
}

} // namespace catalyst
