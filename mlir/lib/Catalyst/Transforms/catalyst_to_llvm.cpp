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

enum NumericType : int8_t {
    index = 0,
    i1,
    i8,
    i16,
    i32,
    i64,
    f32,
    f64,
    c64,
    c128,
};

std::optional<int8_t> encodeNumericType(Type elemType)
{
    int8_t typeEncoding;
    if (isa<IndexType>(elemType)) {
        typeEncoding = NumericType::index;
    }
    else if (auto intType = dyn_cast<IntegerType>(elemType)) {
        switch (intType.getWidth()) {
        case 1:
            typeEncoding = NumericType::i1;
            break;
        case 8:
            typeEncoding = NumericType::i8;
            break;
        case 16:
            typeEncoding = NumericType::i16;
            break;
        case 32:
            typeEncoding = NumericType::i32;
            break;
        case 64:
            typeEncoding = NumericType::i64;
            break;
        default:
            return std::nullopt;
        }
    }
    else if (auto floatType = dyn_cast<FloatType>(elemType)) {
        switch (floatType.getWidth()) {
        case 32:
            typeEncoding = NumericType::f32;
            break;
        case 64:
            typeEncoding = NumericType::f64;
            break;
        default:
            return std::nullopt;
        }
    }
    else if (auto cmplxType = dyn_cast<ComplexType>(elemType)) {
        auto floatType = dyn_cast<FloatType>(cmplxType.getElementType());
        if (!floatType)
            return std::nullopt;

        switch (floatType.getWidth()) {
        case 32:
            typeEncoding = NumericType::c64;
            break;
        case 64:
            typeEncoding = NumericType::c128;
            break;
        default:
            return std::nullopt;
        }
    }
    else {
        return std::nullopt;
    }
    return typeEncoding;
}

struct PrintOpPattern : public OpConversionPattern<PrintOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(PrintOp op, PrintOpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {

        Location loc = op.getLoc();
        MLIRContext *ctx = this->getContext();

        Type voidType = LLVM::LLVMVoidType::get(ctx);
        Type voidPtrType = LLVM::LLVMPointerType::get(ctx);

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
            StringRef qirName = "__quantum__rt__print_tensor";

            // C interface for the print function is an unranked & opaque memref descriptor:
            // {
            //    i64 rank,
            //    void* memref_descriptor,
            //    i8 type_encoding
            // }
            // Where the type_encoding is a simple enum for all supported numeric types:
            //   i1, i16, i32, i64, f32, f64, c64, c128 (see runtime Types.h)
            Type structType = LLVM::LLVMStructType::getLiteral(
                ctx, {IntegerType::get(ctx, 64), voidPtrType, IntegerType::get(ctx, 8)});
            Type structPtrType = LLVM::LLVMPointerType::get(structType);
            Type qirSignature = LLVM::LLVMFunctionType::get(voidType, structPtrType);
            LLVM::LLVMFuncOp fnDecl =
                ensureFunctionDeclaration(rewriter, op, qirName, qirSignature);

            Value memref = op.getVal();
            MemRefType memrefType = cast<MemRefType>(memref.getType());
            Value llvmMemref = adaptor.getVal();
            Type llvmMemrefType = llvmMemref.getType();
            Value structValue = rewriter.create<LLVM::UndefOp>(loc, structType);

            Value rank = rewriter.create<LLVM::ConstantOp>(
                loc, rewriter.getI64IntegerAttr(memrefType.getRank()));
            structValue = rewriter.create<LLVM::InsertValueOp>(loc, structValue, rank, 0);

            Value c1 = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64IntegerAttr(1));
            Value memrefPtr = rewriter.create<LLVM::AllocaOp>(
                loc, LLVM::LLVMPointerType::get(llvmMemrefType), c1);
            rewriter.create<LLVM::StoreOp>(loc, llvmMemref, memrefPtr);
            memrefPtr = rewriter.create<LLVM::BitcastOp>(loc, voidPtrType, memrefPtr);
            structValue = rewriter.create<LLVM::InsertValueOp>(loc, structValue, memrefPtr, 1);

            Type elemType = memrefType.getElementType();
            std::optional<int8_t> typeEncoding = encodeNumericType(elemType);
            if (!typeEncoding.has_value()) {
                return op.emitOpError("Unsupported element type for printing!");
            }
            Value typeValue = rewriter.create<LLVM::ConstantOp>(
                loc, rewriter.getI8IntegerAttr(typeEncoding.value()));
            structValue = rewriter.create<LLVM::InsertValueOp>(loc, structValue, typeValue, 2);

            Value structPtr =
                rewriter.create<LLVM::AllocaOp>(loc, LLVM::LLVMPointerType::get(structType), c1);
            rewriter.create<LLVM::StoreOp>(loc, structValue, structPtr);
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
