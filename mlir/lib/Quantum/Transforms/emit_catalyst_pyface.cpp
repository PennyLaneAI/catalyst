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
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/FormatVariadic.h"

#include "Quantum/Transforms/Passes.h"
#include "Quantum/Transforms/Patterns.h"

using namespace mlir;
using namespace catalyst::quantum;

namespace {

std::optional<LLVM::LLVMFuncOp> getCallee(LLVM::LLVMFuncOp op)
{
    size_t counter = 0;
    std::optional<LLVM::LLVMFuncOp> callee = std::nullopt;
    op.walk([&](LLVM::CallOp callOp) {
        auto calleeAttr = callOp.getCalleeAttr();
        // calleeAttr is optional in case of function pointers.
        if (!calleeAttr)
            return;

        counter++;
        callee = SymbolTable::lookupNearestSymbolFrom<LLVM::LLVMFuncOp>(op, calleeAttr);
    });
    return counter == 1 ? callee : std::nullopt;
}

bool hasCWrapperAttribute(LLVM::LLVMFuncOp op)
{
    return (bool)(op->getAttrOfType<UnitAttr>(LLVM::LLVMDialect::getEmitCWrapperAttrName()));
}

bool hasCalleeCWrapperAttribute(LLVM::LLVMFuncOp op)
{
    std::optional<LLVM::LLVMFuncOp> callee = getCallee(op);
    if (!callee)
        return false;
    return hasCWrapperAttribute(callee.value());
}

bool matchesNamingConvention(LLVM::LLVMFuncOp op)
{
    std::string _mlir_ciface = "_mlir_ciface_";
    size_t _mlir_ciface_len = _mlir_ciface.length();
    auto symName = op.getSymName();
    const char *symNameStr = symName.data();
    size_t symNameLength = strlen(symNameStr);
    // Filter based on name.
    if (symNameLength <= _mlir_ciface_len)
        return false;

    bool nameMatches = 0 == strncmp(symNameStr, _mlir_ciface.c_str(), _mlir_ciface_len);
    return nameMatches;
}

bool isFunctionMLIRCWrapper(LLVM::LLVMFuncOp op)
{
    if (!matchesNamingConvention(op))
        return false;
    if (!hasCalleeCWrapperAttribute(op))
        return false;
    return true;
}

bool functionHasReturns(LLVM::LLVMFuncOp op)
{
    auto functionType = op.getFunctionType();
    return !functionType.getReturnType().isa<LLVM::LLVMVoidType>();
}

bool functionHasInputs(LLVM::LLVMFuncOp op)
{
    auto functionType = op.getFunctionType();
    return !(functionType.getParams().empty());
}

LLVM::LLVMFunctionType convertFunctionTypeCatalystWrapper(PatternRewriter &rewriter,
                                                          LLVM::LLVMFunctionType functionType,
                                                          bool hasReturns, bool hasInputs)
{
    SmallVector<Type, 2> transformedInputs;

    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto inputs = functionType.getParams();
    Type resultType = hasReturns ? inputs.front() : ptrType;
    transformedInputs.push_back(resultType);

    if (hasReturns) {
        inputs = inputs.drop_front();
    }

    LLVMTypeConverter typeConverter(rewriter.getContext());
    Type inputType = hasInputs ? typeConverter.packFunctionResults(inputs) : ptrType;
    bool noChange = inputs.size() == 1;
    if (noChange) {
        // Still wrap the pointer into a struct
        // for uniformity in Python and in the unwrapping.
        inputType = LLVM::LLVMStructType::getLiteral(rewriter.getContext(), inputType);
    }
    if (inputType.isa<LLVM::LLVMStructType>()) {
        inputType = LLVM::LLVMPointerType::get(inputType);
    }
    transformedInputs.push_back(inputType);

    Type voidType = LLVM::LLVMVoidType::get(rewriter.getContext());
    return LLVM::LLVMFunctionType::get(voidType, transformedInputs);
}

void wrapResultsAndArgsInTwoStructs(LLVM::LLVMFuncOp op, PatternRewriter &rewriter,
                                    std::string nameWithoutPrefix)
{
    // Guaranteed by match
    LLVM::LLVMFuncOp callee = getCallee(op).value();
    bool hasReturns = functionHasReturns(callee);
    bool hasInputs = functionHasInputs(callee);

    LLVM::LLVMFunctionType functionType = op.getFunctionType();
    LLVM::LLVMFunctionType wrapperFuncType =
        convertFunctionTypeCatalystWrapper(rewriter, functionType, hasReturns, hasInputs);

    Location loc = op.getLoc();
    auto wrapperFuncOp = rewriter.create<LLVM::LLVMFuncOp>(
        loc, llvm::formatv("_catalyst_pyface_{0}", nameWithoutPrefix).str(), wrapperFuncType,
        LLVM::Linkage::External, /*dsoLocal*/ false,
        /*cconv*/ LLVM::CConv::C);

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(wrapperFuncOp.addEntryBlock());

    auto type = op.getFunctionType();
    auto params = type.getParams();
    SmallVector<Value, 8> args;

    if (hasReturns) {
        args.push_back(wrapperFuncOp.getArgument(0));
        params = params.drop_front();
    }

    if (hasInputs) {
        Value arg = wrapperFuncOp.getArgument(1);
        Value structOfMemrefs = rewriter.create<LLVM::LoadOp>(loc, arg);

        for (size_t idx = 0; idx < params.size(); idx++) {
            Value pointer = rewriter.create<LLVM::ExtractValueOp>(loc, structOfMemrefs, idx);
            args.push_back(pointer);
        }
    }

    auto call = rewriter.create<LLVM::CallOp>(loc, op, args);

    rewriter.create<LLVM::ReturnOp>(loc, call.getResults());
}

struct EmitCatalystPyInterfaceTransform : public OpRewritePattern<LLVM::LLVMFuncOp> {
    using OpRewritePattern<LLVM::LLVMFuncOp>::OpRewritePattern;

    LogicalResult match(LLVM::LLVMFuncOp op) const override;
    void rewrite(LLVM::LLVMFuncOp op, PatternRewriter &rewriter) const override;
};

LogicalResult EmitCatalystPyInterfaceTransform::match(LLVM::LLVMFuncOp op) const
{
    return isFunctionMLIRCWrapper(op) ? success() : failure();
}

void EmitCatalystPyInterfaceTransform::rewrite(LLVM::LLVMFuncOp op, PatternRewriter &rewriter) const
{
    // Find substr after _mlir_ciface_
    std::string _mlir_ciface = "_mlir_ciface_";
    size_t _mlir_ciface_len = _mlir_ciface.length();
    auto symName = op.getSymName();
    const char *symNameStr = symName.data();
    const char *functionNameWithoutPrefix = symNameStr + _mlir_ciface_len;
    auto newName = llvm::formatv("_catalyst_ciface_{0}", functionNameWithoutPrefix).str();

    rewriter.updateRootInPlace(op, [&] { op.setSymName(newName); });
    wrapResultsAndArgsInTwoStructs(op, rewriter, functionNameWithoutPrefix);
}

} // namespace

namespace catalyst {

#define GEN_PASS_DEF_EMITCATALYSTPYINTERFACEPASS
#include "Quantum/Transforms/Passes.h.inc"

struct EmitCatalystPyInterfacePass
    : impl::EmitCatalystPyInterfacePassBase<EmitCatalystPyInterfacePass> {
    using EmitCatalystPyInterfacePassBase::EmitCatalystPyInterfacePassBase;

    void runOnOperation() final
    {
        MLIRContext *context = &getContext();
        RewritePatternSet patterns(context);
        patterns.add<EmitCatalystPyInterfaceTransform>(context);

        if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
            signalPassFailure();
        }
    }
};

std::unique_ptr<Pass> createEmitCatalystPyInterfacePass()
{
    return std::make_unique<EmitCatalystPyInterfacePass>();
}

} // namespace catalyst
