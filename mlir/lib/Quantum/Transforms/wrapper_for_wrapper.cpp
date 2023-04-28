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

namespace catalyst {
namespace quantum {

static size_t countCallsites(LLVM::LLVMFuncOp op)
{
    size_t count = 0;
    op.walk([&](LLVM::CallOp callOp) { count++; });
    return count;
}

static bool callsiteHasAttribute(LLVM::LLVMFuncOp op)
{
    bool retval = false;
    ModuleOp moduleOp = cast<ModuleOp>(op->getParentOp());
    if (!moduleOp)
        return retval;
    op.walk([&](LLVM::CallOp callOp) {
        auto callee = callOp.getCallee();
        if (callee) {
            auto stringRef = callee.value().data();
            Operation *fnDecl = moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(stringRef);
            retval = (bool)(fnDecl->getAttrOfType<UnitAttr>(
                LLVM::LLVMDialect::getEmitCWrapperAttrName()));
        }
    });
    return retval;
}

struct WrapperForWrapperTransform : public OpRewritePattern<LLVM::LLVMFuncOp> {
    using OpRewritePattern<LLVM::LLVMFuncOp>::OpRewritePattern;

    LogicalResult match(LLVM::LLVMFuncOp op) const override;
    void rewrite(LLVM::LLVMFuncOp op, PatternRewriter &rewriter) const override;
};

LogicalResult WrapperForWrapperTransform::match(LLVM::LLVMFuncOp op) const
{
    std::string _mlir_ciface = "_mlir_ciface_";
    size_t _mlir_ciface_len = _mlir_ciface.length();
    auto symName = op.getSymName();
    const char *symNameStr = symName.data();
    size_t symNameLength = strlen(symNameStr);
    // Filter based on name.
    if (symNameLength <= _mlir_ciface_len)
        return failure();

    bool nameMatches = 0 == strncmp(symNameStr, _mlir_ciface.c_str(), _mlir_ciface_len);
    if (!nameMatches)
        return failure();

    // Function must only contain a single function call.
    size_t callsites = countCallsites(op);
    if (callsites != 1)
        return failure();

    // The only call must contain the emitCWrapper attribute
    bool hasAttribute = callsiteHasAttribute(op);

    return hasAttribute ? success() : failure();
}

static bool functionHasReturns(LLVM::LLVMFuncOp op)
{
    bool result = false;
    op.walk([&](LLVM::CallOp callOp) { result = (bool)callOp.getResult(); });
    return result;
}

static bool functionHasInputs(LLVM::LLVMFuncOp op)
{
    bool result = false;
    op.walk([&](LLVM::CallOp callOp) { result = !callOp.getOperandTypes().empty(); });
    return result;
}

static LLVM::LLVMFunctionType
convertFunctionTypeCatalystWrapper(PatternRewriter &rewriter, LLVM::LLVMFunctionType functionType,
                                   bool hasReturns, bool hasInputs)
{
    SmallVector<Type, 2> transformedInputs;

    Type ptr = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto inputs = functionType.getParams();
    Type resultType = hasReturns ? inputs.front() : ptr;
    transformedInputs.push_back(resultType);

    if (hasReturns) {
        inputs = inputs.drop_front();
    }

    LLVMTypeConverter typeConverter(rewriter.getContext());
    Type inputType = hasInputs ? typeConverter.packFunctionResults(inputs) : ptr;
    bool noChange = inputs.size() == 1;
    if (noChange) {
        // Still wrap the pointer into a struct
        // for uniformity in Python and in the unwrapping.
        inputType = LLVM::LLVMStructType::getLiteral(rewriter.getContext(), inputType);
    }
    if (auto structType = inputType.dyn_cast<LLVM::LLVMStructType>()) {
        inputType = LLVM::LLVMPointerType::get(structType);
    }
    transformedInputs.push_back(inputType);

    Type voidType = LLVM::LLVMVoidType::get(rewriter.getContext());
    return LLVM::LLVMFunctionType::get(voidType, transformedInputs);
}

static void wrapResultsAndArgsInTwoStructs(LLVM::LLVMFuncOp op, PatternRewriter &rewriter,
                                           std::string nameWithoutPrefix)
{
    bool hasReturns = functionHasReturns(op);
    bool hasInputs = functionHasInputs(op);
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

        for (auto &en : llvm::enumerate(params)) {
            Value pointer = rewriter.create<LLVM::ExtractValueOp>(loc, structOfMemrefs, en.index());
            args.push_back(pointer);
        }
    }

    auto call = rewriter.create<LLVM::CallOp>(loc, op, args);

    rewriter.create<LLVM::ReturnOp>(loc, call.getResults());
}

void WrapperForWrapperTransform::rewrite(LLVM::LLVMFuncOp op, PatternRewriter &rewriter) const
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

struct WrapperForWrapperPass : public PassWrapper<WrapperForWrapperPass, OperationPass<ModuleOp>> {
    WrapperForWrapperPass() {}

    StringRef getArgument() const override { return "wrapper-for-wrapper"; }

    StringRef getDescription() const override { return "Wrapper around wrapper"; }

    void getDependentDialects(DialectRegistry &registry) const override
    {
        registry.insert<LLVM::LLVMDialect>();
    }

    void runOnOperation() final
    {
        MLIRContext *context = &getContext();
        RewritePatternSet patterns(context);
        patterns.add<WrapperForWrapperTransform>(context);

        if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
            signalPassFailure();
        }
    }
};

} // namespace quantum

std::unique_ptr<Pass> createWrapperForWrapperPass()
{
    return std::make_unique<quantum::WrapperForWrapperPass>();
}

} // namespace catalyst
