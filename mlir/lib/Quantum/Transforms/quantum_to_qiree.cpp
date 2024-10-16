// Copyright 2022-2023 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Quantum/IR/QuantumDialect.h"
#include "Quantum/Transforms/Passes.h"
#include "Quantum/Transforms/Patterns.h"

using namespace mlir;
using namespace catalyst::quantum;

namespace catalyst {
namespace quantum {

#define GEN_PASS_DEF_QIREECONVERSIONPASS
#include "Quantum/Transforms/Passes.h.inc"

struct QIREETypeConverter : public LLVMTypeConverter {

    QIREETypeConverter(MLIRContext *ctx) : LLVMTypeConverter(ctx)
    {
        addConversion([&](QubitType type) { return convertQubitType(type); });
        addConversion([&](QuregType type) { return convertQuregType(type); });
        addConversion([&](ObservableType type) { return convertObservableType(type); });
        addConversion([&](ResultType type) { return convertResultType(type); });
    }

  private:
    Type convertQubitType(Type mlirType)
    {
        return LLVM::LLVMPointerType::get(
            &getContext()); // LLVM::LLVMStructType::getOpaque("Qubit", &getContext());
    }

    Type convertQuregType(Type mlirType)
    {
        return LLVM::LLVMPointerType::get(
            &getContext()); // LLVM::LLVMStructType::getOpaque("Array", &getContext());
    }

    Type convertObservableType(Type mlirType)
    {
        return this->convertType(IntegerType::get(&getContext(), 64));
    }

    Type convertResultType(Type mlirType)
    {
        return LLVM::LLVMPointerType::get(
            &getContext()); // LLVM::LLVMStructType::getOpaque("Result", &getContext());
    }
};

struct QIREEConversionPass : impl::QIREEConversionPassBase<QIREEConversionPass> {
    using QIREEConversionPassBase::QIREEConversionPassBase;

    void runOnOperation() final
    {
        MLIRContext *context = &getContext();
        QIREETypeConverter typeConverter(context);

        RewritePatternSet patterns(context);
        cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);
        populateFuncToLLVMConversionPatterns(typeConverter, patterns);
        populateQIREEConversionPatterns(patterns);

        LLVMConversionTarget target(*context);
        target.addLegalOp<ModuleOp>();

        if (failed(applyFullConversion(getOperation(), target, std::move(patterns)))) {
            signalPassFailure();
        }
    }
};

} // namespace quantum

std::unique_ptr<Pass> createQIREEConversionPass()
{
    return std::make_unique<quantum::QIREEConversionPass>();
}

} // namespace catalyst
