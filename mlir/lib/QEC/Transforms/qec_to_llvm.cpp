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

#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "QEC/IR/QECOps.h"
#include "QEC/Transforms/Patterns.h"
#include "Quantum/IR/QuantumDialect.h"

using namespace mlir;

namespace catalyst {
namespace qec {

#define GEN_PASS_DECL_QECCONVERSIONPASS
#define GEN_PASS_DEF_QECCONVERSIONPASS
#include "QEC/Transforms/Passes.h.inc"

struct QECTypeConverter : public LLVMTypeConverter {
    QECTypeConverter(MLIRContext *ctx) : LLVMTypeConverter(ctx)
    {
        addConversion([&](quantum::QubitType type) { return convertQubitType(type); });
        addConversion([&](quantum::QuregType type) { return convertQuregType(type); });
        addConversion([&](quantum::ResultType type) { return convertResultType(type); });
        addConversion([&](quantum::ObservableType type) { return convertObservableType(type); });
    }

  private:
    Type convertQubitType(Type mlirType) { return LLVM::LLVMPointerType::get(&getContext()); }
    Type convertQuregType(Type mlirType) { return LLVM::LLVMPointerType::get(&getContext()); }
    Type convertResultType(Type mlirType) { return LLVM::LLVMPointerType::get(&getContext()); }
    Type convertObservableType(Type mlirType) { return LLVM::LLVMPointerType::get(&getContext()); }
};

struct QECConversionPass : impl::QECConversionPassBase<QECConversionPass> {
    using QECConversionPassBase::QECConversionPassBase;

    void runOnOperation() final
    {
        MLIRContext *context = &getContext();
        QECTypeConverter typeConverter(context);

        RewritePatternSet patterns(context);

        // Add infrastructure patterns for func.func, control flow, etc.
        mlir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);
        mlir::populateFuncToLLVMConversionPatterns(typeConverter, patterns);
        mlir::cf::populateAssertToLLVMConversionPattern(typeConverter, patterns);

        // Add QEC-specific patterns
        qec::populateConversionPatterns(typeConverter, patterns);

        LLVMConversionTarget target(*context);

        if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
            signalPassFailure();
        }
    }
};

} // namespace qec
} // namespace catalyst
