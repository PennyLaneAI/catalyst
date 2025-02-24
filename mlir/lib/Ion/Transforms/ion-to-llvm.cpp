// Copyright 2025 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Ion/IR/IonOps.h"
#include "Ion/Transforms/Patterns.h"

using namespace mlir;
using namespace catalyst::ion;

namespace catalyst {
namespace ion {

#define GEN_PASS_DECL_IONCONVERSIONPASS
#define GEN_PASS_DEF_IONCONVERSIONPASS
#include "Ion/Transforms/Passes.h.inc"

struct IonTypeConverter : public LLVMTypeConverter {
    IonTypeConverter(MLIRContext *ctx) : LLVMTypeConverter(ctx)
    {
        addConversion([&](IonType type) { return convertIonType(type); });
        addConversion([&](PulseType type) { return convertPulseType(type); });
        addConversion([&](catalyst::quantum::QubitType type) { return convertQubitType(type); });
    }

  private:
    Type convertIonType(Type mlirType) { return LLVM::LLVMPointerType::get(&getContext()); }
    Type convertPulseType(Type mlirType) { return LLVM::LLVMPointerType::get(&getContext()); }
    Type convertQubitType(Type mlirType) { return LLVM::LLVMPointerType::get(&getContext()); }
};

struct IonConversionPass : impl::IonConversionPassBase<IonConversionPass> {
    using IonConversionPassBase::IonConversionPassBase;

    void runOnOperation() final
    {
        MLIRContext *context = &getContext();
        IonTypeConverter typeConverter(context);

        RewritePatternSet patterns(context);
        populateConversionPatterns(typeConverter, patterns);

        LLVMConversionTarget target(*context);
        target.addIllegalDialect<catalyst::ion::IonDialect>();
        target.addLegalDialect<catalyst::quantum::QuantumDialect>();
        target.addLegalDialect<mlir::func::FuncDialect>();
        target.addLegalDialect<arith::ArithDialect>();
        target.addLegalDialect<scf::SCFDialect>();

        if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
            signalPassFailure();
        }
    }
};

} // namespace ion

std::unique_ptr<Pass> createIonConversionPass()
{
    return std::make_unique<ion::IonConversionPass>();
}

} // namespace catalyst
