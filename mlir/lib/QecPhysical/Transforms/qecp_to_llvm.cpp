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

#include <optional>

#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "QecPhysical/IR/QecPhysicalOps.h"
#include "QecPhysical/Transforms/Patterns.h"

using namespace mlir;
using namespace catalyst::qecp;

namespace catalyst {
namespace qecp {

#define GEN_PASS_DEF_QECPHYSICALCONVERSIONPASS
#include "QecPhysical/Transforms/Passes.h.inc"

struct QecPhysicalTypeConverter : public LLVMTypeConverter {
    QecPhysicalTypeConverter(MLIRContext *ctx) : LLVMTypeConverter(ctx)
    {
        addConversion([&](TannerGraphType type) { return convertTannerGraphType(type); });
    }

  private:
    std::optional<Type> convertTannerGraphType(TannerGraphType mlirType)
    {
        auto *ctx = &getContext();

        auto llvmStruct = LLVM::LLVMStructType::getIdentified(ctx, "TannerGraph");

        if (!llvmStruct.isInitialized()) {
            auto elementType = mlirType.getElementType();

            auto rowIdxMemRef = mlir::MemRefType::get({mlirType.getRowIdxSize()}, elementType);
            auto colPtrMemRef = mlir::MemRefType::get({mlirType.getColPtrSize()}, elementType);

            mlir::Type rowIdxStruct = convertType(rowIdxMemRef);
            mlir::Type colPtrStruct = convertType(colPtrMemRef);

            if (!rowIdxStruct || !colPtrStruct) {
                return std::nullopt;
            }

            if (failed(llvmStruct.setBody({rowIdxStruct, colPtrStruct}, /*isPacked=*/false))) {
                return std::nullopt; // Conversion failed
            }
        }
        return llvmStruct;
    }
};

struct QecPhysicalConversionPass : impl::QecPhysicalConversionPassBase<QecPhysicalConversionPass> {
    using QecPhysicalConversionPassBase::QecPhysicalConversionPassBase;

    void runOnOperation() final
    {
        MLIRContext *context = &getContext();
        QecPhysicalTypeConverter typeConverter(context);

        RewritePatternSet patterns(context);

        // Add infrastructure patterns for func.func, control flow, etc.
        cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);
        populateFuncToLLVMConversionPatterns(typeConverter, patterns);
        cf::populateAssertToLLVMConversionPattern(typeConverter, patterns);

        populateQecPhysicalConversionPatterns(typeConverter, patterns);

        LLVMConversionTarget target(*context);
        target
            .addIllegalOp<catalyst::qecp::AssembleTannerGraphOp, catalyst::qecp::DecodeEsmCssOp>();

        if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
            signalPassFailure();
        }
    }
};

} // namespace qecp
} // namespace catalyst
