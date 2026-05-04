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

#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "QecPhysical/IR/QecPhysicalDialect.h"
#include "QecPhysical/Transforms/Patterns.h"

using namespace mlir;
using namespace catalyst::qecp;

namespace catalyst {
namespace qecp {

#define GEN_PASS_DECL_QECPHYSICALCONVERSIONPASS
#define GEN_PASS_DEF_QECPHYSICALCONVERSIONPASS
#include "QecPhysical/Transforms/Passes.h.inc"

struct QecPhysicalTypeConverter : public LLVMTypeConverter {
    QecPhysicalTypeConverter(MLIRContext *ctx) : LLVMTypeConverter(ctx)
    {
        addConversion([&](TannerGraphType type) { return convertTannerGraphType(type); });
    }

  private:
    Type convertTannerGraphType(Type mlirType) { return LLVM::LLVMPointerType::get(&getContext()); }
};

struct QecPhysicalConversionPass : impl::QecPhysicalConversionPassBase<QecPhysicalConversionPass> {
    using QecPhysicalConversionPassBase::QecPhysicalConversionPassBase;

    void runOnOperation() final
    {
        MLIRContext *context = &getContext();
        QecPhysicalTypeConverter typeConverter(context);

        RewritePatternSet patterns(context);
        populateLLVMConversionPatterns(typeConverter, patterns);

        LLVMConversionTarget target(*context);
        // TODOs: We need to uncomment the following line once all qecp-to-llvm patterns are added
        // target.addIllegalDialect<catalyst::qecp::QecPhysicalDialect>();

        if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
            signalPassFailure();
        }
    }
};

} // namespace qecp
} // namespace catalyst
