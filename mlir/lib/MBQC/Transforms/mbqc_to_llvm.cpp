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
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "MBQC/IR/MBQCOps.h"
#include "MBQC/Transforms/Patterns.h"

using namespace mlir;

namespace catalyst {
namespace mbqc {

#define GEN_PASS_DECL_MBQCCONVERSIONPASS
#define GEN_PASS_DEF_MBQCCONVERSIONPASS
#include "MBQC/Transforms/Passes.h.inc"

class MBQCTypeConverter : public LLVMTypeConverter {
  public:
    MBQCTypeConverter(MLIRContext *ctx) : LLVMTypeConverter(ctx)
    {
        addConversion([&](quantum::QubitType type) { return convertQubitType(type); });
        addConversion([&](quantum::ResultType type) { return convertResultType(type); });
    }

  private:
    Type convertQubitType(Type mlirType) { return LLVM::LLVMPointerType::get(&getContext()); }
    Type convertResultType(Type mlirType) { return LLVM::LLVMPointerType::get(&getContext()); }
};

struct MBQCConversionPass : impl::MBQCConversionPassBase<MBQCConversionPass> {
    using MBQCConversionPassBase::MBQCConversionPassBase;

    void runOnOperation() final
    {
        MLIRContext *context = &getContext();
        MBQCTypeConverter typeConverter(context);

        RewritePatternSet patterns(context);
        populateConversionPatterns(typeConverter, patterns);

        LLVMConversionTarget target(*context);
        target.addIllegalDialect<catalyst::mbqc::MBQCDialect>();

        if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
            signalPassFailure();
        }
    }
};

} // namespace mbqc

std::unique_ptr<Pass> createMBQCConversionPass()
{
    return std::make_unique<mbqc::MBQCConversionPass>();
}

} // namespace catalyst
