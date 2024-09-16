// Copyright 2024 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"

#include "Catalyst/Transforms/Passes.h"
#include "Catalyst/Transforms/TBAAUtils.h"

#include "Catalyst/Transforms/Patterns.h"
#include "Gradient/IR/GradientInterfaces.h"

using namespace mlir;

namespace catalyst {

#define GEN_PASS_DEF_MEMREFTOLLVMWITHTBAAPASS
#define GEN_PASS_DECL_MEMREFTOLLVMWITHTBAAPASS
#include "Catalyst/Transforms/Passes.h.inc"

} // namespace catalyst

class MemrefToLLVMWithTBAAPass
    : public catalyst::impl::MemrefToLLVMWithTBAAPassBase<MemrefToLLVMWithTBAAPass> {
  public:
    void runOnOperation() override;

  private:
    void lowerMemrefWithTBAA(ModuleOp module);
};

void MemrefToLLVMWithTBAAPass::runOnOperation()
{
    ModuleOp mod = getOperation();
    bool containGradients = false;
    mod.walk([&](LLVM::LLVMFuncOp op) {
        if (op.getName().starts_with("__enzyme_autodiff")) {
            containGradients = true;
            return WalkResult::interrupt();
        }
        return WalkResult::skip();
    });
    if (containGradients) {
        lowerMemrefWithTBAA(mod);
    }
}

void MemrefToLLVMWithTBAAPass::lowerMemrefWithTBAA(ModuleOp module)
{
    mlir::MLIRContext *ctx = module.getContext();

    auto root = mlir::StringAttr::get(ctx, "Catalyst TBAA");
    auto intName = mlir::StringAttr::get(ctx, "int");
    auto float32Name = mlir::StringAttr::get(ctx, "float");
    auto float64Name = mlir::StringAttr::get(ctx, "double");
    auto pointerName = mlir::StringAttr::get(ctx, "any pointer");

    catalyst::TBAATree tree{ctx, root, intName, float32Name, float64Name, pointerName};
    catalyst::TBAATree &treeRef = tree;
    LLVMTypeConverter typeConverter(ctx);

    RewritePatternSet patterns(&getContext());
    catalyst::populateTBAATagsPatterns(treeRef, typeConverter, patterns);

    LLVMConversionTarget target(*ctx);
    target.addIllegalOp<memref::LoadOp>();
    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
        return signalPassFailure();
    }
}
std::unique_ptr<Pass> catalyst::createMemrefToLLVMWithTBAAPass()
{
    return std::make_unique<MemrefToLLVMWithTBAAPass>();
}
