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

#include "Catalyst/Transforms/Passes.h"
#include "Catalyst/Transforms/TBAAUtils.h"

#include <mlir/Dialect/LLVMIR/LLVMAttrs.h>

using namespace mlir;

namespace catalyst {

#define GEN_PASS_DEF_ADDTBAATAGSPASS
#define GEN_PASS_DECL_ADDTBAATAGSPASS
#include "Catalyst/Transforms/Passes.h.inc"

} // namespace catalyst

class AddTBAATagsPass : public catalyst::impl::AddTBAATagsPassBase<AddTBAATagsPass> {
  public:
    void runOnOperation() override;

  private:
    void createTBAATree(ModuleOp module);
};

void AddTBAATagsPass::runOnOperation()
{
    ModuleOp mod = getOperation();
    createTBAATree(mod);
}

void AddTBAATagsPass::createTBAATree(ModuleOp module)
{
    mlir::MLIRContext *ctx = module.getContext();

    auto root = mlir::StringAttr::get(ctx, "Catalyst TBAA");
    auto intName = mlir::StringAttr::get(ctx, "int");
    auto floatName = mlir::StringAttr::get(ctx, "float");
    auto pointerName = mlir::StringAttr::get(ctx, "any pointer");

    catalyst::TBAATree tree{ctx, root, intName, floatName, pointerName};
}
std::unique_ptr<Pass> catalyst::createAddTBAATagsPass() { return std::make_unique<AddTBAATagsPass>(); }


// struct LoadOpLowering : public LoadStoreOpLowering<memref::LoadOp> {
//   using Base::Base;

//   LogicalResult
//   matchAndRewrite(memref::LoadOp loadOp, OpAdaptor adaptor,
//                   ConversionPatternRewriter &rewriter) const override {
//     auto type = loadOp.getMemRefType();

//     Value dataPtr =
//         getStridedElementPtr(loadOp.getLoc(), type, adaptor.getMemref(),
//                              adaptor.getIndices(), rewriter);
//     auto op = rewriter.replaceOpWithNewOp<LLVM::LoadOp>(
//         loadOp, typeConverter->convertType(type.getElementType()), dataPtr, 0,
//         false, loadOp.getNontemporal());
//     op.setTBAATags();
//     return success();
//   }
// };