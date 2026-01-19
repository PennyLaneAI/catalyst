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

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Catalyst/Transforms/Patterns.h"
#include "Gradient/Transforms/EnzymeConstants.h"

using namespace mlir;

namespace catalyst {

#define GEN_PASS_DEF_REGISTERINACTIVECALLBACKPASS
#include "Catalyst/Transforms/Passes.h.inc"

struct RegisterInactiveCallbackPass
    : impl::RegisterInactiveCallbackPassBase<RegisterInactiveCallbackPass> {
    using RegisterInactiveCallbackPassBase::RegisterInactiveCallbackPassBase;
    void runOnOperation() final
    {
        auto mod = getOperation();
        StringRef inactive_callbackFnName = "__catalyst_inactive_callback";
        auto fnDecl = mod.lookupSymbol<LLVM::LLVMFuncOp>(inactive_callbackFnName);
        if (!fnDecl) {
            return;
        }
        MLIRContext *context = &getContext();
        auto builder = OpBuilder(context);
        builder.setInsertionPointToStart(mod.getBody());
        auto ptrTy = LLVM::LLVMPointerType::get(context);
        auto arrTy = LLVM::LLVMArrayType::get(ptrTy, 1);
        auto loc = mod.getLoc();
        auto isConstant = false;
        auto linkage = LLVM::Linkage::External;
        auto key = catalyst::gradient::enzyme_inactivefn_key;
        auto glb = LLVM::GlobalOp::create(builder, loc, arrTy, isConstant, linkage, key, nullptr);
        // Create a block and push it to the global
        Block *block = new Block();
        glb.getInitializerRegion().push_back(block);
        builder.setInsertionPointToStart(block);
        auto undef = LLVM::UndefOp::create(builder, glb.getLoc(), arrTy);
        auto fnSym = SymbolRefAttr::get(context, inactive_callbackFnName);
        auto fnPtr = LLVM::AddressOfOp::create(builder, glb.getLoc(), ptrTy, fnSym);
        auto filledInArray = LLVM::InsertValueOp::create(builder, glb.getLoc(), undef, fnPtr, 0);
        LLVM::ReturnOp::create(builder, glb.getLoc(), filledInArray);
    }
};

} // namespace catalyst
