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

#include "mlir/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/DialectConversion.h"

#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"

#include "Catalyst/IR/CatalystDialect.h"
#include "Catalyst/IR/CatalystOps.h"

using namespace mlir;
using namespace catalyst;

#define GEN_PASS_DEF_LOWERQUANTUMMODULETOPAYLOADPASS
#include "Catalyst/Transforms/Passes.h.inc"

namespace {
struct LowerQuantumModuleToPayloadRewritePattern : public mlir::OpRewritePattern<ModuleOp> {
    using mlir::OpRewritePattern<ModuleOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(ModuleOp op, mlir::PatternRewriter &rewriter) const override
    {

        ModuleOp parent = op->getParentOfType<ModuleOp>();
        if (!parent)
            return failure();

        llvm::LLVMContext llvmContext;
        auto llvmModule = translateModuleToLLVMIR(op, llvmContext);
        if (!llvmModule)
            return failure();

        std::string targetModule;
        llvm::raw_string_ostream stream(targetModule);
        llvmModule->print(stream, nullptr);

        auto payloadOp =
            rewriter.create<DevicePayloadOp>(op.getLoc(), op.getSymName().value(), stream.str());
        rewriter.replaceOp(op, payloadOp);
        return success();
    }
};

} // namespace

namespace catalyst {

void populateLowerQuantumModuleToPayloadPatterns(RewritePatternSet &patterns)
{
    patterns.add<LowerQuantumModuleToPayloadRewritePattern>(patterns.getContext());
}

} // namespace catalyst
