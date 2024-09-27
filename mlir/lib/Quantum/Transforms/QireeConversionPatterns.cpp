// Copyright 2022-2024 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <string>

#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Quantum/IR/QuantumOps.h"
#include "Quantum/Transforms/Patterns.h"
#include "llvm/ADT/StringRef.h"

using namespace mlir;
using namespace catalyst::quantum;

namespace {

constexpr int64_t UNKNOWN = ShapedType::kDynamic;
constexpr int32_t NO_POSTSELECT = -1;


LLVM::LLVMFuncOp ensureFunctionDeclaration(PatternRewriter &rewriter, Operation *op,
                                           StringRef fnSymbol, Type fnType)
{
    Operation *fnDecl = SymbolTable::lookupNearestSymbolFrom(op, rewriter.getStringAttr(fnSymbol));

    if (!fnDecl) {
        PatternRewriter::InsertionGuard insertGuard(rewriter);
        ModuleOp mod = op->getParentOfType<ModuleOp>();
        rewriter.setInsertionPointToStart(mod.getBody());

        auto fnOp = rewriter.create<LLVM::LLVMFuncOp>(op->getLoc(), fnSymbol, fnType);

        auto entryPoint = rewriter.getStringAttr("entry_point");
        auto numQubit = rewriter.getStringAttr("num_required_qubits");
        auto numQubitVal = rewriter.getStringAttr("2");
        auto outputLabel = rewriter.getStringAttr("output_labeling_schema");
        SmallVector<Attribute> passthrough = {entryPoint, outputLabel};

        fnOp->setAttr("passthrough", ArrayAttr::get(rewriter.getContext(), passthrough));
        
        Block *entryBlock = new Block();
        fnOp.getBody().push_back(entryBlock);
        rewriter.setInsertionPointToEnd(entryBlock);

         // Create a return operation (returning void)
        rewriter.create<LLVM::ReturnOp>(mod.getLoc(), ValueRange{});
        fnDecl = fnOp;
    }
    else {
        assert(isa<LLVM::LLVMFuncOp>(fnDecl) && "QIR function declaration is not a LLVMFuncOp");
    }

    return cast<LLVM::LLVMFuncOp>(fnDecl);
}

////////////////////////
// Runtime Management //
////////////////////////

template <typename T> struct RTBasedPattern : public OpRewritePattern<T> {
    using OpRewritePattern<T>::OpRewritePattern;

    LogicalResult matchAndRewrite(T op,
                                  PatternRewriter &rewriter) const override
    {
        MLIRContext *ctx = this->getContext();

        ModuleOp parentModule = op->template getParentOfType<ModuleOp>();

        if (parentModule) {
            bool insertMain = true;
            for (auto func : parentModule.getOps<LLVM::LLVMFuncOp>()) {
                if (func.getName() == "main") {
                    insertMain = false;
                }
            }
            if (insertMain) {
                StringRef qirName = "main";
                Type qirSignature = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx), {});
                LLVM::LLVMFuncOp fnDecl = ensureFunctionDeclaration(rewriter, op, qirName, qirSignature);
            }
        }

        
        rewriter.eraseOp(op);

        return success();
    }
};

} // namespace

namespace catalyst {
namespace quantum {

void populateQIREEConversionPatterns(RewritePatternSet &patterns)
{
    patterns.add<RTBasedPattern<InitializeOp>>( patterns.getContext());
    patterns.add<RTBasedPattern<FinalizeOp>>(patterns.getContext());
}

} // namespace quantum
} // namespace catalyst
