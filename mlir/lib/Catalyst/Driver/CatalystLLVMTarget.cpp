// Copyright 2023 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "Catalyst/Driver/CatalystLLVMTarget.h"

#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Target/LLVMIR/LLVMTranslationInterface.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"

#include "Gradient/IR/GradientDialect.h"

using namespace mlir;

namespace {
/// Emit the LLVM IR metadata required to register custom gradients in Enzyme.
/// This interface will convert `gradient.augment` and `gradient.vjp` attributes on function-like
/// ops to the metadata read by Enzyme.
class GradientToEnzymeMetadataTranslation : public LLVMTranslationDialectInterface {
    using LLVMTranslationDialectInterface::LLVMTranslationDialectInterface;

    LogicalResult amendOperation(Operation *op, NamedAttribute attribute,
                                 LLVM::ModuleTranslation &moduleTranslation) const override
    {
        auto funcOp = dyn_cast<FunctionOpInterface>(op);
        bool hasAugment = funcOp->hasAttrOfType<FlatSymbolRefAttr>("gradient.augment");
        bool hasVJP = funcOp->hasAttrOfType<FlatSymbolRefAttr>("gradient.vjp");
        bool failedToMatch = !(funcOp && hasAugment && hasVJP);
        if (failedToMatch) {
            // Do nothing.
            return success();
        }

        auto function = moduleTranslation.lookupFunction(funcOp.getName());
        bool alreadyAmended =
            function->hasMetadata("enzyme_augment") && function->hasMetadata("enzyme_gradient");
        if (alreadyAmended) {
            return success();
        }

        auto augmented = moduleTranslation.lookupFunction(
            funcOp->getAttrOfType<FlatSymbolRefAttr>("gradient.augment").getValue());
        auto vjp = moduleTranslation.lookupFunction(
            funcOp->getAttrOfType<FlatSymbolRefAttr>("gradient.vjp").getValue());
        assert(augmented && "gradient.augment did not reference a valid LLVM function");
        assert(vjp && "gradient.vjp did not reference a valid LLVM function");

        llvm::LLVMContext &ctx = moduleTranslation.getLLVMContext();
        function->addMetadata("enzyme_augment",
                              *llvm::MDNode::get(ctx, llvm::ConstantAsMetadata::get(augmented)));
        function->addMetadata("enzyme_gradient",
                              *llvm::MDNode::get(ctx, llvm::ConstantAsMetadata::get(vjp)));
        return success();
    }
};
} // namespace

void catalyst::registerLLVMTranslations(DialectRegistry &registry)
{
    registerLLVMDialectTranslation(registry);
    registerBuiltinDialectTranslation(registry);
    registry.addExtension(+[](MLIRContext *ctx, gradient::GradientDialect *dialect) {
        dialect->addInterfaces<GradientToEnzymeMetadataTranslation>();
    });
}
