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

#include "llvm/IR/LegacyPassManager.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Target/LLVMIR/LLVMTranslationInterface.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"

#include "Catalyst/Driver/CatalystLLVMTarget.h"
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

LogicalResult catalyst::compileObjectFile(const CompilerOptions &options,
                                          std::unique_ptr<llvm::Module> llvmModule,
                                          StringRef filename)
{
    using namespace llvm;

    std::string targetTriple = sys::getDefaultTargetTriple();

    InitializeAllTargetInfos();
    InitializeAllTargets();
    InitializeAllTargetMCs();
    InitializeAllAsmParsers();
    InitializeAllAsmPrinters();

    std::string err;

    auto target = TargetRegistry::lookupTarget(targetTriple, err);

    if (!target) {
        CO_MSG(options, CO_VERB_URGENT, err);
        return failure();
    }

    // Target a generic CPU without any additional features, options, or relocation model
    const char *cpu = "generic";
    const char *features = "";

    TargetOptions opt;
    auto targetMachine =
        target->createTargetMachine(targetTriple, cpu, features, opt, Reloc::Model::PIC_);
    llvmModule->setDataLayout(targetMachine->createDataLayout());
    llvmModule->setTargetTriple(targetTriple);

    std::error_code errCode;
    raw_fd_ostream dest(filename, errCode, sys::fs::OF_None);

    if (errCode) {
        CO_MSG(options, CO_VERB_URGENT, "could not open file: " << errCode.message() << "\n");
        return failure();
    }

    legacy::PassManager pm;
    if (targetMachine->addPassesToEmitFile(pm, dest, nullptr, CGFT_ObjectFile)) {
        CO_MSG(options, CO_VERB_URGENT, "TargetMachine can't emit an .o file\n");
        return failure();
    }

    pm.run(*llvmModule);
    dest.flush();
    return success();
}
