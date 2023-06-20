#include "Catalyst/IR/CatalystDialect.h"
#include "Gradient/IR/GradientDialect.h"
#include "Quantum-c/Dialects.h"
#include "Quantum/IR/QuantumDialect.h"

#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Target/LLVMIR/LLVMTranslationInterface.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "llvm/Support/SourceMgr.h"

#include "memory"

#include "mhlo/IR/register.h"
#include "mhlo/transforms/passes.h"
#include "stablehlo/dialect/Register.h"

using namespace mlir;
using namespace catalyst;

namespace {
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
        function->dump();
        return success();
    }
};

OwningOpRef<ModuleOp> parseSource(MLIRContext *ctx, const char *source)
{
    auto moduleBuffer = llvm::MemoryBuffer::getMemBufferCopy(source, "jit source");
    auto sourceMgr = std::make_shared<llvm::SourceMgr>();
    sourceMgr->AddNewSourceBuffer(std::move(moduleBuffer), SMLoc());

    FallbackAsmResourceMap fallbackResourceMap;
    ParserConfig parserConfig{ctx, /*verifyAfterParse=*/true, &fallbackResourceMap};

    SourceMgrDiagnosticHandler sourceMgrHandler(*sourceMgr, ctx);
    return parseSourceFile<ModuleOp>(sourceMgr, parserConfig);
}

/// Register all dialects required by the Catalyst compiler.
void registerAllCatalystDialects(DialectRegistry &registry)
{
    // MLIR Core dialects
    registerAllDialects(registry);

    // HLO
    mhlo::registerAllMhloDialects(registry);
    stablehlo::registerAllDialects(registry);

    // Catalyst
    registry.insert<CatalystDialect>();
    registry.insert<quantum::QuantumDialect>();
    registry.insert<gradient::GradientDialect>();
}

/// Register the translations needed to convert to LLVM IR.
void registerLLVMTranslations(DialectRegistry &registry)
{
    registerLLVMDialectTranslation(registry);
    registerBuiltinDialectTranslation(registry);
    registry.addExtension(+[](MLIRContext *ctx, gradient::GradientDialect *dialect) {
        dialect->addInterfaces<GradientToEnzymeMetadataTranslation>();
    });
}

LogicalResult runLowering(MLIRContext *ctx, ModuleOp moduleOp)
{
    auto pm = PassManager::on<ModuleOp>(ctx);
    pm.addNestedPass<func::FuncOp>(
        mhlo::createLegalizeHloToLinalgPass(/*enable-primitive-ops=*/false));
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createConvertFuncToLLVMPass());
    return pm.run(moduleOp);
}
} // namespace

const char *Canonicalize(const char *source)
{
    DialectRegistry registry;
    registerAllCatalystDialects(registry);
    MLIRContext context{registry};
    OwningOpRef<ModuleOp> op = parseSource(&context, source);

    auto pm = PassManager::on<ModuleOp>(&context);
    pm.addPass(createCanonicalizerPass());
    if (failed(pm.run(*op))) {
        return "failed to canonicalize";
    }

    std::string output;
    llvm::raw_string_ostream ostream{output};
    op->print(ostream);

    char *outbuf = static_cast<char *>(std::malloc(output.size()));
    std::strcpy(outbuf, output.c_str());
    return outbuf;
}

void QuantumDriverMain(const char *source)
{
    DialectRegistry registry;
    registerAllCatalystDialects(registry);
    registerLLVMTranslations(registry);
    MLIRContext context{registry};
    OwningOpRef<ModuleOp> op = parseSource(&context, source);
    if (!op) {
        return;
    }

    if (failed(runLowering(&context, *op))) {
        return;
    }

    llvm::LLVMContext llvmContext;
    auto llvmModule = translateModuleToLLVMIR(*op, llvmContext);
    if (!llvmModule) {
        return;
    }

    llvmModule->dump();
}
