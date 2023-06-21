#include "Catalyst/IR/CatalystDialect.h"
#include "Catalyst/Transforms/Passes.h"
#include "Gradient/IR/GradientDialect.h"
#include "Gradient/Transforms/Passes.h"
#include "Quantum-c/Dialects.h"
#include "Quantum/IR/QuantumDialect.h"
#include "Quantum/Transforms/Passes.h"

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

void registerAllCatalystPasses()
{
    mlir::registerPass(catalyst::createArrayListToMemRefPass);
    mlir::registerPass(catalyst::createGradientBufferizationPass);
    mlir::registerPass(catalyst::createGradientLoweringPass);
    mlir::registerPass(catalyst::createGradientConversionPass);
    mlir::registerPass(catalyst::createQuantumBufferizationPass);
    mlir::registerPass(catalyst::createQuantumConversionPass);
    mlir::registerPass(catalyst::createEmitCatalystPyInterfacePass);
    mlir::registerPass(catalyst::createCopyGlobalMemRefPass);

    mhlo::registerAllMhloPasses();
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

//===----------------------------------------------------------------------===//
// Lowering pipelines.
//===----------------------------------------------------------------------===//

LogicalResult addMhloToCorePasses(PassManager &pm)
{
    const char *mhloToCorePipeline = "func.func(chlo-legalize-to-hlo),"
                                     "stablehlo-legalize-to-hlo,"
                                     "func.func(mhlo-legalize-control-flow),"
                                     "func.func(hlo-legalize-to-linalg),"
                                     "func.func(mhlo-legalize-to-std),"
                                     "convert-to-signless";
    return parsePassPipeline(mhloToCorePipeline, pm);
}

LogicalResult addQuantumCompilationPasses(PassManager &pm)
{
    const char *quantumPipeline = "lower-gradients,"
                                  "convert-arraylist-to-memref";

    return parsePassPipeline(quantumPipeline, pm);
}

LogicalResult addBufferizationPasses(PassManager &pm)
{
    const char *bufferizationPipeline =
        "inline,"
        "gradient-bufferize,"
        "scf-bufferize,"
        "convert-tensor-to-linalg,"      // tensor.pad
        "convert-elementwise-to-linalg," // Must be run before --arith-bufferize
        "arith-bufferize,"
        "empty-tensor-to-alloc-tensor,"
        "func.func(bufferization-bufferize),"
        "func.func(tensor-bufferize),"
        "func.func(linalg-bufferize),"
        "func.func(tensor-bufferize),"
        "quantum-bufferize,"
        "func-bufferize,"
        "func.func(finalizing-bufferize),"
        // "func.func(buffer-hoisting),"
        "func.func(buffer-loop-hoisting),"
        // "func.func(buffer-deallocation),"
        "convert-bufferization-to-memref,"
        "canonicalize,"
        // "cse,"
        "cp-global-memref";
    return parsePassPipeline(bufferizationPipeline, pm);
}

LogicalResult addLowerToLLVMPasses(PassManager &pm)
{
    const char *lowerToLLVMDialectPipeline =
        "func.func(convert-linalg-to-loops),"
        "convert-scf-to-cf,"
        // This pass expands memref operations that modify the metadata of a memref (sizes, offsets,
        // strides) into a sequence of easier to analyze constructs. In particular, this pass
        // transforms operations into explicit sequence of operations that model the effect of this
        // operation on the different metadata. This pass uses affine constructs to materialize
        // these effects. Concretely, expanded-strided-metadata is used to decompose memref.subview
        // as it has no lowering in -finalize-memref-to-llvm.
        "expand-strided-metadata,"
        "lower-affine,"
        "arith-expand," // some arith ops (ceildivsi) require expansion to be lowered to llvm
        "convert-complex-to-standard," // added for complex.exp lowering
        "convert-complex-to-llvm,"
        "convert-math-to-llvm,"
        // Run after -convert-math-to-llvm as it marks math::powf illegal without converting it.
        "convert-math-to-libm,"
        "convert-arith-to-llvm,"
        "finalize-memref-to-llvm{use-generic-functions},"
        "convert-index-to-llvm,"
        "convert-gradient-to-llvm,"
        "convert-quantum-to-llvm,"
        "emit-catalyst-py-interface,"
        // Remove any dead casts as the final pass expects to remove all existing casts,
        // but only those that form a loop back to the original type.
        "canonicalize,"
        "reconcile-unrealized-casts";
    return parsePassPipeline(lowerToLLVMDialectPipeline, pm);
}

LogicalResult runLowering(MLIRContext *ctx, ModuleOp moduleOp)
{
    auto pm = PassManager::on<ModuleOp>(ctx, PassManager::Nesting::Implicit);
    if (failed(addMhloToCorePasses(pm))) {
        return failure();
    }
    if (failed(addQuantumCompilationPasses(pm))) {
        return failure();
    }
    if (failed(addBufferizationPasses(pm))) {
        return failure();
    }
    if (failed(addLowerToLLVMPasses(pm))) {
        return failure();
    }

    return pm.run(moduleOp);
}
} // namespace

CatalystCReturnCode Canonicalize(const char *source, char **dest)
{
    DialectRegistry registry;
    registerAllCatalystDialects(registry);
    MLIRContext context{registry};
    OwningOpRef<ModuleOp> op = parseSource(&context, source);

    auto pm = PassManager::on<ModuleOp>(&context);
    pm.addPass(createCanonicalizerPass());
    if (failed(pm.run(*op))) {
        return ReturnLoweringFailed;
    }

    std::string output;
    llvm::raw_string_ostream ostream{output};
    op->print(ostream);

    // Need to explicitly allocate the char buffer for C interop
    *dest = static_cast<char *>(std::malloc(output.size() + 1));
    std::strcpy(*dest, output.c_str());
    return ReturnOk;
}

CatalystCReturnCode QuantumDriverMain(const char *source, bool keepIntermediate)
{
    registerAllCatalystPasses();

    DialectRegistry registry;
    registerAllCatalystDialects(registry);
    registerLLVMTranslations(registry);
    MLIRContext context{registry};
    OwningOpRef<ModuleOp> op = parseSource(&context, source);
    if (!op) {
        return ReturnParsingFailed;
    }

    if (failed(runLowering(&context, *op))) {
        return ReturnLoweringFailed;
    }

    llvm::LLVMContext llvmContext;
    auto llvmModule = translateModuleToLLVMIR(*op, llvmContext);
    if (!llvmModule) {
        return ReturnTranslationFailed;
    }

    llvmModule->dump();
    return ReturnOk;
}
