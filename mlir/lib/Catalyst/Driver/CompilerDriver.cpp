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

#include <filesystem>
#include <list>
#include <memory>

#include "gml_st/transforms/passes.h"
#include "mhlo/IR/register.h"
#include "mhlo/transforms/passes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "stablehlo/dialect/Register.h"
#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/SourceMgr.h"

#include "Catalyst/Driver/CatalystLLVMTarget.h"
#include "Catalyst/Driver/CompilerDriver.h"
#include "Catalyst/Driver/Support.h"
#include "Catalyst/IR/CatalystDialect.h"
#include "Catalyst/Transforms/Passes.h"
#include "Gradient/IR/GradientDialect.h"
#include "Gradient/Transforms/Passes.h"
#include "Quantum/IR/QuantumDialect.h"
#include "Quantum/Transforms/Passes.h"

#include "Enzyme.h"
#include "PreserveNVVM.h"

using namespace mlir;
using namespace catalyst;
namespace fs = std::filesystem;

namespace {

std::string joinPasses(const Pipeline::PassList &passes)
{
    std::string joined;
    llvm::raw_string_ostream stream{joined};
    llvm::interleaveComma(passes, stream);
    return joined;
}

struct CatalystIRPrinterConfig : public PassManager::IRPrinterConfig {
    typedef std::function<LogicalResult(Pass *, PrintCallbackFn print)> PrintHandler;
    PrintHandler printHandler;

    CatalystIRPrinterConfig(PrintHandler printHandler)
        : IRPrinterConfig(/*printModuleScope=*/true), printHandler(printHandler)
    {
    }

    void printAfterIfEnabled(Pass *pass, Operation *operation, PrintCallbackFn printCallback) final
    {
        if (failed(printHandler(pass, printCallback))) {
            operation->emitError("IR printing failed");
        }
    }
};

} // namespace

namespace {
/// Parse an MLIR module given in textual ASM representation. Any errors during parsing will be
/// output to diagnosticStream.
OwningOpRef<ModuleOp> parseMLIRSource(MLIRContext *ctx, StringRef source, StringRef moduleName,
                                      raw_ostream &diagnosticStream)
{
    auto moduleBuffer = llvm::MemoryBuffer::getMemBufferCopy(source, moduleName);
    auto sourceMgr = std::make_shared<llvm::SourceMgr>();
    sourceMgr->AddNewSourceBuffer(std::move(moduleBuffer), SMLoc());

    FallbackAsmResourceMap fallbackResourceMap;
    ParserConfig parserConfig{ctx, /*verifyAfterParse=*/true, &fallbackResourceMap};

    SourceMgrDiagnosticHandler sourceMgrHandler(*sourceMgr, ctx, diagnosticStream);
    return parseSourceFile<ModuleOp>(sourceMgr, parserConfig);
}

/// Parse an LLVM module given in textual representation. Any parse errors will be output to
/// the provided SMDiagnostic.
std::shared_ptr<llvm::Module> parseLLVMSource(llvm::LLVMContext &context, StringRef source,
                                              StringRef moduleName, llvm::SMDiagnostic &err)
{
    auto moduleBuffer = llvm::MemoryBuffer::getMemBufferCopy(source, moduleName);
    return llvm::parseIR(llvm::MemoryBufferRef(*moduleBuffer), err, context);
}

/// Register all dialects required by the Catalyst compiler.
void registerAllCatalystDialects(DialectRegistry &registry)
{
    // MLIR Core dialects
    registerAllDialects(registry);
    registerAllExtensions(registry);

    // HLO
    mhlo::registerAllMhloDialects(registry);
    stablehlo::registerAllDialects(registry);

    // Catalyst
    registry.insert<CatalystDialect>();
    registry.insert<quantum::QuantumDialect>();
    registry.insert<gradient::GradientDialect>();
}
} // namespace

FailureOr<llvm::Function *> getJITFunction(MLIRContext *ctx, llvm::Module &llvmModule)
{
    Location loc = NameLoc::get(StringAttr::get(ctx, llvmModule.getName()));
    for (auto &function : llvmModule.functions()) {
        emitRemark(loc) << "Found LLVM function: " << function.getName() << "\n";
        if (function.getName().starts_with("catalyst.entry_point")) {
            return &function;
        }
    }
    emitError(loc, "Failed to find JIT function");
    return failure();
}

LogicalResult inferMLIRReturnTypes(MLIRContext *ctx, llvm::Type *returnType,
                                   Type assumedElementType,
                                   SmallVectorImpl<RankedTensorType> &inferredTypes)
{
    auto inferSingleMemRef = [&](llvm::StructType *descriptorType) {
        SmallVector<int64_t> resultShape;
        assert(descriptorType->getNumElements() >= 3 &&
               "Expected MemRef descriptor struct to have at least 3 entries");
        if (descriptorType->getNumElements() == 3) {
            // resultShape is empty
        }
        else {
            auto *arrayType = cast<llvm::ArrayType>(descriptorType->getTypeAtIndex(3));
            size_t rank = arrayType->getNumElements();
            for (size_t i = 0; i < rank; i++) {
                resultShape.push_back(ShapedType::kDynamic);
            }
        };
        return RankedTensorType::get(resultShape, assumedElementType);
    };
    if (returnType->isVoidTy()) {
        return failure();
    }
    if (auto *structType = dyn_cast<llvm::StructType>(returnType)) {
        // The return type could be a single memref descriptor or a struct of multiple memref
        // descriptors.
        if (isa<llvm::StructType>(structType->getElementType(0))) {
            for (size_t i = 0; i < structType->getNumElements(); i++) {
                inferredTypes.push_back(
                    inferSingleMemRef(cast<llvm::StructType>(structType->getTypeAtIndex(i))));
            }
        }
        else {
            // Assume the function returns a single memref
            inferredTypes.push_back(inferSingleMemRef(structType));
        }
        return success();
    }
    return failure();
}

LogicalResult runLLVMPasses(const CompilerOptions &options,
                            std::shared_ptr<llvm::Module> llvmModule,
                            CompilerOutput::PipelineOutputs &outputs)
{
    // opt -O2
    // As seen here:
    // https://llvm.org/docs/NewPassManager.html#just-tell-me-how-to-run-the-default-optimization-pipeline-with-the-new-pass-manager

    // Create the analysis managers.
    llvm::LoopAnalysisManager LAM;
    llvm::FunctionAnalysisManager FAM;
    llvm::CGSCCAnalysisManager CGAM;
    llvm::ModuleAnalysisManager MAM;
    // Create the new pass manager builder.
    // Take a look at the PassBuilder constructor parameters for more
    // customization, e.g. specifying a TargetMachine or various debugging
    // options.
    llvm::PassBuilder PB;
    // Register all the basic analyses with the managers.
    PB.registerModuleAnalyses(MAM);
    PB.registerCGSCCAnalyses(CGAM);
    PB.registerFunctionAnalyses(FAM);
    PB.registerLoopAnalyses(LAM);
    PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

    // Create the pass manager.
    // This one corresponds to a typical -O2 optimization pipeline.
    llvm::ModulePassManager MPM = PB.buildPerModuleDefaultPipeline(llvm::OptimizationLevel::O2);

    // Optimize the IR!
    MPM.run(*llvmModule.get(), MAM);

    return success();
}

LogicalResult runEnzymePasses(const CompilerOptions &options,
                              std::shared_ptr<llvm::Module> llvmModule,
                              CompilerOutput::PipelineOutputs &outputs)
{
    // Create the new pass manager builder.
    // Take a look at the PassBuilder constructor parameters for more
    // customization, e.g. specifying a TargetMachine or various debugging
    // options.
    llvm::PassBuilder PB;

    // Create the analysis managers.
    llvm::LoopAnalysisManager LAM;
    llvm::FunctionAnalysisManager FAM;
    llvm::CGSCCAnalysisManager CGAM;
    llvm::ModuleAnalysisManager MAM;

    // Register all the basic analyses with the managers.
    PB.registerModuleAnalyses(MAM);
    PB.registerCGSCCAnalyses(CGAM);
    PB.registerFunctionAnalyses(FAM);
    PB.registerLoopAnalyses(LAM);
    PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

    // Call Enzyme specific augmentPassBuilder which will add Enzyme passes.
    augmentPassBuilder(PB);

    // Create the pass manager.
    // This one corresponds to a typical -O2 optimization pipeline.
    llvm::ModulePassManager MPM = PB.buildModuleOptimizationPipeline(
        llvm::OptimizationLevel::O2, llvm::ThinOrFullLTOPhase::None);

    // Optimize the IR!
    MPM.run(*llvmModule.get(), MAM);

    return success();
}

LogicalResult runLowering(const CompilerOptions &options, MLIRContext *ctx, ModuleOp moduleOp,
                          CompilerOutput::PipelineOutputs &outputs)
{
    auto pm = PassManager::on<ModuleOp>(ctx, PassManager::Nesting::Implicit);

    std::unordered_map<const Pass *, std::list<Pipeline::Name>> pipelineTailMarkers;
    for (const auto &pipeline : options.pipelinesCfg) {
        if (failed(parsePassPipeline(joinPasses(pipeline.passes), pm, options.diagnosticStream))) {
            return failure();
        }
        PassManager::pass_iterator p = pm.end();
        const Pass *lastPass = &(*(p - 1));
        pipelineTailMarkers[lastPass].push_back(pipeline.name);
    }

    if (options.keepIntermediate) {

        {
            std::string tmp;
            {
                llvm::raw_string_ostream s{tmp};
                s << moduleOp;
            }
            std::string outFile = fs::path(options.moduleName.str()).replace_extension(".mlir");
            if (failed(catalyst::dumpToFile(options, outFile, tmp))) {
                return failure();
            }
        }

        {
            size_t pipelineIdx = 0;
            auto printHandler =
                [&](Pass *pass, CatalystIRPrinterConfig::PrintCallbackFn print) -> LogicalResult {
                auto res = pipelineTailMarkers.find(pass);
                if (res != pipelineTailMarkers.end()) {
                    for (const auto &pn : res->second) {
                        std::string outFile = fs::path(std::to_string(pipelineIdx++) + "_" + pn)
                                                  .replace_extension(".mlir");
                        {
                            llvm::raw_string_ostream s{outputs[pn]};
                            print(s);
                        }
                        if (failed(catalyst::dumpToFile(options, outFile, outputs[pn]))) {
                            return failure();
                        }
                    }
                }
                return success();
            };

            pm.enableIRPrinting(std::unique_ptr<PassManager::IRPrinterConfig>(
                new CatalystIRPrinterConfig(printHandler)));
        }
    }

    if (failed(pm.run(moduleOp))) {
        return failure();
    }

    return success();
}

LogicalResult QuantumDriverMain(const CompilerOptions &options, CompilerOutput &output)
{
    DialectRegistry registry;
    static bool initialized = false;
    if (!initialized) {
        registerAllPasses();
    }
    initialized |= true;
    registerAllCatalystPasses();
    mhlo::registerAllMhloPasses();
    gml_st::registerGmlStPasses();

    registerAllCatalystDialects(registry);
    registerLLVMTranslations(registry);
    MLIRContext ctx(registry);
    ctx.disableMultithreading();
    ScopedDiagnosticHandler scopedHandler(
        &ctx, [&](Diagnostic &diag) { diag.print(options.diagnosticStream); });

    llvm::LLVMContext llvmContext;
    std::shared_ptr<llvm::Module> llvmModule;

    llvm::raw_string_ostream outIRStream(output.outIR);

    // First attempt to parse the input as an MLIR module.
    OwningOpRef<ModuleOp> op =
        parseMLIRSource(&ctx, options.source, options.moduleName, options.diagnosticStream);
    if (op) {
        if (failed(runLowering(options, &ctx, *op, output.pipelineOutputs))) {
            return failure();
        }

        output.outIR.clear();
        outIRStream << *op;

        if (options.lowerToLLVM) {
            llvmModule = translateModuleToLLVMIR(*op, llvmContext);
            if (!llvmModule) {
                return failure();
            }

            if (options.keepIntermediate) {
                if (failed(catalyst::dumpToFile(options, "llvm_ir.ll", *llvmModule))) {
                    return failure();
                }
            }
        }
    }
    else {
        // If parsing as an MLIR module failed, attempt to parse as an LLVM IR module.
        llvm::SMDiagnostic err;
        llvmModule = parseLLVMSource(llvmContext, options.source, options.moduleName, err);
        if (!llvmModule) {
            // If both MLIR and LLVM failed to parse, exit.
            err.print(options.moduleName.data(), options.diagnosticStream);
            return failure();
        }
    }

    if (llvmModule) {

        if (failed(runLLVMPasses(options, llvmModule, output.pipelineOutputs))) {
            return failure();
        }

        if (failed(runEnzymePasses(options, llvmModule, output.pipelineOutputs))) {
            return failure();
        }

        output.outIR.clear();
        outIRStream << *llvmModule;

        // Attempt to infer the name and return type of the module from LLVM IR. This information is
        // required when executing a module given as textual IR.
        auto function = getJITFunction(&ctx, *llvmModule);
        if (succeeded(function)) {
            output.inferredAttributes.functionName = function.value()->getName().str();

            CO_MSG(options, Verbosity::Debug,
                   "Inferred function name: '" << output.inferredAttributes.functionName << "'\n");

            // When inferring the return type from LLVM, assume a f64
            // element type. This is because the LLVM pointer type is
            // opaque and requires looking into its uses to infer its type.
            SmallVector<RankedTensorType> returnTypes;
            if (failed(inferMLIRReturnTypes(&ctx, function.value()->getReturnType(),
                                            Float64Type::get(&ctx), returnTypes))) {
                // Inferred return types are only required when compiling from textual IR. This
                // inference failing is not a problem when compiling from Python.
                CO_MSG(options, Verbosity::Urgent, "Unable to infer function return type\n");
            }
            else {
                {
                    llvm::raw_string_ostream returnTypeStream(output.inferredAttributes.returnType);
                    llvm::interleaveComma(returnTypes, returnTypeStream,
                                          [&](RankedTensorType t) { t.print(returnTypeStream); });
                }
                CO_MSG(options, Verbosity::Debug,
                       "Inferred function return type: '" << output.inferredAttributes.returnType
                                                          << "'\n");
            }
        }
        else {
            CO_MSG(options, Verbosity::Urgent, "Unable to infer jit_* function attributes\n");
        }

        auto outfile = options.getObjectFile();
        if (failed(compileObjectFile(options, std::move(llvmModule), outfile))) {
            return failure();
        }
        output.objectFilename = outfile;
    }
    return success();
}
