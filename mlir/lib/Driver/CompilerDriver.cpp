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

#include <cassert>

#include <iostream>
#include <memory>
#include <optional>
#include <string>

#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/Transforms/Coroutines/CoroCleanup.h"
#include "llvm/Transforms/Coroutines/CoroConditionalWrapper.h"
#include "llvm/Transforms/Coroutines/CoroEarly.h"
#include "llvm/Transforms/Coroutines/CoroSplit.h"
#include "llvm/Transforms/IPO/GlobalDCE.h"

#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassInstrumentation.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/LLVMIR/Export.h"

#include "stablehlo/dialect/Register.h"
#include "stablehlo/integrations/c/StablehloPasses.h"
#include "stablehlo/transforms/optimization/Passes.h"

#include "Catalyst/IR/CatalystDialect.h"
#include "Catalyst/Transforms/BufferizableOpInterfaceImpl.h"
#include "Driver/CatalystLLVMTarget.h"
#include "Driver/CompilerDriver.h"
#include "Driver/LineUtils.hpp"
#include "Driver/PassInstrumentation.hpp"
#include "Driver/Pipelines.h"
#include "Driver/Support.h"
#include "Driver/Timer.hpp"
#include "Gradient/IR/GradientDialect.h"
#include "Gradient/IR/GradientInterfaces.h"
#include "Gradient/Transforms/BufferizableOpInterfaceImpl.h"
#include "Ion/IR/IonDialect.h"
#include "MBQC/IR/MBQCDialect.h"
#include "Mitigation/IR/MitigationDialect.h"
#include "PauliFrame/IR/PauliFrameDialect.h"
#include "QEC/IR/QECDialect.h"
#include "Quantum/IR/QuantumDialect.h"
#include "Quantum/Transforms/BufferizableOpInterfaceImpl.h"
#include "RTIO/IR/RTIODialect.h"
#include "RegisterAllPasses.h"

#include "Enzyme.h"

using namespace mlir;
using namespace catalyst;
using namespace catalyst::driver;
namespace cl = llvm::cl;

namespace {

std::string joinPasses(const llvm::SmallVector<std::string> &passes)
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

/// The upstream MLIR Test dialect does not have a header we can include
/// We must declare the registration function, and link to the corresponding upstream target
/// in CMake.
namespace test {
void registerTestDialect(mlir::DialectRegistry &);
} // namespace test

namespace {
/// Parse an MLIR module given in textual ASM representation. Any errors during parsing will be
/// output to diagnosticStream.
OwningOpRef<ModuleOp> parseMLIRSource(MLIRContext *ctx, const llvm::SourceMgr &sourceMgr)
{
    FallbackAsmResourceMap fallbackResourceMap;
    ParserConfig parserConfig{ctx, /*verifyAfterParse=*/true, &fallbackResourceMap};

    return parseSourceFile<ModuleOp>(sourceMgr, parserConfig);
}

/// From the MLIR module it checks if gradients operations are in the program.
bool containsGradients(mlir::ModuleOp moduleOp)
{
    bool contain = false;
    moduleOp.walk([&](catalyst::gradient::GradientOpInterface op) {
        contain = true;
        return WalkResult::interrupt();
    });
    return contain;
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
    ::test::registerTestDialect(registry);

    // HLO
    stablehlo::registerAllDialects(registry);

    // Catalyst
    registry.insert<CatalystDialect>();
    registry.insert<quantum::QuantumDialect>();
    registry.insert<qec::QECDialect>();
    registry.insert<mbqc::MBQCDialect>();
    registry.insert<ion::IonDialect>();
    registry.insert<rtio::RTIODialect>();
    registry.insert<gradient::GradientDialect>();
    registry.insert<mitigation::MitigationDialect>();
    registry.insert<pauli_frame::PauliFrameDialect>();
}
} // namespace

// Determines if the compilation stage should be executed if a checkpointStage is given
bool shouldRunStage(const CompilerOptions &options, CompilerOutput &output,
                    const std::string &stageName)
{
    if (options.checkpointStage.empty()) {
        return true;
    }
    if (!output.isCheckpointFound) {
        output.isCheckpointFound = (options.checkpointStage == stageName);
        return false;
    }
    return true;
}

LogicalResult runCoroLLVMPasses(const CompilerOptions &options,
                                std::shared_ptr<llvm::Module> llvmModule, CompilerOutput &output)
{
    if (!shouldRunStage(options, output, "CoroOpt")) {
        return success();
    }

    // Create a pass to lower LLVM coroutines (similar to what happens in O0)
    llvm::ModulePassManager CoroPM;
    CoroPM.addPass(llvm::CoroEarlyPass());
    llvm::CGSCCPassManager CGPM;
    CGPM.addPass(llvm::CoroSplitPass());
    CoroPM.addPass(llvm::createModuleToPostOrderCGSCCPassAdaptor(std::move(CGPM)));
    CoroPM.addPass(llvm::CoroCleanupPass());
    CoroPM.addPass(llvm::GlobalDCEPass());

    // Create the analysis managers.
    llvm::LoopAnalysisManager LAM;
    llvm::FunctionAnalysisManager FAM;
    llvm::CGSCCAnalysisManager CGAM;
    llvm::ModuleAnalysisManager MAM;

    llvm::PassBuilder PB;
    PB.registerModuleAnalyses(MAM);
    PB.registerCGSCCAnalyses(CGAM);
    PB.registerFunctionAnalyses(FAM);
    PB.registerLoopAnalyses(LAM);
    PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

    // Optimize the IR!
    CoroPM.run(*llvmModule.get(), MAM);

    if (options.keepIntermediate) {
        output.setStage("CoroOpt");
        std::string tmp;
        llvm::raw_string_ostream rawStringOstream{tmp};
        llvmModule->print(rawStringOstream, nullptr);
        auto outFile = output.nextPipelineSummaryFilename("CoroOptPasses", ".ll");
        dumpToFile(options, outFile, tmp);
    }

    return success();
}

LogicalResult runO2LLVMPasses(const CompilerOptions &options,
                              std::shared_ptr<llvm::Module> llvmModule, CompilerOutput &output)
{
    // opt -O2
    // As seen here:
    // https://llvm.org/docs/NewPassManager.html#just-tell-me-how-to-run-the-default-optimization-pipeline-with-the-new-pass-manager
    if (!shouldRunStage(options, output, "O2Opt")) {
        return success();
    }

    // Create the analysis managers.
    llvm::LoopAnalysisManager LAM;
    llvm::FunctionAnalysisManager FAM;
    llvm::CGSCCAnalysisManager CGAM;
    llvm::ModuleAnalysisManager MAM;
    // Create the new pass manager builder.
    // Take a look at the PassBuilder constructor parameters for more
    // customization, e.g. specifying a TargetMachine or various debugging
    // options.
    llvm::PassInstrumentationCallbacks PIC;
    PIC.registerShouldRunOptionalPassCallback([](llvm::StringRef P, llvm::Any) {
        if (P == "MemCpyOptPass") {
            return false;
        }
        return true;
    });
    llvm::PassBuilder PB(nullptr, llvm::PipelineTuningOptions(), std::nullopt, &PIC);
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

    if (options.keepIntermediate) {
        output.setStage("O2Opt");
        std::string tmp;
        llvm::raw_string_ostream rawStringOstream{tmp};
        llvmModule->print(rawStringOstream, nullptr);
        auto outFile = output.nextPipelineSummaryFilename("O2OptPasses", ".ll");
        dumpToFile(options, outFile, tmp);
    }

    return success();
}

LogicalResult runEnzymePasses(const CompilerOptions &options,
                              std::shared_ptr<llvm::Module> llvmModule, CompilerOutput &output)
{
    if (!shouldRunStage(options, output, "Enzyme")) {
        return success();
    }

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

    if (options.keepIntermediate) {
        output.setStage("Enzyme");
        std::string tmp;
        llvm::raw_string_ostream rawStringOstream{tmp};
        llvmModule->print(rawStringOstream, nullptr);
        auto outFile = output.nextPipelineSummaryFilename("EnzymePasses", ".ll");
        dumpToFile(options, outFile, tmp);
    }

    return success();
}

std::string readInputFile(const std::string &filename)
{
    if (filename == "-") {
        std::stringstream buffer;
        std::istreambuf_iterator<char> begin(std::cin), end;
        buffer << std::string(begin, end);
        return buffer.str();
    }
    std::ifstream file(filename);
    if (!file.is_open()) {
        return "";
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

LogicalResult preparePassManager(PassManager &pm, const CompilerOptions &options,
                                 CompilerOutput &output, catalyst::utils::Timer<> &timer,
                                 TimingScope &timing)
{
    MlirOptMainConfig config = MlirOptMainConfig::createFromCLOptions();
    pm.enableVerifier(config.shouldVerifyPasses());
    if (failed(applyPassManagerCLOptions(pm)))
        return failure();
    if (failed(config.setupPassPipeline(pm)))
        return failure();
    pm.enableTiming(timing);
    pm.addInstrumentation(std::unique_ptr<PassInstrumentation>(
        new CatalystPassInstrumentation(options, output, timer)));

    return success();
}

LogicalResult configurePipeline(PassManager &pm, const CompilerOptions &options, Pipeline &pipeline,
                                bool clHasManualPipeline)
{
    pm.clear();
    if (!clHasManualPipeline && failed(pipeline.addPipeline(pm))) {
        llvm::errs() << "Pipeline creation function not found: " << pipeline.getName() << "\n";
        return failure();
    }
    if (clHasManualPipeline &&
        failed(parsePassPipeline(joinPasses(pipeline.getPasses()), pm, options.diagnosticStream))) {
        return failure();
    }
    if (options.dumpPassPipeline) {
        pm.dump();
        llvm::errs() << "\n";
    }
    return success();
}

LogicalResult runPipeline(PassManager &pm, const CompilerOptions &options, CompilerOutput &output,
                          Pipeline &pipeline, bool clHasManualPipeline, ModuleOp moduleOp)
{
    if (!shouldRunStage(options, output, pipeline.getName()) || pipeline.getPasses().size() == 0) {
        return success();
    }

    output.setStage(pipeline.getName());

    if (failed(configurePipeline(pm, options, pipeline, clHasManualPipeline))) {
        llvm::errs() << "Failed to run pipeline: " << pipeline.getName() << "\n";
        return failure();
    }
    if (failed(pm.run(moduleOp))) {
        llvm::errs() << "Failed to run pipeline: " << pipeline.getName() << "\n";
        return failure();
    }
    if (options.keepIntermediate && (options.checkpointStage.empty() || output.isCheckpointFound)) {
        std::string tmp;
        llvm::raw_string_ostream s{tmp};
        s << moduleOp;
        dumpToFile(options, output.nextPipelineSummaryFilename(pipeline.getName(), ".mlir"), tmp);
    }
    return success();
}

LogicalResult runLowering(const CompilerOptions &options, MLIRContext *ctx, ModuleOp moduleOp,
                          CompilerOutput &output, TimingScope &timing)

{
    catalyst::utils::Timer<> timer{};

    auto pm = PassManager::on<ModuleOp>(ctx, PassManager::Nesting::Implicit);
    if (failed(preparePassManager(pm, options, output, timer, timing))) {
        llvm::errs() << "Failed to setup pass manager\n";
        return failure();
    }

    bool clHasIndividualPass = pm.size() > 0;
    bool clHasManualPipeline = !options.pipelinesCfg.empty();
    if (clHasIndividualPass && clHasManualPipeline) {
        llvm::errs() << "--catalyst-pipeline option can't be used with individual pass options "
                        "or -pass-pipeline.\n";
        return failure();
    }

    // If individual passes are configured, run them
    if (clHasIndividualPass) {
        if (options.dumpPassPipeline) {
            pm.dump();
            llvm::errs() << "\n";
        }
        return pm.run(moduleOp);
    }

    // If pipelines are not configured explicitly, use the catalyst default pipeline
    std::vector<Pipeline> UserPipeline =
        clHasManualPipeline ? options.pipelinesCfg : getDefaultPipeline();
    for (auto &pipeline : UserPipeline) {
        if (failed(catalyst::utils::Timer<>::timer(runPipeline, pipeline.getName(),
                                                   /* add_endl */ false, pm, options, output,
                                                   pipeline, clHasManualPipeline, moduleOp))) {
            return failure();
        }
        catalyst::utils::LinesCount::call(moduleOp);
    }
    return success();
}

LogicalResult verifyInputType(const CompilerOptions &options, InputType inType)
{
    if (inType == InputType::OTHER) {
        CO_MSG(options, Verbosity::Urgent, "Wrong or unsupported input\n");
        return failure();
    }
    if (options.loweringAction == Action::LLC && inType != InputType::LLVMIR) {
        CO_MSG(options, Verbosity::Urgent, "Expected LLVM IR input but received MLIR input.\n");
        return failure();
    }
    if (options.loweringAction < Action::LLC && inType != InputType::MLIR) {
        CO_MSG(options, Verbosity::Urgent, "Expected MLIR input but received LLVM IR input.\n");
        return failure();
    }
    return success();
}

LogicalResult QuantumDriverMain(const CompilerOptions &options, CompilerOutput &output,
                                DialectRegistry &registry)
{
    using timer = catalyst::utils::Timer<>;

    OpPrintingFlags opPrintingFlags{};
    if (options.useNameLocAsPrefix) {
        opPrintingFlags.printNameLocAsPrefix();
    }

    MLIRContext ctx(registry);
    ctx.printOpOnDiagnostic(true);
    ctx.printStackTraceOnDiagnostic(options.verbosity >= Verbosity::Debug);
    // TODO: FIXME:
    // Let's try to enable multithreading. Do not forget to protect the printing.
    ctx.disableMultithreading();
    // The transform dialect doesn't appear to load dependent dialects
    // fpr named passes.
    ctx.loadAllAvailableDialects();

    ScopedDiagnosticHandler scopedHandler(
        &ctx, [&](Diagnostic &diag) { diag.print(options.diagnosticStream); });

    llvm::LLVMContext llvmContext;
    std::shared_ptr<llvm::Module> llvmModule;

    llvm::raw_string_ostream outIRStream(output.outIR);

    auto moduleBuffer = llvm::MemoryBuffer::getMemBufferCopy(options.source, options.moduleName);
    auto sourceMgr = std::make_shared<llvm::SourceMgr>();
    sourceMgr->AddNewSourceBuffer(std::move(moduleBuffer), SMLoc());
    SourceMgrDiagnosticHandler sourceMgrHandler(*sourceMgr, &ctx, options.diagnosticStream);

    DefaultTimingManager tm;
    applyDefaultTimingManagerCLOptions(tm);
    TimingScope timing = tm.getRootScope();

    TimingScope parserTiming = timing.nest("Parser");
    OwningOpRef<ModuleOp> mlirModule =
        timer::timer(parseMLIRSource, "parseMLIRSource", /* add_endl */ false, &ctx, *sourceMgr);

    enum InputType inType = InputType::OTHER;
    if (mlirModule) {
        inType = InputType::MLIR;
        catalyst::utils::LinesCount::call(*mlirModule);
        output.isCheckpointFound = options.checkpointStage == "mlir";
    }
    else {
        llvm::SMDiagnostic err;
        llvmModule = timer::timer(parseLLVMSource, "parseLLVMSource", false, llvmContext,
                                  options.source, options.moduleName, err);

        if (!llvmModule) {
            err.print(options.moduleName.data(), options.diagnosticStream);
            CO_MSG(options, Verbosity::Urgent, "Failed to parse module as LLVM or MLIR source\n");
            return failure();
        }
        inType = InputType::LLVMIR;
        output.isCheckpointFound = options.checkpointStage == "LLVMIRTranslation";
        catalyst::utils::LinesCount::call(*llvmModule);
    }
    if (failed(verifyInputType(options, inType))) {
        return failure();
    }
    parserTiming.stop();

    // Enzyme always happens after O2Opt. If the checkpoint is O2Opt, enzymeRun must be set to
    // true so that the enzyme pass can be executed.
    bool enzymeRun = options.checkpointStage == "O2Opt";

    bool runAll = (options.loweringAction == Action::All);
    bool runOpt = (options.loweringAction == Action::OPT) || runAll;
    bool runTranslate = (options.loweringAction == Action::Translate) || runAll;
    bool runLLC = (options.loweringAction == Action::LLC) || runAll;

    if (runOpt && (inType == InputType::MLIR)) {
        TimingScope optTiming = timing.nest("Optimization");
        // TODO: The enzymeRun flag will not travel correctly in the case where different
        // stages of compilation are executed independently via the Catalyst CLI.
        // Ideally, It should be added to the IR via an attribute.
        enzymeRun = containsGradients(*mlirModule);
        if (failed(runLowering(options, &ctx, *mlirModule, output, optTiming))) {
            CO_MSG(options, Verbosity::Urgent, "Failed to lower MLIR module\n");
            return failure();
        }
        output.outIR.clear();
        if (options.keepIntermediate) {
            mlirModule->print(outIRStream, opPrintingFlags);
        }
        optTiming.stop();
    }

    if (runTranslate && (inType == InputType::MLIR)) {
        TimingScope translateTiming = timing.nest("Translate");
        llvmModule =
            timer::timer(translateModuleToLLVMIR, "translateModuleToLLVMIR",
                         /* add_endl */ false, *mlirModule, llvmContext, "LLVMDialectModule",
                         /* disableVerification */ true);
        if (!llvmModule) {
            CO_MSG(options, Verbosity::Urgent, "Failed to translate LLVM module\n");
            return failure();
        }

        inType = InputType::LLVMIR;
        catalyst::utils::LinesCount::call(*llvmModule);

        if (options.keepIntermediate) {
            output.setStage("LLVMIRTranslation");
            std::string tmp;
            llvm::raw_string_ostream rawStringOstream{tmp};
            llvmModule->print(rawStringOstream, nullptr);
            auto outFile = output.nextPipelineSummaryFilename("LLVMIRTranslation", ".ll");
            dumpToFile(options, outFile, tmp);
        }
        output.outIR.clear();
        if (options.keepIntermediate) {
            outIRStream << *llvmModule;
        }
        translateTiming.stop();
    }

    if (runLLC && (inType == InputType::LLVMIR)) {
        TimingScope llcTiming = timing.nest("llc");
        // Set data layout before LLVM passes or the default one is used.
        llvm::Triple targetTriple(llvm::sys::getDefaultTargetTriple());

        llvm::InitializeAllTargetInfos();
        llvm::InitializeAllTargets();
        llvm::InitializeAllTargetMCs();
        llvm::InitializeAllAsmParsers();
        llvm::InitializeAllAsmPrinters();

        std::string err;
        auto target = llvm::TargetRegistry::lookupTarget(targetTriple, err);
        llvm::TargetOptions opt;
        const char *cpu = "generic";
        const char *features = "";
        auto targetMachine =
            target->createTargetMachine(targetTriple, cpu, features, opt, llvm::Reloc::Model::PIC_);
        targetMachine->setOptLevel(llvm::CodeGenOptLevel::None);
        llvmModule->setDataLayout(targetMachine->createDataLayout());
        llvmModule->setTargetTriple(targetTriple);

        if (options.asyncQnodes) {
            TimingScope coroLLVMPassesTiming = llcTiming.nest("LLVM coroutine passes");
            if (failed(timer::timer(runCoroLLVMPasses, "runCoroLLVMPasses", /* add_endl */ false,
                                    options, llvmModule, output))) {
                return failure();
            }
            coroLLVMPassesTiming.stop();
            catalyst::utils::LinesCount::call(*llvmModule.get());
        }

        if (enzymeRun) {
            TimingScope o2PassesTiming = llcTiming.nest("LLVM O2 passes");
            if (failed(timer::timer(runO2LLVMPasses, "runO2LLVMPasses", /* add_endl */ false,
                                    options, llvmModule, output))) {
                return failure();
            }
            o2PassesTiming.stop();
            catalyst::utils::LinesCount::call(*llvmModule.get());

            TimingScope enzymePassesTiming = llcTiming.nest("Enzyme passes");
            if (failed(timer::timer(runEnzymePasses, "runEnzymePasses", /* add_endl */ false,
                                    options, llvmModule, output))) {
                return failure();
            }
            enzymePassesTiming.stop();
            catalyst::utils::LinesCount::call(*llvmModule.get());
        }

        std::string errorMessage;
        auto outfile = openOutputFile(output.outputFilename, &errorMessage);
        if (output.outputFilename == "-" && llvmModule) {
            // Do not generate file if outputting to stdout.
            outfile->os() << *llvmModule;
            outfile->keep();
            // early exit
            return success();
        }

        TimingScope outputTiming = llcTiming.nest("compileObject");
        output.outIR.clear();
        if (options.keepIntermediate) {
            outIRStream << *llvmModule;
        }

        if (failed(timer::timer(compileObjectFile, "compileObjFile", /* add_endl */ true, options,
                                llvmModule, targetMachine, options.getObjectFile()))) {
            return failure();
        }
        outputTiming.stop();
        llcTiming.stop();
    }

    std::string errorMessage;
    auto outfile = openOutputFile(output.outputFilename, &errorMessage);
    if (!outfile) {
        llvm::errs() << errorMessage << "\n";
        return failure();
    }
    else if (output.outputFilename == "-" && llvmModule) {
        // already handled
    }
    else if (output.outputFilename == "-" && mlirModule) {
        mlirModule->print(outfile->os(), opPrintingFlags);
        outfile->keep();
    }

    if (options.keepIntermediate and output.outputFilename != "-") {
        outfile->os() << output.outIR;
        outfile->keep();
    }

    return success();
}

size_t findMatchingClosingParen(llvm::StringRef str, size_t openParenPos)
{
    int parenCount = 1;
    for (size_t pos = openParenPos + 1; pos < str.size(); pos++) {
        if (str[pos] == '(') {
            parenCount++;
        }
        else if (str[pos] == ')') {
            parenCount--;
            if (parenCount == 0) {
                return pos;
            }
        }
    }
    return llvm::StringRef::npos;
}

std::vector<Pipeline> parsePipelines(const cl::list<std::string> &catalystPipeline)
{
    std::vector<Pipeline> allPipelines;
    for (const auto &pipelineStr : catalystPipeline) {
        llvm::StringRef pipelineRef = llvm::StringRef(pipelineStr).trim();

        if (pipelineRef.empty()) {
            continue;
        }

        size_t openParenPos = pipelineRef.find('(');
        size_t closeParenPos = findMatchingClosingParen(pipelineRef, openParenPos);

        if (openParenPos == llvm::StringRef::npos || closeParenPos == llvm::StringRef::npos) {
            llvm::errs() << "Error: Invalid pipeline format: " << pipelineStr << "\n";
            continue;
        }

        // Extract pipeline name
        llvm::StringRef pipelineName = pipelineRef.slice(0, openParenPos).trim();
        llvm::StringRef passesStr = pipelineRef.slice(openParenPos + 1, closeParenPos).trim();
        llvm::SmallVector<llvm::StringRef, 8> passList;
        passesStr.split(passList, ';', /*MaxSplit=*/-1, /*KeepEmpty=*/false);

        llvm::SmallVector<std::string> passes;
        for (auto &pass : passList) {
            passes.push_back(pass.trim().str());
        }

        Pipeline pipeline;
        pipeline.setName(pipelineName.str());
        pipeline.setPasses(passes);
        allPipelines.push_back(std::move(pipeline));
    }
    return allPipelines;
}

int QuantumDriverMainFromCL(int argc, char **argv)
{
    // Command-line options

    // ATTENTION
    // ---------
    // Any modifications made to the command-line interface should be documented in
    // doc/catalyst-cli/catalyst-cli.rst
    cl::OptionCategory CatalystCat("Catalyst CLI Options", "");
    cl::opt<std::string> WorkspaceDir("workspace", cl::desc("Workspace directory"), cl::init("."),
                                      cl::cat(CatalystCat));
    cl::opt<std::string> ModuleName("module-name", cl::desc("Module name"),
                                    cl::init("catalyst_module"), cl::cat(CatalystCat));

    cl::opt<enum SaveTemps> SaveAfterEach(
        "save-ir-after-each", cl::desc("Keep intermediate files after each pass or pipeline"),
        cl::values(clEnumValN(SaveTemps::AfterPassChanged, "changed",
                              "Save IR after each pass (only if changed)")),
        cl::values(clEnumValN(SaveTemps::AfterPass, "pass",
                              "Save IR after each pass (even if unchanged)")),
        cl::values(clEnumValN(SaveTemps::AfterPipeline, "pipeline", "Save IR after each pipeline")),
        cl::init(SaveTemps::None), cl::cat(CatalystCat));
    cl::opt<bool> KeepIntermediate(
        "keep-intermediate", cl::desc("Keep intermediate files"), cl::init(false),
        cl::callback([&](const bool &) { SaveAfterEach.setValue(SaveTemps::AfterPipeline); }),
        cl::cat(CatalystCat));
    cl::opt<bool> UseNameLocAsPrefix("use-nameloc-as-prefix",
                                     cl::desc("Use name location as prefix"), cl::init(false),
                                     cl::cat(CatalystCat));
    cl::opt<bool> AsyncQNodes("async-qnodes", cl::desc("Enable asynchronous QNodes"),
                              cl::init(false), cl::cat(CatalystCat));
    cl::opt<bool> Verbose("verbose", cl::desc("Set verbose"), cl::init(false),
                          cl::cat(CatalystCat));
    cl::list<std::string> CatalystPipeline(
        "catalyst-pipeline", cl::desc("Catalyst Compiler pass pipelines"), cl::ZeroOrMore,
        cl::CommaSeparated, cl::cat(CatalystCat));
    cl::opt<std::string> CheckpointStage("checkpoint-stage", cl::desc("Checkpoint stage"),
                                         cl::init(""), cl::cat(CatalystCat));
    cl::opt<enum Action> LoweringAction(
        "tool", cl::desc("Select the tool to isolate"),
        cl::values(clEnumValN(Action::OPT, "opt", "run quantum-opt on the MLIR input")),
        cl::values(clEnumValN(Action::Translate, "translate",
                              "run mlir-translate on the MLIR LLVM dialect")),
        cl::values(clEnumValN(Action::LLC, "llc", "run llc on the llvm IR input")),
        cl::values(clEnumValN(Action::All, "all",
                              "run quantum-opt, mlir-translate, and llc on the MLIR input")),
        cl::init(Action::All), cl::cat(CatalystCat));
    cl::opt<bool> DumpPassPipeline("dump-catalyst-pipeline",
                                   cl::desc("Print the pipeline that will be run"), cl::init(false),
                                   cl::cat(CatalystCat));
    cl::opt<bool> DumpModuleScope("dump-module-scope",
                                  cl::desc("Print the whole module in intermediate files"),
                                  cl::init(true), cl::cat(CatalystCat));

    // Create dialect registry
    DialectRegistry registry;
    mlir::registerAllPasses();
    catalyst::registerAllPasses();
    registerAllCatalystPipelines();
    mlirRegisterAllStablehloPasses();
    mlir::stablehlo::registerOptimizationPasses();
    registerAllCatalystDialects(registry);
    registerLLVMTranslations(registry);

    // Register bufferization interfaces
    catalyst::registerBufferizableOpInterfaceExternalModels(registry);
    catalyst::gradient::registerBufferizableOpInterfaceExternalModels(registry);
    catalyst::quantum::registerBufferizableOpInterfaceExternalModels(registry);

    // Register and parse command line options.
    std::string inputFilename, outputFilename;
    std::string helpStr = "Catalyst Command Line Interface options. \n"
                          "Below, there is a complete list of options for the Catalyst CLI tool"
                          "In the first section, you can find the options that are used to"
                          "configure the Catalyst compiler. Next, you can find the options"
                          "specific to the mlir-opt tool.\n";
    std::tie(inputFilename, outputFilename) =
        registerAndParseCLIOptions(argc, argv, helpStr, registry);
    llvm::InitLLVM y(argc, argv);
    MlirOptMainConfig config = MlirOptMainConfig::createFromCLOptions();

    // Read the input IR file
    std::string source = readInputFile(inputFilename);
    if (source.empty()) {
        llvm::errs() << "Error: Unable to read input file: " << inputFilename << "\n";
        return 1;
    }

    std::unique_ptr<CompilerOutput> output(new CompilerOutput());
    assert(output);
    output->outputFilename = outputFilename;
    llvm::raw_string_ostream errStream{output->diagnosticMessages};

    CompilerOptions options{.source = source,
                            .workspace = WorkspaceDir,
                            .moduleName = ModuleName,
                            .diagnosticStream = errStream,
                            .keepIntermediate = SaveAfterEach,
                            .dumpModuleScope = DumpModuleScope,
                            .useNameLocAsPrefix = UseNameLocAsPrefix,
                            .asyncQnodes = AsyncQNodes,
                            .verbosity = Verbose ? Verbosity::All : Verbosity::Urgent,
                            .pipelinesCfg = parsePipelines(CatalystPipeline),
                            .checkpointStage = CheckpointStage,
                            .loweringAction = LoweringAction,
                            .dumpPassPipeline = DumpPassPipeline};

    mlir::LogicalResult result = QuantumDriverMain(options, *output, registry);

    errStream.flush();

    if (mlir::failed(result)) {
        llvm::errs() << "Compilation failed:\n" << output->diagnosticMessages << "\n";
        return 1;
    }

    if (Verbose)
        llvm::outs() << "Compilation successful:\n" << output->diagnosticMessages << "\n";
    return 0;
}
