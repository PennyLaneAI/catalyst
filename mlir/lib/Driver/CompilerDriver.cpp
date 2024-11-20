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

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <list>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>

#include "mhlo/IR/register.h"
#include "mhlo/transforms/passes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "stablehlo/dialect/Register.h"
#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/FileSystem.h"
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

#include "Catalyst/IR/CatalystDialect.h"
#include "Catalyst/Transforms/Passes.h"
#include "Driver/CatalystLLVMTarget.h"
#include "Driver/CompilerDriver.h"
#include "Driver/Pipelines.h"
#include "Driver/Support.h"
#include "Gradient/IR/GradientDialect.h"
#include "Gradient/IR/GradientInterfaces.h"
#include "Gradient/Transforms/Passes.h"
#include "Mitigation/IR/MitigationDialect.h"
#include "Mitigation/Transforms/Passes.h"
#include "Quantum/IR/QuantumDialect.h"
#include "Quantum/Transforms/Passes.h"

#include "Enzyme.h"
#include "Timer.hpp"

using namespace mlir;
using namespace catalyst;
using namespace catalyst::driver;
namespace cl = llvm::cl;

namespace catalyst::utils {

/**
 * LinesCount : A utility class to count the number of lines of embedded programs
 * in different compilation stages.
 *
 * You can dump the program-size embedded in an `Operation`, `ModuleOp`, or
 * `llvm::Module` using the static methods in this class.
 *
 * To display results, run the driver with the `ENABLE_DIAGNOSTICS=ON` variable.
 * To store results in YAML format, use `DIAGNOSTICS_RESULTS_PATH=/path/to/file.yml`
 * along with `ENABLE_DIAGNOSTICS=ON`.
 */
class LinesCount {
  private:
    inline static void print(const std::string &opStrBuf, const std::string &name)
    {
        const auto num_lines = std::count(opStrBuf.cbegin(), opStrBuf.cend(), '\n');
        if (!name.empty()) {
            std::cerr << "[DIAGNOSTICS] After " << std::setw(25) << std::left << name;
        }
        std::cerr << "\t" << std::fixed << "programsize: " << num_lines << std::fixed << " lines\n";
    }

    inline static void store(const std::string &opStrBuf, const std::string &name,
                             const std::filesystem::path &file_path)
    {
        const auto num_lines = std::count(opStrBuf.cbegin(), opStrBuf.cend(), '\n');

        const std::string_view key_padding = "          ";
        const std::string_view val_padding = "              ";

        if (!std::filesystem::exists(file_path)) {
            std::ofstream ofile(file_path);
            assert(ofile.is_open() && "Invalid file to store timer results");
            if (!name.empty()) {
                ofile << key_padding << "- " << name << ":\n";
            }
            ofile << val_padding << "programsize: " << num_lines << "\n";
            ofile.close();
            return;
        }
        // else

        // Second, update the file
        std::ofstream ofile(file_path, std::ios::app);
        assert(ofile.is_open() && "Invalid file to store timer results");
        if (!name.empty()) {
            ofile << key_padding << "- " << name << ":\n";
        }
        ofile << val_padding << "programsize: " << num_lines << "\n";
        ofile.close();
    }

    inline static void dump(const std::string &opStrBuf, const std::string &name = {})
    {
        char *file = getenv("DIAGNOSTICS_RESULTS_PATH");
        if (!file) {
            print(opStrBuf, name);
            return;
        }
        // else
        store(opStrBuf, name, std::filesystem::path{file});
    }

  public:
    [[nodiscard]] inline static bool is_diagnostics_enabled()
    {
        char *value = getenv("ENABLE_DIAGNOSTICS");
        return value && std::string(value) == "ON";
    }

    static void Operation(Operation *op, const std::string &name = {})
    {
        if (!is_diagnostics_enabled()) {
            return;
        }

        std::string opStrBuf;
        llvm::raw_string_ostream rawStrBef{opStrBuf};
        rawStrBef << *op;

        dump(opStrBuf, name);
    }

    static void ModuleOp(const ModuleOp &op, const std::string &name = {})
    {
        if (!is_diagnostics_enabled()) {
            return;
        }

        std::string modStrBef;
        llvm::raw_string_ostream rawStrBef{modStrBef};
        op->print(rawStrBef);

        dump(modStrBef, name);
    }

    static void Module(const llvm::Module &llvmModule, const std::string &name = {})
    {
        if (!is_diagnostics_enabled()) {
            return;
        }

        std::string modStrBef;
        llvm::raw_string_ostream rawStrBef{modStrBef};
        llvmModule.print(rawStrBef, nullptr);

        dump(modStrBef, name);
    }
};

} // namespace catalyst::utils

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

struct CatalystPassInstrumentation : public PassInstrumentation {
    typedef std::function<void(Pass *pass, Operation *operation)> PassCallback;
    PassCallback beforePassCallback;
    PassCallback afterPassCallback;
    PassCallback afterPassFailedCallback;

    CatalystPassInstrumentation(PassCallback beforePassCallback, PassCallback afterPassCallback,
                                PassCallback afterPassFailedCallback)
        : beforePassCallback(beforePassCallback), afterPassCallback(afterPassCallback),
          afterPassFailedCallback(afterPassFailedCallback)
    {
    }

    void runBeforePass(Pass *pass, Operation *operation) override
    {
        this->beforePassCallback(pass, operation);
    }

    void runAfterPass(Pass *pass, Operation *operation) override
    {
        this->afterPassCallback(pass, operation);
    }

    void runAfterPassFailed(Pass *pass, Operation *operation) override
    {
        this->afterPassFailedCallback(pass, operation);
    }
};

} // namespace

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

    // HLO
    mhlo::registerAllMhloDialects(registry);
    stablehlo::registerAllDialects(registry);

    // Catalyst
    registry.insert<CatalystDialect>();
    registry.insert<quantum::QuantumDialect>();
    registry.insert<gradient::GradientDialect>();
    registry.insert<mitigation::MitigationDialect>();
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
        std::string tmp;
        llvm::raw_string_ostream rawStringOstream{tmp};
        llvmModule->print(rawStringOstream, nullptr);
        auto outFile = output.nextPipelineDumpFilename("CoroOpt", ".ll");
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
        std::string tmp;
        llvm::raw_string_ostream rawStringOstream{tmp};
        llvmModule->print(rawStringOstream, nullptr);
        auto outFile = output.nextPipelineDumpFilename("O2Opt", ".ll");
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
        std::string tmp;
        llvm::raw_string_ostream rawStringOstream{tmp};
        llvmModule->print(rawStringOstream, nullptr);
        auto outFile = output.nextPipelineDumpFilename("Enzyme", ".ll");
        dumpToFile(options, outFile, tmp);
    }

    return success();
}

std::string readInputFile(const std::string &filename)
{
    std::ifstream file(filename);
    if (!file.is_open()) {
        return "";
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

LogicalResult preparePassManager(PassManager &pm, const CompilerOptions &options,
                                 CompilerOutput &output, catalyst::utils::Timer &timer,
                                 TimingScope &timing)
{
    auto beforePassCallback = [&](Pass *pass, Operation *op) {
        if (!timer.is_active()) {
            timer.start();
        }
    };

    // For each pipeline-terminating pass, print the IR into the corresponding dump file and
    // into a diagnostic output buffer. Note that one pass can terminate multiple pipelines.
    auto afterPassCallback = [&](Pass *pass, Operation *op) {
        auto pipelineName = pass->getName();
        timer.dump(pipelineName.str(), /*add_endl */ false);
        catalyst::utils::LinesCount::Operation(op);
        if (options.keepIntermediate >= SaveTemps::AfterPass) {
            std::string tmp;
            llvm::raw_string_ostream s{tmp};
            s << *op;
            dumpToFile(options, output.nextPipelineDumpFilename(pipelineName.str()), tmp);
        }
    };

    // For each failed pass, print the owner pipeline name into a diagnostic stream.
    auto afterPassFailedCallback = [&](Pass *pass, Operation *op) {
        options.diagnosticStream << "While processing '" << pass->getName().str() << "' pass ";
        std::string tmp;
        llvm::raw_string_ostream s{tmp};
        s << *op;
        if (options.keepIntermediate) {
            dumpToFile(options, output.nextPipelineDumpFilename(pass->getName().str() + "_FAILED"),
                       tmp);
        }
    };

    MlirOptMainConfig config = MlirOptMainConfig::createFromCLOptions();
    pm.enableVerifier(config.shouldVerifyPasses());
    if (failed(applyPassManagerCLOptions(pm)))
        return failure();
    if (failed(config.setupPassPipeline(pm)))
        return failure();
    pm.enableTiming(timing);
    pm.addInstrumentation(std::unique_ptr<PassInstrumentation>(new CatalystPassInstrumentation(
        beforePassCallback, afterPassCallback, afterPassFailedCallback)));
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

LogicalResult runLowering(const CompilerOptions &options, MLIRContext *ctx, ModuleOp moduleOp,
                          CompilerOutput &output, TimingScope &timing)

{
    if (options.keepIntermediate && options.checkpointStage.empty()) {
        std::string tmp;
        llvm::raw_string_ostream s{tmp};
        s << moduleOp;
        dumpToFile(options, output.nextPipelineDumpFilename(options.moduleName.str(), ".mlir"),
                   tmp);
    }

    catalyst::utils::Timer timer{};

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
        if (!shouldRunStage(options, output, pipeline.getName()) ||
            pipeline.getPasses().size() == 0) {
            continue;
        }
        if (failed(configurePipeline(pm, options, pipeline, clHasManualPipeline))) {
            llvm::errs() << "Failed to run pipeline: " << pipeline.getName() << "\n";
            return failure();
        }

        if (failed(pm.run(moduleOp)))
            return failure();

        if (options.keepIntermediate && options.checkpointStage.empty()) {
            std::string tmp;
            llvm::raw_string_ostream s{tmp};
            s << moduleOp;
            dumpToFile(options, output.nextPipelineDumpFilename(pipeline.getName(), ".mlir"), tmp);
        }
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
    using timer = catalyst::utils::Timer;

    MLIRContext ctx(registry);
    ctx.printOpOnDiagnostic(true);
    ctx.printStackTraceOnDiagnostic(options.verbosity >= Verbosity::Debug);
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
        catalyst::utils::LinesCount::ModuleOp(*mlirModule);
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
        output.isCheckpointFound = options.checkpointStage == "llvm_ir";
        catalyst::utils::LinesCount::Module(*llvmModule);
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
        // stages of compilation are executed independently via the catalyst-cli executable.
        // Ideally, It should be added to the IR via an attribute.
        enzymeRun = containsGradients(*mlirModule);
        if (failed(runLowering(options, &ctx, *mlirModule, output, optTiming))) {
            CO_MSG(options, Verbosity::Urgent, "Failed to lower MLIR module\n");
            return failure();
        }
        output.outIR.clear();
        outIRStream << *mlirModule;
        optTiming.stop();
    }

    if (runTranslate && (inType == InputType::MLIR)) {
        TimingScope translateTiming = timing.nest("Translate");
        llvmModule =
            timer::timer(translateModuleToLLVMIR, "translateModuleToLLVMIR",
                         /* add_endl */ false, *mlirModule, llvmContext, "LLVMDialectModule");
        if (!llvmModule) {
            CO_MSG(options, Verbosity::Urgent, "Failed to translate LLVM module\n");
            return failure();
        }

        inType = InputType::LLVMIR;
        catalyst::utils::LinesCount::Module(*llvmModule);

        if (options.keepIntermediate) {
            std::string tmp;
            llvm::raw_string_ostream rawStringOstream{tmp};
            llvmModule->print(rawStringOstream, nullptr);
            auto outFile = output.nextPipelineDumpFilename("llvm_ir", ".ll");
            dumpToFile(options, outFile, tmp);
        }
        output.outIR.clear();
        outIRStream << *llvmModule;
        translateTiming.stop();
    }

    if (runLLC && (inType == InputType::LLVMIR)) {
        TimingScope llcTiming = timing.nest("llc");
        // Set data layout before LLVM passes or the default one is used.
        std::string targetTriple = llvm::sys::getDefaultTargetTriple();

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

        catalyst::utils::LinesCount::Module(*llvmModule.get());

        TimingScope coroLLVMPassesTiming = llcTiming.nest("LLVM coroutine passes");
        if (options.asyncQnodes &&
            failed(timer::timer(runCoroLLVMPasses, "runCoroLLVMPasses", /* add_endl */ false,
                                options, llvmModule, output))) {
            return failure();
        }
        coroLLVMPassesTiming.stop();

        if (enzymeRun) {
            TimingScope o2PassesTiming = llcTiming.nest("LLVM O2 passes");
            if (failed(timer::timer(runO2LLVMPasses, "runO2LLVMPasses", /* add_endl */ false,
                                    options, llvmModule, output))) {
                return failure();
            }
            o2PassesTiming.stop();
            catalyst::utils::LinesCount::Module(*llvmModule.get());

            TimingScope enzymePassesTiming = llcTiming.nest("Enzyme passes");
            if (failed(timer::timer(runEnzymePasses, "runEnzymePasses", /* add_endl */ false,
                                    options, llvmModule, output))) {
                return failure();
            }
            enzymePassesTiming.stop();
            catalyst::utils::LinesCount::Module(*llvmModule.get());
        }

        TimingScope outputTiming = llcTiming.nest("compileObject");
        output.outIR.clear();
        outIRStream << *llvmModule;

        if (failed(timer::timer(compileObjectFile, "compileObjFile", /* add_endl */ true, options,
                                std::move(llvmModule), targetMachine, options.getObjectFile()))) {
            return failure();
        }
        outputTiming.stop();
        llcTiming.stop();
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
    cl::opt<std::string> WorkspaceDir("workspace", cl::desc("Workspace directory"), cl::init("."));
    cl::opt<std::string> ModuleName("module-name", cl::desc("Module name"),
                                    cl::init("catalyst_module"));

    cl::opt<enum SaveTemps> SaveAfterEach(
        "save-ir-after-each", cl::desc("Keep intermediate files after each pass or pipeline"),
        cl::values(clEnumValN(SaveTemps::AfterPass, "pass", "Save IR after each pass")),
        cl::values(clEnumValN(SaveTemps::AfterPipeline, "pipeline", "Save IR after each pipeline")),
        cl::init(SaveTemps::None));
    cl::opt<bool> KeepIntermediate(
        "keep-intermediate", cl::desc("Keep intermediate files"), cl::init(false),
        cl::callback([&](const bool &) { SaveAfterEach.setValue(SaveTemps::AfterPipeline); }));
    cl::opt<bool> AsyncQNodes("async-qnodes", cl::desc("Enable asynchronous QNodes"),
                              cl::init(false));
    cl::opt<bool> Verbose("verbose", cl::desc("Set verbose"), cl::init(false));
    cl::list<std::string> CatalystPipeline("catalyst-pipeline",
                                           cl::desc("Catalyst Compiler pass pipelines"),
                                           cl::ZeroOrMore, cl::CommaSeparated);
    cl::opt<std::string> CheckpointStage("checkpoint-stage", cl::desc("Checkpoint stage"),
                                         cl::init(""));
    cl::opt<enum Action> LoweringAction(
        "tool", cl::desc("Select the tool to isolate"),
        cl::values(clEnumValN(Action::OPT, "opt", "run quantum-opt on the MLIR input")),
        cl::values(clEnumValN(Action::Translate, "translate",
                              "run mlir-translate on the MLIR LLVM dialect")),
        cl::values(clEnumValN(Action::LLC, "llc", "run llc on the llvm IR input")),
        cl::values(clEnumValN(Action::All, "all",
                              "run quantum-opt, mlir-translate, and llc on the MLIR input")),
        cl::init(Action::All));
    cl::opt<bool> DumpPassPipeline(
        "dump-catalyst-pipeline", cl::desc("Print the pipeline that will be run"), cl::init(false));

    // Create dialect registry
    DialectRegistry registry;
    registerAllPasses();
    registerAllCatalystPasses();
    registerAllCatalystPipelines();
    mhlo::registerAllMhloPasses();
    registerAllCatalystDialects(registry);
    registerLLVMTranslations(registry);

    // Register and parse command line options.
    std::string inputFilename, outputFilename;
    std::tie(inputFilename, outputFilename) =
        registerAndParseCLIOptions(argc, argv, "quantum compiler", registry);
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

    // If not creating object file, output the IR to the specified file.
    std::string errorMessage;
    auto outfile = openOutputFile(outputFilename, &errorMessage);
    if (!outfile) {
        llvm::errs() << errorMessage << "\n";
        return 1;
    }
    outfile->os() << output->outIR;
    outfile->keep();
    if (Verbose)
        llvm::outs() << "Compilation successful:\n" << output->diagnosticMessages << "\n";
    return 0;
}
