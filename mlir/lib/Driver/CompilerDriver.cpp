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
#include "Driver/CompilerDriver.hpp"
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

namespace {

using namespace mlir;
using namespace catalyst;
using namespace catalyst::driver;
namespace cl = llvm::cl;

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

namespace catalyst::driver {
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
} // namespace catalyst::driver

// Determines if the compilation stage should be executed if a checkpointStage is given
bool catalyst::driver::shouldRunStage(const CompilerOptions &options, CompilerOutput &output,
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

llvm::LogicalResult catalyst::driver::runCoroLLVMPasses(const CompilerOptions &options,
                                                        std::shared_ptr<llvm::Module> llvmModule,
                                                        CompilerOutput &output)
{
    if (!catalyst::driver::shouldRunStage(options, output, "CoroOpt")) {
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

llvm::LogicalResult catalyst::driver::runO2LLVMPasses(const CompilerOptions &options,
                                                      std::shared_ptr<llvm::Module> llvmModule,
                                                      CompilerOutput &output)
{
    // opt -O2
    // As seen here:
    // https://llvm.org/docs/NewPassManager.html#just-tell-me-how-to-run-the-default-optimization-pipeline-with-the-new-pass-manager
    if (!catalyst::driver::shouldRunStage(options, output, "O2Opt")) {
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

llvm::LogicalResult catalyst::driver::runEnzymePasses(const CompilerOptions &options,
                                                      std::shared_ptr<llvm::Module> llvmModule,
                                                      CompilerOutput &output)
{
    if (!catalyst::driver::shouldRunStage(options, output, "Enzyme")) {
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

std::string catalyst::driver::readInputFile(const std::string &filename)
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

llvm::LogicalResult catalyst::driver::preparePassManager(PassManager &pm,
                                                         const CompilerOptions &options,
                                                         CompilerOutput &output,
                                                         catalyst::utils::Timer<> &timer,
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

llvm::LogicalResult catalyst::driver::configurePipeline(PassManager &pm,
                                                        const CompilerOptions &options,
                                                        Pipeline &pipeline,
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

llvm::LogicalResult catalyst::driver::runPipeline(PassManager &pm, const CompilerOptions &options,
                                                  CompilerOutput &output, Pipeline &pipeline,
                                                  bool clHasManualPipeline, ModuleOp moduleOp)
{
    if (!catalyst::driver::shouldRunStage(options, output, pipeline.getName()) ||
        pipeline.getPasses().size() == 0) {
        return success();
    }

    output.setStage(pipeline.getName());

    if (failed(catalyst::driver::configurePipeline(pm, options, pipeline, clHasManualPipeline))) {
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

llvm::LogicalResult catalyst::driver::runLowering(const CompilerOptions &options, MLIRContext *ctx,
                                                  ModuleOp moduleOp, CompilerOutput &output,
                                                  TimingScope &timing)

{
    catalyst::utils::Timer<> timer{};

    auto pm = PassManager::on<ModuleOp>(ctx, PassManager::Nesting::Implicit);
    if (failed(catalyst::driver::preparePassManager(pm, options, output, timer, timing))) {
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
        if (failed(catalyst::utils::Timer<>::timer(catalyst::driver::runPipeline,
                                                   pipeline.getName(),
                                                   /* add_endl */ false, pm, options, output,
                                                   pipeline, clHasManualPipeline, moduleOp))) {
            return failure();
        }
        catalyst::utils::LinesCount::call(moduleOp);
    }
    return success();
}

llvm::LogicalResult catalyst::driver::verifyInputType(const CompilerOptions &options,
                                                      InputType inType)
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

size_t catalyst::driver::findMatchingClosingParen(llvm::StringRef str, size_t openParenPos)
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

std::vector<Pipeline>
catalyst::driver::parsePipelines(const cl::list<std::string> &catalystPipeline)
{
    std::vector<Pipeline> allPipelines;
    for (const auto &pipelineStr : catalystPipeline) {
        llvm::StringRef pipelineRef = llvm::StringRef(pipelineStr).trim();

        if (pipelineRef.empty()) {
            continue;
        }

        size_t openParenPos = pipelineRef.find('(');
        size_t closeParenPos =
            catalyst::driver::findMatchingClosingParen(pipelineRef, openParenPos);

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

///////////////////////////////////////////////////////////////////////////////////////////

std::string CompilerOptions::getObjectFile() const
{
    using path = std::filesystem::path;
    return path(workspace.str()) / path(moduleName.str() + ".o");
}

std::string CompilerOutput::nextPassDumpFilename(const std::string &pipelineName,
                                                 const std::string &ext)
{
    return std::filesystem::path(currentStage) /
           std::filesystem::path(std::to_string(this->passCounter++) + "_" + pipelineName)
               .replace_extension(ext);
}

std::string CompilerOutput::nextPipelineSummaryFilename(const std::string &pipelineName,
                                                        const std::string &ext)
{
    return std::filesystem::path(std::to_string(this->globalPipelineCounter) + "_After" +
                                 pipelineName)
        .replace_extension(ext);
}

void CompilerOutput::setStage(const std::string &stageName)
{
    ++globalPipelineCounter;
    currentStage = std::to_string(globalPipelineCounter) + "_" + stageName;
    passCounter = 1;
}
