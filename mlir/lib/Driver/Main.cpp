#include "Catalyst/Transforms/BufferizableOpInterfaceImpl.h"
#include "Driver/Support.h"
#include "Gradient/Transforms/BufferizableOpInterfaceImpl.h"
#include "Quantum/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/IR/BuiltinOps.h"  // mlir::ModuleOp
#include "mlir/IR/Diagnostics.h" // mlir::Diagnostic
#include "mlir/IR/MLIRContext.h" // mlir::MLIRContext
#include "mlir/IR/OwningOpRef.h" // mlir::OwningOpRef
#include "mlir/Support/Timing.h" // mlir::DefaultTimingManager, mlir::TimingScope
#include "llvm/IR/LLVMContext.h" // llvm::LLVMContext
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/LogicalResult.h" // llvm::LogicalResult
#include "llvm/Support/MemoryBuffer.h"  // llvm::MemoryBuffer
#include "llvm/Support/SMLoc.h"         // llvm::SMLoc
#include "llvm/Support/SourceMgr.h"     // llvm::SourceMgr
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/TargetParser/Host.h"
#include <llvm/ADT/STLForwardCompat.h>
#include <llvm/Support/ToolOutputFile.h>

#include <mlir/InitAllPasses.h>

#include "mlir/Target/LLVMIR/ModuleTranslation.h"

#include "mlir/Support/FileUtilities.h"

#include "stablehlo/integrations/c/StablehloPasses.h"

#include "RegisterAllPasses.h"

#include "Driver/CatalystLLVMTarget.h"
#include "Driver/CompilerDriver.hpp"
#include "Driver/LineUtils.hpp"
#include "Driver/Timer.hpp"
#include "stablehlo/transforms/optimization/Passes.h"

namespace {
using namespace catalyst;
using namespace catalyst::driver;
} // namespace

llvm::LogicalResult QuantumDriverMain(const CompilerOptions &options, CompilerOutput &output,
                                      mlir::DialectRegistry &registry)
{
    using timer = catalyst::utils::Timer<>;

    mlir::OpPrintingFlags opPrintingFlags{};
    if (options.useNameLocAsPrefix) {
        opPrintingFlags.printNameLocAsPrefix();
    }

    mlir::MLIRContext ctx(registry);
    ctx.printOpOnDiagnostic(true);
    ctx.printStackTraceOnDiagnostic(options.verbosity >= Verbosity::Debug);
    // TODO: FIXME:
    // Let's try to enable multithreading. Do not forget to protect the printing.
    ctx.disableMultithreading();
    // The transform dialect doesn't appear to load dependent dialects
    // fpr named passes.
    ctx.loadAllAvailableDialects();

    mlir::ScopedDiagnosticHandler scopedHandler(
        &ctx, [&](mlir::Diagnostic &diag) { diag.print(options.diagnosticStream); });

    llvm::LLVMContext llvmContext;
    std::shared_ptr<llvm::Module> llvmModule;

    llvm::raw_string_ostream outIRStream(output.outIR);

    auto moduleBuffer = llvm::MemoryBuffer::getMemBufferCopy(options.source, options.moduleName);
    auto sourceMgr = std::make_shared<llvm::SourceMgr>();
    sourceMgr->AddNewSourceBuffer(std::move(moduleBuffer), llvm::SMLoc());
    mlir::SourceMgrDiagnosticHandler sourceMgrHandler(*sourceMgr, &ctx, options.diagnosticStream);

    mlir::DefaultTimingManager tm;
    applyDefaultTimingManagerCLOptions(tm);
    mlir::TimingScope timing = tm.getRootScope();

    mlir::TimingScope parserTiming = timing.nest("Parser");
    mlir::OwningOpRef<mlir::ModuleOp> mlirModule =
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
            return llvm::failure();
        }
        inType = InputType::LLVMIR;
        output.isCheckpointFound = options.checkpointStage == "LLVMIRTranslation";
        catalyst::utils::LinesCount::call(*llvmModule);
    }
    if (failed(verifyInputType(options, inType))) {
        return llvm::failure();
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
        mlir::TimingScope optTiming = timing.nest("Optimization");
        // TODO: The enzymeRun flag will not travel correctly in the case where different
        // stages of compilation are executed independently via the Catalyst CLI.
        // Ideally, It should be added to the IR via an attribute.
        enzymeRun = containsGradients(*mlirModule);
        if (failed(runLowering(options, &ctx, *mlirModule, output, optTiming))) {
            CO_MSG(options, Verbosity::Urgent, "Failed to lower MLIR module\n");
            return llvm::failure();
        }
        output.outIR.clear();
        if (options.keepIntermediate) {
            mlirModule->print(outIRStream, opPrintingFlags);
        }
        optTiming.stop();
    }

    if (runTranslate && (inType == InputType::MLIR)) {
        mlir::TimingScope translateTiming = timing.nest("Translate");
        llvmModule =
            timer::timer(mlir::translateModuleToLLVMIR, "translateModuleToLLVMIR",
                         /* add_endl */ false, *mlirModule, llvmContext, "LLVMDialectModule",
                         /* disableVerification */ true);
        if (!llvmModule) {
            CO_MSG(options, Verbosity::Urgent, "Failed to translate LLVM module\n");
            return llvm::failure();
        }

        inType = InputType::LLVMIR;
        catalyst::utils::LinesCount::call(*llvmModule);

        if (options.keepIntermediate) {
            output.setStage("LLVMIRTranslation");
            std::string tmp;
            llvm::raw_string_ostream rawStringOstream{tmp};
            llvmModule->print(rawStringOstream, nullptr);
            auto outFile = output.nextPipelineSummaryFilename("LLVMIRTranslation", ".ll");
            catalyst::driver::dumpToFile(options, outFile, tmp);
        }
        output.outIR.clear();
        if (options.keepIntermediate) {
            outIRStream << *llvmModule;
        }
        translateTiming.stop();
    }

    if (runLLC && (inType == InputType::LLVMIR)) {
        mlir::TimingScope llcTiming = timing.nest("llc");
        // Set data layout before LLVM passes or the default one is used.
        llvm::Triple targetTriple{llvm::sys::getDefaultTargetTriple()};

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
            mlir::TimingScope coroLLVMPassesTiming = llcTiming.nest("LLVM coroutine passes");
            if (failed(timer::timer(runCoroLLVMPasses, "runCoroLLVMPasses", /* add_endl */ false,
                                    options, llvmModule, output))) {
                return llvm::failure();
            }
            coroLLVMPassesTiming.stop();
            catalyst::utils::LinesCount::call(*llvmModule.get());
        }

        if (enzymeRun) {
            mlir::TimingScope o2PassesTiming = llcTiming.nest("LLVM O2 passes");
            if (failed(timer::timer(runO2LLVMPasses, "runO2LLVMPasses", /* add_endl */ false,
                                    options, llvmModule, output))) {
                return llvm::failure();
            }
            o2PassesTiming.stop();
            catalyst::utils::LinesCount::call(*llvmModule.get());

            mlir::TimingScope enzymePassesTiming = llcTiming.nest("Enzyme passes");
            if (failed(timer::timer(runEnzymePasses, "runEnzymePasses", /* add_endl */ false,
                                    options, llvmModule, output))) {
                return llvm::failure();
            }
            enzymePassesTiming.stop();
            catalyst::utils::LinesCount::call(*llvmModule.get());
        }

        std::string errorMessage;
        auto outfile = mlir::openOutputFile(output.outputFilename, &errorMessage);
        if (output.outputFilename == "-" && llvmModule) {
            // Do not generate file if outputting to stdout.
            outfile->os() << *llvmModule;
            outfile->keep();
            // early exit
            return llvm::success();
        }

        mlir::TimingScope outputTiming = llcTiming.nest("compileObject");
        output.outIR.clear();
        if (options.keepIntermediate) {
            outIRStream << *llvmModule;
        }

        if (failed(timer::timer(catalyst::driver::compileObjectFile, "compileObjFile",
                                /* add_endl */ true, options, llvmModule, targetMachine,
                                options.getObjectFile()))) {
            return llvm::failure();
        }
        outputTiming.stop();
        llcTiming.stop();
    }

    std::string errorMessage;
    auto outfile = mlir::openOutputFile(output.outputFilename, &errorMessage);
    if (!outfile) {
        llvm::errs() << errorMessage << "\n";
        return llvm::failure();
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

    return llvm::success();
}

int QuantumDriverMainFromCL(int argc, char **argv)
{
    namespace cl = llvm::cl;
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
    mlir::DialectRegistry registry;
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
        return llvm::to_underlying(ErrorCode::Failure);
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
        return llvm::to_underlying(ErrorCode::Failure);
    }

    if (Verbose)
        llvm::outs() << "Compilation successful:\n" << output->diagnosticMessages << "\n";
    return llvm::to_underlying(ErrorCode::Success);
}