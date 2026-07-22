// Copyright 2026 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringExtras.h" // llvm::join
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/Triple.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Target/LLVMIR/Import.h" // translateDataLayout

#include "Catalyst/IR/CatalystDialect.h"
#include "Catalyst/IR/CatalystOps.h"
#include "Driver/DefaultPipelines/DefaultPipelines.h"
#include "Gradient/IR/GradientDialect.h"
#include "Ion/IR/IonDialect.h"
#include "MBQC/IR/MBQCDialect.h"
#include "Mitigation/IR/MitigationDialect.h"
#include "PBC/IR/PBCDialect.h"
#include "PauliFrame/IR/PauliFrameDialect.h"
#include "QRef/IR/QRefDialect.h"
#include "QecLogical/IR/QecLogicalDialect.h"
#include "QecPhysical/IR/QecPhysicalDialect.h"
#include "Quantum/IR/QuantumDialect.h"
#include "RTIO/IR/RTIODialect.h"

using namespace mlir;

namespace catalyst {

#define GEN_PASS_DECL_CROSSCOMPILETARGETSPASS
#define GEN_PASS_DEF_CROSSCOMPILETARGETSPASS
#include "Catalyst/Transforms/Passes.h.inc"

namespace {

// Make `fn` externally callable through the C ABI
void exposeEntryViaCInterface(func::FuncOp fn)
{
    OpBuilder builder(fn.getContext());
    fn.setVisibility(SymbolTable::Visibility::Public);
    fn->setAttr("llvm.emit_c_interface", builder.getUnitAttr());
    fn->removeAttr("llvm.linkage");
}

// The default lowering applied to a catalyst.target module: bufferization +
// LLVM-dialect lowering, reusing the same stage definitions as the host pipeline.
std::vector<std::string> defaultLoweringPassList()
{
    auto buf = driver::getBufferizationStage();
    auto llvmPasses = driver::getLLVMDialectLoweringStage();
    std::vector<std::string> passes;
    passes.reserve(buf.size() + llvmPasses.size());
    passes.insert(passes.end(), buf.begin(), buf.end());
    for (const auto &passName : llvmPasses) {
        if (passName != "convert-executor-to-llvm") {
            passes.push_back(passName);
        }
    }
    return passes;
}

// Lower a *local* (non-dispatch) target module. Its cross-compiled object is statically linked into
// the final binary, so each host-side launch_kernel into it is rewritten to a flat func.call
// against an external declaration of the entry (resolved at link time by the object's native
// symbol). The object path is recorded on the root module for the linker, and the now-empty module
// is erased.
LogicalResult lowerLocalTargetCalls(ModuleOp host, ModuleOp nested,
                                    SmallVectorImpl<Attribute> &objectFiles)
{
    MLIRContext *ctx = host.getContext();

    // A statically-linked object must be built for the host architecture. A different triple can
    // only be reached via executor dispatch.
    if (auto targetAttr = nested->getAttrOfType<DictionaryAttr>("catalyst.target")) {
        if (auto triple = targetAttr.getAs<StringAttr>("triple")) {
            if (!triple.getValue().empty() &&
                triple.getValue() != llvm::sys::getDefaultTargetTriple()) {
                nested.emitError("local (non-executor) target must use the host triple '")
                    << llvm::sys::getDefaultTargetTriple()
                    << "'; use executor() to dispatch a cross-compiled target";
                return failure();
            }
        }
    }

    StringRef moduleName = nested.getSymName().value_or("");
    SmallVector<catalyst::LaunchKernelOp> launches;
    host.walk([&](catalyst::LaunchKernelOp launchKernel) {
        if (launchKernel.getCalleeModuleName().getValue() == moduleName) {
            launches.push_back(launchKernel);
        }
    });
    for (catalyst::LaunchKernelOp launchKernel : launches) {
        StringRef entry = launchKernel.getCalleeName().getValue();
        // One external declaration per entry, resolved against the linked object's native symbol.
        auto decl = host.lookupSymbol<func::FuncOp>(entry);
        if (!decl) {
            OpBuilder rootBuilder(host.getBody(), host.getBody()->begin());
            auto fnTy = FunctionType::get(ctx, launchKernel.getOperandTypes(),
                                          launchKernel.getResultTypes());
            decl = func::FuncOp::create(rootBuilder, launchKernel.getLoc(), entry, fnTy);
            decl.setPrivate();
        }
        OpBuilder callBuilder(launchKernel);
        auto call = func::CallOp::create(callBuilder, launchKernel.getLoc(), decl,
                                         launchKernel.getOperands());
        launchKernel.replaceAllUsesWith(call.getResults());
        launchKernel.erase();
    }

    if (auto objPath = nested->getAttrOfType<StringAttr>("catalyst.object_file")) {
        objectFiles.push_back(objPath);
    }
    nested.erase();
    return success();
}

struct CrossCompileTargetsPass : impl::CrossCompileTargetsPassBase<CrossCompileTargetsPass> {
    using CrossCompileTargetsPassBase::CrossCompileTargetsPassBase;

    void getDependentDialects(DialectRegistry &registry) const override
    {
        mlir::registerLLVMDialectTranslation(registry);
        mlir::registerBuiltinDialectTranslation(registry);
        mlir::registerAllDialects(registry);
        registry.insert<catalyst::CatalystDialect, catalyst::quantum::QuantumDialect,
                        catalyst::qref::QRefDialect, catalyst::pbc::PBCDialect,
                        catalyst::gradient::GradientDialect, catalyst::mbqc::MBQCDialect,
                        catalyst::mitigation::MitigationDialect,
                        catalyst::pauli_frame::PauliFrameDialect, catalyst::ion::IonDialect,
                        catalyst::rtio::RTIODialect, catalyst::qecl::QecLogicalDialect,
                        catalyst::qecp::QecPhysicalDialect>();
    }

    void runOnOperation() final
    {
        ModuleOp host = getOperation();

        SmallVector<ModuleOp> targetMods;
        for (auto &op : host.getBody()->getOperations()) {
            if (auto mod = dyn_cast<ModuleOp>(&op)) {
                if (mod->hasAttr("catalyst.target")) {
                    targetMods.push_back(mod);
                }
            }
        }

        if (targetMods.empty()) {
            return;
        }

        if (workspace.empty()) {
            host.emitError("Missing `workspace` option for target cross-compilation");
            return signalPassFailure();
        }

        llvm::InitializeAllTargetInfos();
        llvm::InitializeAllTargets();
        llvm::InitializeAllTargetMCs();
        llvm::InitializeAllAsmParsers();
        llvm::InitializeAllAsmPrinters();

        // Compile each target module to an object, then dispose of it per delivery mode:
        //  - executor (catalyst.dispatch): leave the module intact for dispatch-executor-targets.
        //  - local: statically link — flatten its host calls and record the object for the linker.
        SmallVector<Attribute> localObjectFiles;
        for (auto nested : targetMods) {
            FailureOr<std::string> objPath = compileTargetModule(nested);
            if (failed(objPath)) {
                return signalPassFailure();
            }
            nested->setAttr("catalyst.object_file", StringAttr::get(&getContext(), *objPath));
            if (!nested->hasAttr("catalyst.dispatch")) {
                if (failed(lowerLocalTargetCalls(host, nested, localObjectFiles))) {
                    return signalPassFailure();
                }
            }
        }

        // Record the objects to statically link, for the driver to hand to the linker.
        if (!localObjectFiles.empty()) {
            host->setAttr("catalyst.object_files", ArrayAttr::get(&getContext(), localObjectFiles));
        }
    }

    // Returns the kernel-specific subdirectory {workspace}/{name}/, creating it if needed.
    std::string makeKernelDir(StringRef name)
    {
        llvm::SmallString<128> dir(workspace);
        llvm::sys::path::append(dir, name);
        (void)llvm::sys::fs::create_directories(dir);
        return std::string(dir.str());
    }

    // Write `op`/`mod` to {dir}/{filename} (used only when dump-intermediate is set).
    void dumpMLIR(mlir::Operation *op, StringRef dir, StringRef filename)
    {
        llvm::SmallString<128> path(dir);
        llvm::sys::path::append(path, filename);
        std::error_code ec;
        llvm::raw_fd_ostream os(path, ec, llvm::sys::fs::OF_None);
        if (!ec) {
            op->print(os);
        }
    }

    void dumpLLVMIR(llvm::Module &mod, StringRef dir, StringRef filename)
    {
        llvm::SmallString<128> path(dir);
        llvm::sys::path::append(path, filename);
        std::error_code ec;
        llvm::raw_fd_ostream os(path, ec, llvm::sys::fs::OF_None);
        if (!ec) {
            mod.print(os, nullptr);
        }
    }

    /**
     * @brief Build a TargetMachine for the given triple (generic CPU, no extra features).
     *
     * @return The TargetMachine, or nullptr (with a diagnostic on stderr) if the
     *         triple is not available in this LLVM build.
     */
    std::unique_ptr<llvm::TargetMachine> createTargetMachine(StringRef triple)
    {
        llvm::Triple parsedTriple{triple};
        std::string err;
        const llvm::Target *llvmTarget = llvm::TargetRegistry::lookupTarget(parsedTriple, err);
        if (!llvmTarget) {
            llvm::errs() << "Target triple '" << triple
                         << "' not registered in this LLVM build: " << err << "\n";
            return nullptr;
        }
        llvm::TargetOptions opt;
        std::unique_ptr<llvm::TargetMachine> targetMachine(llvmTarget->createTargetMachine(
            parsedTriple, /*cpu=*/"generic", /*features=*/"", opt, llvm::Reloc::Model::PIC_));
        if (!targetMachine) {
            llvm::errs() << "Could not create TargetMachine for triple '" << triple << "'\n";
            return nullptr;
        }
        targetMachine->setOptLevel(llvm::CodeGenOptLevel::Aggressive);
        return targetMachine;
    }

    /**
     * @brief Emit a `.o` file from an LLVM module using a prepared TargetMachine.
     *
     * The module's triple and data layout were stamped on the source MLIR module before
     * lowering and carried into `llvmModule` by translateModuleToLLVMIR, so they already
     * match `targetMachine` — no need to re-set them here.
     *
     * @param llvmModule The LLVM module to emit.
     * @param name The name used as the object file's basename.
     * @param dir The directory to write the object file into.
     * @param targetMachine The target machine to emit for.
     * @return std::string The path to the emitted object file, or "" on failure.
     */
    std::string emitObjectFile(std::unique_ptr<llvm::Module> &&llvmModule, StringRef name,
                               StringRef dir, llvm::TargetMachine &targetMachine)
    {
        llvm::SmallString<128> p(dir);
        llvm::sys::path::append(p, name.str() + ".o");
        std::string objPath = std::string(p.str());

        std::error_code errCode;
        llvm::raw_fd_ostream dest(objPath, errCode, llvm::sys::fs::OF_None);
        if (errCode) {
            llvm::errs() << "Cannot open " << objPath << " for writing: " << errCode.message()
                         << "\n";
            return "";
        }
        llvm::legacy::PassManager codegenPM;
        if (targetMachine.addPassesToEmitFile(codegenPM, dest, nullptr,
                                              llvm::CodeGenFileType::ObjectFile)) {
            llvm::errs() << "TargetMachine cannot emit an object file\n";
            return "";
        }
        codegenPM.run(*llvmModule);
        dest.flush();
        return objPath;
    }

    // Cross-compile a catalyst.target module to a `.o` for its triple and return the path.
    FailureOr<std::string> compileTargetModule(ModuleOp nested)
    {
        MLIRContext *ctx = &getContext();
        StringRef name = nested.getSymName().value_or("unnamed");

        auto targetAttr = nested->getAttrOfType<DictionaryAttr>("catalyst.target");
        if (!targetAttr) {
            nested.emitError("catalyst.target module missing catalyst.target dict attr");
            return failure();
        }

        // Triple selection: the optional 'triple' key on the catalyst.target dict
        // wins; otherwise the host triple.
        std::string moduleTarget;
        if (auto tripleAttr = targetAttr.getAs<StringAttr>("triple")) {
            moduleTarget = tripleAttr.getValue().str();
        }
        if (moduleTarget.empty()) {
            moduleTarget = llvm::sys::getDefaultTargetTriple();
        }

        std::string kernelDir = makeKernelDir(name);

        std::unique_ptr<llvm::TargetMachine> targetMachine = createTargetMachine(moduleTarget);
        if (!targetMachine) {
            nested.emitError("failed to create target machine for triple '" + moduleTarget +
                             "' (target module: " + name.str() + ")");
            return failure();
        }
        llvm::DataLayout dataLayout = targetMachine->createDataLayout();

        // Clone the target module into an unparented root module: leaves `nested` intact in the
        // host (its host launch_kernel is consumed later by local flattening or dispatch) and gives
        // the sub-pipeline / translateModuleToLLVMIR a top-level module to operate on.
        OpBuilder builder(ctx);
        mlir::OwningOpRef<mlir::ModuleOp> standalone(cast<ModuleOp>(nested->clone()));

        Operation *moduleOp = standalone->getOperation();
        moduleOp->setAttr(LLVM::LLVMDialect::getTargetTripleAttrName(),
                          builder.getStringAttr(moduleTarget));
        moduleOp->setAttr(LLVM::LLVMDialect::getDataLayoutAttrName(),
                          builder.getStringAttr(dataLayout.getStringRepresentation()));
        moduleOp->setAttr(DLTIDialect::kDataLayoutAttrName,
                          mlir::translateDataLayout(dataLayout, ctx));

        // The entry points are exactly the functions the host calls into this module, named by the
        // surviving launch_kernel call edges. Expose those through the C ABI; privatize the rest so
        // they can be internalized / DCE'd and don't leak as exported symbols.
        StringRef moduleName = nested.getSymName().value_or("");
        llvm::SmallSet<StringRef, 8> entries;
        if (auto host = nested->getParentOfType<ModuleOp>()) {
            host.walk([&](catalyst::LaunchKernelOp launchKernel) {
                if (launchKernel.getCalleeModuleName().getValue() == moduleName) {
                    entries.insert(launchKernel.getCalleeName().getValue());
                }
            });
        }
        for (auto fn : standalone->getOps<func::FuncOp>()) {
            if (entries.contains(fn.getName())) {
                exposeEntryViaCInterface(fn);
            }
            else {
                fn.setPrivate();
            }
        }

        if (dumpIntermediate) {
            dumpMLIR(*standalone, kernelDir, "extracted.mlir");
        }

        // Lower the extracted module. A 'pipeline' key on catalyst.target selects a named
        // target-lowering pipeline resolved against the host pipeline registry.
        // Without it, fall back to the default bufferization + LLVM-dialect lowering.
        std::string pipelineSpec;
        if (auto pipelineAttr = targetAttr.getAs<StringAttr>("pipeline")) {
            pipelineSpec = pipelineAttr.getValue().str();
        }
        if (pipelineSpec.empty()) {
            pipelineSpec = llvm::join(defaultLoweringPassList(), ",");
        }
        PassManager subPM(ctx);
        if (failed(parsePassPipeline(pipelineSpec, subPM))) {
            nested.emitError("failed to build the target-lowering pipeline '" + pipelineSpec + "'");
            return failure();
        }
        if (failed(subPM.run(*standalone))) {
            nested.emitError("failed to lower target module to LLVM dialect: " + name.str());
            return failure();
        }

        // Translate to LLVM IR and emit the object file.
        llvm::LLVMContext llvmCtx;
        std::unique_ptr<llvm::Module> llvmModule =
            translateModuleToLLVMIR(*standalone, llvmCtx, name);
        if (!llvmModule) {
            nested.emitError("failed to translate target module to LLVM IR: " + name.str());
            return failure();
        }
        if (dumpIntermediate) {
            dumpLLVMIR(*llvmModule, kernelDir, name.str() + ".ll");
        }
        std::string objPath =
            emitObjectFile(std::move(llvmModule), name, kernelDir, *targetMachine);
        if (objPath.empty()) {
            nested.emitError("failed to emit object file for target module: " + name.str());
            return failure();
        }
        return objPath;
    }
};

} // namespace

} // namespace catalyst
