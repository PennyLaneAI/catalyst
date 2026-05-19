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
#include "llvm/ADT/StringExtras.h"
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
#include "llvm/TargetParser/Triple.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"

#include "stablehlo/dialect/StablehloOps.h"

#include "Catalyst/IR/CatalystOps.h"
#include "Driver/DefaultPipelines/DefaultPipelines.h"
#include "Gradient/IR/GradientDialect.h"
#include "PBC/IR/PBCDialect.h"
#include "Quantum/IR/QuantumDialect.h"

using namespace mlir;

namespace catalyst {

#define GEN_PASS_DECL_CROSSCOMPILEREMOTEKERNELSPASS
#define GEN_PASS_DEF_CROSSCOMPILEREMOTEKERNELSPASS
#include "Catalyst/Transforms/Passes.h.inc"

namespace {

// Writes intermediate MLIR after each top-level pass in the target-module sub-PM.
// Files land in {targetDir}/lowering/{N}_{pass-arg}.mlir.
struct TargetModuleLoweringInstrumentation : public mlir::PassInstrumentation {
    std::string loweringDir;
    int passIdx = 0;

    TargetModuleLoweringInstrumentation(StringRef targetDir)
    {
        llvm::SmallString<128> dir(targetDir);
        llvm::sys::path::append(dir, "lowering");
        (void)llvm::sys::fs::create_directories(dir);
        loweringDir = std::string(dir.str());
    }

    void runAfterPass(mlir::Pass *pass, mlir::Operation *op) override
    {
        if (!isa<mlir::ModuleOp>(op))
            return;
        ++passIdx;
        std::string arg = pass->getArgument().str();
        if (arg.empty())
            arg = "adaptor";
        llvm::SmallString<128> path(loweringDir);
        llvm::sys::path::append(path, std::to_string(passIdx) + "_" + arg + ".mlir");
        std::error_code ec;
        llvm::raw_fd_ostream os(path, ec, llvm::sys::fs::OF_None);
        if (!ec)
            op->print(os);
    }
};

/**
 * @brief Prepare a function for cross-compilation into a standalone remote kernel.
 *
 * The conversion takes place in the following steps:
 *   1. Set public visibility so the symbol is exported in the emitted .o and the
 *      ORC JIT executor can look it up by name at load time.
 *   2. Add `llvm.emit_c_interface` so the runtime can call the function via ORC
 *      without knowing the LLVM calling convention.
 *   3. Strip `llvm.linkage` — internal linkage would hide the symbol outside the
 *      .o; removing it gives the function default (external) linkage.
 *
 * @example
 * Before sanitization:
 * ```mlir
 * func.func @circuit(%arg0: memref<1xi64>) -> memref<f64>
 *     attributes {llvm.linkage = #llvm.linkage<internal>} {
 *   return %0 : memref<f64>
 * }
 * ```
 * After sanitization:
 * ```mlir
 * func.func public @circuit(%arg0: memref<1xi64>) -> memref<f64>
 *     attributes {llvm.emit_c_interface} {
 *   return %0 : memref<f64>
 * }
 * ```
 *
 * @param fn The function to sanitize.
 */
void sanitizeForRemoteExecution(func::FuncOp fn)
{
    OpBuilder builder(fn.getContext());
    fn.setVisibility(SymbolTable::Visibility::Public);
    fn->setAttr("llvm.emit_c_interface", builder.getUnitAttr());
    fn->removeAttr("llvm.linkage");
}

struct CrossCompileRemoteKernelsPass
    : impl::CrossCompileRemoteKernelsPassBase<CrossCompileRemoteKernelsPass> {
    using CrossCompileRemoteKernelsPassBase::CrossCompileRemoteKernelsPassBase;

    // Declare all dialects the LLVM-dialect-lowering sub-PM uses so they are
    // pre-instantiated before the outer pass manager enters multi-threaded mode.
    // Lazy dialect loading is forbidden in a multi-threaded context, and this pass
    // runs a nested PassManager. Note that this is mostly for testing purposes using quantum-opt.
    void getDependentDialects(DialectRegistry &registry) const override
    {
        mlir::registerLLVMDialectTranslation(registry);
        mlir::registerBuiltinDialectTranslation(registry);
        registry.insert<
            affine::AffineDialect,
            arith::ArithDialect,
            bufferization::BufferizationDialect,
            cf::ControlFlowDialect,
            complex::ComplexDialect,
            func::FuncDialect,
            index::IndexDialect,
            linalg::LinalgDialect,
            LLVM::LLVMDialect,
            math::MathDialect,
            memref::MemRefDialect,
            scf::SCFDialect,
            tensor::TensorDialect,
            vector::VectorDialect,
            mlir::stablehlo::StablehloDialect,
            catalyst::CatalystDialect,
            catalyst::gradient::GradientDialect,
            catalyst::pbc::PBCDialect,
            catalyst::quantum::QuantumDialect>();
    }

    void runOnOperation() final
    {
        ModuleOp host = getOperation();

        SmallVector<ModuleOp> targetMods;
        for (auto &op : host.getBody()->getOperations())
            if (auto mod = dyn_cast<ModuleOp>(&op))
                if (mod->hasAttr("catalyst.target"))
                    targetMods.push_back(mod);

        if (targetMods.empty())
            return;

        if (workspace.empty()) {
            host.emitError("Missing `workspace` option for remote kernel cross-compilation");
            return signalPassFailure();
        }

        llvm::InitializeAllTargetInfos();
        llvm::InitializeAllTargets();
        llvm::InitializeAllTargetMCs();
        llvm::InitializeAllAsmParsers();
        llvm::InitializeAllAsmPrinters();

        // Track which addresses have already had remote_open injected so that multiple
        // catalyst.target modules sharing an executor get only one remote_open call.
        llvm::SmallSet<std::string, 4> openedAddresses;

        for (auto nested : targetMods) {
            if (failed(compileTargetModule(host, nested, openedAddresses)))
                return signalPassFailure();
        }

        // The runtime closes all open sessions with one call; inject it once.
        if (!openedAddresses.empty())
            injectRemoteCloseIntoTeardown(host);
    }

    // Insert a no-operand CustomCallOp before the terminator of `funcName` in `host`.
    // Returns the op so the caller can attach attributes. Returns nullptr if the function
    // is absent or bodyless.
    catalyst::CustomCallOp injectCustomCallInto(ModuleOp host, StringRef funcName,
                                                StringRef callName)
    {
        auto fn = host.lookupSymbol<func::FuncOp>(funcName);
        if (!fn || fn.getBody().empty())
            return nullptr;
        Operation *terminator = fn.getBody().front().getTerminator();
        if (!terminator)
            return nullptr;
        OpBuilder b(terminator);
        return catalyst::CustomCallOp::create(b, fn.getLoc(), TypeRange{}, ValueRange{},
                                              callName, nullptr);
    }

    void injectRemoteOpenIntoSetup(ModuleOp host, StringRef addr)
    {
        auto op = injectCustomCallInto(host, "setup", "remote_open");
        if (op)
            op->setAttr("catalyst.remote_address", StringAttr::get(&getContext(), addr));
    }

    void injectRemoteCloseIntoTeardown(ModuleOp host)
    {
        // remote_close closes all open sessions; no address needed.
        injectCustomCallInto(host, "teardown", "remote_close");
    }

    void injectRemoteSendBinaryIntoSetup(ModuleOp host, StringAttr addressAttr,
                                         StringAttr pathAttr, StringAttr calleeAttr)
    {
        auto op = injectCustomCallInto(host, "setup", "remote_send_binary");
        if (op) {
            op->setAttr("catalyst.remote_address", addressAttr);
            op->setAttr("catalyst.remote_kernel_path", pathAttr);
            op->setAttr("catalyst.remote_kernel_callee", calleeAttr);
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

    void dumpMLIR(mlir::Operation *op, StringRef dir, StringRef filename)
    {
        llvm::SmallString<128> path(dir);
        llvm::sys::path::append(path, filename);
        std::error_code ec;
        llvm::raw_fd_ostream os(path, ec, llvm::sys::fs::OF_None);
        if (!ec)
            op->print(os);
    }

    void dumpLLVMIR(llvm::Module &mod, StringRef dir, StringRef filename)
    {
        llvm::SmallString<128> path(dir);
        llvm::sys::path::append(path, filename);
        std::error_code ec;
        llvm::raw_fd_ostream os(path, ec, llvm::sys::fs::OF_None);
        if (!ec)
            mod.print(os, nullptr);
    }

    std::string emitObjectFile(std::unique_ptr<llvm::Module> &&llvmModule, StringRef name,
                               StringRef triple, StringRef dir)
    {
        llvm::Triple parsedTriple{triple};
        std::string err;
        const llvm::Target *llvmTarget = llvm::TargetRegistry::lookupTarget(parsedTriple, err);
        if (!llvmTarget) {
            llvm::errs() << "Target triple '" << triple
                         << "' not registered in this LLVM build: " << err << "\n";
            return "";
        }
        llvm::TargetOptions opt;
        std::unique_ptr<llvm::TargetMachine> targetMachine(llvmTarget->createTargetMachine(
            parsedTriple, "generic", "", opt, llvm::Reloc::Model::PIC_));
        if (!targetMachine) {
            llvm::errs() << "Could not create TargetMachine for triple '" << triple << "'\n";
            return "";
        }
        targetMachine->setOptLevel(llvm::CodeGenOptLevel::Aggressive);
        llvmModule->setDataLayout(targetMachine->createDataLayout());
        llvmModule->setTargetTriple(parsedTriple);

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
        if (targetMachine->addPassesToEmitFile(codegenPM, dest, nullptr,
                                               llvm::CodeGenFileType::ObjectFile)) {
            llvm::errs() << "TargetMachine cannot emit an object file\n";
            return "";
        }
        codegenPM.run(*llvmModule);
        dest.flush();
        return objPath;
    }

    // Compile a catalyst.target nested module to a .o, inject ORC JIT calls into setup(),
    // replace func.call sites with remote_call custom_calls, and erase the nested module.
    LogicalResult compileTargetModule(ModuleOp host, ModuleOp nested,
                                      llvm::SmallSet<std::string, 4> &openedAddresses)
    {
        MLIRContext *ctx = &getContext();
        StringRef name = nested.getName().value_or("unnamed");

        // Extract address from catalyst.target dict attr.
        auto targetAttr = nested->getAttrOfType<DictionaryAttr>("catalyst.target");
        if (!targetAttr) {
            nested.emitError("catalyst.target module missing catalyst.target dict attr");
            return failure();
        }
        auto addrAttr = targetAttr.getAs<StringAttr>("address");
        if (!addrAttr || addrAttr.getValue().empty()) {
            nested.emitError(
                "catalyst.target module requires 'address' key in catalyst.target attribute");
            return failure();
        }
        std::string moduleAddress = addrAttr.getValue().str();

        // Allow an optional 'triple' override in the catalyst.target dict; fall back to
        // the pass-level target option (which defaults to the host triple).
        std::string moduleTarget = target;
        if (auto tripleAttr = targetAttr.getAs<StringAttr>("triple"))
            if (!tripleAttr.getValue().empty())
                moduleTarget = tripleAttr.getValue().str();

        std::string kernelDir = makeKernelDir(name);

        // Clone nested module into a standalone top-level module so that
        // translateModuleToLLVMIR sees a root module rather than a nested one.
        OpBuilder builder(ctx);
        mlir::OwningOpRef<mlir::ModuleOp> standalone =
            ModuleOp::create(builder.getUnknownLoc(), name);
        for (auto &op : nested.getBody()->getOperations())
            standalone->getBody()->push_back(op.clone());

        // Sanitize all functions in the clone.
        for (auto fn : standalone->getOps<func::FuncOp>())
            sanitizeForRemoteExecution(fn);

        dumpMLIR(*standalone, kernelDir, "extracted.mlir");

        // Lower the standalone clone: bufferize first (one-shot-bufferize doesn't
        // descend into nested modules, so the target module content is still in tensor
        // form here), then run the standard LLVM-dialect lowering stage.
        PassManager subPM(ctx);
        auto bufPasses = catalyst::driver::getBufferizationStage();
        auto llvmPasses = catalyst::driver::getLLVMDialectLoweringStage();
        bufPasses.insert(bufPasses.end(), llvmPasses.begin(), llvmPasses.end());
        std::string passList = llvm::join(bufPasses, ",");
        if (failed(parsePassPipeline(passList, subPM))) {
            nested.emitError("failed to parse LLVM lowering pipeline for target module: " +
                             name.str());
            return failure();
        }
        subPM.addInstrumentation(
            std::make_unique<TargetModuleLoweringInstrumentation>(kernelDir));
        if (failed(subPM.run(*standalone))) {
            nested.emitError("failed to lower target module to LLVM dialect: " + name.str());
            return failure();
        }

        // Translate to LLVM IR and emit .o (plus a .ll for inspection).
        llvm::LLVMContext llvmCtx;
        std::unique_ptr<llvm::Module> llvmModule =
            translateModuleToLLVMIR(*standalone, llvmCtx, name);
        if (!llvmModule) {
            nested.emitError("failed to translate target module to LLVM IR: " + name.str());
            return failure();
        }
        dumpLLVMIR(*llvmModule, kernelDir, name.str() + ".ll");
        std::string objPath = emitObjectFile(std::move(llvmModule), name, moduleTarget, kernelDir);
        if (objPath.empty()) {
            nested.emitError("failed to emit object file for target module: " + name.str());
            return failure();
        }

        auto pathAttr = StringAttr::get(ctx, objPath);
        auto addressAttr = StringAttr::get(ctx, moduleAddress);

        // Inject remote_open (setup) once per unique address.
        if (!openedAddresses.count(moduleAddress)) {
            openedAddresses.insert(moduleAddress);
            injectRemoteOpenIntoSetup(host, moduleAddress);
        }

        // For each function in the nested module, replace host-side func.call with remote_call
        // and inject remote_send_binary into setup().
        for (auto nestedFn : nested.getOps<func::FuncOp>()) {
            StringRef fnName = nestedFn.getName();
            auto decl = host.lookupSymbol<func::FuncOp>(fnName);
            if (!decl || !decl.isExternal())
                continue;

            auto calleeAttr = StringAttr::get(ctx, fnName);

            SmallVector<func::CallOp> calls;
            if (auto uses = SymbolTable::getSymbolUses(decl.getNameAttr(), host)) {
                for (const SymbolTable::SymbolUse &use : *uses) {
                    if (auto call = dyn_cast<func::CallOp>(use.getUser()))
                        calls.push_back(call);
                }
            }
            for (func::CallOp call : calls) {
                OpBuilder b(call);
                auto custom = catalyst::CustomCallOp::create(
                    b, call.getLoc(), call.getResultTypes(), call.getOperands(), "remote_call",
                    nullptr);
                custom->setAttr("catalyst.remote_kernel_path", pathAttr);
                custom->setAttr("catalyst.remote_address", addressAttr);
                custom->setAttr("catalyst.remote_kernel_callee", calleeAttr);
                call.replaceAllUsesWith(custom.getResults());
                call.erase();
            }

            injectRemoteSendBinaryIntoSetup(host, addressAttr, pathAttr, calleeAttr);
            decl.erase();
        }

        nested.erase();
        return success();
    }
};

} // namespace

} // namespace catalyst
