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
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Export.h"

#include "Catalyst/IR/CatalystOps.h"
#include "Driver/DefaultPipelines/DefaultPipelines.h"

using namespace mlir;

namespace catalyst {

#define GEN_PASS_DECL_CROSSCOMPILEREMOTEKERNELSPASS
#define GEN_PASS_DEF_CROSSCOMPILEREMOTEKERNELSPASS
#include "Catalyst/Transforms/Passes.h.inc"

namespace {

// Sanitize the qnode for compiling the qnode into a standalone module.

/**
 * @brief Sanitize the qnode for compiling the qnode into a standalone module.
 * The conversion takes place in the following steps:
 *     1. Strip `llvm.linkage`
 *     2. Set it to public visibility.
 *     3. Add `llvm.emit_c_interface` to make the qnode callable from C.
 * We then further wrap it in a `builtin.module` and emit the `.o` file.
 *
 * @example
 * Before sanitization:
 * ```mlir
 * func.func @qnode_0(%arg0: memref<1xi64>) -> memref<f64>
 * attributes {llvm.linkage = #llvm.linkage<internal>, qnode} {
 *   return %0 : memref<f64>
 * }
 * ```
 * After sanitization:
 * ```mlir
 * func.func @qnode_0(%arg0: memref<1xi64>) -> memref<f64>
 * attributes {llvm.emit_c_interface, catalyst.remote_kernel} {
 *   return %0 : memref<f64>
 * }
 * ```
 * @param fn The qnode to sanitize.
 */
void sanitizeQNode(func::FuncOp fn)
{
    OpBuilder builder(fn.getContext());
    fn.setVisibility(SymbolTable::Visibility::Public);
    fn->setAttr("llvm.emit_c_interface", builder.getUnitAttr());
    fn->setAttr("catalyst.remote_kernel", builder.getUnitAttr());
    fn->removeAttr("llvm.linkage");
    fn->removeAttr("qnode");
}

struct CrossCompileRemoteKernelsPass
    : impl::CrossCompileRemoteKernelsPassBase<CrossCompileRemoteKernelsPass> {
    using CrossCompileRemoteKernelsPassBase::CrossCompileRemoteKernelsPassBase;

    void runOnOperation() final
    {
        ModuleOp host = getOperation();

        SmallVector<func::FuncOp> qnodes =
            llvm::filter_to_vector(host.getOps<func::FuncOp>(), [&](func::FuncOp fn) {
                return fn->hasAttr("qnode") && !fn->hasAttr("catalyst.remote_kernel");
            });

        if (qnodes.empty()) {
            return;
        }

        if (workspace.empty()) {
            host.emitError("Missing `workspace` option for remote kernel cross-compilation");
            return signalPassFailure();
        }

        // For cross-compilation, we need to initialize the LLVM target registry.
        llvm::InitializeAllTargetInfos();
        llvm::InitializeAllTargets();
        llvm::InitializeAllTargetMCs();
        llvm::InitializeAllAsmParsers();
        llvm::InitializeAllAsmPrinters();

        injectRemoteOpenIntoSetup(host);

        for (auto qnode : qnodes) {
            if (failed(compileQNode(host, qnode))) {
                return signalPassFailure();
            }
        }

        attachAddressToPluginCalls(host);
    }

    // For each `catalyst.custom_call fn("remote_lib_call")`,
    // attach the executor's `catalyst.remote_address` for remote library calls.
    void attachAddressToPluginCalls(ModuleOp host)
    {
        auto addressAttr = StringAttr::get(&getContext(), address);
        host.walk([&](catalyst::CustomCallOp call) {
            if (call.getCallTargetName() == "remote_lib_call") {
                call->setAttr("catalyst.remote_address", addressAttr);
            }
        });
    }

    // Insert `__catalyst__remote__open(addr)` into the host's `setup` function exactly once.
    void injectRemoteOpenIntoSetup(ModuleOp host)
    {
        auto setupFn = host.lookupSymbol<func::FuncOp>("setup");
        if (!setupFn || setupFn.getBody().empty()) {
            return;
        }
        Block &setupBody = setupFn.getBody().front();
        Operation *terminator = setupBody.getTerminator();
        if (!terminator) {
            return;
        }
        OpBuilder b(terminator);
        Location loc = setupFn.getLoc();
        auto openOp = catalyst::CustomCallOp::create(
            b, loc, /*resultTypes=*/TypeRange{}, /*inputs=*/ValueRange{},
            /*call_target_name=*/"remote_open", /*number_original_arg=*/nullptr);
        openOp->setAttr("catalyst.remote_address", StringAttr::get(&getContext(), address));
    }

    // Insert `__catalyst__remote__send_binary(addr, path)` into the host's `setup` function.
    void injectRemoteSendBinaryIntoSetup(ModuleOp host, StringAttr addressAttr, StringAttr pathAttr,
                                         StringAttr calleeAttr)
    {
        auto setupFn = host.lookupSymbol<func::FuncOp>("setup");
        if (!setupFn || setupFn.getBody().empty()) {
            return;
        }
        Block &setupBody = setupFn.getBody().front();
        Operation *terminator = setupBody.getTerminator();
        if (!terminator) {
            return;
        }
        OpBuilder b(terminator);
        Location loc = setupFn.getLoc();
        auto sendOp = catalyst::CustomCallOp::create(b, loc, /*resultTypes=*/TypeRange{},
                                                     /*inputs=*/ValueRange{},
                                                     /*call_target_name=*/"remote_send_binary",
                                                     /*number_original_arg=*/nullptr);
        sendOp->setAttr("catalyst.remote_address", addressAttr);
        sendOp->setAttr("catalyst.remote_kernel_path", pathAttr);
        sendOp->setAttr("catalyst.remote_kernel_callee", calleeAttr);
    }

    /**
     * @brief Lower the qnode to LLVM IR.
     * This function takes the qnode and lowers it to LLVM IR for cross-compilation.
     *
     * @param qnode The qnode to lower.
     * @param llvmCtx Context that owns the produced module; must outlive the returned Module.
     * @return std::unique_ptr<llvm::Module> The LLVM module containing the lowered qnode.
     */
    std::unique_ptr<llvm::Module> loweringQNode(func::FuncOp qnode, llvm::LLVMContext &llvmCtx)
    {
        MLIRContext *ctx = &getContext();
        StringRef qnodeName = qnode.getName();

        // Step 1: Clone the qnode into a fresh standalone module.
        OpBuilder builder(ctx);
        auto clone = ModuleOp::create(builder.getUnknownLoc(), qnodeName);
        auto qnodeClone = cast<func::FuncOp>(qnode->clone());
        clone.getBody()->push_back(qnodeClone);
        sanitizeQNode(qnodeClone);

        // Step 2: Run the LLVM-dialect lowering stage on the clone.
        PassManager nested(ctx);
        std::string passList = llvm::join(catalyst::driver::getLLVMDialectLoweringStage(), ",");
        if (failed(parsePassPipeline(passList, nested))) {
            qnode.emitError("Failed to parse LLVM-dialect lowering pipeline");
            clone->erase();
            return nullptr;
        }
        if (failed(nested.run(clone))) {
            qnode.emitError("Lowering failed on cloned qnode");
            clone->erase();
            return nullptr;
        }

        // Step 3: Translate the lowered clone to LLVM IR.
        std::unique_ptr<llvm::Module> llvmModule =
            translateModuleToLLVMIR(clone, llvmCtx, /*name=*/qnodeName);
        if (!llvmModule) {
            qnode.emitError("Failed to translate lowered qnode to LLVM IR");
            clone->erase();
            return nullptr;
        }
        clone->erase();
        return llvmModule;
    }

    /**
     * @brief Emit the LLVM module to an object file.
     * This function takes the LLVM module and emits it to an object file.
     *
     * @param llvmModule The LLVM module to emit.
     * @param qnodeName The name of the qnode.
     * @param target The target triple to emit the object file for.
     * @param cpu The CPU model to tune codegen for (`generic` for baseline,
     *        `cortex-a72` for Versal Premium APU, etc.).
     * @param features Comma-separated `+feat`/`-feat` tokens, or empty to let
     *        the CPU pick its own feature set.
     * @return std::string The path to the emitted object file.
     */
    std::string emitObjectFile(std::unique_ptr<llvm::Module> &&llvmModule, StringRef qnodeName,
                               StringRef target, StringRef cpu, StringRef features)
    {
        std::string objPath;

        // Build a TargetMachine + object emission.
        llvm::Triple parsedTriple{target};
        std::string err;
        const llvm::Target *llvmTarget = llvm::TargetRegistry::lookupTarget(parsedTriple, err);
        if (!llvmTarget) {
            llvm::errs() << "Target triple '" << target
                         << "' not registered in this LLVM build: " << err << "\n";
            return "";
        }
        llvm::TargetOptions opt;
        std::unique_ptr<llvm::TargetMachine> targetMachine(llvmTarget->createTargetMachine(
            parsedTriple, cpu, features, opt, llvm::Reloc::Model::PIC_));
        if (!targetMachine) {
            llvm::errs() << "Could not create TargetMachine for triple '" << target
                         << "' cpu='" << cpu << "' features='" << features << "'\n";
            return "";
        }

        targetMachine->setOptLevel(llvm::CodeGenOptLevel::Aggressive); // -O3
        llvmModule->setDataLayout(targetMachine->createDataLayout());
        llvmModule->setTargetTriple(parsedTriple);

        llvm::SmallString<128> p(workspace);
        llvm::sys::path::append(p, qnodeName.str() + ".o");
        objPath = std::string(p.str());

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
            llvm::errs() << "TargetMachine cannot emit an object file for the requested target"
                         << "\n";
            return "";
        }
        codegenPM.run(*llvmModule);
        dest.flush();

        return objPath;
    }

    /**
     * @brief Compile the qnode as a standalone module, emit the object file, and replace the
     * host-side calls with custom calls.
     *
     * @param host The host module containing the qnode.
     * @param qnode The qnode to compile.
     * @return LogicalResult Success if the qnode is compiled successfully, failure otherwise.
     */
    LogicalResult compileQNode(ModuleOp host, func::FuncOp qnode)
    {
        MLIRContext *ctx = &getContext();
        StringRef qnodeName = qnode.getName();

        llvm::LLVMContext llvmCtx;
        auto llvmModule = loweringQNode(qnode, llvmCtx);
        if (!llvmModule) {
            return failure();
        }

        // Build a TargetMachine + object emission.
        std::string objPath =
            emitObjectFile(std::move(llvmModule), qnodeName, target, cpu, features);
        if (objPath.empty()) {
            return failure();
        }

        // Replace every host-side `func.call @<qnode>(...)` with a
        // `catalyst.custom_call fn("remote_call")(...)`
        SmallVector<func::CallOp> callsToReplace;
        if (auto uses = SymbolTable::getSymbolUses(qnode.getNameAttr(), host)) {
            for (const SymbolTable::SymbolUse &use : *uses) {
                if (auto call = dyn_cast<func::CallOp>(use.getUser())) {
                    callsToReplace.push_back(call);
                }
            }
        }

        auto pathAttr = StringAttr::get(ctx, objPath);
        auto addressAttr = StringAttr::get(ctx, address);
        auto calleeAttr = StringAttr::get(ctx, qnodeName);
        for (func::CallOp call : callsToReplace) {
            OpBuilder builder(call);

            auto custom = catalyst::CustomCallOp::create(builder, call.getLoc(),
                                                         /*resultTypes=*/call.getResultTypes(),
                                                         call.getOperands(),
                                                         /*call_target_name=*/"remote_call",
                                                         /*number_original_arg=*/nullptr);
            custom->setAttr("catalyst.remote_kernel_path", pathAttr);
            custom->setAttr("catalyst.remote_address", addressAttr);
            custom->setAttr("catalyst.remote_kernel_callee", calleeAttr);
            call.replaceAllUsesWith(custom.getResults());
            call.erase();
        }

        // Inject binary send into setup function.
        injectRemoteSendBinaryIntoSetup(host, addressAttr, pathAttr, calleeAttr);

        qnode.erase();
        return success();
    }
};

} // namespace

} // namespace catalyst
