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

#include "Remote/IR/RemoteOps.h"
#include "Remote/Transforms/Passes.h"

using namespace mlir;

namespace catalyst {
namespace remote {

#define GEN_PASS_DEF_CROSSCOMPILEREMOTEKERNELSPASS
#include "Remote/Transforms/Passes.h.inc"

namespace {

/// Sanitize the qnode so it can be compiled into a standalone module:
///   1. Drop `llvm.linkage` and `qnode`.
///   2. Promote to public visibility.
///   3. Add `llvm.emit_c_interface` so it's callable from C as
///      `_mlir_ciface_<name>` (and `_catalyst_pyface_<name>` is provided by
///      downstream wrapping).
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

        SmallVector<catalyst::CustomCallOp> calls;
        host.walk([&](catalyst::CustomCallOp call) {
            if (call.getCallTargetName() == "remote_call") {
                calls.push_back(call);
            }
        });

        if (qnodes.empty() && calls.empty()) {
            return;
        }

        if (!qnodes.empty()) {
            if (workspace.empty()) {
                host.emitError("Missing `workspace` option for remote kernel cross-compilation");
                return signalPassFailure();
            }

            llvm::InitializeAllTargetInfos();
            llvm::InitializeAllTargets();
            llvm::InitializeAllTargetMCs();
            llvm::InitializeAllAsmParsers();
            llvm::InitializeAllAsmPrinters();
        }

        injectRemoteOpenIntoSetup(host);

        for (auto qnode : qnodes) {
            if (failed(compileQNode(host, qnode))) {
                return signalPassFailure();
            }
        }

        rewriteLegacyLibCalls(calls);
    }

    /// Rewrite legacy `catalyst.custom_call fn("remote_call")` ops into
    /// typed `remote.call` ops carrying the executor address.
    void rewriteLegacyLibCalls(ArrayRef<catalyst::CustomCallOp> libCalls)
    {
        MLIRContext *ctx = &getContext();
        auto addressAttr = StringAttr::get(ctx, address);
        for (catalyst::CustomCallOp call : libCalls) {
            auto symAttr = call->getAttrOfType<StringAttr>("catalyst.remote_symbol");
            if (!symAttr) {
                call->emitOpError("legacy remote_call missing `catalyst.remote_symbol` attribute");
                return signalPassFailure();
            }
            OpBuilder b(call);
            IntegerAttr numInputAttr = nullptr;
            if (auto n = call.getNumberOriginalArg()) {
                numInputAttr = b.getI32IntegerAttr(*n);
            }
            auto remoteCall =
                remote::CallOp::create(b, call.getLoc(), call.getResultTypes(), call.getOperands(),
                                       /*address=*/addressAttr, /*symbol=*/symAttr,
                                       /*num_input_args=*/numInputAttr);
            call.replaceAllUsesWith(remoteCall.getResults());
            call.erase();
        }
    }

    /// Insert `remote.open` into the host's `setup` function exactly once.
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
        remote::OpenOp::create(b, setupFn.getLoc(),
                               /*address=*/StringAttr::get(&getContext(), address));
    }

    /// Insert `remote.send_binary` into the host's `setup` function for the
    /// kernel we just produced.
    void injectRemoteSendBinaryIntoSetup(ModuleOp host, StringAttr addressAttr, StringAttr pathAttr)
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
        remote::SendBinaryOp::create(b, setupFn.getLoc(),
                                     /*address=*/addressAttr,
                                     /*binary_path=*/pathAttr);
    }

    /// Clone the qnode into a fresh module, sanitize it, run the LLVM-dialect
    /// lowering stage, then translate to LLVM IR.
    std::unique_ptr<llvm::Module> loweringQNode(func::FuncOp qnode, llvm::LLVMContext &llvmCtx)
    {
        MLIRContext *ctx = &getContext();
        StringRef qnodeName = qnode.getName();

        OpBuilder builder(ctx);
        auto clone = ModuleOp::create(builder.getUnknownLoc(), qnodeName);
        auto qnodeClone = cast<func::FuncOp>(qnode->clone());
        clone.getBody()->push_back(qnodeClone);
        sanitizeQNode(qnodeClone);

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

    /// Emit the LLVM module to `<workspace>/<qnode>.o`.
    std::string emitObjectFile(std::unique_ptr<llvm::Module> &&llvmModule, StringRef qnodeName,
                               StringRef targetTriple, StringRef cpuModel, StringRef featureList)
    {
        llvm::Triple parsedTriple{targetTriple};
        std::string err;
        const llvm::Target *llvmTarget = llvm::TargetRegistry::lookupTarget(parsedTriple, err);
        if (!llvmTarget) {
            llvm::errs() << "Target triple '" << targetTriple
                         << "' not registered in this LLVM build: " << err << "\n";
            return "";
        }
        llvm::TargetOptions opt;
        std::unique_ptr<llvm::TargetMachine> targetMachine(llvmTarget->createTargetMachine(
            parsedTriple, cpuModel, featureList, opt, llvm::Reloc::Model::PIC_));
        if (!targetMachine) {
            llvm::errs() << "Could not create TargetMachine for triple '" << targetTriple
                         << "' cpu='" << cpuModel << "' features='" << featureList << "'\n";
            return "";
        }

        targetMachine->setOptLevel(llvm::CodeGenOptLevel::Aggressive);
        llvmModule->setDataLayout(targetMachine->createDataLayout());
        llvmModule->setTargetTriple(parsedTriple);

        llvm::SmallString<128> p(workspace);
        llvm::sys::path::append(p, qnodeName.str() + ".o");
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
            llvm::errs() << "TargetMachine cannot emit an object file for the requested target"
                         << "\n";
            return "";
        }
        codegenPM.run(*llvmModule);
        dest.flush();

        return objPath;
    }

    /// Compile a qnode to a `.o`, then replace every host-side `func.call` to
    /// it with a `remote.launch` op and inject the matching `remote.send_binary`
    /// into setup.
    LogicalResult compileQNode(ModuleOp host, func::FuncOp qnode)
    {
        MLIRContext *ctx = &getContext();
        StringRef qnodeName = qnode.getName();

        llvm::LLVMContext llvmCtx;
        auto llvmModule = loweringQNode(qnode, llvmCtx);
        if (!llvmModule) {
            return failure();
        }

        std::string objPath =
            emitObjectFile(std::move(llvmModule), qnodeName, target, cpu, features);
        if (objPath.empty()) {
            return failure();
        }

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
            auto launch = remote::LaunchOp::create(builder, call.getLoc(), call.getResultTypes(),
                                                   call.getOperands(), addressAttr, calleeAttr);
            call.replaceAllUsesWith(launch.getResults());
            call.erase();
        }

        injectRemoteSendBinaryIntoSetup(host, addressAttr, pathAttr);

        qnode.erase();
        return success();
    }
};

} // namespace

} // namespace remote
} // namespace catalyst
