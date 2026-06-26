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
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include "Catalyst/IR/CatalystOps.h"
#include "Remote/IR/RemoteOps.h"
#include "Remote/Transforms/Passes.h"

using namespace mlir;

namespace catalyst {
namespace remote {

#define GEN_PASS_DEF_DISPATCHREMOTETARGETSPASS
#include "Remote/Transforms/Passes.h.inc"

namespace {

// Ships cross-compiled `catalyst.target` modules to a remote executor using the `remote` dialect.
//
// This pass should be run after `cross-compile-targets`, which records each module's object file in
// `catalyst.object_file`. For every nested module carrying a `catalyst.dispatch` attribute this pass:
//   1. Injects `remote.open` into `setup()` (once per unique address) and `remote.send_binary`
//      into `setup()` (once per module; the object holds every entry).
//   2. Rewrites every host-side `func.call` to an entry function into a `remote.launch` op carrying
//      the executor address and the entry callee. Its lowering resolves `_catalyst_pyface_<callee>`
//      in the shipped object.
//   3. Erases the bodyless external declarations and the nested module from the host.
//
// Session teardown is handled by the runtime, which closes every open session when the process
// exits, so no explicit close op is emitted (matching `cross-compile-remote-kernels`).
struct DispatchRemoteTargetsPass
    : impl::DispatchRemoteTargetsPassBase<DispatchRemoteTargetsPass> {
    using DispatchRemoteTargetsPassBase::DispatchRemoteTargetsPassBase;

    void runOnOperation() final
    {
        ModuleOp host = getOperation();

        SmallVector<ModuleOp> targetMods;
        for (auto &op : host.getBody()->getOperations()) {
            if (auto mod = dyn_cast<ModuleOp>(&op)) {
                if (mod->hasAttr("catalyst.dispatch")) {
                    targetMods.push_back(mod);
                }
            }
        }

        // catalyst.custom_call ops whose backend_config carries a `dispatch` entry
        // The call-target name is the executor-side symbol.
        SmallVector<catalyst::CustomCallOp> libCalls;
        host.walk([&](catalyst::CustomCallOp call) {
            if (remoteDispatchOf(call)) {
                libCalls.push_back(call);
            }
        });

        if (targetMods.empty() && libCalls.empty()) {
            return;
        }

        // Modules sharing an executor get a single remote.open.
        llvm::SmallSet<std::string, 4> openedAddresses;
        StringAttr executorAddress;

        for (auto nested : targetMods) {
            if (failed(dispatchViaOrcRemote(host, nested, openedAddresses, executorAddress))) {
                return signalPassFailure();
            }
            nested.erase();
        }

        const size_t numQnodeExecutors = openedAddresses.size();

        if (!libCalls.empty()) {
            for (catalyst::CustomCallOp call : libCalls) {
                StringAttr dispatch = remoteDispatchOf(call);
                if ((!dispatch || dispatch.getValue().empty()) && numQnodeExecutors > 1) {
                    call.emitOpError("ambiguous remote executor");
                    return signalPassFailure();
                }
                StringAttr addrAttr = libCallAddress(call, executorAddress);
                if (!addrAttr) {
                    call.emitOpError("remote runtime_call has no executor address");
                    return signalPassFailure();
                }
                if (openedAddresses.insert(addrAttr.str()).second) {
                    injectRemoteOpenIntoSetup(host, addrAttr);
                }
            }
            if (failed(rewriteRemoteLibCalls(libCalls, executorAddress))) {
                return signalPassFailure();
            }
        }
    }

    static StringAttr remoteDispatchOf(catalyst::CustomCallOp call)
    {
        if (auto cfg = call.getBackendConfigAttr()) {
            return cfg.getAs<StringAttr>("dispatch");
        }
        return nullptr;
    }

    static StringAttr libCallAddress(catalyst::CustomCallOp call, StringAttr fallbackAddress)
    {
        if (StringAttr dispatch = remoteDispatchOf(call)) {
            if (!dispatch.getValue().empty()) {
                return dispatch;
            }
        }
        return fallbackAddress;
    }

    LogicalResult rewriteRemoteLibCalls(ArrayRef<catalyst::CustomCallOp> libCalls,
                                        StringAttr fallbackAddress)
    {
        MLIRContext *ctx = &getContext();
        for (catalyst::CustomCallOp call : libCalls) {
            StringAttr addressAttr = libCallAddress(call, fallbackAddress);
            if (!addressAttr) {
                call.emitOpError("remote runtime_call has no executor address");
                return failure();
            }
            auto symAttr = StringAttr::get(ctx, call.getCallTargetName());
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
        return success();
    }

    // Insert `remote.open` before the terminator of `setup`. No-op if setup is absent or bodyless.
    void injectRemoteOpenIntoSetup(ModuleOp host, StringAttr addressAttr)
    {
        auto setupFn = host.lookupSymbol<func::FuncOp>("setup");
        if (!setupFn || setupFn.getBody().empty()) {
            return;
        }
        Operation *terminator = setupFn.getBody().front().getTerminator();
        if (!terminator) {
            return;
        }
        OpBuilder b(terminator);
        remote::OpenOp::create(b, setupFn.getLoc(), addressAttr);
    }

    // Insert `remote.send_binary` before the terminator of `setup`. No-op if setup is absent or
    // bodyless. Always called after `injectRemoteOpenIntoSetup` so the session is opened first.
    void injectRemoteSendBinaryIntoSetup(ModuleOp host, StringAttr addressAttr, StringAttr pathAttr)
    {
        auto setupFn = host.lookupSymbol<func::FuncOp>("setup");
        if (!setupFn || setupFn.getBody().empty()) {
            return;
        }
        Operation *terminator = setupFn.getBody().front().getTerminator();
        if (!terminator) {
            return;
        }
        OpBuilder b(terminator);
        remote::SendBinaryOp::create(b, setupFn.getLoc(), addressAttr, pathAttr);
    }

    // Open a session (once per address), ship the object recorded in
    // `catalyst.object_file`, and rewrite each host-side func.call into a `remote.launch`.
    LogicalResult dispatchViaOrcRemote(ModuleOp host, ModuleOp nested,
                                       llvm::SmallSet<std::string, 4> &openedAddresses,
                                       StringAttr &executorAddress)
    {
        MLIRContext *ctx = &getContext();

        // The object path is produced by the cross-compile-targets pass.
        auto objPathAttr = nested->getAttrOfType<StringAttr>("catalyst.object_file");
        if (!objPathAttr || objPathAttr.getValue().empty()) {
            nested.emitError("remote dispatch requires a non-empty 'catalyst.object_file' "
                             "attribute (run cross-compile-targets first)");
            return failure();
        }

        auto dispatchAttr = nested->getAttrOfType<DictionaryAttr>("catalyst.dispatch");
        auto addrAttr = dispatchAttr ? dispatchAttr.getAs<StringAttr>("address") : nullptr;
        if (!addrAttr || addrAttr.getValue().empty()) {
            nested.emitError("remote dispatch requires a non-empty 'address' key in the "
                             "catalyst.dispatch attribute");
            return failure();
        }
        std::string moduleAddress = addrAttr.getValue().str();

        auto pathAttr = StringAttr::get(ctx, objPathAttr.getValue());
        auto addressAttr = StringAttr::get(ctx, moduleAddress);
        // Remember the executor address so standalone remote lib calls reuse it.
        executorAddress = addressAttr;

        // Inject remote.open (setup) once per unique address.
        if (!openedAddresses.count(moduleAddress)) {
            openedAddresses.insert(moduleAddress);
            injectRemoteOpenIntoSetup(host, addressAttr);
        }

        // Ship the object once per module: a single `catalyst.object_file` holds every
        // entry function, and remote.launch resolves individual symbols within it.
        injectRemoteSendBinaryIntoSetup(host, addressAttr, pathAttr);

        // Rewrite each host-side launch_kernel targeting this module into a remote.launch.
        StringRef moduleName = nested.getSymName().value_or("");
        SmallVector<catalyst::LaunchKernelOp> launches;
        host.walk([&](catalyst::LaunchKernelOp launchKernel) {
            if (launchKernel.getCalleeModuleName().getValue() == moduleName) {
                launches.push_back(launchKernel);
            }
        });
        for (catalyst::LaunchKernelOp launchKernel : launches) {
            // remote.launch marshals memref descriptors, so its lowering only accepts memref-typed
            // operands and results. Reject anything else here with a clear error rather than
            // crashing later in convert-remote-to-llvm. This runs after bufferization, so a
            // well-formed entry call is already memref-typed.
            auto isMemref = [](Type ty) { return isa<MemRefType>(ty); };
            if (!llvm::all_of(launchKernel.getOperandTypes(), isMemref) ||
                !llvm::all_of(launchKernel.getResultTypes(), isMemref)) {
                launchKernel.emitOpError("remote dispatch of '")
                    << launchKernel.getCalleeName().getValue()
                    << "' requires memref-typed operands and results";
                return failure();
            }
            auto calleeAttr = StringAttr::get(ctx, launchKernel.getCalleeName().getValue());
            OpBuilder b(launchKernel);
            // `pathAttr` (the object-file path, same value shipped by send_binary) keys the
            // per-kernel JITDylib on the executor so the entry resolves in its own object.
            auto launch = remote::LaunchOp::create(
                b, launchKernel.getLoc(), launchKernel.getResultTypes(),
                launchKernel.getOperands(), addressAttr, calleeAttr, pathAttr);
            launchKernel.replaceAllUsesWith(launch.getResults());
            launchKernel.erase();
        }
        return success();
    }
};

} // namespace

} // namespace remote
} // namespace catalyst
