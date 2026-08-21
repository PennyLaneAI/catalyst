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

#include "Executor/IR/ExecutorOps.h"
#include "Executor/Transforms/Passes.h"

using namespace mlir;

namespace catalyst {
namespace executor {

#define GEN_PASS_DEF_DISPATCHEXECUTORTARGETSPASS
#include "Executor/Transforms/Passes.h.inc"

namespace {

// Ships cross-compiled target modules to an executor using the `executor` dialect.
//
// This pass runs after `cross-compile-targets`, which records each module's object file in
// `catalyst.object_file`. For every nested module carrying a `catalyst.dispatch` attribute this
// pass:
//   1. Opens a session (`executor.open`) once per host function and ships the module's object
//      (`executor.send_binary`) once per function, before its launches.
//   2. Rewrites every host-side `catalyst.launch_kernel` targeting the module into an
//      `executor.launch` carrying the session value, the entry callee and the object path.
//   3. Erases the nested module from the host after its `catalyst.launch_kernel`s are rewritten.
//
// A `catalyst.custom_call` carrying a `dispatch` entry in its `backend_config` is rewritten into an
// `executor.call` reusing the same per-function session.
struct DispatchExecutorTargetsPass
    : impl::DispatchExecutorTargetsPassBase<DispatchExecutorTargetsPass> {
    using DispatchExecutorTargetsPassBase::DispatchExecutorTargetsPassBase;

    void runOnOperation() final {
        ModuleOp host = getOperation();

        SmallVector<ModuleOp> targetMods;
        for (auto mod : host.getBody()->getOps<ModuleOp>()) {
            if (mod->hasAttr("catalyst.dispatch")) {
                targetMods.push_back(mod);
            }
        }

        // catalyst.custom_call ops whose backend_config carries a `dispatch` entry
        // The call-target name is the executor-side symbol.
        SmallVector<catalyst::CustomCallOp> libCalls;
        host.walk([&](catalyst::CustomCallOp call) {
            if (executorDispatchOf(call)) {
                libCalls.push_back(call);
            }
        });

        if (targetMods.empty() && libCalls.empty()) {
            return;
        }

        backlineBringupMode = false;
        host.walk([&](func::FuncOp fn) {
            if (fn->hasAttr("catalyst.backline_bringup")) {
                backlineBringupMode = true;
            }
        });

        // Modules sharing an executor get a single executor.open.
        llvm::SmallSet<std::string, 4> openedAddresses;
        StringAttr executorAddress;

        for (auto nested : targetMods) {
            if (failed(dispatchTargetModule(host, nested, openedAddresses, executorAddress))) {
                return signalPassFailure();
            }
            nested.erase();
        }

        const size_t numQnodeExecutors = openedAddresses.size();

        if (!libCalls.empty()) {
            for (catalyst::CustomCallOp call : libCalls) {
                StringAttr dispatch = executorDispatchOf(call);
                if ((!dispatch || dispatch.getValue().empty()) && numQnodeExecutors > 1) {
                    call.emitOpError("ambiguous executor");
                    return signalPassFailure();
                }
                StringAttr addrAttr = libCallAddress(call, executorAddress);
                if (!addrAttr) {
                    call.emitOpError("custom_call dispatch has no executor address");
                    return signalPassFailure();
                }
            }
            if (failed(rewriteExecutorLibCalls(libCalls, executorAddress))) {
                return signalPassFailure();
            }
        }

        // A function that issued async launches must wait for them before it returns,
        // so the launched entry is fully established before any later work runs.
        //
        // TODO: replace this with a `catalyst.launch_kernel_async` op (+ `catalyst.await`) so
        // inject-transport-session places the await itself and this pass lowers 1:1, deleting the
        // block below.
        WalkResult joined = host.walk([&](func::FuncOp fn) {
            SmallVector<Value> tokens;
            fn.walk([&](executor::LaunchAsyncOp launch) { tokens.push_back(launch.getToken()); });
            if (tokens.empty()) {
                return WalkResult::advance();
            }
            Block *entry = &fn.getBody().front();
            bool inEntryBlock = true;
            for (Value token : tokens) {
                if (token.getDefiningOp()->getBlock() != entry) {
                    inEntryBlock = false;
                    break;
                }
            }
            if (!inEntryBlock) {
                fn.emitOpError("async launch must be in the function's entry block");
                return WalkResult::interrupt();
            }
            OpBuilder b(entry->getTerminator());
            for (Value token : tokens) {
                executor::AwaitOp::create(b, fn.getLoc(), token);
            }
            return WalkResult::advance();
        });
        if (joined.wasInterrupted()) {
            return signalPassFailure();
        }
    }

    static StringAttr executorDispatchOf(catalyst::CustomCallOp call) {
        if (auto cfg = call.getBackendConfigAttr()) {
            return cfg.getAs<StringAttr>("dispatch");
        }
        return nullptr;
    }

    static StringAttr libCallAddress(catalyst::CustomCallOp call, StringAttr fallbackAddress) {
        if (StringAttr dispatch = executorDispatchOf(call)) {
            if (!dispatch.getValue().empty()) {
                return dispatch;
            }
        }
        return fallbackAddress;
    }

    bool backlineBringupMode = false;

    // Sessions opened per (function, address).
    llvm::DenseMap<std::pair<Operation *, Attribute>, Value> sessionCache;

    // Return the session handle for `addressAttr` in `user`'s function, opening one at the function
    // entry on first use and caching it for subsequent sites.
    Value getOrOpenSession(Operation *user, StringAttr addressAttr) {
        auto func = user->getParentOfType<func::FuncOp>();
        std::pair<Operation *, Attribute> key{func.getOperation(), addressAttr};
        if (Value cached = sessionCache.lookup(key)) {
            return cached;
        }
        Block &entry = func.getBody().front();
        OpBuilder b(&entry, entry.begin());
        Value session =
            executor::OpenOp::create(b, func.getLoc(), SessionType::get(&getContext()), addressAttr)
                .getSession();
        sessionCache[key] = session;
        return session;
    }

    LogicalResult rewriteExecutorLibCalls(ArrayRef<catalyst::CustomCallOp> libCalls,
                                          StringAttr fallbackAddress) {
        MLIRContext *ctx = &getContext();
        for (catalyst::CustomCallOp call : libCalls) {
            StringAttr addressAttr = libCallAddress(call, fallbackAddress);
            if (!addressAttr) {
                call.emitOpError("custom_call dispatch has no executor address");
                return failure();
            }
            auto symAttr = StringAttr::get(ctx, call.getCallTargetName());
            OpBuilder b(call);
            IntegerAttr numInputAttr = nullptr;
            if (auto n = call.getNumberOriginalArg()) {
                numInputAttr = b.getI32IntegerAttr(*n);
            }
            Value session = getOrOpenSession(call, addressAttr);
            auto executorCall = executor::CallOp::create(
                b, call.getLoc(), call.getResultTypes(), session, call.getOperands(),
                /*symbol=*/symAttr, /*num_input_args=*/numInputAttr);
            call.replaceAllUsesWith(executorCall.getResults());
            call.erase();
        }
        return success();
    }

    // Objects already shipped per (function, object), so `send_binary` is emitted once per host
    // function even when that function launches the module more than once.
    llvm::DenseSet<std::pair<Operation *, Attribute>> shippedObjects;

    // Ship `pathAttr` over the session for `addressAttr`, once per (function, object). Emitted
    // right after the function's `executor.open` so it precedes every launch of the object.
    void ensureBinaryShipped(Operation *user, StringAttr addressAttr, StringAttr pathAttr) {
        auto func = user->getParentOfType<func::FuncOp>();
        std::pair<Operation *, Attribute> key{func.getOperation(), pathAttr};
        if (!shippedObjects.insert(key).second) {
            return;
        }
        Value session = getOrOpenSession(user, addressAttr);
        Operation *openOp = session.getDefiningOp();
        OpBuilder b(openOp);
        b.setInsertionPointAfter(openOp);
        executor::SendBinaryOp::create(b, func.getLoc(), session, pathAttr);
    }

    // Rewrite each host-side launch_kernel targeting `nested` into an `executor.launch`, opening a
    // session and shipping the object recorded in `catalyst.object_file` within the launching
    // function.
    LogicalResult dispatchTargetModule(ModuleOp host, ModuleOp nested,
                                       llvm::SmallSet<std::string, 4> &openedAddresses,
                                       StringAttr &executorAddress) {
        MLIRContext *ctx = &getContext();

        // The object path is produced by the cross-compile-targets pass.
        auto objPathAttr = nested->getAttrOfType<StringAttr>("catalyst.object_file");
        if (!objPathAttr || objPathAttr.getValue().empty()) {
            nested.emitError("executor dispatch requires a non-empty 'catalyst.object_file' "
                             "attribute (run cross-compile-targets first)");
            return failure();
        }

        auto dispatchAttr = nested->getAttrOfType<DictionaryAttr>("catalyst.dispatch");
        auto addrAttr = dispatchAttr ? dispatchAttr.getAs<StringAttr>("address") : nullptr;
        if (!addrAttr || addrAttr.getValue().empty()) {
            nested.emitError("executor dispatch requires a non-empty 'address' key in the "
                             "catalyst.dispatch attribute");
            return failure();
        }
        std::string moduleAddress = addrAttr.getValue().str();

        auto pathAttr = StringAttr::get(ctx, objPathAttr.getValue());
        auto addressAttr = StringAttr::get(ctx, moduleAddress);
        // Remember the executor address so standalone executor lib calls reuse it.
        executorAddress = addressAttr;

        // Track unique executor addresses (used to detect ambiguous lib-call dispatch).
        openedAddresses.insert(moduleAddress);

        // Rewrite each host-side launch_kernel targeting this module into an executor.launch.
        StringRef moduleName = nested.getSymName().value_or("");
        SmallVector<catalyst::LaunchKernelOp> launches;
        host.walk([&](catalyst::LaunchKernelOp launchKernel) {
            if (launchKernel.getCalleeModuleName().getValue() == moduleName) {
                launches.push_back(launchKernel);
            }
        });
        for (catalyst::LaunchKernelOp launchKernel : launches) {
            // executor.launch marshals memref descriptors, so its lowering only accepts
            // memref-typed operands and results. Reject anything else here with a clear error
            // rather than crashing later in convert-executor-to-llvm. This runs after
            // bufferization, so a well-formed entry call is already memref-typed.
            auto isMemref = [](Type ty) { return isa<MemRefType>(ty); };
            if (!llvm::all_of(launchKernel.getOperandTypes(), isMemref) ||
                !llvm::all_of(launchKernel.getResultTypes(), isMemref)) {
                launchKernel.emitOpError("executor dispatch of '")
                    << launchKernel.getCalleeName().getValue()
                    << "' requires memref-typed operands and results";
                return failure();
            }
            auto calleeAttr = StringAttr::get(ctx, launchKernel.getCalleeName().getValue());
            OpBuilder b(launchKernel);
            // Open the session and ship the object within this launch's own function, then reuse
            // the session for the launch. `pathAttr` (the object-file path shipped by send_binary)
            // identifies which object the entry resolves in.
            Value session = getOrOpenSession(launchKernel, addressAttr);
            bool isBringup =
                launchKernel->getParentOfType<func::FuncOp>()->hasAttr("catalyst.backline_bringup");
            if (!backlineBringupMode || isBringup) {
                ensureBinaryShipped(launchKernel, addressAttr, pathAttr);
            }
            if (launchKernel->hasAttr("catalyst.nonblocking")) {
                if (launchKernel->getNumOperands() != 0 || launchKernel->getNumResults() != 0) {
                    launchKernel.emitOpError("nonblocking dispatch requires a '()->()' callee");
                    return failure();
                }
                executor::LaunchAsyncOp::create(b, launchKernel.getLoc(),
                                                executor::TokenType::get(ctx), session, calleeAttr,
                                                /*object=*/pathAttr);
                launchKernel.erase();
                continue;
            }
            auto launch = executor::LaunchOp::create(
                b, launchKernel.getLoc(), launchKernel.getResultTypes(), session,
                launchKernel.getOperands(), calleeAttr, /*object=*/pathAttr);
            launchKernel.replaceAllUsesWith(launch.getResults());
            launchKernel.erase();
        }
        return success();
    }
};

} // namespace

} // namespace executor
} // namespace catalyst
