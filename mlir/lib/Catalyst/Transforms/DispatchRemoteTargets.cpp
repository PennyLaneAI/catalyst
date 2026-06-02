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

#include "Catalyst/IR/CatalystDialect.h"
#include "Catalyst/IR/CatalystOps.h"

using namespace mlir;

namespace catalyst {

#define GEN_PASS_DECL_DISPATCHREMOTETARGETSPASS
#define GEN_PASS_DEF_DISPATCHREMOTETARGETSPASS
#include "Catalyst/Transforms/Passes.h.inc"

namespace {

// Functions marked as the object's externally-callable entry points.
bool isEntryPoint(func::FuncOp fn)
{
    return fn->hasAttr("catalyst.entry_point");
}

// Ships cross-compiled `catalyst.target` modules to a remote executor.
//
// This pass should be run after `cross-compile-targets`, which records each module's object file in
// `catalyst.object_file`. For every nested module carrying a `catalyst.dispatch` attribute this pass:
//   1. Injects `remote_open` into `setup()` (once per unique address) and
//      `remote_send_binary` into `setup()` (once per module; the object holds every entry).
//   2. Rewrites every host-side `func.call` to an entry function into a
//      `catalyst.custom_call fn("remote_call")` carrying the object path, address,
//      and callee.
//   3. Injects `remote_close` into `teardown()` once any session was opened.
//   4. Erases the bodyless external declarations and the nested module from the host.
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

        if (targetMods.empty()) {
            return;
        }

        // Modules sharing an executor get a single remote_open.
        llvm::SmallSet<std::string, 4> openedAddresses;

        for (auto nested : targetMods) {
            if (failed(dispatchViaOrcRemote(host, nested, openedAddresses))) {
                return signalPassFailure();
            }
            nested.erase();
        }

        // One remote_close covers every open session.
        if (!openedAddresses.empty()) {
            injectRemoteCloseIntoTeardown(host);
        }
    }

    // Insert a no-operand CustomCallOp before the terminator of `funcName` in `host`.
    // Returns the op so the caller can attach attributes, or nullptr if the function
    // is absent or bodyless.
    catalyst::CustomCallOp injectCustomCallInto(ModuleOp host, StringRef funcName,
                                                StringRef callName)
    {
        auto fn = host.lookupSymbol<func::FuncOp>(funcName);
        if (!fn || fn.getBody().empty()) {
            return nullptr;
        }
        Operation *terminator = fn.getBody().front().getTerminator();
        if (!terminator) {
            return nullptr;
        }
        OpBuilder b(terminator);
        return catalyst::CustomCallOp::create(b, fn.getLoc(), TypeRange{}, ValueRange{},
                                              callName, nullptr);
    }

    void injectRemoteOpenIntoSetup(ModuleOp host, StringRef addr)
    {
        auto op = injectCustomCallInto(host, "setup", "remote_open");
        if (op) {
            op->setAttr("catalyst.remote_address", StringAttr::get(&getContext(), addr));
        }
    }

    void injectRemoteCloseIntoTeardown(ModuleOp host)
    {
        // remote_close closes all open sessions; no address needed.
        injectCustomCallInto(host, "teardown", "remote_close");
    }

    void injectRemoteSendBinaryIntoSetup(ModuleOp host, StringAttr addressAttr, StringAttr pathAttr)
    {
        auto op = injectCustomCallInto(host, "setup", "remote_send_binary");
        if (op) {
            op->setAttr("catalyst.remote_address", addressAttr);
            op->setAttr("catalyst.remote_kernel_path", pathAttr);
        }
    }

    // Open a session (once per address), ship the object recorded in
    // `catalyst.object_file`, and rewrite each host-side func.call into a `remote_call`.
    LogicalResult dispatchViaOrcRemote(ModuleOp host, ModuleOp nested,
                                       llvm::SmallSet<std::string, 4> &openedAddresses)
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

        // Inject remote_open (setup) once per unique address.
        if (!openedAddresses.count(moduleAddress)) {
            openedAddresses.insert(moduleAddress);
            injectRemoteOpenIntoSetup(host, moduleAddress);
        }

        // Ship the object once per module: a single `catalyst.object_file` holds every
        // entry function, and remote_call resolves individual symbols within it.
        injectRemoteSendBinaryIntoSetup(host, addressAttr, pathAttr);

        // Per entry-marked function: rewrite host-side calls to remote_call.
        for (auto nestedFn : nested.getOps<func::FuncOp>()) {
            if (!isEntryPoint(nestedFn)) {
                continue;
            }
            StringRef fnName = nestedFn.getName();
            auto decl = host.lookupSymbol<func::FuncOp>(fnName);
            if (!decl || !decl.isExternal()) {
                continue;
            }

            auto calleeAttr = StringAttr::get(ctx, fnName);

            SmallVector<func::CallOp> calls;
            if (auto uses = SymbolTable::getSymbolUses(decl.getNameAttr(), host)) {
                for (const SymbolTable::SymbolUse &use : *uses) {
                    if (auto call = dyn_cast<func::CallOp>(use.getUser())) {
                        calls.push_back(call);
                    }
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

            decl.erase();
        }
        return success();
    }
};

} // namespace

} // namespace catalyst
