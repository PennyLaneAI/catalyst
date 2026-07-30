// Copyright 2026 Xanadu Quantum Technologies Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// inject-transport-session: read the `catalyst.backline` module attribute and emit the transport
// session lifecycle into @setup/@teardown.

#include <string>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/Pass.h"

#include "Catalyst/IR/CatalystOps.h"

#include "Transport/IR/TransportOps.h"
#include "Transport/Transforms/Passes.h"

using namespace mlir;
using namespace catalyst::transport;

namespace catalyst {
namespace transport {

#define GEN_PASS_DEF_INJECTTRANSPORTSESSIONPASS
#include "Transport/Transforms/Passes.h.inc"

namespace {

constexpr llvm::StringRef kBacklineAttr = "catalyst.backline";

// Which backline node a target module belongs to.
constexpr llvm::StringRef kRoleAttr = "catalyst.backline_role";
constexpr llvm::StringRef kControllerRole = "controller";
constexpr llvm::StringRef kCoprocessorRole = "coprocessor";
constexpr llvm::StringRef kDefaultDataPath = "cpu_verbs";

struct InjectTransportSessionPass
    : public impl::InjectTransportSessionPassBase<InjectTransportSessionPass> {
    using InjectTransportSessionPassBase::InjectTransportSessionPassBase;

    void runOnOperation() override
    {
        ModuleOp mod = getOperation();
        auto backline = mod->getAttrOfType<BacklineAttr>(kBacklineAttr);
        if (!backline)
            return;

        MLIRContext *ctx = &getContext();
        Location loc = mod.getLoc();
        auto ctrlTy = SessionType::get(ctx, Role::Controller);
        auto coTy = SessionType::get(ctx, Role::Coprocessor);
        auto tokTy = TokenType::get(ctx);
        // The attribute verifier guarantees a controller, and a peer and symbol per coprocessor.
        NodeAttr ctrl = backline.getController();
        ArrayRef<NodeAttr> coprocs = backline.getCoprocessors();

        // An empty `() -> ()` public func in `target`.
        auto makeVoidFunc = [&](const Twine &name, ModuleOp target) -> func::FuncOp {
            OpBuilder mb(ctx);
            mb.setInsertionPointToEnd(target.getBody());
            auto fn = func::FuncOp::create(mb, loc, name.str(), mb.getFunctionType({}, {}));
            fn.setPublic();
            Block *blk = fn.addEntryBlock();
            OpBuilder rb(blk, blk->begin());
            func::ReturnOp::create(rb, loc);
            return fn;
        };

        // Remoteness comes from the attribute, not from whether a target module is present.
        bool remoteController = ctrl.isRemote();

        ModuleOp ctrlMod;
        if (remoteController) {
            for (auto m : mod.getOps<ModuleOp>()) {
                if (m->getAttrOfType<StringAttr>(kRoleAttr) == kControllerRole) {
                    ctrlMod = m;
                    break;
                }
            }
            if (!ctrlMod) {
                mod.emitError() << "remote controller has no module tagged " << kRoleAttr << " = \""
                                << kControllerRole << "\"";
                return signalPassFailure();
            }
        }

        // The runtime calls @setup/@teardown around every execution; create them if absent.
        auto findOrCreate = [&](StringRef name) -> func::FuncOp {
            if (auto fn = mod.lookupSymbol<func::FuncOp>(name))
                return fn;
            return makeVoidFunc(name, mod);
        };
        func::FuncOp hostSetup = findOrCreate("setup");
        func::FuncOp hostTeardown = findOrCreate("teardown");

        // A remote controller's ops go into its target module, which @setup/@teardown launch.
        func::FuncOp setupFn =
            remoteController ? makeVoidFunc("setup_transport", ctrlMod) : hostSetup;
        func::FuncOp teardownFn =
            remoteController ? makeVoidFunc("teardown_transport", ctrlMod) : hostTeardown;

        // Launch a `() -> ()` func in a dispatched module.
        auto launchVoid = [&](OpBuilder &lb, StringAttr modName, StringAttr fnName,
                              bool nonblocking = false) {
            auto callee = SymbolRefAttr::get(modName, {FlatSymbolRefAttr::get(ctx, fnName)});
            auto lk =
                catalyst::LaunchKernelOp::create(lb, loc, TypeRange{}, callee, ValueRange{},
                                                 /*arg_attrs=*/nullptr, /*res_attrs=*/nullptr);
            // Serve waits for a connect a later op makes, so it must not block here.
            if (nonblocking)
                lk->setAttr("catalyst.nonblocking", UnitAttr::get(ctx));
        };

        // ---- bring-up ----
        OpBuilder b(ctx);
        b.setInsertionPoint(setupFn.getBody().front().getTerminator());
        auto i16A = [&](int64_t v) { return b.getIntegerAttr(b.getIntegerType(16), v); };
        auto commit = [&](Value ct) {
            CommitWorkItemOp::create(b, loc, ct, b.getI32IntegerAttr(ctrl.workItemIdx()),
                                     b.getI64IntegerAttr(ctrl.inBytes()),
                                     b.getI64IntegerAttr(ctrl.outBytes()));
        };

        // Every session, with the func that stops it: each is released where it was created.
        struct KeyedSession {
            Type role;
            StringAttr key;
            func::FuncOp releaseIn;
        };
        SmallVector<KeyedSession> keyed;

        // Funcs the host launches, per remote coprocessor.
        struct RemoteCoproc {
            StringAttr module;
            StringAttr serve;
            StringAttr stop;
        };
        SmallVector<RemoteCoproc> remoteCoprocs;

        // A host-process coprocessor under a remote controller, whose bring-up spans the launch.
        struct PendingLocalCoproc {
            Value session;
            Value token;
            NodeAttr node;
        };
        SmallVector<PendingLocalCoproc> pendingLocal;

        if (coprocs.empty()) {
            // Controller-only
            StringAttr key = ctrl.keyOr("controller");
            Value ct = CreateOp::create(b, loc, ctrlTy, ctrl.getBackendLib(), ctrl.getConfig(), key)
                           .getSession();
            ConnectOp::create(b, loc, ct, ctrl.getPeer(), i16A(ctrl.oobPort()));
            ExchangeKeysOp::create(b, loc, ct);
            EstablishChannelOp::create(b, loc, ct, ctrl.dataPathOr(kDefaultDataPath));
            commit(ct);
            StartOp::create(b, loc, ct);
            keyed.push_back({ctrlTy, key, teardownFn});
        }
        for (auto [i, coproc] : llvm::enumerate(coprocs)) {
            // Both sessions share this key, so teardown and the qnode resolve them.
            StringAttr key = coproc.keyOr("coprocessor." + std::to_string(i));

            if (coproc.isRemote()) {
                std::string sfx = coprocs.size() > 1 ? ("." + std::to_string(i)) : std::string();

                // The coprocessor's own ops go into its target module, cross-compiled to its triple
                // and dispatched by the host.
                OpBuilder mmb(ctx);
                mmb.setInsertionPointToEnd(mod.getBody());
                std::string coprocModName = ("module_coproc" + sfx);
                auto coprocMod = ModuleOp::create(mmb, loc, StringRef(coprocModName));
                coprocMod->setAttr("catalyst.target",
                                   mmb.getDictionaryAttr({NamedAttribute(
                                       mmb.getStringAttr("triple"), coproc.getTriple())}));
                coprocMod->setAttr(kRoleAttr, mmb.getStringAttr(kCoprocessorRole));
                if (auto addr = coproc.getAddress(); !addr.getValue().empty()) {
                    coprocMod->setAttr("catalyst.dispatch",
                                       mmb.getDictionaryAttr(
                                           {NamedAttribute(mmb.getStringAttr("address"), addr)}));
                }

                func::FuncOp serveFn = makeVoidFunc("coproc_serve" + sfx, coprocMod);
                OpBuilder cb(ctx);
                cb.setInsertionPoint(serveFn.getBody().front().getTerminator());
                Value co =
                    CreateOp::create(cb, loc, coTy, coproc.getBackendLib(), coproc.getConfig(), key)
                        .getSession();
                Value tok = ConnectAsyncOp::create(cb, loc, tokTy, co, coproc.getPeer(),
                                                   i16A(coproc.oobPort()))
                                .getToken();
                BarrierOp::create(cb, loc, tok);
                ExchangeKeysOp::create(cb, loc, co);
                EstablishChannelOp::create(cb, loc, co, coproc.dataPathOr(kDefaultDataPath));
                SetCoprocessorFnOp::create(cb, loc, co, coproc.getSymbol());
                StartOp::create(cb, loc, co);

                func::FuncOp stopFn = makeVoidFunc("coproc_stop" + sfx, coprocMod);
                OpBuilder sb(ctx);
                sb.setInsertionPoint(stopFn.getBody().front().getTerminator());
                Value cs = GetSessionOp::create(sb, loc, coTy, key).getSession();
                StopOp::create(sb, loc, cs);
                DestroyOp::create(sb, loc, cs);

                StringAttr coprocModNameAttr = mmb.getStringAttr(coprocModName);
                if (remoteController) {
                    // Both roles are dispatched, so host orchestration launches them in order.
                    remoteCoprocs.push_back(
                        {coprocModNameAttr, serveFn.getSymNameAttr(), stopFn.getSymNameAttr()});
                }
                else {
                    // The controller dials inline below, so serve must already be running; and its
                    // stop must precede the controller's, which the teardown loop appends after.
                    OpBuilder hb(hostSetup.getBody().front().getTerminator());
                    launchVoid(hb, coprocModNameAttr, serveFn.getSymNameAttr(),
                               /*nonblocking=*/true);
                    OpBuilder htb(hostTeardown.getBody().front().getTerminator());
                    launchVoid(htb, coprocModNameAttr, stopFn.getSymNameAttr());
                }

                Value ctr =
                    CreateOp::create(b, loc, ctrlTy, ctrl.getBackendLib(), ctrl.getConfig(), key)
                        .getSession();
                ConnectOp::create(b, loc, ctr, coproc.getPeer(), i16A(coproc.oobPort()));
                ExchangeKeysOp::create(b, loc, ctr);
                EstablishChannelOp::create(b, loc, ctr, ctrl.dataPathOr(kDefaultDataPath));
                commit(ctr);
                StartOp::create(b, loc, ctr);
                keyed.push_back({ctrlTy, key, teardownFn});
                continue;
            }

            if (remoteController) {
                // The controller's ops go to its target module; the coprocessor listens here, and
                // the rest of its bring-up follows the launch that dials it.
                Value ctr =
                    CreateOp::create(b, loc, ctrlTy, ctrl.getBackendLib(), ctrl.getConfig(), key)
                        .getSession();
                ConnectOp::create(b, loc, ctr, coproc.getPeer(), i16A(coproc.oobPort()));
                ExchangeKeysOp::create(b, loc, ctr);
                EstablishChannelOp::create(b, loc, ctr, ctrl.dataPathOr(kDefaultDataPath));
                commit(ctr);
                StartOp::create(b, loc, ctr);
                keyed.push_back({ctrlTy, key, teardownFn});

                OpBuilder hb(hostSetup.getBody().front().getTerminator());
                Value lco =
                    CreateOp::create(hb, loc, coTy, coproc.getBackendLib(), coproc.getConfig(), key)
                        .getSession();
                Value ltok = ConnectAsyncOp::create(hb, loc, tokTy, lco, coproc.getPeer(),
                                                    i16A(coproc.oobPort()))
                                 .getToken();
                SetCoprocessorFnOp::create(hb, loc, lco, coproc.getSymbol());
                pendingLocal.push_back({lco, ltok, coproc});
                keyed.push_back({coTy, key, hostTeardown});
                continue;
            }

            Value ct = CreateOp::create(b, loc, ctrlTy, ctrl.getBackendLib(), ctrl.getConfig(), key)
                           .getSession();
            Value co =
                CreateOp::create(b, loc, coTy, coproc.getBackendLib(), coproc.getConfig(), key)
                    .getSession();
            Value t1 =
                ConnectAsyncOp::create(b, loc, tokTy, co, coproc.getPeer(), i16A(coproc.oobPort()))
                    .getToken();
            ConnectOp::create(b, loc, ct, coproc.getPeer(), i16A(coproc.oobPort()));
            BarrierOp::create(b, loc, t1);
            Value t2 = ExchangeKeysAsyncOp::create(b, loc, tokTy, co).getToken();
            ExchangeKeysOp::create(b, loc, ct);
            BarrierOp::create(b, loc, t2);
            EstablishChannelOp::create(b, loc, co, coproc.dataPathOr(kDefaultDataPath));
            EstablishChannelOp::create(b, loc, ct, ctrl.dataPathOr(kDefaultDataPath));
            SetCoprocessorFnOp::create(b, loc, co, coproc.getSymbol());
            commit(ct);
            StartOp::create(b, loc, co);
            StartOp::create(b, loc, ct);
            keyed.push_back({coTy, key, teardownFn});
            keyed.push_back({ctrlTy, key, teardownFn});
        }

        // ---- teardown: resolve each session by key, stop + destroy ----
        for (auto it = keyed.rbegin(); it != keyed.rend(); ++it) {
            OpBuilder tb(it->releaseIn.getBody().front().getTerminator());
            Value s = GetSessionOp::create(tb, loc, it->role, it->key).getSession();
            StopOp::create(tb, loc, s);
            DestroyOp::create(tb, loc, s);
        }

        // ---- host orchestration ----
        // @setup/@teardown launch the dispatched lifecycle funcs in dependency order.
        if (remoteController) {
            StringAttr ctrlModName = ctrlMod.getSymNameAttr();

            // @setup reaches every role, so it is the single point that ships objects.
            hostSetup->setAttr("catalyst.backline_bringup", UnitAttr::get(ctx));
            OpBuilder hb(ctx);
            hb.setInsertionPoint(hostSetup.getBody().front().getTerminator());
            for (const RemoteCoproc &c : remoteCoprocs)
                launchVoid(hb, c.module, c.serve, /*nonblocking=*/true);
            launchVoid(hb, ctrlModName, setupFn.getSymNameAttr());

            // The controller has dialed, so a host-process coprocessor's handshake can complete.
            for (const PendingLocalCoproc &c : pendingLocal) {
                BarrierOp::create(hb, loc, c.token);
                ExchangeKeysOp::create(hb, loc, c.session);
                EstablishChannelOp::create(hb, loc, c.session, c.node.dataPathOr(kDefaultDataPath));
                StartOp::create(hb, loc, c.session);
            }

            // Coprocessors stop first: their pending receive needs the controller's transport.
            OpBuilder htb(ctx);
            htb.setInsertionPoint(hostTeardown.getBody().front().getTerminator());
            for (const RemoteCoproc &c : remoteCoprocs)
                launchVoid(htb, c.module, c.stop);
            launchVoid(htb, ctrlModName, teardownFn.getSymNameAttr());
        }
    }
};

} // namespace
} // namespace transport
} // namespace catalyst
