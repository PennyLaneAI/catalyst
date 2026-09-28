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

// A session and the func that releases it: each is torn down where it was created.
struct KeyedSession {
    Type role;
    StringAttr key;
    func::FuncOp releaseIn;
};

// A dispatched coprocessor the host launches, named by its module and lifecycle entry points.
struct RemoteCoproc {
    StringAttr module;
    StringAttr serve;
    StringAttr stop;
};

// A host-process coprocessor under a remote controller, whose bring-up spans the launch that dials
// it.
struct PendingLocalCoproc {
    Value session;
    Value token;
};

// Emits the transport session lifecycle for one backline placement.
//
// A node is either
// - local:
// meaning it runs in the host process, so its session ops go straight into the host
// @setup/@teardown
// - remote: it is cross-compiled into its own target module and dispatched to an executor,
// so its session ops go into that module and the host only launches them.
// If a controller and coprocessor are both local are co-located they share the host process,
// so the controller dials the coprocessor inline.
//
// Each combination has its own method:
//
//   ctrl   | coproc         | method                           | ctrl ops    | coproc ops
//   -------+----------------+----------------------------------+-------------+--------------
//   any    | none, peer set | emitControllerOnly               | @setup or   | n/a
//          |                | (dials its own peer)             | (module_ctrl|
//          |                |                                  |  if remote) |
//   any    | none, no peer  | none, nothing to dial            | n/a         | n/a
//   local  | local          | emitColocated                    | @setup      | @setup
//   local  | remote         | emitLocalControllerRemoteCoproc  | @setup      | module_coproc
//   remote | local          | emitRemoteControllerLocalCoproc  | module_ctrl | @setup
//   remote | remote         | emitRemoteControllerRemoteCoproc | module_ctrl | module_coproc
//
// Controller remoteness is placement-wide, so every row a placement uses shares its controller
// kind; coprocessor remoteness is per node, so a placement can still mix local and remote
// coprocessors within that kind.
class SessionEmitter {
  public:
    SessionEmitter(ModuleOp mod, StringAttr transport, NodeAttr ctrl, bool dispatchedController,
                   ModuleOp ctrlMod)
        : ctx(mod.getContext()), loc(mod.getLoc()), mod(mod), ctrl(ctrl),
          dispatchedController(dispatchedController), ctrlMod(ctrlMod),
          ctrlTy(SessionType::get(ctx, Role::Controller)),
          coTy(SessionType::get(ctx, Role::Coprocessor)), tokTy(TokenType::get(ctx)),
          transport(transport.getValue()), b(ctx) {
        // The runtime calls @setup/@teardown around every execution; reuse them if present.
        hostSetup = findOrCreate("setup");
        hostTeardown = findOrCreate("teardown");
        // A remote controller's ops go into its target module, which @setup/@teardown launch.
        if (dispatchedController) {
            setTargetFromNode(ctrlMod, ctrl);
        }
        setupFn = dispatchedController ? makeVoidFunc("setup_transport", ctrlMod) : hostSetup;
        teardownFn =
            dispatchedController ? makeVoidFunc("teardown_transport", ctrlMod) : hostTeardown;
        b.setInsertionPoint(terminatorOf(setupFn));
    }

    // Emit the whole session lifecycle: bring up each participant, then teardown and host
    // orchestration.
    void run(ArrayRef<NodeAttr> coprocs) {
        if (coprocs.empty()) {
            emitControllerOnly();
        } else {
            for (auto [i, coproc] : llvm::enumerate(coprocs)) {
                emitCoproc(coproc, i, coprocs.size());
            }
        }
        finalize();
    }

  private:
    // Controller-only: a single controller session dialing its own peer.
    void emitControllerOnly() {
        StringAttr key = ctrl.keyOr("controller");
        Value ct = createSession(b, ctrlTy, ctrl, key);
        ConnectOp::create(b, loc, ct, peerFor(ctrl), portFor(ctrl));
        ExchangeKeysOp::create(b, loc, ct);
        EstablishChannelOp::create(b, loc, ct, b.getStringAttr(transport));
        commit(ct);
        StartOp::create(b, loc, ct);
        keyed.push_back({ctrlTy, key, teardownFn});
    }

    // Dispatch one coprocessor to the handler for its (controller, coprocessor) remoteness.
    void emitCoproc(NodeAttr coproc, size_t index, size_t count) {
        StringAttr key = coproc.keyOr("coprocessor." + std::to_string(index));
        if (coproc.isOutOfProcess()) {
            std::string sfx = count > 1 ? ("." + std::to_string(index)) : std::string();
            if (dispatchedController) {
                emitRemoteControllerRemoteCoproc(coproc, key, sfx);
            } else {
                emitLocalControllerRemoteCoproc(coproc, key, sfx);
            }
        } else if (dispatchedController) {
            emitRemoteControllerLocalCoproc(coproc, key);
        } else {
            emitColocated(coproc, key);
        }
    }

    // Teardown every session by key, then the host orchestration that drives dispatched roles.
    void finalize() {
        for (auto it = keyed.rbegin(); it != keyed.rend(); ++it) {
            OpBuilder tb(terminatorOf(it->releaseIn));
            Value s = GetSessionOp::create(tb, loc, it->role, it->key).getSession();
            StopOp::create(tb, loc, s);
            DestroyOp::create(tb, loc, s);
        }
        if (dispatchedController) {
            emitHostOrchestration();
        }
    }

    // An empty `() -> ()` public func in `target`.
    func::FuncOp makeVoidFunc(const Twine &name, ModuleOp target) {
        OpBuilder mb(ctx);
        mb.setInsertionPointToEnd(target.getBody());
        auto fn = func::FuncOp::create(mb, loc, name.str(), mb.getFunctionType({}, {}));
        fn.setPublic();
        Block *blk = fn.addEntryBlock();
        OpBuilder rb(blk, blk->begin());
        func::ReturnOp::create(rb, loc);
        return fn;
    }

    func::FuncOp findOrCreate(StringRef name) {
        if (auto fn = mod.lookupSymbol<func::FuncOp>(name)) {
            return fn;
        }
        return makeVoidFunc(name, mod);
    }

    // Launch a `() -> ()` func in a dispatched module.
    void launchVoid(OpBuilder &lb, StringAttr modName, StringAttr fnName,
                    bool nonblocking = false) {
        auto callee = SymbolRefAttr::get(modName, {FlatSymbolRefAttr::get(ctx, fnName)});
        auto lk = catalyst::LaunchKernelOp::create(lb, loc, TypeRange{}, callee, ValueRange{},
                                                   /*arg_attrs=*/nullptr, /*res_attrs=*/nullptr);
        // Serve waits for a connect a later op makes, so it must not block here.
        if (nonblocking) {
            lk->setAttr("catalyst.nonblocking", UnitAttr::get(ctx));
        }
    }

    // Ports are carried as i32 so a value above 32767 reads as itself rather than as a negative
    // number; the LLVM lowering narrows to the runtime's uint16_t.
    IntegerAttr portAttr(int64_t v) {
        return b.getIntegerAttr(b.getIntegerType(16, /*isSigned=*/false), v);
    }

    // In-process transports (memcpy) pair on the session key and never dial peer:oob_port, so
    // emitted transport.connect / connect_async ops carry no peer / oob_port.
    bool needsOob() const { return transport != "memcpy"; }
    StringAttr peerFor(NodeAttr node) { return needsOob() ? node.getPeer() : StringAttr{}; }
    IntegerAttr portFor(NodeAttr node) {
        return needsOob() ? portAttr(node.oobPort()) : IntegerAttr{};
    }

    void commit(Value ct) {
        SetMessageSizesOp::create(b, loc, ct, b.getI32IntegerAttr(ctrl.workItemIdx()),
                                  b.getI64IntegerAttr(ctrl.inBytes()),
                                  b.getI64IntegerAttr(ctrl.outBytes()));
    }

    // The return op of a single-block func: new body ops are inserted before it.
    static Operation *terminatorOf(func::FuncOp fn) { return fn.getBody().front().getTerminator(); }

    // Create a `role` session for `node`, keyed so teardown and the qnode can resolve it.
    Value createSession(OpBuilder &bld, Type role, NodeAttr node, StringAttr key) {
        return CreateOp::create(bld, loc, role, node.getBackendLib(), node.getConfig(), key)
            .getSession();
    }

    // The controller-side session dialing a coprocessor's peer, released in teardown.
    void emitControllerDial(NodeAttr coproc, StringAttr key) {
        Value ctr = createSession(b, ctrlTy, ctrl, key);
        ConnectOp::create(b, loc, ctr, peerFor(coproc), portFor(coproc));
        ExchangeKeysOp::create(b, loc, ctr);
        EstablishChannelOp::create(b, loc, ctr, b.getStringAttr(transport));
        commit(ctr);
        StartOp::create(b, loc, ctr);
        keyed.push_back({ctrlTy, key, teardownFn});
    }

    // Fill a node's nested module in with the triple it is cross-compiled to and the address it is
    // dispatched to, both taken from its entry in the placement.
    void setTargetFromNode(ModuleOp nested, NodeAttr node) {
        OpBuilder b(ctx);
        nested->setAttr(
            "catalyst.target",
            b.getDictionaryAttr({NamedAttribute(b.getStringAttr("triple"), node.getTriple())}));
        if (auto addr = node.getAddress(); !addr.getValue().empty()) {
            nested->setAttr(
                "catalyst.dispatch",
                b.getDictionaryAttr({NamedAttribute(b.getStringAttr("address"), addr)}));
        }
    }

    // A remote coprocessor's target module, cross-compiled to its triple and dispatched by the
    // host, returned as the serve/stop launch targets the host orchestration drives.
    RemoteCoproc emitCoprocModule(NodeAttr coproc, StringAttr key, const std::string &sfx) {
        OpBuilder mmb(ctx);
        mmb.setInsertionPointToEnd(mod.getBody());
        std::string coprocModName = ("module_coproc" + sfx);
        auto coprocMod = ModuleOp::create(mmb, loc, StringRef(coprocModName));
        coprocMod->setAttr(kRoleAttr, mmb.getStringAttr(kCoprocessorRole));
        setTargetFromNode(coprocMod, coproc);

        func::FuncOp serveFn = makeVoidFunc("coproc_serve" + sfx, coprocMod);
        OpBuilder cb(ctx);
        cb.setInsertionPoint(terminatorOf(serveFn));
        Value co = createSession(cb, coTy, coproc, key);
        Value tok =
            ConnectAsyncOp::create(cb, loc, tokTy, co, peerFor(coproc), portFor(coproc)).getToken();
        AwaitOp::create(cb, loc, tok);
        ExchangeKeysOp::create(cb, loc, co);
        EstablishChannelOp::create(cb, loc, co, cb.getStringAttr(transport));
        SetCoprocessorFnOp::create(cb, loc, co, coproc.getSymbol());
        StartOp::create(cb, loc, co);

        func::FuncOp stopFn = makeVoidFunc("coproc_stop" + sfx, coprocMod);
        OpBuilder sb(ctx);
        sb.setInsertionPoint(terminatorOf(stopFn));
        Value cs = GetSessionOp::create(sb, loc, coTy, key).getSession();
        StopOp::create(sb, loc, cs);
        DestroyOp::create(sb, loc, cs);

        return {mmb.getStringAttr(coprocModName), serveFn.getSymNameAttr(),
                stopFn.getSymNameAttr()};
    }

    // Remote controller, remote coprocessor: both dispatched, so host orchestration launches them.
    void emitRemoteControllerRemoteCoproc(NodeAttr coproc, StringAttr key, const std::string &sfx) {
        remoteCoprocs.push_back(emitCoprocModule(coproc, key, sfx));
        emitControllerDial(coproc, key);
    }

    // Local controller, remote coprocessor: the controller dials inline below, so serve must
    // already be running, and its stop must precede the controller's.
    void emitLocalControllerRemoteCoproc(NodeAttr coproc, StringAttr key, const std::string &sfx) {
        RemoteCoproc cm = emitCoprocModule(coproc, key, sfx);
        OpBuilder hb(terminatorOf(hostSetup));
        launchVoid(hb, cm.module, cm.serve, /*nonblocking=*/true);
        OpBuilder htb(terminatorOf(hostTeardown));
        launchVoid(htb, cm.module, cm.stop);
        emitControllerDial(coproc, key);
    }

    // Remote controller, host-process coprocessor: the controller dials from its target module, and
    // the coprocessor's handshake completes after the launch that dials it (see
    // emitHostOrchestration()).
    void emitRemoteControllerLocalCoproc(NodeAttr coproc, StringAttr key) {
        emitControllerDial(coproc, key);

        OpBuilder hb(terminatorOf(hostSetup));
        Value lco = createSession(hb, coTy, coproc, key);
        Value ltok = ConnectAsyncOp::create(hb, loc, tokTy, lco, peerFor(coproc), portFor(coproc))
                         .getToken();
        SetCoprocessorFnOp::create(hb, loc, lco, coproc.getSymbol());
        pendingLocal.push_back({lco, ltok});
        keyed.push_back({coTy, key, hostTeardown});
    }

    // Co-located controller and coprocessor: both brought up inline with an async handshake.
    void emitColocated(NodeAttr coproc, StringAttr key) {
        Value ct = createSession(b, ctrlTy, ctrl, key);
        Value co = createSession(b, coTy, coproc, key);
        Value t1 =
            ConnectAsyncOp::create(b, loc, tokTy, co, peerFor(coproc), portFor(coproc)).getToken();
        ConnectOp::create(b, loc, ct, peerFor(coproc), portFor(coproc));
        AwaitOp::create(b, loc, t1);
        Value t2 = ExchangeKeysAsyncOp::create(b, loc, tokTy, co).getToken();
        ExchangeKeysOp::create(b, loc, ct);
        AwaitOp::create(b, loc, t2);
        EstablishChannelOp::create(b, loc, co, b.getStringAttr(transport));
        EstablishChannelOp::create(b, loc, ct, b.getStringAttr(transport));
        SetCoprocessorFnOp::create(b, loc, co, coproc.getSymbol());
        commit(ct);
        StartOp::create(b, loc, co);
        StartOp::create(b, loc, ct);
        keyed.push_back({coTy, key, teardownFn});
        keyed.push_back({ctrlTy, key, teardownFn});
    }

    // @setup/@teardown launch the dispatched lifecycle funcs in dependency order.
    void emitHostOrchestration() {
        StringAttr ctrlModName = ctrlMod.getSymNameAttr();

        // @setup reaches every role, so it is the single point that ships objects.
        hostSetup->setAttr("catalyst.backline_bringup", UnitAttr::get(ctx));
        OpBuilder hb(ctx);
        hb.setInsertionPoint(terminatorOf(hostSetup));
        for (const RemoteCoproc &c : remoteCoprocs) {
            launchVoid(hb, c.module, c.serve, /*nonblocking=*/true);
        }
        launchVoid(hb, ctrlModName, setupFn.getSymNameAttr());

        // The controller has dialed, so a host-process coprocessor's handshake can complete.
        for (const PendingLocalCoproc &c : pendingLocal) {
            AwaitOp::create(hb, loc, c.token);
            ExchangeKeysOp::create(hb, loc, c.session);
            EstablishChannelOp::create(hb, loc, c.session, hb.getStringAttr(transport));
            StartOp::create(hb, loc, c.session);
        }

        // Coprocessors stop first: their pending receive needs the controller's transport.
        OpBuilder htb(ctx);
        htb.setInsertionPoint(terminatorOf(hostTeardown));
        for (const RemoteCoproc &c : remoteCoprocs) {
            launchVoid(htb, c.module, c.stop);
        }
        launchVoid(htb, ctrlModName, teardownFn.getSymNameAttr());
    }

    MLIRContext *ctx;
    Location loc;
    ModuleOp mod;
    NodeAttr ctrl;
    bool dispatchedController;
    ModuleOp ctrlMod;
    Type ctrlTy;
    Type coTy;
    Type tokTy;
    llvm::StringRef transport;
    func::FuncOp hostSetup;
    func::FuncOp hostTeardown;
    func::FuncOp setupFn;
    func::FuncOp teardownFn;
    OpBuilder b;
    SmallVector<KeyedSession> keyed;
    SmallVector<RemoteCoproc> remoteCoprocs;
    SmallVector<PendingLocalCoproc> pendingLocal;
};

struct InjectTransportSessionPass
    : public impl::InjectTransportSessionPassBase<InjectTransportSessionPass> {
    using InjectTransportSessionPassBase::InjectTransportSessionPassBase;

    void runOnOperation() override {
        ModuleOp mod = getOperation();
        auto backline = mod->getAttrOfType<BacklineAttr>(kBacklineAttr);
        if (!backline) {
            return;
        }

        // The attribute verifier guarantees a controller, and a peer and symbol per coprocessor.
        NodeAttr ctrl = backline.getController();
        ArrayRef<NodeAttr> coprocs = backline.getCoprocessors();

        // A lone controller with no peer has nothing to dial, so no transport session is emitted.
        // A peer still brings one up, supporting self-dial tests.
        StringAttr ctrlPeer = ctrl.getPeer();
        if (coprocs.empty() && (!ctrlPeer || ctrlPeer.getValue().empty())) {
            return;
        }

        // Remoteness comes from the attribute, not from whether a target module is present.
        bool dispatchedController = ctrl.isOutOfProcess();

        ModuleOp ctrlMod;
        if (dispatchedController) {
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

        SessionEmitter(mod, backline.getTransport(), ctrl, dispatchedController, ctrlMod)
            .run(coprocs);
    }
};

} // namespace
} // namespace transport
} // namespace catalyst
