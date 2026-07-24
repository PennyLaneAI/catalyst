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
// session lifecycle into the HOST entry function.

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <string>

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
constexpr llvm::StringRef kEmitCInterface = "llvm.emit_c_interface";

struct InjectTransportSessionPass
    : public impl::InjectTransportSessionPassBase<InjectTransportSessionPass> {
    using InjectTransportSessionPassBase::InjectTransportSessionPassBase;

    void runOnOperation() override
    {
        ModuleOp mod = getOperation();
        auto backline = mod->getAttrOfType<DictionaryAttr>(kBacklineAttr);
        if (!backline)
            return;
        auto ctrl = backline.getAs<DictionaryAttr>("controller");
        if (!ctrl)
            return;

        func::FuncOp host;
        mod.walk([&](func::FuncOp fn) {
            if (fn->hasAttr(kEmitCInterface) && !fn.getBody().empty())
                host = fn;
        });
        if (!host) {
            mod.emitError("inject-transport-session: no host (llvm.emit_c_interface) function found");
            return signalPassFailure();
        }

        MLIRContext *ctx = &getContext();
        Location loc = host.getLoc();
        auto ctrlTy = SessionType::get(ctx, Role::Controller);
        auto coTy = SessionType::get(ctx, Role::Coprocessor);
        auto tokTy = TokenType::get(ctx);
        SmallVector<DictionaryAttr> coprocs;
        if (auto arr = backline.getAs<ArrayAttr>("coprocessors")) {
            for (Attribute a : arr)
                if (auto d = dyn_cast<DictionaryAttr>(a))
                    coprocs.push_back(d);
        }
        else if (auto d = backline.getAs<DictionaryAttr>("coprocessor")) {
            coprocs.push_back(d);
        }

        auto strOf = [&](DictionaryAttr d, StringRef key) -> StringAttr {
            if (d)
                if (auto a = d.getAs<StringAttr>(key))
                    return a;
            return StringAttr::get(ctx, "");
        };
        auto i64Of = [&](DictionaryAttr d, StringRef key, int64_t dflt) -> int64_t {
            if (d)
                if (auto a = d.getAs<IntegerAttr>(key))
                    return a.getInt();
            return dflt;
        };
        auto dataPathOf = [&](DictionaryAttr d, StringRef dflt) -> StringAttr {
            auto a = d ? d.getAs<StringAttr>("data_path") : StringAttr();
            return StringAttr::get(ctx, a ? a.getValue() : dflt);
        };
        auto keyOf = [&](DictionaryAttr d, StringRef fallback) -> StringAttr {
            if (auto n = d ? d.getAs<StringAttr>("name") : StringAttr(); n && !n.getValue().empty())
                return n;
            return StringAttr::get(ctx, fallback);
        };

        Block &entry = host.getBody().front();

        // ---- bring-up ----
        OpBuilder b(ctx);
        b.setInsertionPointToStart(&entry);
        auto i16A = [&](int64_t v) { return b.getIntegerAttr(b.getIntegerType(16), v); };
        auto commit = [&](Value ct) {
            CommitWorkItemOp::create(b, loc, ct, b.getI32IntegerAttr(i64Of(ctrl, "work_item_idx", 0)),
                                     b.getI64IntegerAttr(i64Of(ctrl, "in_bytes", 8)),
                                     b.getI64IntegerAttr(i64Of(ctrl, "out_bytes", 8)));
        };

        SmallVector<Value> sessions;

        if (coprocs.empty()) {
            // Controller-only
            Value ct = CreateOp::create(b, loc, ctrlTy, strOf(ctrl, "backend_lib"),
                                        strOf(ctrl, "config"), keyOf(ctrl, "controller"))
                           .getSession();
            ConnectOp::create(b, loc, ct, strOf(ctrl, "peer"), i16A(i64Of(ctrl, "oob_port", 0)));
            ExchangeKeysOp::create(b, loc, ct);
            EstablishChannelOp::create(b, loc, ct, dataPathOf(ctrl, "cpu_verbs"));
            commit(ct);
            StartOp::create(b, loc, ct);
            sessions.push_back(ct);
        }
        for (auto [i, coproc] : llvm::enumerate(coprocs)) {
            StringAttr key = keyOf(coproc, "coprocessor." + std::to_string(i));
            Value ct = CreateOp::create(b, loc, ctrlTy, strOf(ctrl, "backend_lib"),
                                        strOf(ctrl, "config"), key)
                           .getSession();
            Value co = CreateOp::create(b, loc, coTy, strOf(coproc, "backend_lib"),
                                        strOf(coproc, "config"), /*key=*/StringAttr::get(ctx, ""))
                           .getSession();
            Value t1 = ConnectAsyncOp::create(b, loc, tokTy, co, strOf(coproc, "peer"),
                                              i16A(i64Of(coproc, "oob_port", 0)))
                           .getToken();
            ConnectOp::create(b, loc, ct, strOf(coproc, "peer"), i16A(i64Of(coproc, "oob_port", 0)));
            BarrierOp::create(b, loc, t1);
            Value t2 = ExchangeKeysAsyncOp::create(b, loc, tokTy, co).getToken();
            ExchangeKeysOp::create(b, loc, ct);
            BarrierOp::create(b, loc, t2);
            EstablishChannelOp::create(b, loc, co, dataPathOf(coproc, "gpu_engine"));
            EstablishChannelOp::create(b, loc, ct, dataPathOf(ctrl, "cpu_verbs"));
            SetCoprocessorFnOp::create(b, loc, co, strOf(coproc, "symbol"));
            commit(ct);
            StartOp::create(b, loc, co);
            StartOp::create(b, loc, ct);
            sessions.push_back(co);
            sessions.push_back(ct);
        }

        // ---- teardown ----
        b.setInsertionPoint(entry.getTerminator());
        for (Value s : llvm::reverse(sessions)) {
            StopOp::create(b, loc, s);
            DestroyOp::create(b, loc, s);
        }
    }
};

} // namespace
} // namespace transport
} // namespace catalyst
