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
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include "Catalyst/IR/CatalystOps.h"

#include "Executor/IR/ExecutorOps.h"
#include "Executor/Transforms/Passes.h"

using namespace mlir;

namespace catalyst {
namespace executor {

#define GEN_PASS_DEF_LOWERRUNTIMEDISPATCHPASS
#include "Executor/Transforms/Passes.h.inc"

namespace {

// Width of the fixed NUL-padded field a `str` argument occupies in the wire layout.
// It must match CATALYST_TRANSPORT_STR_BYTES in runtime/include/TransportABI.h.
constexpr int64_t kStrFieldBytes = 256;

// Rewrites dispatched `catalyst.runtime_call` ops into `executor.call`.
struct LowerRuntimeDispatchPass : impl::LowerRuntimeDispatchPassBase<LowerRuntimeDispatchPass> {
    using LowerRuntimeDispatchPassBase::LowerRuntimeDispatchPassBase;

    void runOnOperation() final {
        ModuleOp host = getOperation();

        SmallVector<catalyst::RuntimeCallOp> dispatched;
        host.walk([&](catalyst::RuntimeCallOp call) {
            if (call.isDispatched()) {
                dispatched.push_back(call);
            }
        });
        if (dispatched.empty()) {
            return;
        }

        // The executor a QNode targets, used by a call that names no address of its own.
        llvm::SmallSet<StringRef, 4> qnodeAddresses;
        for (auto nested : host.getBody()->getOps<ModuleOp>()) {
            auto dispatchAttr = nested->getAttrOfType<DictionaryAttr>("catalyst.dispatch");
            if (!dispatchAttr) {
                continue;
            }
            if (auto address = dispatchAttr.getAs<StringAttr>("address")) {
                qnodeAddresses.insert(address.getValue());
            }
        }

        for (catalyst::RuntimeCallOp call : dispatched) {
            if (failed(rewrite(call, qnodeAddresses))) {
                return signalPassFailure();
            }
        }
    }

  private:
    // Sessions opened per (function, address).
    llvm::DenseMap<std::pair<Operation *, Attribute>, Value> sessionCache;

    // Get the executor address for `call`.
    FailureOr<StringAttr> addressOf(catalyst::RuntimeCallOp call,
                                    const llvm::SmallSet<StringRef, 4> &qnodeAddresses) {
        StringAttr dispatch = call.getDispatchAttr();
        if (!dispatch.getValue().empty()) {
            return dispatch;
        }
        if (qnodeAddresses.size() > 1) {
            return call.emitOpError("ambiguous executor: the program targets ")
                   << qnodeAddresses.size()
                   << " executors, so a dispatched call has to name the one it wants";
        }
        if (qnodeAddresses.empty()) {
            return call.emitOpError("dispatch has no executor address");
        }
        return StringAttr::get(&getContext(), *qnodeAddresses.begin());
    }

    // Return the session handle for `addressAttr` in `user`'s function, opening one at the function
    // entry on first use and caching it for subsequent sites.
    Value getOrOpenSession(Operation *user, StringAttr address) {
        auto func = user->getParentOfType<func::FuncOp>();
        std::pair<Operation *, Attribute> key{func.getOperation(), address};
        if (Value cached = sessionCache.lookup(key)) {
            return cached;
        }
        Block &entry = func.getBody().front();

        // Check if a session for this address is already in the entry block.
        // If so, just reuse it.
        for (auto open : entry.getOps<OpenOp>()) {
            if (open.getAddressAttr() == address) {
                sessionCache[key] = open.getSession();
                return open.getSession();
            }
        }

        // Otherwise, create a new session.
        OpBuilder b(&entry, entry.begin());
        Value session =
            OpenOp::create(b, func.getLoc(), SessionType::get(&getContext()), address).getSession();
        sessionCache[key] = session;
        return session;
    }

    // Create a string field for the wire layout.
    static Value stringField(OpBuilder &b, Location loc, StringRef text) {
        SmallVector<int8_t> field(kStrFieldBytes, 0);
        llvm::copy(text, field.begin());
        auto type = RankedTensorType::get({kStrFieldBytes}, b.getI8Type());
        return arith::ConstantOp::create(b, loc, DenseElementsAttr::get(type, ArrayRef(field)))
            .getResult();
    }

    LogicalResult rewrite(catalyst::RuntimeCallOp call,
                          const llvm::SmallSet<StringRef, 4> &qnodeAddresses) {
        FailureOr<StringAttr> address = addressOf(call, qnodeAddresses);
        if (failed(address)) {
            return failure();
        }

        Location loc = call.getLoc();
        OpBuilder b(call);

        SmallVector<Value> inputs;
        unsigned inputIndex = 0;
        unsigned stringIndex = 0;
        ArrayAttr strings = call.getCStringsAttr();
        for (Attribute attr : call.getCParams()) {
            StringRef param = cast<StringAttr>(attr).getValue();
            if (param == "out") {
                continue;
            }
            if (param == "str") {
                StringRef text = cast<StringAttr>(strings[stringIndex++]).getValue();
                inputs.push_back(stringField(b, loc, text));
                continue;
            }
            Value scalar = call.getInputs()[inputIndex++];
            auto type = RankedTensorType::get({1}, scalar.getType());
            inputs.push_back(tensor::FromElementsOp::create(b, loc, type, scalar).getResult());
        }

        SmallVector<Type> replyTypes;
        bool hasResult = !call.getScalarResult().empty();
        if (hasResult) {
            replyTypes.push_back(RankedTensorType::get({1}, call.getScalarResult()[0].getType()));
        }
        for (Value out : call.getOutTensors()) {
            replyTypes.push_back(out.getType());
        }

        Value session = getOrOpenSession(call, *address);
        auto executorCall =
            CallOp::create(b, loc, replyTypes, session, inputs, call.getCalleeAttr(),
                           b.getI32IntegerAttr(static_cast<int32_t>(inputs.size())));

        SmallVector<Value> replacements;
        unsigned replyIndex = 0;
        if (hasResult) {
            Value zero = arith::ConstantOp::create(b, loc, b.getIndexAttr(0)).getResult();
            replacements.push_back(
                tensor::ExtractOp::create(b, loc, executorCall.getResult(replyIndex++), zero)
                    .getResult());
        }
        for (unsigned i = replyIndex, e = executorCall.getNumResults(); i < e; ++i) {
            replacements.push_back(executorCall.getResult(i));
        }

        call.replaceAllUsesWith(replacements);
        call.erase();
        return success();
    }
};

} // namespace

} // namespace executor
} // namespace catalyst
