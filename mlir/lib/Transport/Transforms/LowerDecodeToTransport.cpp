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

// lower-decode-to-transport: replace each bufferized `qecp.decode_esm_css` op with a transport
// round over its buffers:
//     %s = transport.get_session : !transport.session<controller>
//     transport.kick    %s, %syndrome
//     transport.collect %s, %correction

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"

#include <string>

#include "Transport/IR/TransportOps.h"
#include "Transport/Transforms/Passes.h"

using namespace mlir;
using namespace catalyst::transport;

namespace catalyst {
namespace transport {

#define GEN_PASS_DEF_LOWERDECODETOTRANSPORTPASS
#include "Transport/Transforms/Passes.h.inc"

namespace {

constexpr llvm::StringRef kBacklineAttr = "catalyst.backline";

struct LowerDecodeToTransportPass
    : public impl::LowerDecodeToTransportPassBase<LowerDecodeToTransportPass> {
    using LowerDecodeToTransportPassBase::LowerDecodeToTransportPassBase;

    void runOnOperation() override
    {
        ModuleOp mod = getOperation();
        auto backline = mod->getAttrOfType<DictionaryAttr>(kBacklineAttr);
        if (!backline)
            return;

        MLIRContext *ctx = &getContext();
        auto ctrlTy = SessionType::get(ctx, Role::Controller);

        SmallVector<std::string> peerKeys;
        if (auto arr = backline.getAs<ArrayAttr>("coprocessors")) {
            for (size_t i = 0; i < arr.size(); ++i) {
                StringRef name;
                if (auto d = dyn_cast<DictionaryAttr>(arr[i]))
                    if (auto n = d.getAs<StringAttr>("name"); n && !n.getValue().empty())
                        name = n.getValue();
                peerKeys.push_back(name.empty() ? ("coprocessor." + std::to_string(i)) : name.str());
            }
        }
        if (peerKeys.empty()) {
            StringRef name;
            if (auto c = backline.getAs<DictionaryAttr>("controller"))
                if (auto n = c.getAs<StringAttr>("name"); n && !n.getValue().empty())
                    name = n.getValue();
            peerKeys.push_back(name.empty() ? "controller" : name.str());
        }

        auto emitRound = [&](Operation *anchor, Value syndrome, Value correction, StringRef key) {
            OpBuilder b(anchor);
            Value s = GetSessionOp::create(b, anchor->getLoc(), ctrlTy, b.getStringAttr(key))
                          .getSession();
            KickOp::create(b, anchor->getLoc(), s, syndrome, b.getI32IntegerAttr(0));
            CollectOp::create(b, anchor->getLoc(), TypeRange{}, ValueRange{s, correction});
            anchor->erase();
        };

        SmallVector<Operation *> anchors;
        mod.walk([&](Operation *op) {
            if (op->getName().getStringRef() == "qecp.decode_esm_css" && op->getNumResults() == 0 &&
                op->getNumOperands() == 3)
                anchors.push_back(op);
        });
        for (size_t k = 0; k < anchors.size(); ++k) {
            Operation *anchor = anchors[k];
            std::string key;
            if (auto tag = anchor->getAttrOfType<StringAttr>("transport.peer"))
                key = tag.getValue().str();
            else
                key = peerKeys[k % peerKeys.size()];
            emitRound(anchor, anchor->getOperand(0), anchor->getOperand(2), key);
        }
    }
};

} // namespace
} // namespace transport
} // namespace catalyst
