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

#include <string>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/Pass.h"

#include "QecPhysical/IR/QecPhysicalOps.h"

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
        auto backline = mod->getAttrOfType<BacklineAttr>(kBacklineAttr);
        if (!backline) {
            return;
        }

        SmallVector<std::string> peerKeys;
        for (auto [i, coproc] : llvm::enumerate(backline.getCoprocessors())) {
            peerKeys.push_back(coproc.keyOr("coprocessor." + std::to_string(i)).str());
        }
        // The controller is not an offload target, so with no coprocessors declared,
        // there is nowhere to send the decode, and it stays local.
        if (peerKeys.empty()) {
            return;
        }

        MLIRContext *ctx = &getContext();
        auto ctrlTy = SessionType::get(ctx, Role::Controller);

        auto emitRound = [&](Operation *anchor, Value syndrome, Value correction, StringRef key) {
            OpBuilder b(anchor);
            Value s = GetSessionOp::create(b, anchor->getLoc(), ctrlTy, b.getStringAttr(key))
                          .getSession();
            KickOp::create(b, anchor->getLoc(), s, syndrome, b.getI32IntegerAttr(0));
            CollectOp::create(b, anchor->getLoc(), TypeRange{}, ValueRange{s, correction});
            anchor->erase();
        };

        SmallVector<qecp::DecodeEsmCssOp> anchors;
        mod.walk([&](qecp::DecodeEsmCssOp op) {
            if (op.isBufferized() && op.getErrIdxIn()) {
                anchors.push_back(op);
            }
        });
        // Distribute the decoding tasks across the coprocessors in a round-robin fashion.
        for (size_t k = 0; k < anchors.size(); ++k) {
            qecp::DecodeEsmCssOp anchor = anchors[k];
            emitRound(anchor, anchor.getEsm(), anchor.getErrIdxIn(), peerKeys[k % peerKeys.size()]);
        }
    }
};

} // namespace
} // namespace transport
} // namespace catalyst
