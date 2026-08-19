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
// round over its buffers.
//
//     %buf = memref.alloc()                 %s    = transport.get_session
//     qecp.decode_esm_css %esm in (%buf)    transport.stage_payload %s, %esm
//     %v   = memref.load %buf          ->   transport.post %s
//     memref.dealloc %buf                   %slot = transport.reply_slot %s
//                                           transport.collect %s, %slot
//                                           %v    = memref.load %slot
//
// Whether the reply is collected in the transport's ring slot, as above, or in the buffer the
// program already had, depends on what the round can prove about that buffer; see emitRound.

#include <string>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSwitch.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Interfaces/CallInterfaces.h"
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

    void runOnOperation() override {
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

        auto emitRound = [&](qecp::DecodeEsmCssOp anchor, Value syndrome, Value correction,
                             StringRef key) {
            OpBuilder b(anchor);
            Value s = GetSessionOp::create(b, anchor->getLoc(), ctrlTy, b.getStringAttr(key))
                          .getSession();
            // Each check type is decoded by its own peer-side decoder, and the id travels in
            // the frame, so it is settled when the payload is staged.
            const std::int32_t decoderSlot =
                llvm::StringSwitch<std::int32_t>(
                    cast<qecp::DecodeEsmCssOp>(anchor).getCheckType().value_or(""))
                    .Case("x", 0)
                    .Case("z", 1)
                    .Default(0);
            StagePayloadOp::create(b, anchor->getLoc(), s, syndrome,
                                   b.getI32IntegerAttr(decoderSlot));
            PostOp::create(b, anchor->getLoc(), s);

            // The correction buffer exists only to receive the peer's reply, and the reply
            // already lands in the transport's ring. So the round has a choice of where to
            // collect it, and both of these are valid lowerings of the decode:
            //
            //   into the program's buffer          into the ring slot
            //   -------------------------          ---------------------------------
            //   %buf = memref.alloc()              %slot = transport.reply_slot %s
            //   transport.collect %s, %buf         transport.collect %s, %slot
            //   %v   = memref.load %buf            %v    = memref.load %slot
            //   memref.dealloc %buf
            //
            // The right-hand one costs neither the allocation nor the backend's copy out of the
            // ring, but it hands %slot to code that was written against %buf, so it is valid
            // only while nothing can tell the two apart. Three conditions establish that.

            // (1) The buffer is ours to retire: allocated in this function, and used only by
            //     the decode, by reads, and by its own free. Any other user is a way the
            //     rewrite goes wrong.
            Operation *alloc = correction.getDefiningOp();
            SmallVector<Operation *> frees;
            SmallVector<Operation *> reads;
            bool ownsBuffer = alloc && isa<memref::AllocOp, memref::AllocaOp>(alloc);
            for (Operation *user : correction.getUsers()) {
                if (isa<memref::DeallocOp>(user)) {
                    frees.push_back(user);
                } else if (isa<memref::LoadOp>(user)) {
                    reads.push_back(user);
                } else if (user != anchor) {
                    ownsBuffer = false;
                }
            }

            // (2) Every read still sees this round's reply. The ring recycles a slot every
            //     K_RING_SLOTS rounds, so a read left until later rounds have run picks up one
            //     of their replies, where a buffer would still hold ours. Rounds are a runtime
            //     count, so the rule is one a pass can check: every read is in the anchor's
            //     block, after it, with no further round posted in between. Not "reads dominate
            //     the next post", which rejects a decode in a loop, where the next iteration's
            //     post sits above the read.
            auto opensAnotherRound = [](Operation *op) {
                // An emitted post, a decode not yet lowered, or anything that could hide either.
                return isa<PostOp, qecp::DecodeEsmCssOp>(op) || isa<CallOpInterface>(op) ||
                       op->getNumRegions() > 0;
            };
            size_t promptReads = 0;
            for (Operation *op = anchor->getNextNode(); op && !opensAnotherRound(op);
                 op = op->getNextNode()) {
                promptReads += llvm::is_contained(reads, op);
            }
            const bool readsBeatRecycle = promptReads == reads.size();

            // (3) A slot can describe the buffer's type. reply_slot hands back one ring slot as
            //     a contiguous 1-D span from its base, so a strided, higher-rank or dynamically
            //     shaped buffer has no slot that stands for it.
            auto memTy = dyn_cast<MemRefType>(correction.getType());
            const bool fitsASlot = memTy && memTy.hasStaticShape() && memTy.getRank() == 1 &&
                                   memTy.getLayout().isIdentity();

            if (!ownsBuffer || !readsBeatRecycle || !fitsASlot) {
                // Keep the program's buffer, and let the backend copy the reply out into it.
                CollectOp::create(b, anchor->getLoc(), TypeRange{}, ValueRange{s, correction});
                anchor->erase();
                return;
            }

            Value slot = ReplySlotOp::create(b, anchor->getLoc(), memTy, s).getSlot();
            CollectOp::create(b, anchor->getLoc(), TypeRange{}, ValueRange{s, slot});
            anchor->erase();

            for (Operation *free : frees) {
                free->erase();
            }
            correction.replaceAllUsesWith(slot);
            assert(alloc->use_empty() && "correction buffer outlived the uses (1) allowed");
            alloc->erase();
        };

        SmallVector<qecp::DecodeEsmCssOp> anchors;
        mod.walk([&](qecp::DecodeEsmCssOp op) {
            if (op.isBufferized() && op.getErrIdxIn()) {
                anchors.push_back(op);
            }
        });
        // TODO: This is a naive approach. This strategy should be further analysed and refined.
        // Decodes are handed to the coprocessors round-robin.
        for (auto [k, anchor] : llvm::enumerate(anchors)) {
            emitRound(anchor, anchor.getEsm(), anchor.getErrIdxIn(), peerKeys[k % peerKeys.size()]);
        }
    }
};

} // namespace
} // namespace transport
} // namespace catalyst
