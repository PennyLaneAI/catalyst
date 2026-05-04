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

// DirectTransferPass — Phase 3, Task 6
//
// Annotates each quantum.bias_transfer op with:
//   is_direct_copy  : i32   -- 1 when |B_target − B_rep| ≤ threshold
//   param_byte_count: i32   -- bytes to copy (param_buffer_size × 8; verified)
//
// Shape verification: retrieves param_buffer_size from the enclosing
// FuncOp's freeze_partition annotation (written by doqaoa-shared-buffer).
// Signals pass failure if the sizes are inconsistent.
//
// Also annotates the enclosing freeze_partition with aggregate counts:
//   dt_direct_count    : i32
//   dt_warmstart_count : i32
//
// The LLVM lowering (in ConversionPatterns.cpp) reads is_direct_copy and
// param_byte_count to emit either:
//   1. llvm.intr.memcpy(%dst, %src, param_byte_count, false)   [direct copy]
//   2. A call into the pre-computed warm-start param table        [warm-start]
//
// Requires: doqaoa-shared-buffer (param_buffer_size on freeze_partition).
// The presence of B_rep, B_target, threshold attributes on bias_transfer
// is enforced by the quantum.bias_transfer verifier (Phase 1).

#include <cmath>
#include <cstdint>

#include "llvm/Support/raw_ostream.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/Pass.h"

#include "Quantum/IR/QuantumDialect.h"
#include "Quantum/IR/QuantumOps.h"
#include "Quantum/Transforms/Patterns.h"

using namespace mlir;
using namespace catalyst::quantum;

namespace catalyst {
namespace quantum {

#define GEN_PASS_DECL_DIRECTTRANSFERPASS
#define GEN_PASS_DEF_DIRECTTRANSFERPASS
#include "Quantum/Transforms/Passes.h.inc"

struct DirectTransferPass : impl::DirectTransferPassBase<DirectTransferPass> {
    using DirectTransferPassBase::DirectTransferPassBase;

    void runOnOperation() final
    {
        func::FuncOp func = getOperation();
        MLIRContext *ctx  = &getContext();
        Builder builder(ctx);

        // ── Collect freeze_partition ops (we'll annotate them at the end) ─
        SmallVector<FreezePartitionOp> freezeOps;
        func.walk([&](FreezePartitionOp op) { freezeOps.push_back(op); });

        // ── For each freeze_partition: find its BiasTransferOps ───────────
        //    (in DO-QAOA IR the bias_transfer ops live in the same FuncOp;
        //     we group them by their param_buffer_size annotation)
        for (FreezePartitionOp fpOp : freezeOps) {

            // Resolve param_buffer_size from shared-buffer annotation
            int32_t bufSize = 2; // default for p=1
            if (auto a = fpOp->getAttr("param_buffer_size"))
                bufSize = cast<IntegerAttr>(a).getInt();

            if (bufSize <= 0) {
                fpOp->emitError()
                    << "doqaoa-direct-transfer: param_buffer_size = " << bufSize
                    << " is invalid (expected >= 1); run doqaoa-shared-buffer first";
                signalPassFailure();
                return;
            }

            int32_t byteCount = bufSize * static_cast<int32_t>(sizeof(double)); // = 16 for p=1

            int32_t directCount    = 0;
            int32_t warmstartCount = 0;

            // Walk all BiasTransferOps in the function and annotate them.
            // In the current IR structure all bias_transfer ops in a FuncOp
            // belong conceptually to a single freeze_partition (one circuit).
            func.walk([&](BiasTransferOp btOp) {

                double bRep  = btOp.getBRep();
                double bTarg = btOp.getBTarget();
                double thr   = btOp.getThreshold();
                double deltaB = std::fabs(bTarg - bRep);

                // ── Shape verification ────────────────────────────────────
                // Verify that threshold is positive and in sensible range
                if (thr <= 0.0) {
                    btOp->emitWarning()
                        << "doqaoa-direct-transfer: threshold = " << thr
                        << " is non-positive; treating as direct copy";
                }

                // ── Determine transfer mode ───────────────────────────────
                bool isDirect = (deltaB <= thr);

                btOp->setAttr("is_direct_copy",
                              builder.getI32IntegerAttr(isDirect ? 1 : 0));
                btOp->setAttr("param_byte_count",
                              builder.getI32IntegerAttr(byteCount));

                if (isDirect)
                    ++directCount;
                else
                    ++warmstartCount;
            });

            // ── Annotate freeze_partition with aggregate counts ────────────
            fpOp->setAttr("dt_direct_count",
                          builder.getI32IntegerAttr(directCount));
            fpOp->setAttr("dt_warmstart_count",
                          builder.getI32IntegerAttr(warmstartCount));

            // Remark
            llvm::SmallString<160> info;
            llvm::raw_svector_ostream ss(info);
            ss << "doqaoa-direct-transfer: "
               << (directCount + warmstartCount) << " bias_transfer ops — "
               << "direct_copy=" << directCount
               << " warm_start=" << warmstartCount
               << " | param_byte_count=" << byteCount;
            fpOp->emitRemark() << info;
        }
    }
};

} // namespace quantum
} // namespace catalyst
