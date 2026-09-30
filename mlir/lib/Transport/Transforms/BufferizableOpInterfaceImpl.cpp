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

// BufferizableOpInterface external models for the transport ops, so kick/collect can be created in
// their tensor (value-semantics) forms before bufferization and lowered to the memref dest-passing
// forms by one-shot bufferize. See:
//   https://mlir.llvm.org/docs/Bufferization/#extending-one-shot-bufferize

#include "Transport/Transforms/BufferizableOpInterfaceImpl.h"

#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"

#include "Transport/IR/TransportDialect.h"
#include "Transport/IR/TransportOps.h"

using namespace mlir;
using namespace catalyst::transport;

namespace {

struct StagePayloadOpInterface
    : public bufferization::BufferizableOpInterface::ExternalModel<StagePayloadOpInterface,
                                                                   StagePayloadOp> {
    bool bufferizesToMemoryRead(Operation *, OpOperand &,
                                const bufferization::AnalysisState &) const {
        return true;
    }
    bool bufferizesToMemoryWrite(Operation *, OpOperand &,
                                 const bufferization::AnalysisState &) const {
        return false;
    }
    bufferization::AliasingValueList getAliasingValues(Operation *, OpOperand &,
                                                       const bufferization::AnalysisState &) const {
        return {};
    }
    LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                            const bufferization::BufferizationOptions &options,
                            bufferization::BufferizationState &state) const {
        auto stageOp = cast<StagePayloadOp>(op);
        if (!isa<RankedTensorType>(stageOp.getPayload().getType())) {
            return success(); // already a memref
        }
        Location loc = op->getLoc();

        FailureOr<Value> payloadBuffer = getBuffer(rewriter, stageOp.getPayload(), options, state);
        if (failed(payloadBuffer)) {
            return failure();
        }

        // The transport backend reads the payload as a contiguous block through the memref's
        // aligned pointer, ignoring its strides and offset. Copy any non-identity-layout payload
        // into a fresh contiguous buffer first so the intended elements are sent.
        Value buffer = *payloadBuffer;
        auto memrefTy = cast<MemRefType>(buffer.getType());
        if (!memrefTy.getLayout().isIdentity()) {
            MemRefType contiguousTy =
                MemRefType::get(memrefTy.getShape(), memrefTy.getElementType());
            auto alloc = memref::AllocOp::create(rewriter, loc, contiguousTy);
            memref::CopyOp::create(rewriter, loc, buffer, alloc.getResult());
            buffer = alloc.getResult();
        }

        StagePayloadOp::create(rewriter, loc, stageOp.getSession(), buffer);
        rewriter.eraseOp(op);
        return success();
    }
};

struct CollectOpInterface
    : public bufferization::BufferizableOpInterface::ExternalModel<CollectOpInterface, CollectOp> {
    bool bufferizesToAllocation(Operation *, Value) const { return true; }
    bool bufferizesToMemoryRead(Operation *, OpOperand &,
                                const bufferization::AnalysisState &) const {
        return false;
    }
    bool bufferizesToMemoryWrite(Operation *, OpOperand &,
                                 const bufferization::AnalysisState &) const {
        return false;
    }
    bufferization::AliasingValueList getAliasingValues(Operation *, OpOperand &,
                                                       const bufferization::AnalysisState &) const {
        return {};
    }
    LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                            const bufferization::BufferizationOptions &options,
                            bufferization::BufferizationState &state) const {
        auto collectOp = cast<CollectOp>(op);
        if (!collectOp.getResult()) {
            return success(); // already dest-passing
        }
        Location loc = op->getLoc();
        auto tensorTy = cast<RankedTensorType>(collectOp.getResult().getType());

        FailureOr<Value> tensorAlloc = bufferization::allocateTensorForShapedValue(
            rewriter, loc, collectOp.getResult(), options, state, /*copy=*/false);
        if (failed(tensorAlloc)) {
            return failure();
        }
        MemRefType memTy = MemRefType::get(tensorTy.getShape(), tensorTy.getElementType());
        Value buffer =
            bufferization::ToBufferOp::create(rewriter, loc, memTy, *tensorAlloc).getResult();
        CollectOp::create(rewriter, loc, TypeRange{}, ValueRange{collectOp.getSession(), buffer});
        bufferization::replaceOpWithBufferizedValues(rewriter, op, buffer);
        return success();
    }
};

} // namespace

void catalyst::transport::registerBufferizableOpInterfaceExternalModels(DialectRegistry &registry) {
    registry.addExtension(+[](MLIRContext *ctx, TransportDialect *dialect) {
        StagePayloadOp::attachInterface<StagePayloadOpInterface>(*ctx);
        CollectOp::attachInterface<CollectOpInterface>(*ctx);
    });
}
