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

struct KickOpInterface
    : public bufferization::BufferizableOpInterface::ExternalModel<KickOpInterface, KickOp> {
    bool bufferizesToMemoryRead(Operation *, OpOperand &,
                                const bufferization::AnalysisState &) const
    {
        return true;
    }
    bool bufferizesToMemoryWrite(Operation *, OpOperand &,
                                 const bufferization::AnalysisState &) const
    {
        return false;
    }
    bufferization::AliasingValueList getAliasingValues(Operation *, OpOperand &,
                                                       const bufferization::AnalysisState &) const
    {
        return {};
    }
    LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                            const bufferization::BufferizationOptions &options,
                            bufferization::BufferizationState &state) const
    {
        auto kickOp = cast<KickOp>(op);
        auto tensorTy = dyn_cast<RankedTensorType>(kickOp.getPayload().getType());
        if (!tensorTy)
            return success(); // already a memref
        Location loc = op->getLoc();
        MemRefType memTy = MemRefType::get(tensorTy.getShape(), tensorTy.getElementType());
        auto toBuffer =
            bufferization::ToBufferOp::create(rewriter, loc, memTy, kickOp.getPayload());
        KickOp::create(rewriter, loc, kickOp.getSession(), toBuffer.getResult(),
                       kickOp.getWorkItemIdxAttr());
        rewriter.eraseOp(op);
        return success();
    }
};

struct CollectOpInterface
    : public bufferization::BufferizableOpInterface::ExternalModel<CollectOpInterface, CollectOp> {
    bool bufferizesToMemoryRead(Operation *, OpOperand &,
                                const bufferization::AnalysisState &) const
    {
        return false;
    }
    bool bufferizesToMemoryWrite(Operation *, OpOperand &,
                                 const bufferization::AnalysisState &) const
    {
        return false;
    }
    bufferization::AliasingValueList getAliasingValues(Operation *, OpOperand &,
                                                       const bufferization::AnalysisState &) const
    {
        return {};
    }
    LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                            const bufferization::BufferizationOptions &options,
                            bufferization::BufferizationState &state) const
    {
        auto collectOp = cast<CollectOp>(op);
        if (!collectOp.getResult())
            return success(); // already dest-passing
        Location loc = op->getLoc();
        auto tensorTy = cast<RankedTensorType>(collectOp.getResult().getType());
        MemRefType memTy = MemRefType::get(tensorTy.getShape(), tensorTy.getElementType());
        Value buffer = memref::AllocOp::create(rewriter, loc, memTy);
        CollectOp::create(rewriter, loc, TypeRange{}, ValueRange{collectOp.getSession(), buffer});
        bufferization::replaceOpWithBufferizedValues(rewriter, op, buffer);
        return success();
    }
};

} // namespace

void catalyst::transport::registerBufferizableOpInterfaceExternalModels(DialectRegistry &registry)
{
    registry.addExtension(+[](MLIRContext *ctx, TransportDialect *dialect) {
        KickOp::attachInterface<KickOpInterface>(*ctx);
        CollectOp::attachInterface<CollectOpInterface>(*ctx);
    });
}
