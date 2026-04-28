// Copyright 2024-2025 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "QecPhysical/Transforms/BufferizableOpInterfaceImpl.h"

#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include "QecPhysical/IR/QecPhysicalDialect.h"
#include "QecPhysical/IR/QecPhysicalOps.h"

#include <mlir/IR/Builders.h>

using namespace mlir;
using namespace catalyst::qecp;

/**
 * Implementation of the BufferizableOpInterface for use with one-shot bufferization.
 * For more information on the interface, refer to the documentation below:
 *  https://mlir.llvm.org/docs/Bufferization/#extending-one-shot-bufferize
 *  https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Dialect/Bufferization/IR/BufferizableOpInterface.td#L14
 */

namespace {

// Bufferization of qecp.decode_esm_css.
//   - Convert tensor of ESMs to memref.
//   - Bufferize result tensor of error indicies with a corresponding memref.alloc; users of the
//     result tensor are updated to use the new memref.
struct DecodeEsmCssOpInterface
    : public bufferization::BufferizableOpInterface::ExternalModel<DecodeEsmCssOpInterface,
                                                                   DecodeEsmCssOp> {
    bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                                const bufferization::AnalysisState &state) const
    {
        return true;
    }

    bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                                 const bufferization::AnalysisState &state) const
    {
        return false;
    }

    bool bufferizesToAllocation(Operation *op, Value value) const { return true; }

    bufferization::AliasingValueList
    getAliasingValues(Operation *op, OpOperand &opOperand,
                      const bufferization::AnalysisState &state) const
    {
        return {};
    }

    LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                            const bufferization::BufferizationOptions &options,
                            bufferization::BufferizationState &state) const
    {
        auto decodeEsmCssOp = cast<DecodeEsmCssOp>(op);
        Location loc = op->getLoc();

        auto esmTensorType = cast<RankedTensorType>(decodeEsmCssOp.getEsm().getType());
        MemRefType esmMemRefType =
            MemRefType::get(esmTensorType.getShape(), esmTensorType.getElementType());
        auto esmToBufferOp = bufferization::ToBufferOp::create(rewriter, loc, esmMemRefType,
                                                               decodeEsmCssOp.getEsm());

        auto errIdxTensorType = cast<RankedTensorType>(decodeEsmCssOp.getErrIdx().getType());
        MemRefType errIdxMemRefType =
            MemRefType::get(errIdxTensorType.getShape(), errIdxTensorType.getElementType());
        Value errIdxBuffer = memref::AllocOp::create(rewriter, loc, errIdxMemRefType);

        DecodeEsmCssOp::create(
            rewriter, loc, TypeRange{},
            esmToBufferOp.getResult(), decodeEsmCssOp.getTannerGraph(), errIdxBuffer);

        bufferization::replaceOpWithBufferizedValues(rewriter, op, errIdxBuffer);
        return success();
    }
};

} // namespace

void catalyst::qecp::registerBufferizableOpInterfaceExternalModels(DialectRegistry &registry)
{
    registry.addExtension(+[](MLIRContext *ctx, catalyst::qecp::QecPhysicalDialect *dialect) {
        DecodeEsmCssOp::attachInterface<DecodeEsmCssOpInterface>(*ctx);
    });
}
