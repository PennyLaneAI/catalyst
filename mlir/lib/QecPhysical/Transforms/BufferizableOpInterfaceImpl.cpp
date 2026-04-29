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

#include "QecPhysical/Transforms/BufferizableOpInterfaceImpl.h"

#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"

#include "QecPhysical/IR/QecPhysicalDialect.h"
#include "QecPhysical/IR/QecPhysicalOps.h"

using namespace mlir;
using namespace catalyst::qecp;

/**
 * Implementation of the BufferizableOpInterface for use with one-shot bufferization.
 * For more information on the interface, refer to the documentation below:
 *  https://mlir.llvm.org/docs/Bufferization/#extending-one-shot-bufferize
 *  https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Dialect/Bufferization/IR/BufferizableOpInterface.td#L14
 */

namespace {

// Bufferization of qecp.assemble_tanner.
//   - Convert tensor of row_idx to memref.
//   - Convert tensor of col_ptr to memref.
struct AssembleTannerGraphOpInterface
    : public bufferization::BufferizableOpInterface::ExternalModel<AssembleTannerGraphOpInterface,
                                                                   AssembleTannerGraphOp> {
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
        auto assembleTannerOp = cast<AssembleTannerGraphOp>(op);
        Location loc = op->getLoc();

        auto rowIdxTensorType = cast<RankedTensorType>(assembleTannerOp.getRowIdx().getType());
        MemRefType rowIdxMemRefType =
            MemRefType::get(rowIdxTensorType.getShape(), rowIdxTensorType.getElementType());
        auto rowIdxToBufferOp = bufferization::ToBufferOp::create(rewriter, loc, rowIdxMemRefType,
                                                                  assembleTannerOp.getRowIdx());

        auto colPtrTensorType = cast<RankedTensorType>(assembleTannerOp.getColPtr().getType());
        MemRefType colPtrMemRefType =
            MemRefType::get(colPtrTensorType.getShape(), colPtrTensorType.getElementType());
        auto colPtrToBufferOp = bufferization::ToBufferOp::create(rewriter, loc, colPtrMemRefType,
                                                                  assembleTannerOp.getColPtr());

        auto newAssembleTannerOp = AssembleTannerGraphOp::create(
            rewriter, loc, assembleTannerOp.getResult().getType(), rowIdxToBufferOp.getResult(),
            colPtrToBufferOp.getResult());

        bufferization::replaceOpWithBufferizedValues(rewriter, op, newAssembleTannerOp.getResult());
        return success();
    }
};

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

        DecodeEsmCssOp::create(rewriter, loc, TypeRange{}, esmToBufferOp.getResult(),
                               decodeEsmCssOp.getTannerGraph(), errIdxBuffer);

        bufferization::replaceOpWithBufferizedValues(rewriter, op, errIdxBuffer);
        return success();
    }
};

// Bufferization of qecp.decode_physical_meas.
//   - Convert tensor of physical measurements to memref.
//   - Bufferize result tensor of logical measurements with a corresponding memref.alloc; users of
//     the result tensor are updated to use the new memref.
struct DecodePhysicalMeasurementOpInterface
    : public bufferization::BufferizableOpInterface::ExternalModel<
          DecodePhysicalMeasurementOpInterface, DecodePhysicalMeasurementOp> {
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
        auto decodePhysMeasOp = cast<DecodePhysicalMeasurementOp>(op);
        Location loc = op->getLoc();

        auto physMeasTensorType =
            cast<RankedTensorType>(decodePhysMeasOp.getPhysicalMeasurements().getType());
        MemRefType physMeasMemRefType =
            MemRefType::get(physMeasTensorType.getShape(), physMeasTensorType.getElementType());
        auto physMeasToBufferOp = bufferization::ToBufferOp::create(
            rewriter, loc, physMeasMemRefType, decodePhysMeasOp.getPhysicalMeasurements());

        auto logiMeasTensorType =
            cast<RankedTensorType>(decodePhysMeasOp.getLogicalMeasurements().getType());
        MemRefType logiMeasMemRefType =
            MemRefType::get(logiMeasTensorType.getShape(), logiMeasTensorType.getElementType());
        Value logiMeasBuffer = memref::AllocOp::create(rewriter, loc, logiMeasMemRefType);

        DecodePhysicalMeasurementOp::create(rewriter, loc, TypeRange{},
                                            physMeasToBufferOp.getResult(), logiMeasBuffer);

        bufferization::replaceOpWithBufferizedValues(rewriter, op, logiMeasBuffer);
        return success();
    }
};

} // namespace

void catalyst::qecp::registerBufferizableOpInterfaceExternalModels(DialectRegistry &registry)
{
    registry.addExtension(+[](MLIRContext *ctx, catalyst::qecp::QecPhysicalDialect *dialect) {
        AssembleTannerGraphOp::attachInterface<AssembleTannerGraphOpInterface>(*ctx);
        DecodeEsmCssOp::attachInterface<DecodeEsmCssOpInterface>(*ctx);
        DecodePhysicalMeasurementOp::attachInterface<DecodePhysicalMeasurementOpInterface>(*ctx);
    });
}
