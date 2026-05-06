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
#include "mlir/IR/BuiltinTypes.h"

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

/// If \p v is a ranked tensor, bufferize it with `bufferization::ToBufferOp` and return the
/// memref. If it is already a memref, return it unchanged.
static FailureOr<Value> bufferizeToMemRef(RewriterBase &rewriter, Location loc, Value v)
{
    if (auto tensorTy = dyn_cast<RankedTensorType>(v.getType())) {
        MemRefType memrefTy =
            MemRefType::get(tensorTy.getShape(), tensorTy.getElementType());
        return bufferization::ToBufferOp::create(rewriter, loc, memrefTy, v).getResult();
    }
    if (isa<MemRefType>(v.getType()))
        return v;
    return failure();
}

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

        FailureOr<Value> rowIdxBuf =
            bufferizeToMemRef(rewriter, loc, assembleTannerOp.getRowIdx());
        FailureOr<Value> colPtrBuf =
            bufferizeToMemRef(rewriter, loc, assembleTannerOp.getColPtr());
        if (failed(rowIdxBuf) || failed(colPtrBuf))
            return assembleTannerOp.emitOpError(
                "row_idx and col_ptr must be ranked tensors or 1-D memrefs");

        auto newAssembleTannerOp = AssembleTannerGraphOp::create(
            rewriter, loc, assembleTannerOp.getResult().getType(), *rowIdxBuf, *colPtrBuf);

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

        FailureOr<Value> esmBuf = bufferizeToMemRef(rewriter, loc, decodeEsmCssOp.getEsm());
        if (failed(esmBuf))
            return decodeEsmCssOp.emitOpError("ESM operand must be a ranked tensor or 1-D memref");

        Value errIdxBuffer;
        if (Value inBuf = decodeEsmCssOp.getErrIdxIn()) {
            errIdxBuffer = inBuf;
        }
        else {
            if (decodeEsmCssOp->getNumResults() == 0)
                return decodeEsmCssOp.emitOpError(
                    "bufferization requires a tensor result or err_idx_in memref");
            auto errIdxTensorTy =
                dyn_cast<RankedTensorType>(decodeEsmCssOp.getResultTypes().front());
            if (!errIdxTensorTy)
                return decodeEsmCssOp.emitOpError("tensor result must be a ranked tensor");
            MemRefType errIdxMemRefType =
                MemRefType::get(errIdxTensorTy.getShape(), errIdxTensorTy.getElementType());
            errIdxBuffer = memref::AllocOp::create(rewriter, loc, errIdxMemRefType);
        }

        DecodeEsmCssOp::create(rewriter, loc, TypeRange{}, *esmBuf,
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

        FailureOr<Value> physMeasBuf =
            bufferizeToMemRef(rewriter, loc, decodePhysMeasOp.getPhysicalMeasurements());
        if (failed(physMeasBuf))
            return decodePhysMeasOp.emitOpError(
                "physical_measurements must be a ranked tensor or 1-D memref");

        Value logiMeasBuffer;
        if (Value inBuf = decodePhysMeasOp.getLogicalMeasurementsIn()) {
            logiMeasBuffer = inBuf;
        }
        else {
            if (decodePhysMeasOp->getNumResults() == 0)
                return decodePhysMeasOp.emitOpError(
                    "bufferization requires a tensor result or logical_measurements_in memref");
            auto logiMeasTensorTy =
                dyn_cast<RankedTensorType>(decodePhysMeasOp.getResultTypes().front());
            if (!logiMeasTensorTy)
                return decodePhysMeasOp.emitOpError("tensor result must be a ranked tensor");
            MemRefType logiMeasMemRefType =
                MemRefType::get(logiMeasTensorTy.getShape(), logiMeasTensorTy.getElementType());
            logiMeasBuffer = memref::AllocOp::create(rewriter, loc, logiMeasMemRefType);
        }

        DecodePhysicalMeasurementOp::create(rewriter, loc, TypeRange{}, *physMeasBuf,
                                            logiMeasBuffer);

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
