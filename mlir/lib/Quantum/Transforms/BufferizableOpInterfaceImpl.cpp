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

#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include "Quantum/IR/QuantumOps.h"
#include "Quantum/Transforms/BufferizableOpInterfaceImpl.h"

using namespace mlir;
using namespace catalyst::quantum;

/**
 * Implementation of the BufferizableOpInterface for use with one-shot bufferization.
 * For more information on the interface, refer to the documentation below:
 *  https://mlir.llvm.org/docs/Bufferization/#extending-one-shot-bufferize
 *  https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Dialect/Bufferization/IR/BufferizableOpInterface.td#L14
 */

namespace {

// Bufferization of quantum.unitary.
// Convert Matrix into memref.
struct QubitUnitaryOpInterface
    : public bufferization::BufferizableOpInterface::ExternalModel<QubitUnitaryOpInterface,
                                                                   QubitUnitaryOp> {
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
                            const bufferization::BufferizationOptions &options) const
    {
        auto qubitUnitaryOp = cast<QubitUnitaryOp>(op);
        Location loc = op->getLoc();
        auto tensorType = cast<RankedTensorType>(qubitUnitaryOp.getMatrix().getType());
        MemRefType memrefType = MemRefType::get(tensorType.getShape(), tensorType.getElementType());
        auto toMemrefOp =
            rewriter.create<bufferization::ToMemrefOp>(loc, memrefType, qubitUnitaryOp.getMatrix());
        auto memref = toMemrefOp.getResult();
        bufferization::replaceOpWithNewBufferizedOp<QubitUnitaryOp>(
            rewriter, op, qubitUnitaryOp.getOutQubits().getTypes(),
            qubitUnitaryOp.getOutCtrlQubits().getTypes(), memref, qubitUnitaryOp.getInQubits(),
            qubitUnitaryOp.getAdjointAttr(), qubitUnitaryOp.getInCtrlQubits(),
            qubitUnitaryOp.getInCtrlValues());
        return success();
    }
};

// Bufferization of quantum.hermitian.
// Convert Matrix into memref.
struct HermitianOpInterface
    : public bufferization::BufferizableOpInterface::ExternalModel<HermitianOpInterface,
                                                                   HermitianOp> {
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
                            const bufferization::BufferizationOptions &options) const
    {
        auto hermitianOp = cast<HermitianOp>(op);
        Location loc = op->getLoc();
        auto tensorType = cast<RankedTensorType>(hermitianOp.getMatrix().getType());
        MemRefType memrefType = MemRefType::get(tensorType.getShape(), tensorType.getElementType());
        auto toMemrefOp =
            rewriter.create<bufferization::ToMemrefOp>(loc, memrefType, hermitianOp.getMatrix());
        auto memref = toMemrefOp.getResult();
        auto newHermitianOp = rewriter.create<HermitianOp>(loc, hermitianOp.getType(), memref,
                                                           hermitianOp.getQubits());
        bufferization::replaceOpWithBufferizedValues(rewriter, op, newHermitianOp.getObs());

        return success();
    }
};

// Bufferization of quantum.hamiltonian.
// Convert coefficient tensor into memref.
struct HamiltonianOpInterface
    : public bufferization::BufferizableOpInterface::ExternalModel<HamiltonianOpInterface,
                                                                   HamiltonianOp> {
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
                            const bufferization::BufferizationOptions &options) const
    {
        auto hamiltonianOp = cast<HamiltonianOp>(op);
        Location loc = op->getLoc();
        auto tensorType = cast<RankedTensorType>(hamiltonianOp.getCoeffs().getType());
        MemRefType memrefType = MemRefType::get(tensorType.getShape(), tensorType.getElementType());
        auto toMemrefOp =
            rewriter.create<bufferization::ToMemrefOp>(loc, memrefType, hamiltonianOp.getCoeffs());
        auto memref = toMemrefOp.getResult();
        auto newHamiltonianOp = rewriter.create<HamiltonianOp>(loc, hamiltonianOp.getType(), memref,
                                                               hamiltonianOp.getTerms());
        bufferization::replaceOpWithBufferizedValues(rewriter, op, newHamiltonianOp.getObs());

        return success();
    }
};

// Bufferization of quantum.sample.
// Result tensor of quantum.sample is bufferized with a corresponding memref.alloc.
// Users of the result tensor are updated to use the new memref.
struct SampleOpInterface
    : public bufferization::BufferizableOpInterface::ExternalModel<SampleOpInterface, SampleOp> {
    bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                                const bufferization::AnalysisState &state) const
    {
        return false;
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
                            const bufferization::BufferizationOptions &options) const
    {
        auto sampleOp = cast<SampleOp>(op);
        Location loc = op->getLoc();
        auto tensorType = cast<RankedTensorType>(sampleOp.getSamples().getType());
        MemRefType resultType = MemRefType::get(tensorType.getShape(), tensorType.getElementType());

        SmallVector<Value> allocSizes;
        for (Value dynShapeDimension : sampleOp.getDynamicShape()) {
            auto indexCastOp =
                rewriter.create<index::CastSOp>(loc, rewriter.getIndexType(), dynShapeDimension);
            allocSizes.push_back(indexCastOp);
        }

        Value allocVal = rewriter.create<memref::AllocOp>(loc, resultType, allocSizes);
        auto allocedSampleOp = rewriter.create<SampleOp>(
            loc, TypeRange{}, ValueRange{sampleOp.getObs(), allocVal}, op->getAttrs());
        allocedSampleOp->setAttr("operandSegmentSizes", rewriter.getDenseI32ArrayAttr({1, 0, 1}));
        bufferization::replaceOpWithBufferizedValues(rewriter, op, allocVal);
        return success();
    }
};

// Bufferization of quantum.counts.
// Result tensors of quantum.counts are bufferized with corresponding memref.alloc ops.
// Users of the result tensors are updated to use the new memrefs.
struct CountsOpInterface
    : public bufferization::BufferizableOpInterface::ExternalModel<CountsOpInterface, CountsOp> {
    bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                                const bufferization::AnalysisState &state) const
    {
        return false;
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
                            const bufferization::BufferizationOptions &options) const
    {
        auto countsOp = cast<CountsOp>(op);
        Location loc = op->getLoc();

        SmallVector<Value> buffers;
        for (size_t i : {0, 1}) {
            auto tensorType = cast<RankedTensorType>(countsOp.getType(i));
            MemRefType resultType =
                MemRefType::get(tensorType.getShape(), tensorType.getElementType());
            auto shape = cast<mlir::RankedTensorType>(tensorType).getShape();

            Value allocVal;
            if (shape[0] == ShapedType::kDynamic) {
                auto indexCastOp = rewriter.create<index::CastSOp>(loc, rewriter.getIndexType(),
                                                                   countsOp.getDynamicShape());
                allocVal =
                    rewriter.create<memref::AllocOp>(loc, resultType, ValueRange{indexCastOp});
            }
            else {
                allocVal = rewriter.create<memref::AllocOp>(loc, resultType);
            }
            buffers.push_back(allocVal);
        }

        rewriter.create<CountsOp>(loc, nullptr, nullptr, countsOp.getObs(), nullptr, buffers[0],
                                  buffers[1]);
        bufferization::replaceOpWithBufferizedValues(rewriter, op, buffers);

        return success();
    }
};

// Bufferization of quantum.probs.
// Result tensor of quantum.probs is bufferized with a corresponding memref.alloc.
// Users of the result tensor are updated to use the new memref.
struct ProbsOpInterface
    : public bufferization::BufferizableOpInterface::ExternalModel<ProbsOpInterface, ProbsOp> {
    bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                                const bufferization::AnalysisState &state) const
    {
        return false;
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
                            const bufferization::BufferizationOptions &options) const
    {
        auto probsOp = cast<ProbsOp>(op);
        Location loc = op->getLoc();
        auto tensorType = cast<RankedTensorType>(probsOp.getProbabilities().getType());
        MemRefType resultType = MemRefType::get(tensorType.getShape(), tensorType.getElementType());

        Value buffer;
        auto shape = cast<mlir::RankedTensorType>(tensorType).getShape();
        if (shape[0] == ShapedType::kDynamic) {
            auto indexCastOp = rewriter.create<index::CastSOp>(loc, rewriter.getIndexType(),
                                                               probsOp.getDynamicShape());
            buffer = rewriter.create<memref::AllocOp>(loc, resultType, ValueRange{indexCastOp});
        }
        else {
            buffer = rewriter.create<memref::AllocOp>(loc, resultType);
        }

        auto allocedProbsOp =
            rewriter.create<ProbsOp>(loc, TypeRange{}, ValueRange{probsOp.getObs(), buffer});
        allocedProbsOp->setAttr("operandSegmentSizes", rewriter.getDenseI32ArrayAttr({1, 0, 1}));
        bufferization::replaceOpWithBufferizedValues(rewriter, op, buffer);
        return success();
    }
};

// Bufferization of quantum.state.
// Result tensor of quantum.state is bufferized with a corresponding memref.alloc.
// Users of the result tensor are updated to use the new memref.
struct StateOpInterface
    : public bufferization::BufferizableOpInterface::ExternalModel<StateOpInterface, StateOp> {
    bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                                const bufferization::AnalysisState &state) const
    {
        return false;
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
                            const bufferization::BufferizationOptions &options) const
    {
        auto stateOp = cast<StateOp>(op);
        Location loc = op->getLoc();
        auto tensorType = cast<RankedTensorType>(stateOp.getState().getType());
        MemRefType resultType = MemRefType::get(tensorType.getShape(), tensorType.getElementType());

        Value buffer;
        auto shape = cast<RankedTensorType>(tensorType).getShape();
        if (shape[0] == ShapedType::kDynamic) {
            auto indexCastOp = rewriter.create<index::CastSOp>(loc, rewriter.getIndexType(),
                                                               stateOp.getDynamicShape());
            buffer = rewriter.create<memref::AllocOp>(loc, resultType, ValueRange{indexCastOp});
        }
        else {
            buffer = rewriter.create<memref::AllocOp>(loc, resultType);
        }

        auto allocedStateOp =
            rewriter.create<StateOp>(loc, TypeRange{}, ValueRange{stateOp.getObs(), buffer});
        allocedStateOp->setAttr("operandSegmentSizes", rewriter.getDenseI32ArrayAttr({1, 0, 1}));
        bufferization::replaceOpWithBufferizedValues(rewriter, op, buffer);
        return success();
    }
};

// Bufferization of quantum.set_state.
// Convert InState into memref.
struct SetStateOpInterface
    : public bufferization::BufferizableOpInterface::ExternalModel<SetStateOpInterface,
                                                                   SetStateOp> {
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
                            const bufferization::BufferizationOptions &options) const
    {
        auto setStateOp = cast<SetStateOp>(op);
        Location loc = op->getLoc();
        auto tensorType = cast<RankedTensorType>(setStateOp.getInState().getType());
        MemRefType memrefType = MemRefType::get(tensorType.getShape(), tensorType.getElementType());

        auto toMemrefOp =
            rewriter.create<bufferization::ToMemrefOp>(loc, memrefType, setStateOp.getInState());
        auto memref = toMemrefOp.getResult();
        auto newSetStateOp = rewriter.create<SetStateOp>(loc, setStateOp.getOutQubits().getTypes(),
                                                         memref, setStateOp.getInQubits());
        bufferization::replaceOpWithBufferizedValues(rewriter, op, newSetStateOp.getOutQubits());
        return success();
    }
};

// Bufferization of quantum.set_basis_state.
// Convert BasisState into memref.
struct SetBasisStateOpInterface
    : public bufferization::BufferizableOpInterface::ExternalModel<SetBasisStateOpInterface,
                                                                   SetBasisStateOp> {
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
                            const bufferization::BufferizationOptions &options) const
    {
        auto setBasisStateOp = cast<SetBasisStateOp>(op);
        Location loc = op->getLoc();
        auto tensorType = cast<RankedTensorType>(setBasisStateOp.getBasisState().getType());
        MemRefType memrefType = MemRefType::get(tensorType.getShape(), tensorType.getElementType());

        auto toMemrefOp = rewriter.create<bufferization::ToMemrefOp>(
            loc, memrefType, setBasisStateOp.getBasisState());
        auto memref = toMemrefOp.getResult();
        auto newSetStateOp = rewriter.create<SetBasisStateOp>(
            loc, setBasisStateOp.getOutQubits().getTypes(), memref, setBasisStateOp.getInQubits());
        bufferization::replaceOpWithBufferizedValues(rewriter, op, newSetStateOp.getOutQubits());
        return success();
    }
};

} // namespace

void catalyst::quantum::registerBufferizableOpInterfaceExternalModels(DialectRegistry &registry)
{
    registry.addExtension(+[](MLIRContext *ctx, catalyst::quantum::QuantumDialect *dialect) {
        QubitUnitaryOp::attachInterface<QubitUnitaryOpInterface>(*ctx);
        HermitianOp::attachInterface<HermitianOpInterface>(*ctx);
        HamiltonianOp::attachInterface<HamiltonianOpInterface>(*ctx);
        SampleOp::attachInterface<SampleOpInterface>(*ctx);
        CountsOp::attachInterface<CountsOpInterface>(*ctx);
        ProbsOp::attachInterface<ProbsOpInterface>(*ctx);
        StateOp::attachInterface<StateOpInterface>(*ctx);
        SetStateOp::attachInterface<SetStateOpInterface>(*ctx);
        SetBasisStateOp::attachInterface<SetBasisStateOpInterface>(*ctx);
    });
}
