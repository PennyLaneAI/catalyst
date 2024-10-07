#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Quantum/IR/QuantumOps.h"
#include "Quantum/Transforms/BufferizableOpInterfaceImpl.h"

using namespace mlir;
using namespace catalyst::quantum;

namespace {
/**
 * The new bufferization interface requires `bufferizesToMemoryRead`, `bufferizesToMemoryWrite`,
 * and `getAliasingValues`.
 *
 * `bufferizesToMemoryRead`: Return `true` if the buffer of the given tensor OpOperand is read.
 *
 * `bufferizesToMemoryWrite`: Return `true` if the buffer of the given tensor OpOperand is written
 * (if bufferizing in-place).
 *
 * `getAliasingOpOperands`: Return the OpResults that may share the same buffer as the given
 * OpOperand. Note that MLIR documentation does not mention `getAliasingValues` but it seems to
 * serve the same purpose.
 *
 * Link: https://mlir.llvm.org/docs/Bufferization/#extending-one-shot-bufferize
 */

/// Bufferization of catalyst.quantum.unitary. Convert Matrix into memref.
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

/// Bufferization of catalyst.quantum.hermitian. Convert Matrix into memref.
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

/// Bufferization of catalyst.quantum.hamiltonian. Convert Matrix into memref.
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

/// Bufferization of catalyst.quantum.sample. Replace with memref.alloc and a new
/// catalyst.quantum.sample that uses the memory allocated by memref.alloc.
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

        Value allocVal = rewriter.create<memref::AllocOp>(loc, resultType);
        rewriter.create<SampleOp>(loc, TypeRange{}, ValueRange{sampleOp.getObs(), allocVal},
                                  sampleOp->getAttrs());
        bufferization::replaceOpWithBufferizedValues(rewriter, op, allocVal);

        return success();
    }
};

/// Bufferization of catalyst.quantum.state. Replace with memref.alloc and a new
/// catalyst.quantum.state that uses the memory allocated by memref.alloc.
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

        Value allocVal = rewriter.create<memref::AllocOp>(loc, resultType);
        rewriter.create<StateOp>(loc, TypeRange{}, ValueRange{stateOp.getObs(), allocVal});
        bufferization::replaceOpWithBufferizedValues(rewriter, op, allocVal);

        return success();
    }
};

/// Bufferization of catalyst.quantum.probs. Replace with memref.alloc and a new
/// catalyst.quantum.probs that uses the memory allocated by memref.alloc.
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

        Value allocVal = rewriter.create<memref::AllocOp>(loc, resultType);
        rewriter.create<ProbsOp>(loc, TypeRange{}, ValueRange{probsOp.getObs(), allocVal});
        bufferization::replaceOpWithBufferizedValues(rewriter, op, allocVal);

        return success();
    }
};

/// Bufferization of catalyst.quantum.counts. Replace with memref.allocs and a new
/// catalyst.quantum.counts that uses the memory allocated by memref.allocs.
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
        auto tensorType0 = cast<RankedTensorType>(countsOp.getEigvals().getType());
        auto tensorType1 = cast<RankedTensorType>(countsOp.getCounts().getType());
        MemRefType resultType0 =
            MemRefType::get(tensorType0.getShape(), tensorType0.getElementType());
        MemRefType resultType1 =
            MemRefType::get(tensorType1.getShape(), tensorType1.getElementType());

        Value allocVal0 = rewriter.create<memref::AllocOp>(loc, resultType0);
        Value allocVal1 = rewriter.create<memref::AllocOp>(loc, resultType1);
        rewriter.create<CountsOp>(loc, nullptr, nullptr, countsOp.getObs(), allocVal0, allocVal1,
                                  countsOp.getShotsAttr());
        bufferization::replaceOpWithBufferizedValues(rewriter, op,
                                                     ValueRange{allocVal0, allocVal1});

        return success();
    }
};

/// Bufferization of catalyst.quantum.set_state. Convert InState into memref.
struct SetStateOpInterface
    : public bufferization::BufferizableOpInterface::ExternalModel<SetStateOpInterface,
                                                                   SetStateOp> {
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

/// Bufferization of catalyst.quantum.set_basic_state. Convert BasisState into memref.
struct SetBasisStateOpInterface
    : public bufferization::BufferizableOpInterface::ExternalModel<SetBasisStateOpInterface,
                                                                   SetBasisStateOp> {
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
    registry.addExtension(+[](MLIRContext *ctx, QuantumDialect *dialect) {
        QubitUnitaryOp::attachInterface<QubitUnitaryOpInterface>(*ctx);
        HermitianOp::attachInterface<HermitianOpInterface>(*ctx);
        HamiltonianOp::attachInterface<HamiltonianOpInterface>(*ctx);
        SampleOp::attachInterface<SampleOpInterface>(*ctx);
        StateOp::attachInterface<StateOpInterface>(*ctx);
        ProbsOp::attachInterface<ProbsOpInterface>(*ctx);
        CountsOp::attachInterface<CountsOpInterface>(*ctx);
        SetStateOp::attachInterface<SetStateOpInterface>(*ctx);
        SetBasisStateOp::attachInterface<SetBasisStateOpInterface>(*ctx);
    });
}