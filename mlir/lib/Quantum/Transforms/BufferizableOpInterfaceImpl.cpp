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
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "Quantum/IR/QuantumOps.h"
#include "Quantum/Transforms/BufferizableOpInterfaceImpl.h"

using namespace mlir;
using namespace mlir::bufferization;
using namespace catalyst::quantum;

namespace {

/// Bufferization of catalyst.quantum.set_state. Convert InState into memref.
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

/// Bufferization of catalyst.quantum.set_basis_state. Convert BasisState into memref.
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
        SetStateOp::attachInterface<SetStateOpInterface>(*ctx);
        SetBasisStateOp::attachInterface<SetBasisStateOpInterface>(*ctx);
    });
}
