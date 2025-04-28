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
using namespace catalyst::quantum;

namespace {

/// Bufferization of tensor.extract. Replace with memref.load.
// struct ExtractOpInterface
//     : public mlir::bufferization::BufferizableOpInterface::ExternalModel<ExtractOpInterface,
//                                                     catalyst::quantum::ExtractOp> {
//   bool bufferizesToMemoryRead(mlir::Operation *op, mlir::OpOperand &opOperand,
//                               const mlir::bufferization::AnalysisState &state) const {
//     return true;
//   }

//   bool bufferizesToMemoryWrite(mlir::Operation *op, mlir::OpOperand &opOperand,
//                                const mlir::bufferization::AnalysisState &state) const {
//     return false;
//   }

//   mlir::bufferization::AliasingValueList getAliasingValues(mlir::Operation *op,
//                                       mlir::OpOperand &opOperand,
//                                       const mlir::bufferization::AnalysisState &state) const {
//     return {};
//   }

//   LogicalResult bufferize(mlir::Operation *op, RewriterBase &rewriter,
//                           const mlir::bufferization::BufferizationOptions &options) const {
//     //auto extractOp = cast<catalyst::quantum::ExtractOp>(op);

//     return success();
//   }
// };

/// Bufferization of catalyst.quantum.set_state. Convert InState into memref.
struct SetStateOpInterface : public mlir::bufferization::BufferizableOpInterface::ExternalModel<
                                 SetStateOpInterface, catalyst::quantum::SetStateOp> {
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

} // namespace

void catalyst::quantum::registerBufferizableOpInterfaceExternalModels(DialectRegistry &registry)
{
    registry.addExtension(+[](MLIRContext *ctx, catalyst::quantum::QuantumDialect *dialect) {
        // ExtractOp::attachInterface<ExtractOpInterface>(*ctx);
        SetStateOp::attachInterface<SetStateOpInterface>(*ctx);
    });
}
