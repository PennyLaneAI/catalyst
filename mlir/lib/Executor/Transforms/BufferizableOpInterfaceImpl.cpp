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

// BufferizableOpInterface external model for executor.call, so `lower-runtime-dispatch` can emit
// it in its value-semantics form and one-shot bufferize converts it to the dest-passing form that
// `convert-executor-to-llvm` lowers. See:
//   https://mlir.llvm.org/docs/Bufferization/#extending-one-shot-bufferize

#include "Executor/Transforms/BufferizableOpInterfaceImpl.h"

#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/IR/Builders.h"

#include "Catalyst/Utils/BufferizationUtils.h"

#include "Executor/IR/ExecutorDialect.h"
#include "Executor/IR/ExecutorOps.h"

using namespace mlir;
using namespace catalyst::executor;

namespace {

struct ExecutorCallOpInterface
    : public bufferization::BufferizableOpInterface::ExternalModel<ExecutorCallOpInterface,
                                                                   CallOp> {
    bool bufferizesToAllocation(Operation *op, Value value) const { return true; }
    bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                                const bufferization::AnalysisState &) const {
        return !isDestBuffer(op, opOperand) &&
               isa<RankedTensorType, MemRefType>(opOperand.get().getType());
    }

    bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                                 const bufferization::AnalysisState &) const {
        return isDestBuffer(op, opOperand);
    }

    bufferization::AliasingValueList getAliasingValues(Operation *, OpOperand &,
                                                       const bufferization::AnalysisState &) const {
        return {};
    }

    LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                            const bufferization::BufferizationOptions &options,
                            bufferization::BufferizationState &state) const {
        auto callOp = cast<CallOp>(op);
        if (callOp.isBufferized()) {
            return success();
        }
        Location loc = op->getLoc();

        SmallVector<Value> inputs;
        for (Value input : callOp.getInputs()) {
            if (!isa<RankedTensorType>(input.getType())) {
                inputs.push_back(input);
                continue;
            }
            FailureOr<Value> buffer = getBuffer(rewriter, input, options, state);
            if (failed(buffer)) {
                return failure();
            }

            // The wire layout is a flat copy of each input's elements.
            FailureOr<Value> contiguous = catalyst::makeContiguous(rewriter, *buffer, options);
            if (failed(contiguous)) {
                return failure();
            }
            inputs.push_back(*contiguous);
        }

        SmallVector<Value> outputs;
        for (Value result : callOp.getResults()) {
            auto tensorType = cast<RankedTensorType>(result.getType());
            FailureOr<Value> tensorAlloc = bufferization::allocateTensorForShapedValue(
                rewriter, loc, result, options, state, /*copy=*/false);
            if (failed(tensorAlloc)) {
                return failure();
            }
            MemRefType memrefType =
                MemRefType::get(tensorType.getShape(), tensorType.getElementType());
            outputs.push_back(
                bufferization::ToBufferOp::create(rewriter, loc, memrefType, *tensorAlloc)
                    .getResult());
        }

        // `num_input_args` marks where the output buffers start
        int32_t numInputs = static_cast<int32_t>(inputs.size());
        llvm::append_range(inputs, outputs);

        auto bufferized =
            CallOp::create(rewriter, loc, TypeRange{}, callOp.getSession(), inputs,
                           callOp.getSymbolAttr(), rewriter.getI32IntegerAttr(numInputs));
        bufferized->setDiscardableAttrs(callOp->getDiscardableAttrDictionary());
        bufferization::replaceOpWithBufferizedValues(rewriter, op, outputs);
        return success();
    }

  private:
    // Whether `opOperand` is one of the output buffers of a bufferized call.
    static bool isDestBuffer(Operation *op, OpOperand &opOperand) {
        auto callOp = cast<CallOp>(op);
        std::optional<uint32_t> numInputs = callOp.getNumInputArgs();
        if (!callOp.isBufferized() || !numInputs) {
            return false;
        }

        // The output buffers are the trailing operands of the call.
        return opOperand.getOperandNumber() >=
               callOp.getInputs().getBeginOperandIndex() + *numInputs;
    }
};

} // namespace

void catalyst::executor::registerBufferizableOpInterfaceExternalModels(DialectRegistry &registry) {
    registry.addExtension(+[](MLIRContext *ctx, ExecutorDialect *dialect) {
        CallOp::attachInterface<ExecutorCallOpInterface>(*ctx);
    });
}
