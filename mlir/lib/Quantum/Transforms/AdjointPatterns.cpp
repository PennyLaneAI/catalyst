// Copyright 2023 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#define DEBUG_TYPE "adjoint"

#include <algorithm>
#include <iterator>
#include <string>
#include <unordered_map>
#include <vector>

#include "llvm/Support/Debug.h"
#include "llvm/Support/Errc.h"

#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Catalyst/IR/CatalystOps.h"
#include "Quantum/IR/QuantumInterfaces.h"
#include "Quantum/IR/QuantumOps.h"
#include "Quantum/Transforms/Patterns.h"
#include "Quantum/Utils/QuantumSplitting.h"

using llvm::dbgs;
using namespace mlir;
using namespace catalyst;
using namespace catalyst::quantum;

namespace {

/// Clone the region of the adjoint operation `op` to the insertion point specified by the
/// `builder`. Build and return the value mapping `mapping`.
Value cloneAdjointRegion(AdjointOp op, OpBuilder &builder, IRMapping &mapping)
{
    Block &block = op.getRegion().front();
    for (Operation &op : block.without_terminator()) {
        builder.clone(op, mapping);
    }
    auto yieldOp = cast<quantum::YieldOp>(block.getTerminator());
    return mapping.lookupOrDefault(yieldOp.getOperand(0));
}

/// A class that generates the quantum "backwards pass" of the adjoint operation using the stored
/// gate parameters and cached control flow.
class AdjointGenerator {
  public:
    AdjointGenerator(IRMapping &remappedValues, QuantumCache &cache)
        : remappedValues(remappedValues), cache(cache)
    {
    }

    /// Recursively generate the adjoint version of `region` with reversed control flow and adjoint
    /// quantum gates.
    LogicalResult generate(Region &region, OpBuilder &builder)
    {
        generateImpl(region, builder);
        return failure(generationFailed);
    }

  private:
    void generateImpl(Region &region, OpBuilder &builder)
    {
        assert(region.hasOneBlock() &&
               "Expected only structured control flow (each region should have a single block)");

        for (Operation &op : llvm::reverse(region.front().without_terminator())) {
            LLVM_DEBUG(dbgs() << "generating adjoint for: " << op << "\n");
            if (auto callOp = dyn_cast<func::CallOp>(op)) {
                visitOperation(callOp, builder);
            }
            else if (auto forOp = dyn_cast<scf::ForOp>(op)) {
                visitOperation(forOp, builder);
            }
            else if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
                visitOperation(ifOp, builder);
            }
            else if (auto whileOp = dyn_cast<scf::WhileOp>(op)) {
                visitOperation(whileOp, builder);
            }
            else if (auto insertOp = dyn_cast<quantum::InsertOp>(op)) {
                Value dynamicWire = getDynamicWire(insertOp, builder);
                auto extractOp = builder.create<quantum::ExtractOp>(
                    insertOp.getLoc(), insertOp.getQubit().getType(),
                    remappedValues.lookup(insertOp.getOutQreg()), dynamicWire,
                    insertOp.getIdxAttrAttr());
                remappedValues.map(insertOp.getQubit(), extractOp.getResult());
                remappedValues.map(insertOp.getInQreg(),
                                   remappedValues.lookup(insertOp.getOutQreg()));
            }
            else if (auto extractOp = dyn_cast<quantum::ExtractOp>(op)) {
                Value dynamicWire = getDynamicWire(extractOp, builder);
                auto insertOp = builder.create<quantum::InsertOp>(
                    extractOp.getLoc(), extractOp.getQreg().getType(),
                    remappedValues.lookup(extractOp.getQreg()), dynamicWire,
                    extractOp.getIdxAttrAttr(), remappedValues.lookup(extractOp.getQubit()));
                remappedValues.map(extractOp.getQreg(), insertOp.getResult());
            }
            else if (auto gate = dyn_cast<quantum::QuantumGate>(op)) {
                visitOperation(gate, builder);
            }
            else if (auto adjointOp = dyn_cast<quantum::AdjointOp>(&op)) {
                BlockArgument regionArg = adjointOp.getRegion().getArgument(0);
                Value result = adjointOp.getResult();
                remappedValues.map(regionArg, remappedValues.lookup(result));
                Value reversedResult = cloneAdjointRegion(adjointOp, builder, remappedValues);
                remappedValues.map(adjointOp.getQreg(), reversedResult);
            }
            else if (isa<QuantumDialect>(op.getDialect())) {
                op.emitError("Unhandled operation in adjoint region");
                generationFailed = true;
                return;
            }
        }
    }

    template <typename IndexingOp> Value getDynamicWire(IndexingOp op, OpBuilder &builder)
    {
        Value dynamicWire;
        if (!op.getIdxAttr().has_value()) {
            dynamicWire = builder.create<ListPopOp>(op.getLoc(), cache.wireVector);
        }
        return dynamicWire;
    }

    std::optional<Value> getQuantumReg(ValueRange values)
    {
        for (Value value : values) {
            if (isa<quantum::QuregType>(value.getType())) {
                return value;
            }
        }
        return std::nullopt;
    }

    void visitOperation(quantum::QuantumGate gate, OpBuilder &builder)
    {
        for (const auto &[qubitResult, qubitOperand] :
             llvm::zip(gate.getQubitResults(), gate.getQubitOperands())) {
            remappedValues.map(qubitOperand, remappedValues.lookup(qubitResult));
        }

        auto clone = cast<QuantumGate>(builder.clone(*gate, remappedValues));
        clone.setAdjointFlag(!gate.getAdjointFlag());

        // Read cached parameters from the recorded parameter vector.
        mlir::Operation *operation = gate;
        if (auto parametrizedGate = dyn_cast<quantum::ParametrizedGate>(operation)) {
            OpBuilder::InsertionGuard insertionGuard(builder);
            builder.setInsertionPoint(clone);
            ValueRange params = parametrizedGate.getAllParams();
            size_t numParams = params.size();
            SmallVector<Value> cachedParams(numParams);
            // popping gives the parameters in reverse
            for (auto [idx, param] : llvm::enumerate(llvm::reverse(params))) {
                Type paramType = param.getType();
                verifyTypeIsCacheable(paramType, operation);
                if (paramType.isF64()) {
                    cachedParams[numParams - 1 - idx] =
                        builder.create<ListPopOp>(parametrizedGate.getLoc(), cache.paramVector);
                    continue;
                }

                // Guaranteed by verifyTypeIsPoppable above.
                auto aTensorType = paramType.cast<RankedTensorType>();
                ArrayRef<int64_t> shape = aTensorType.getShape();
                Type elementType = aTensorType.getElementType();
                // Constants
                auto loc = parametrizedGate.getLoc();
                Value c0 = builder.create<index::ConstantOp>(loc, 0);
                Value c1 = builder.create<index::ConstantOp>(loc, 1);
                // TODO: Generalize to all possible dimensions
                bool isDim0Static = ShapedType::kDynamic != shape[0];
                bool isDim1Static = ShapedType::kDynamic != shape[1];
                Value dim0Length = isDim0Static
                                       ? (Value)builder.create<index::ConstantOp>(loc, shape[0])
                                       : (Value)builder.create<tensor::DimOp>(loc, param, c0);
                Value dim1Length = isDim1Static
                                       ? (Value)builder.create<index::ConstantOp>(loc, shape[1])
                                       : (Value)builder.create<tensor::DimOp>(loc, param, c1);

                // Renaming for legibility
                // Note: Since this is a square matrix, upperBound for both loops is the
                // same value.
                Value lowerBoundDim0 = c0;
                Value upperBoundDim0 = dim0Length;
                Value stepDim0 = c1;
                Value lowerBoundDim1 = c0;
                Value upperBoundDim1 = dim1Length;
                Value stepDim1 = c1;
                Value beginningTensor = builder.create<tensor::EmptyOp>(loc, shape, elementType);
                // This time, we are in reverse, so we need to start
                // with N-1 since MLIR does not allow for loops with negative step sizes.
                SmallVector<Value> initialValues = {beginningTensor};

                scf::ForOp iForLoop = builder.create<scf::ForOp>(
                    loc, lowerBoundDim0, upperBoundDim0, stepDim0, initialValues);
                {
                    OpBuilder::InsertionGuard afterIForLoop(builder);
                    builder.setInsertionPointToStart(iForLoop.getBody());
                    auto iIterArgs = iForLoop.getRegionIterArgs();
                    Value currIthTensor = iIterArgs.front();

                    Value i = iForLoop.getInductionVar();
                    Value iPlusOne = builder.create<index::AddOp>(loc, i, c1);
                    Value nMinusIMinusOne = builder.create<index::SubOp>(loc, dim0Length, iPlusOne);
                    // Just for legibility
                    Value iTensorIndex = nMinusIMinusOne;

                    scf::ForOp jForLoop = builder.create<scf::ForOp>(
                        loc, lowerBoundDim1, upperBoundDim1, stepDim1, currIthTensor);
                    {
                        OpBuilder::InsertionGuard afterJForLoop(builder);
                        builder.setInsertionPointToStart(jForLoop.getBody());
                        auto jIterArgs = jForLoop.getRegionIterArgs();
                        assert(jIterArgs.size() == 1 &&
                               "jForLoop has more induction variables than necessary.");
                        Value currIthJthTensor = jIterArgs.front();

                        Value imag = builder.create<ListPopOp>(loc, cache.paramVector);
                        Value real = builder.create<ListPopOp>(loc, cache.paramVector);
                        Value element =
                            builder.create<complex::CreateOp>(loc, elementType, real, imag);

                        // TODO: Generalize to types which are not complex
                        Value j = jForLoop.getInductionVar();
                        Value jPlusOne = builder.create<index::AddOp>(loc, j, c1);
                        Value nMinusJMinusOne =
                            builder.create<index::SubOp>(loc, dim1Length, jPlusOne);
                        // Just for legibility
                        Value jTensorIndex = nMinusJMinusOne;
                        SmallVector<Value> indices = {iTensorIndex, jTensorIndex};

                        Value updatedIthJthTensor = builder.create<tensor::InsertOp>(
                            loc, element, currIthJthTensor, indices);
                        builder.create<scf::YieldOp>(loc, updatedIthJthTensor);
                    }

                    Value ithTensor = jForLoop.getResult(0);
                    builder.create<scf::YieldOp>(loc, ithTensor);
                }

                Value recreatedTensor = iForLoop.getResult(0);
                cachedParams[numParams - 1 - idx] = recreatedTensor;
            }
            MutableOperandRange(clone, parametrizedGate.getParamOperandIdx(), params.size())
                .assign(cachedParams);
        }

        for (const auto &[qubitResult, qubitOperand] :
             llvm::zip(clone.getQubitResults(), gate.getQubitOperands())) {
            remappedValues.map(qubitOperand, qubitResult);
        }
    }

    void visitOperation(func::CallOp callOp, OpBuilder &builder)
    {
        // Get the the original function
        SymbolRefAttr symbol = dyn_cast_if_present<SymbolRefAttr>(callOp.getCallableForCallee());
        func::FuncOp funcOp =
            dyn_cast_or_null<func::FuncOp>(SymbolTable::lookupNearestSymbolFrom(callOp, symbol));
        assert(funcOp != nullptr && "The funcOp is null and therefore not supported.");

        auto resultTypes = funcOp.getResultTypes();
        bool multiReturns = resultTypes.size() > 1;

        bool quantum = std::any_of(resultTypes.begin(), resultTypes.end(),
                                   [](const auto &value) { return isa<QuregType>(value); });
        assert(!(quantum && multiReturns) && "Adjoint does not support functions with multiple "
                                             "returns that contain a quantum register.");

        if (!quantum) {
            // This operation is purely classical
            return;
        }

        // Save the insertion point to come back to it after creating the adjoint of the function
        auto insertionSaved = builder.saveInsertionPoint();
        MLIRContext *ctx = builder.getContext();

        // Create the adjoint of the original function at the module level
        ModuleOp moduleOp = funcOp->getParentOfType<ModuleOp>();
        builder.setInsertionPointToStart(moduleOp.getBody());

        Block *originalBlock = &funcOp.front();
        Operation *originalTerminator = originalBlock->getTerminator();
        ValueRange originalArguments = originalBlock->getArguments();

        // Get the arguments and outputs types from the original function (same signature)
        FunctionType adjointFnType =
            FunctionType::get(ctx, /*inputs=*/
                              originalArguments.getTypes(),
                              /*outputs=*/originalTerminator->getOperandTypes());
        std::string adjointName = funcOp.getName().str() + ".adjoint";
        Location loc = funcOp.getLoc();
        func::FuncOp adjointFnOp = builder.create<func::FuncOp>(loc, adjointName, adjointFnType);
        adjointFnOp.setPrivate();

        // Create the block of the adjoint function
        Block *adjointFnBlock = adjointFnOp.addEntryBlock();
        builder.setInsertionPointToStart(adjointFnBlock);
        Type qregType = quantum::QuregType::get(builder.getContext());
        auto argumentsSize = adjointFnOp.getArguments().size();

        // The last argument is the quantum register
        Value lastArg = adjointFnOp.getArgument(argumentsSize - 1);
        assert(isa<QuregType>(lastArg.getType()) &&
               "The last argument of the function must be the quantum register.");
        quantum::AdjointOp adjointOp = builder.create<quantum::AdjointOp>(loc, qregType, lastArg);

        Region *adjointRegion = &adjointOp.getRegion();
        Region &originalRegion = funcOp.getRegion();

        // Map the arguments with the original function
        IRMapping map;
        for (size_t i = 0; i < funcOp.getArguments().size() - 1; i++) {
            Value arg = adjointFnOp.front().getArgument(i);
            Value argBlock = originalRegion.front().getArgument(i);
            map.map(argBlock, arg);
        }

        // Copy the original region in the adjoint region
        originalRegion.cloneInto(adjointRegion, map);

        // Replace the return of the quantum register with a quantum yield
        auto terminator = adjointRegion->front().getTerminator();
        ValueRange res = terminator->getOperands();
        TypeRange resTypes = terminator->getResultTypes();
        builder.setInsertionPointAfter(terminator);
        builder.create<quantum::YieldOp>(loc, resTypes, res);

        // Return the adjoint operation in the adjoint function
        IRRewriter rewriter(builder);
        rewriter.eraseOp(terminator);
        builder.setInsertionPointAfter(adjointOp);
        builder.create<func::ReturnOp>(loc, adjointOp.getResult());

        // Leave the adjoint func op to go back at the saved insertion
        builder.restoreInsertionPoint(insertionSaved);

        // Get the reversed result
        Value reversedResult = remappedValues.lookup(getQuantumReg(callOp.getResults()).value());
        std::vector<Value> args = {callOp.getArgOperands().begin(), callOp.getArgOperands().end()};
        args.pop_back();
        args.push_back(reversedResult);
        // Call the adjoint func op
        auto adjointCallOp = builder.create<func::CallOp>(loc, adjointFnOp, args);
        ValueRange initQreg = callOp.getArgOperands();
        // Map the initial quantum register with the adjoint result
        remappedValues.map(getQuantumReg(initQreg).value(), adjointCallOp.getResult(0));
    }

    void visitOperation(scf::ForOp forOp, OpBuilder &builder)
    {
        std::optional<Value> yieldedQureg =
            getQuantumReg(forOp.getBody()->getTerminator()->getOperands());
        if (!yieldedQureg.has_value()) {
            // This operation is purely classical
            return;
        }

        Value tape = cache.controlFlowTapes.at(forOp);
        // Popping the start, stop, and step implies that these are backwards relative to
        // the order they were pushed.
        Value step = builder.create<ListPopOp>(forOp.getLoc(), tape);
        Value stop = builder.create<ListPopOp>(forOp.getLoc(), tape);
        Value start = builder.create<ListPopOp>(forOp.getLoc(), tape);

        Value reversedResult = remappedValues.lookup(getQuantumReg(forOp.getResults()).value());
        auto replacedFor = builder.create<scf::ForOp>(
            forOp.getLoc(), start, stop, step, /*iterArgsInit=*/reversedResult,
            [&](OpBuilder &bodyBuilder, Location loc, Value iv, ValueRange iterArgs) {
                OpBuilder::InsertionGuard insertionGuard(builder);
                builder.restoreInsertionPoint(bodyBuilder.saveInsertionPoint());

                remappedValues.map(yieldedQureg.value(), iterArgs[0]);
                generateImpl(forOp.getBodyRegion(), builder);
                builder.create<scf::YieldOp>(
                    loc, remappedValues.lookup(getQuantumReg(forOp.getRegionIterArgs()).value()));
            });
        remappedValues.map(getQuantumReg(forOp.getInitArgs()).value(), replacedFor.getResult(0));
    }

    void visitOperation(scf::IfOp ifOp, OpBuilder &builder)
    {
        std::optional<Value> qureg = getQuantumReg(ifOp.getResults());
        if (!qureg.has_value()) {
            // This operation is purely classical
            return;
        }

        Value tape = cache.controlFlowTapes.at(ifOp);
        Value condition = builder.create<ListPopOp>(ifOp.getLoc(), tape);
        condition = builder.create<index::CastSOp>(ifOp.getLoc(), builder.getI1Type(), condition);
        Value reversedResult = remappedValues.lookup(getQuantumReg(ifOp.getResults()).value());

        // The quantum register is captured from outside rather than passed in through a
        // basic block argument. We thus need to traverse the region to look for it.
        auto findOldestQuregInRegion = [&](Region &region) {
            for (Operation &innerOp : region.getOps()) {
                for (Value operand : innerOp.getOperands()) {
                    if (isa<quantum::QuregType>(operand.getType())) {
                        return operand;
                    }
                }
            }
            llvm_unreachable("failed to find qureg in scf.if region");
        };
        auto getRegionBuilder = [&](Region &oldRegion) {
            return [&](OpBuilder &bodyBuilder, Location loc) {
                OpBuilder::InsertionGuard insertionGuard(builder);
                builder.restoreInsertionPoint(bodyBuilder.saveInsertionPoint());

                std::optional<Value> yieldedQureg =
                    getQuantumReg(oldRegion.front().getTerminator()->getOperands());
                remappedValues.map(yieldedQureg.value(), reversedResult);
                generateImpl(oldRegion, builder);
                builder.create<scf::YieldOp>(
                    loc, remappedValues.lookup(findOldestQuregInRegion(oldRegion)));
            };
        };
        auto reversedIf = builder.create<scf::IfOp>(ifOp.getLoc(), condition,
                                                    getRegionBuilder(ifOp.getThenRegion()),
                                                    getRegionBuilder(ifOp.getElseRegion()));
        Value startingThenQureg = findOldestQuregInRegion(ifOp.getThenRegion());
        Value startingElseQureg = findOldestQuregInRegion(ifOp.getElseRegion());
        assert(startingThenQureg == startingElseQureg &&
               "Expected the same input register for both scf.if branches");
        remappedValues.map(startingThenQureg, reversedIf.getResult(0));
    }

    void visitOperation(scf::WhileOp whileOp, OpBuilder &builder)
    {
        std::optional<Value> yieldedQureg =
            getQuantumReg(whileOp.getAfter().front().getTerminator()->getOperands());
        if (!yieldedQureg.has_value()) {
            // This operation is purely classical
            return;
        }

        Value tape = cache.controlFlowTapes.at(whileOp);
        Value numIterations = builder.create<ListPopOp>(whileOp.getLoc(), tape);
        Value c0 = builder.create<index::ConstantOp>(whileOp.getLoc(), 0);
        Value c1 = builder.create<index::ConstantOp>(whileOp.getLoc(), 1);

        Value iterArgInit = remappedValues.lookup(getQuantumReg(whileOp.getResults()).value());
        auto replacedWhile = builder.create<scf::ForOp>(
            whileOp.getLoc(), /*start=*/c0, /*stop=*/numIterations, /*step=*/c1, iterArgInit,
            /*bodyBuilder=*/
            [&](OpBuilder &bodyBuilder, Location loc, Value iv, ValueRange iterArgs) {
                OpBuilder::InsertionGuard insertionGuard(builder);
                builder.restoreInsertionPoint(bodyBuilder.saveInsertionPoint());

                remappedValues.map(yieldedQureg.value(), iterArgs[0]);
                generateImpl(whileOp.getAfter(), builder);
                builder.create<scf::YieldOp>(
                    loc, remappedValues.lookup(
                             getQuantumReg(whileOp.getAfter().front().getArguments()).value()));
            });
        remappedValues.map(getQuantumReg(whileOp.getInits()).value(), replacedWhile.getResult(0));
    }

  private:
    IRMapping &remappedValues;
    QuantumCache &cache;
    bool generationFailed = false;
};

struct AdjointSingleOpRewritePattern : public mlir::OpRewritePattern<AdjointOp> {
    using mlir::OpRewritePattern<AdjointOp>::OpRewritePattern;

    /// We build a map from values mentioned in the source data flow to the values of
    /// the program where quantum control flow is reversed. Most of the time, there is a 1-to-1
    /// correspondence with a notable exception caused by `insert`/`extract` API asymmetry.
    mlir::LogicalResult matchAndRewrite(AdjointOp adjoint,
                                        mlir::PatternRewriter &rewriter) const override
    {
        LLVM_DEBUG(dbgs() << "Adjointing the following:\n" << adjoint << "\n");
        auto cache = QuantumCache::initialize(adjoint.getRegion(), rewriter, adjoint.getLoc());
        // First, copy the classical computations directly to the target insertion point.
        IRMapping oldToCloned;
        AugmentedCircuitGenerator augmentedGenerator{oldToCloned, cache};
        augmentedGenerator.generate(adjoint.getRegion(), rewriter);

        // Initialize the backward pass with the operand of the quantum.yield
        auto yieldOp = cast<quantum::YieldOp>(adjoint.getRegion().front().getTerminator());
        assert(yieldOp.getNumOperands() == 1 && "Expected quantum.yield to have one operand");
        oldToCloned.map(yieldOp.getOperands().front(), adjoint.getQreg());

        // Emit the adjoint quantum operations and reversed control flow, using cached values.
        AdjointGenerator adjointGenerator{oldToCloned, cache};
        if (failed(adjointGenerator.generate(adjoint.getRegion(), rewriter))) {
            return failure();
        }

        // Explicitly free the memory of the caches.
        cache.emitDealloc(rewriter, adjoint.getLoc());
        // The final register is the re-mapped region argument of the original adjoint op.
        SmallVector<Value> reversedOutputs;
        for (BlockArgument arg : adjoint.getRegion().getArguments()) {
            reversedOutputs.push_back(oldToCloned.lookup(arg));
        }
        rewriter.replaceOp(adjoint, reversedOutputs);
        return success();
    }
};

} // namespace

namespace catalyst {
namespace quantum {

void populateAdjointPatterns(RewritePatternSet &patterns)
{
    patterns.add<AdjointSingleOpRewritePattern>(patterns.getContext(), 1);
}

} // namespace quantum
} // namespace catalyst
