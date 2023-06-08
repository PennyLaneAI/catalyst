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

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Quantum/IR/QuantumOps.h"
#include "Quantum/Transforms/Passes.h"
#include "Quantum/Transforms/Patterns.h"

using namespace mlir;
using namespace catalyst::quantum;

namespace {

bool hasBasicBlockCounterAttribute(func::FuncOp &op)
{
    StringAttr bbCounterAttr = StringAttr::get(op->getContext(), "catalyst.bbcounter");
    return op->hasAttr(bbCounterAttr);
}

void removeBasicBlockCounterAttribute(func::FuncOp &op)
{
    StringAttr bbCounterAttr = StringAttr::get(op->getContext(), "catalyst.bbcounter");
    op->removeAttr(bbCounterAttr);
}

std::vector<Block *> getBasicBlocksInPostOrder(func::FuncOp &op)
{
    // Walk is post-order, so all basic blocks encountered are in post-order.
    // The last block in nested_basic_blocks will be the block for the whole function.
    std::vector<Block *> blocks;
    op.walk([&](mlir::Operation *nestedOp) {
        for (Region &region : nestedOp->getRegions()) {
            for (Block &nestedBlock : region.getBlocks()) {
                blocks.push_back(&nestedBlock);
            }
        }
    });
    return blocks;
}

std::vector<DifferentiableGate> getDifferentiableGates(func::FuncOp &op)
{
    std::vector<DifferentiableGate> ops;
    op.walk([&](mlir::Operation *nestedOp) {
        if (DifferentiableGate gate = dyn_cast<DifferentiableGate>(nestedOp)) {
            ops.push_back(gate);
        }
    });
    return ops;
}

int getDefinitionsBlockOffset(Value value, std::unordered_map<Block *, int> &blockOffsetMap)
{
    mlir::Operation *definingOp = value.getDefiningOp();
    Block *definingBlock = definingOp->getBlock();
    return blockOffsetMap.at(definingBlock);
}

std::unordered_map<void *, int>
getDifferentiableParamsBasicBlockOffsetMap(DifferentiableGate gate,
                                           std::unordered_map<Block *, int> &blockOffsetMap)
{
    std::unordered_map<void *, int> map;
    ValueRange diffParams = gate.getDiffParams();
    for (Value diffParam : diffParams) {
        int definingBlockOffset = getDefinitionsBlockOffset(diffParam, blockOffsetMap);
        map.insert({diffParam.getAsOpaquePointer(), definingBlockOffset});
    }
    return map;
}

std::set<mlir::Operation *> getDifferentiableParamDefinitions(DifferentiableGate gate)
{
    std::set<mlir::Operation *> definitions;
    ValueRange diffParams = gate.getDiffParams();
    for (Value diffParam : diffParams) {
        mlir::Operation *definingOp = diffParam.getDefiningOp();
        definitions.insert(definingOp);
    }
    return definitions;
}

std::set<mlir::Operation *> getDifferentiableParamDefinitions(std::vector<DifferentiableGate> gates)
{
    std::set<mlir::Operation *> definitions;
    for (DifferentiableGate gate : gates) {
        std::set<mlir::Operation *> definitions_in_gate = getDifferentiableParamDefinitions(gate);
        definitions.merge(definitions_in_gate);
    }
    return definitions;
}

std::unordered_map<void *, int>
getDifferentiableParamBasicBlockOffsetMap(std::vector<DifferentiableGate> gates,
                                          std::unordered_map<Block *, int> &blockOffsetMap)
{
    std::unordered_map<void *, int> map;
    for (DifferentiableGate gate : gates) {
        std::unordered_map<void *, int> instructionMap =
            getDifferentiableParamsBasicBlockOffsetMap(gate, blockOffsetMap);
        map.merge(instructionMap);
    }
    return map;
}

struct BasicBlockCounterTransform : public OpRewritePattern<func::FuncOp> {
    using OpRewritePattern<func::FuncOp>::OpRewritePattern;

    LogicalResult match(func::FuncOp op) const override;
    void rewrite(func::FuncOp op, PatternRewriter &rewriter) const override;

  private:
    Value allocateBasicBlockCounterArray(func::FuncOp &op, PatternRewriter &rewriter,
                                         size_t arraySize) const;
    void addIncrements(PatternRewriter &rewriter, Location loc, std::vector<Block *> &,
                       Value &) const;
};

Value BasicBlockCounterTransform::allocateBasicBlockCounterArray(func::FuncOp &op,
                                                                 PatternRewriter &rewriter,
                                                                 size_t arraySize) const
{
    Region &firstRegion = op.getRegion();
    Block &firstBlock = firstRegion.front();
    rewriter.setInsertionPointToStart(&firstBlock);
    Type i64Type = IntegerType::get(rewriter.getContext(), 64);
    Value c1 = rewriter.create<LLVM::ConstantOp>(op->getLoc(), rewriter.getI64IntegerAttr(1));
    Type arrayType = LLVM::LLVMArrayType::get(i64Type, arraySize);
    Type ptrToArrayType = LLVM::LLVMPointerType::get(arrayType);
    return rewriter.create<LLVM::AllocaOp>(op->getLoc(), ptrToArrayType, c1);
}

void BasicBlockCounterTransform::addIncrements(PatternRewriter &rewriter, Location loc,
                                               std::vector<Block *> &blocks,
                                               Value &blockCounterArray) const
{
    std::vector<Value> bbCounters;
    Block *entryBlock = blocks.back();
    size_t blocksSize = blocks.size();
    mlir::Operation *definitionOfArray = blockCounterArray.getDefiningOp();
    Value c1 = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64IntegerAttr(1));
    for (size_t i = 0; i < blocksSize; i++) {
        Block *block = blocks[i];

        // The entry block, the next instructions have to be placed after
        // the definition of the block counter.
        if (block == entryBlock) {
            rewriter.setInsertionPointAfter(definitionOfArray);
        }
        else {
            rewriter.setInsertionPointToStart(block);
        }

        Value loadArray = rewriter.create<LLVM::LoadOp>(loc, blockCounterArray);
        Value currentCount = rewriter.create<LLVM::ExtractValueOp>(loc, loadArray, i);
        Value newCount = rewriter.create<LLVM::AddOp>(loc, c1, currentCount);
        Value newArray = rewriter.create<LLVM::InsertValueOp>(loc, loadArray, newCount, i);
        rewriter.create<LLVM::StoreOp>(loc, newArray, blockCounterArray);
    }
}

LogicalResult BasicBlockCounterTransform::match(func::FuncOp op) const
{
    return hasBasicBlockCounterAttribute(op) ? success() : failure();
}

void BasicBlockCounterTransform::rewrite(func::FuncOp op, PatternRewriter &rewriter) const
{
    auto blocks = getBasicBlocksInPostOrder(op);
    size_t howManyBlocks = blocks.size();
    Value blockCounterArray = this->allocateBasicBlockCounterArray(op, rewriter, howManyBlocks);
    this->addIncrements(rewriter, op->getLoc(), blocks, blockCounterArray);
    std::unordered_map<Block *, int> blockNumberingMap;
    for (size_t i = 0; i < howManyBlocks; i++) {
        blockNumberingMap.insert({blocks[i], i});
    }

    auto differentiableGates = getDifferentiableGates(op);
    auto definitionVector = getDifferentiableParamDefinitions(differentiableGates);
    std::unordered_map<mlir::Operation *, int> definitionNumberMap;
    size_t i = 0;
    for (mlir::Operation *operation : definitionVector) {
        definitionNumberMap.insert({operation, i++});
    }

    auto valueDefiningBlockMap =
        getDifferentiableParamBasicBlockOffsetMap(differentiableGates, blockNumberingMap);

    for (auto gate : differentiableGates) {
        rewriter.setInsertionPoint(gate);
        auto diffParams = gate.getDiffParams();
        std::vector<Value> paramIds;
        std::vector<Value> paramRuntimeDefinitions;
        for (auto diffParam : diffParams) {
            // So, what I need to do is
            // for each diffParam, I need to get the corresponding
            // static definition
            mlir::Operation *operation = diffParam.getDefiningOp();
            int paramIdInt = definitionNumberMap.at(operation);
            Value paramId = rewriter.create<arith::ConstantOp>(
                gate->getLoc(), rewriter.getI64Type(),
                rewriter.getIntegerAttr(rewriter.getI64Type(), paramIdInt));
            paramIds.push_back(paramId);

            int64_t offset = valueDefiningBlockMap.at(diffParam.getAsOpaquePointer());
            ArrayRef<int64_t> basicBlockIndex{static_cast<int64_t>(offset)};
            Value loadArray = rewriter.create<LLVM::LoadOp>(gate->getLoc(), blockCounterArray);
            Value runtimeCount =
                rewriter.create<LLVM::ExtractValueOp>(gate->getLoc(), loadArray, offset);

            paramRuntimeDefinitions.push_back(runtimeCount);
        }

        // Now that we have paramIds and paramRuntimeDefinitions
        // we can substitute the op.
        CustomOp customOp = cast<CustomOp>(gate);
        rewriter.replaceOpWithNewOp<TraceCustomOp>(
            gate, customOp.getResultTypes(), customOp.getParams(), paramIds,
            paramRuntimeDefinitions, customOp.getInQubits(), customOp.getGateName());
    }

    removeBasicBlockCounterAttribute(op);
}

} // namespace

namespace catalyst {

struct BasicBlockCounterPass : public PassWrapper<BasicBlockCounterPass, OperationPass<ModuleOp>> {
    BasicBlockCounterPass() {}

    StringRef getArgument() const override { return "bb-counter"; }

    StringRef getDescription() const override
    {
        return "Add basic-block counters to all functions.";
    }

    void getDependentDialects(DialectRegistry &registry) const override
    {
        registry.insert<memref::MemRefDialect>();
        registry.insert<func::FuncDialect>();
        registry.insert<scf::SCFDialect>();
        registry.insert<LLVM::LLVMDialect>();
    }

    void runOnOperation() final
    {
        MLIRContext *context = &getContext();
        RewritePatternSet patterns(context);
        patterns.add<BasicBlockCounterTransform>(context);

        if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
            signalPassFailure();
        }
    }
};

std::unique_ptr<Pass> createBasicBlockCounterPass()
{
    return std::make_unique<BasicBlockCounterPass>();
}

} // namespace catalyst
