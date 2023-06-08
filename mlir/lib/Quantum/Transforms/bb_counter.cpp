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
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Quantum/Transforms/Passes.h"
#include "Quantum/Transforms/Patterns.h"

using namespace mlir;

namespace {

bool
hasBasicBlockCounterAttribute(func::FuncOp &op) {
  StringAttr bbCounterAttr = StringAttr::get(op->getContext(), "catalyst.bbcounter");
  return op->hasAttr(bbCounterAttr);
}

void
removeBasicBlockCounterAttribute(func::FuncOp &op) {
  StringAttr bbCounterAttr = StringAttr::get(op->getContext(), "catalyst.bbcounter");
  op->removeAttr(bbCounterAttr);
}

std::vector<Block *>
getBasicBlocksInPostOrder(func::FuncOp &op) {
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


struct BasicBlockCounterTransform : public OpRewritePattern<func::FuncOp> {
    using OpRewritePattern<func::FuncOp>::OpRewritePattern;

    LogicalResult match(func::FuncOp op) const override;
    void rewrite(func::FuncOp op, PatternRewriter &rewriter) const override;
private:
    Value allocateBasicBlockCounterArray(func::FuncOp &op, PatternRewriter &rewriter, size_t arraySize) const;
    void addIncrements(PatternRewriter &rewriter, Location loc, std::vector<Block *> &, Value &) const;
};

Value
BasicBlockCounterTransform::allocateBasicBlockCounterArray(func::FuncOp &op, PatternRewriter &rewriter, size_t arraySize) const {
  Region &firstRegion = op.getRegion();
  Block &firstBlock = firstRegion.front();
  rewriter.setInsertionPointToStart(&firstBlock);
  Type i64Type = IntegerType::get(rewriter.getContext(), 64);
  Value c1 = rewriter.create<LLVM::ConstantOp>(op->getLoc(), rewriter.getI64IntegerAttr(1));
  Type arrayType = LLVM::LLVMArrayType::get(i64Type, arraySize);
  return rewriter.create<LLVM::AllocaOp>(op->getLoc(), arrayType, c1);
}

void
BasicBlockCounterTransform::addIncrements(PatternRewriter &rewriter, Location loc, std::vector<Block *> &blocks, Value &blockCounterArray) const {
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
    } else {
      rewriter.setInsertionPointToStart(block);
    }
 
    Value currentCount = rewriter.create<LLVM::ExtractValueOp>(loc, blockCounterArray, ArrayRef<int64_t>{(int64_t)i});
    Value newCount = rewriter.create<LLVM::AddOp>(loc, c1, currentCount);
    rewriter.create<LLVM::InsertValueOp>(loc, blockCounterArray, newCount, ArrayRef<int64_t>{(int64_t)i});
  }
}

LogicalResult
BasicBlockCounterTransform::match(func::FuncOp op) const {
  return hasBasicBlockCounterAttribute(op) ? success() : failure();
}

void
BasicBlockCounterTransform::rewrite(func::FuncOp op, PatternRewriter &rewriter) const {
  auto blocks = getBasicBlocksInPostOrder(op);
  size_t howManyBlocks = blocks.size();
  Value blockCounterArray = this->allocateBasicBlockCounterArray(op, rewriter, howManyBlocks);
  this->addIncrements(rewriter, op->getLoc(), blocks, blockCounterArray);
  std::unordered_map<Block *, int> blockNumberingMap;
  for (size_t i = 0; i < howManyBlocks; i++)
  {
    blockNumberingMap.insert({blocks[i], i});
  }

  op->emitRemark() << "remark";

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

	mlir::Operation *op = getOperation();
	op->emitRemark() << "running on...";

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
