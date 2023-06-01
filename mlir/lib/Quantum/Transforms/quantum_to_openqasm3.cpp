// Copyright 2022-2023 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <set>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/RegionUtils.h"



#include "Quantum/IR/QuantumOps.h"
#include "Quantum/Transforms/Passes.h"
#include "Quantum/Transforms/Patterns.h"

#include "llvm/ADT/SetVector.h"


using namespace mlir;
using namespace catalyst::quantum;

namespace {

bool hasDeviceAttribute(func::FuncOp op)
{
  StringAttr device = StringAttr::get(op->getContext(), "catalyst.device");
  if (!op->hasAttr(device)) return false;

  StringAttr deviceName = op->getAttrOfType<StringAttr>(device);
  StringAttr braketSimulatorAttr = StringAttr::get(op->getContext(), "braket.simulator");
  bool isBraketSimulator = 0 == deviceName.compare(braketSimulatorAttr);
  return isBraketSimulator ? true : false;
}

bool
hasUsesInQuantumOperations(mlir::Operation *op)
{
  for (auto i = op->user_begin(), e = op->user_end(); i != e; ++i) {
     auto user = *i;
     Dialect *dialect = user->getDialect();
     bool isUserInQuantumDialect = isa<QuantumDialect>(dialect);
     if (isUserInQuantumDialect) return true;
  }
  return false;
}

FailureOr<func::FuncOp>
outlineSingleBlockRegion(RewriterBase &rewriter,
	Location loc,
	Region &region,
	StringRef funcName,
	func::CallOp *callOp) {
  assert(!funcName.empty() && "funcName cannot be empty");
  if (!region.hasOneBlock())
    return failure();

  Block *originalBlock = &region.front();
  Operation *originalTerminator = originalBlock->getTerminator();

  // Outline before current function.
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(region.getParentOfType<func::FuncOp>());

  SetVector<Value> captures;
  getUsedValuesDefinedAbove(region, captures);

  ValueRange outlinedValues(captures.getArrayRef());
  SmallVector<Type> outlinedFuncArgTypes;
  SmallVector<Location> outlinedFuncArgLocs;
  // Region's arguments are exactly the first block's arguments as per
  // Region::getArguments().
  // Func's arguments are cat(regions's arguments, captures arguments).
  for (BlockArgument arg : region.getArguments()) {
    outlinedFuncArgTypes.push_back(arg.getType());
    outlinedFuncArgLocs.push_back(arg.getLoc());
  }
  for (Value value : outlinedValues) {
    outlinedFuncArgTypes.push_back(value.getType());
    outlinedFuncArgLocs.push_back(value.getLoc());
  }
  FunctionType outlinedFuncType =
      FunctionType::get(rewriter.getContext(), outlinedFuncArgTypes,
                        originalTerminator->getOperandTypes());
  auto outlinedFunc =
      rewriter.create<func::FuncOp>(loc, funcName, outlinedFuncType);
  Block *outlinedFuncBody = outlinedFunc.addEntryBlock();

  // Merge blocks while replacing the original block operands.
  // Warning: `mergeBlocks` erases the original block, reconstruct it later.
  int64_t numOriginalBlockArguments = originalBlock->getNumArguments();
  auto outlinedFuncBlockArgs = outlinedFuncBody->getArguments();
  {
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointToEnd(outlinedFuncBody);
    rewriter.mergeBlocks(
        originalBlock, outlinedFuncBody,
        outlinedFuncBlockArgs.take_front(numOriginalBlockArguments));
    // Explicitly set up a new ReturnOp terminator.
    rewriter.setInsertionPointToEnd(outlinedFuncBody);
    rewriter.create<func::ReturnOp>(loc, originalTerminator->getResultTypes(),
                                    originalTerminator->getOperands());
  }

  // Reconstruct the block that was deleted and add a
  // terminator(call_results).
  Block *newBlock = rewriter.createBlock(
      &region, region.begin(),
      TypeRange{outlinedFuncArgTypes}.take_front(numOriginalBlockArguments),
      ArrayRef<Location>(outlinedFuncArgLocs)
          .take_front(numOriginalBlockArguments));
  {
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointToEnd(newBlock);
    SmallVector<Value> callValues;
    llvm::append_range(callValues, newBlock->getArguments());
    llvm::append_range(callValues, outlinedValues);
    auto call = rewriter.create<func::CallOp>(loc, outlinedFunc, callValues);
    if (callOp)
      *callOp = call;

    // `originalTerminator` was moved to `outlinedFuncBody` and is still valid.
    // Clone `originalTerminator` to take the callOp results then erase it from
    // `outlinedFuncBody`.
    BlockAndValueMapping bvm;
    bvm.map(originalTerminator->getOperands(), call->getResults());
    rewriter.clone(*originalTerminator, bvm);
    rewriter.eraseOp(originalTerminator);
  }

  // Lastly, explicit RAUW outlinedValues, only for uses within `outlinedFunc`.
  // Clone the `arith::ConstantIndexOp` at the start of `outlinedFuncBody`.
  for (auto it : llvm::zip(outlinedValues, outlinedFuncBlockArgs.take_back(
                                               outlinedValues.size()))) {
    Value orig = std::get<0>(it);
    Value repl = std::get<1>(it);
    {
      OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPointToStart(outlinedFuncBody);
      if (Operation *cst = orig.getDefiningOp<arith::ConstantIndexOp>()) {
        BlockAndValueMapping bvm;
        repl = rewriter.clone(*cst, bvm)->getResult(0);
      }
    }
    orig.replaceUsesWithIf(repl, [&](OpOperand &opOperand) {
      return outlinedFunc->isProperAncestor(opOperand.getOwner());
    });
  }

  return outlinedFunc;
}

Block *
rewriteQuantumCircuitAsInlinedFunction(PatternRewriter &rewriter, func::FuncOp op) {

  func::ReturnOp returnOp;
  op.walk([&](func::ReturnOp op) {
    returnOp = op;
  });


  Block *firstBlock = &op.getRegion().front();
  Block *secondBlock = rewriter.createBlock(&op.getRegion());
  Block *thirdBlock = rewriter.createBlock(&op.getRegion());

  // The first block must jump to the second block
  rewriter.setInsertionPointToEnd(firstBlock);
  rewriter.create<cf::BranchOp>(op->getLoc(), secondBlock);
  
  // The second block must jump to the third block
  rewriter.setInsertionPointToEnd(secondBlock);
  cf::BranchOp secondBlockTerminator = rewriter.create<cf::BranchOp>(op->getLoc(), thirdBlock);

  returnOp->moveBefore(thirdBlock, thirdBlock->end());

  bool afterFirstQuantumDialectOp = false;
  std::vector<mlir::Operation *> secondBlockOps;
  std::vector<mlir::Operation *> thirdBlockOps;

  op.walk([&](mlir::Operation *nestedOp) {
    Dialect *dialect = nestedOp->getDialect();
    bool isCFDialect = isa<mlir::cf::ControlFlowDialect>(dialect);
    bool isFuncDialect = isa<func::FuncDialect>(dialect);
    bool isInvalidDialect = isCFDialect || isFuncDialect;
    if (isInvalidDialect) return;

    bool isQuantumOp = isa<QuantumDialect>(dialect);
    afterFirstQuantumDialectOp |= isQuantumOp;

    if (!afterFirstQuantumDialectOp) return;

    if (isQuantumOp) {
      secondBlockOps.push_back(nestedOp);
    } else if (!hasUsesInQuantumOperations(nestedOp)) {
      // If it doesn't have uses in QuantumOps, then it needs to be
      // pushed to the third block.
      thirdBlockOps.push_back(nestedOp);
    }
  });

  for (mlir::Operation *operation : secondBlockOps) {
    operation->moveBefore(secondBlockTerminator);
  }

  for (mlir::Operation *operation : thirdBlockOps) {
    operation->moveBefore(returnOp);
  }

  return secondBlock;
}



struct QuantumToOpenQASM3Transform : public OpRewritePattern<func::FuncOp> {
    using OpRewritePattern<func::FuncOp>::OpRewritePattern;

    LogicalResult match(func::FuncOp op) const override;
    void rewrite(func::FuncOp op, PatternRewriter &rewriter) const override;
};

LogicalResult
QuantumToOpenQASM3Transform::match(func::FuncOp op) const { return hasDeviceAttribute(op) ? success() : failure(); }

void
QuantumToOpenQASM3Transform::rewrite(func::FuncOp op, PatternRewriter &rewriter) const  {
 
  // We've separated the quantum instructions into their own block.
  Block *quantumBlock = rewriteQuantumCircuitAsInlinedFunction(rewriter, op);

  // Let's get the operand's uses
  SetVector<Value> returnValues;
  for (mlir::Operation &currentOp: quantumBlock->getOperations())
  {
    bool usesOutside = currentOp.isUsedOutsideOfBlock(quantumBlock);
    if (!usesOutside) continue;

    returnValues.insert(currentOp.getResults().begin(), currentOp.getResults().end());
  }


  Block *firstBlock = &op.getRegion().front();
  Block *lastBlock = &op.getRegion().back();
  mlir::Operation &firstBlockTerminator = firstBlock->back();
  rewriter.eraseOp(&firstBlockTerminator);
  rewriter.setInsertionPointToEnd(firstBlock);
  rewriter.create<cf::BranchOp>(op->getLoc(), lastBlock);

  ArrayRef<Value> returnValuesVector = returnValues.getArrayRef();
  TypeRange valueTypes = TypeRange(returnValuesVector);
  rewriter.setInsertionPointToStart(lastBlock);
  scf::ExecuteRegionOp executeRegionOp = rewriter.create<scf::ExecuteRegionOp>(op->getLoc(), valueTypes);
  executeRegionOp.getRegion().emplaceBlock();
  Block &executeRegionBlock = executeRegionOp.getRegion().front();
  executeRegionBlock.getOperations().splice(executeRegionBlock.begin(), quantumBlock->getOperations());

  executeRegionBlock.back().erase();
  rewriter.setInsertionPointToEnd(&executeRegionBlock);
  rewriter.create<scf::YieldOp>(op->getLoc(), returnValuesVector);
  quantumBlock->erase();

  SmallPtrSet<Operation *, 4> usesInsideQuantumBlock;
  for (mlir::Operation &op : executeRegionBlock.getOperations()) {
    usesInsideQuantumBlock.insert(&op);
  }

  for (auto it: llvm::zip(returnValuesVector, executeRegionOp.getResults())) {
    Value orig = std::get<0>(it);
    Value repl = std::get<1>(it);
    orig.replaceAllUsesExcept(repl, usesInsideQuantumBlock);
  }

  func::CallOp callOp;
  FailureOr<func::FuncOp> outlined = outlineSingleBlockRegion(rewriter, op->getLoc(), executeRegionOp.getRegion(), "functionName", &callOp);
  //op.emitRemark() << *outlined;
  StringAttr deviceAttr = StringAttr::get(op->getContext(), "catalyst.device");
  op->removeAttr(deviceAttr);

}

} // namespace

namespace catalyst {
namespace quantum {

struct QuantumToOpenQasm3Pass
    : public PassWrapper<QuantumToOpenQasm3Pass, OperationPass<ModuleOp>> {
    QuantumToOpenQasm3Pass() {}

    StringRef getArgument() const override { return "convert-quantum-to-openqasm3"; }

    StringRef getDescription() const override { return "Convert quantum dialect to openqasm3."; }

    void getDependentDialects(DialectRegistry &registry) const override
    {
        registry.insert<func::FuncDialect>();
        registry.insert<cf::ControlFlowDialect>();
        registry.insert<scf::SCFDialect>();
        registry.insert<arith::ArithDialect>();
	registry.insert<QuantumDialect>();
    }

    void runOnOperation() final
    {
        MLIRContext *context = &getContext();
        RewritePatternSet patterns(context);
        patterns.add<QuantumToOpenQASM3Transform>(context);

        if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
            signalPassFailure();
	}
    }
};

} // namespace quantum

std::unique_ptr<Pass> createQuantumToOpenQasm3Pass()
{
    return std::make_unique<quantum::QuantumToOpenQasm3Pass>();
}

} // namespace catalyst
