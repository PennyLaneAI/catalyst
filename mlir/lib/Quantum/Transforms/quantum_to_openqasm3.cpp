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
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

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

std::pair<DeviceOp, DeallocOp>
sinkQuantumOps(func::FuncOp op) {
  // Let's do a little bit of TypeState analysis (but not really).
  // We just need to make sure that the current quantum device is set
  // and where it ends. All operations in the middle which do not belong
  // to the quantum dialect can be lifted to before the quantum device selection.

  bool afterQuantumDevice = false;
  bool afterDeallocOp = false;
  DeviceOp deviceOp;
  DeallocOp deallocOp;

  op.walk([&](mlir::Operation *nestedOp) {
    // We are not going to do anything unless we are past the first quantum device op.
    bool isQuantumDeviceOp = isa<DeviceOp>(nestedOp);
    if (isQuantumDeviceOp) deviceOp = cast<DeviceOp>(nestedOp);

    afterQuantumDevice |= isQuantumDeviceOp;
    // We are also not going to do anything to operations after the dealloc operation.
    bool isQuantumDeallocOp = isa<DeallocOp>(nestedOp);
    if (isQuantumDeallocOp) deallocOp = cast<DeallocOp>(nestedOp);
    afterDeallocOp |= isQuantumDeallocOp;


    Dialect *dialect = nestedOp->getDialect();
    bool isOpInQuantumDialect = isa<QuantumDialect>(dialect);
    bool isOpInFuncDialect = isa<func::FuncDialect>(dialect);
    bool shouldLiftThisOperation = !isOpInQuantumDialect && !isOpInFuncDialect && afterQuantumDevice && !afterDeallocOp;
    if (!shouldLiftThisOperation) return;

    nestedOp->moveBefore(deviceOp);
  });

  return { deviceOp, deallocOp };

}

bool
hasUsesInOperationsOutsideOfQuantumDialect(mlir::Operation *op)
{
  for (auto i = op->user_begin(), e = op->user_end(); i != e; ++i) {
     auto user = *i;
     Dialect *dialect = user->getDialect();
     bool isUserInQuantumDialect = isa<QuantumDialect>(dialect);
     if (!isUserInQuantumDialect) return true;
  }
  return false;
}

void
rewriteQuantumCircuitAsInlinedFunction(PatternRewriter &rewriter, func::FuncOp op) {

  auto [deviceOp, deallocOp] = sinkQuantumOps(op);

  func::ReturnOp returnOp;
  op.walk([&](func::ReturnOp op) {
    returnOp = op;
  });


  Block *firstBlock = &op.getRegion().front();
  Block *secondBlock = rewriter.createBlock(&op.getRegion());
  Block *thirdBlock = rewriter.createBlock(&op.getRegion());

  // The first block must jump to the second block
  rewriter.setInsertionPointToEnd(firstBlock);
  cf::BranchOp firstBlockTerminator = rewriter.create<cf::BranchOp>(op->getLoc(), secondBlock);
  
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
      nestedOp->emitRemark() << "second";
    } else {
      thirdBlockOps.push_back(nestedOp);
      nestedOp->emitRemark() << "third";
    }
  });

  for (mlir::Operation *operation : secondBlockOps) {
    operation->moveBefore(secondBlockTerminator);
  }

  for (mlir::Operation *operation : thirdBlockOps) {
    operation->moveBefore(returnOp);
  }
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
  // We are essentially going to outline the quantum portion...
  
  StringAttr deviceAttr = StringAttr::get(op->getContext(), "catalyst.device");
  rewriteQuantumCircuitAsInlinedFunction(rewriter, op);
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
        registry.insert<arith::ArithDialect>();
        registry.insert<cf::ControlFlowDialect>();
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
