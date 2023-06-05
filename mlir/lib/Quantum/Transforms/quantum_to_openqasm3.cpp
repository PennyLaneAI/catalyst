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
#include <map>

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

int
isParameterToFunction(Value val)
{
  if (!isa<BlockArgument>(val)) return -1;

  return cast<BlockArgument>(val).getArgNumber();
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
 
  
  std::vector<DifferentiableGate> differentiableGates;
  op.walk([&](mlir::Operation *nestedOp) {
     if (DifferentiableGate gate = dyn_cast<DifferentiableGate>(nestedOp)) {
       differentiableGates.push_back(gate);
     }
  });

  for (auto gate : differentiableGates) {
    if (isa<CustomOp>(gate)) {
      ValueRange gateParams = gate.getDiffParams();
      rewriter.setInsertionPoint(gate);
      std::vector<Value> isGateParamFunctionParam;
      for (auto gateParam : gateParams) {
        int isFunctionParam = isParameterToFunction(gateParam);
	Type i64 = rewriter.getI64Type();
	Value paramVal = rewriter.create<arith::ConstantOp>(gate.getLoc(), i64, rewriter.getIntegerAttr(i64, isFunctionParam));
	isGateParamFunctionParam.push_back(paramVal);
      }

      CustomOp customOp = cast<CustomOp>(gate);
      OpenQASM3CustomOp newOp = rewriter.replaceOpWithNewOp<OpenQASM3CustomOp>(gate, customOp.getResultTypes(), gateParams, isGateParamFunctionParam, customOp.getInQubits(), customOp.getGateName());
    }
  }

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
