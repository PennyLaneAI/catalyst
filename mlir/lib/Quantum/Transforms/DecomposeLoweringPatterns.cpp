// Copyright 2025 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#define DEBUG_TYPE "decompose-lowering"

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"

#include "DecomposeLoweringImpl.hpp"
#include "Quantum/IR/QuantumInterfaces.h"
#include "Quantum/IR/QuantumOps.h"

using namespace mlir;
using namespace catalyst::quantum;

namespace catalyst {
namespace quantum {

struct DLCustomOpPattern : public OpRewritePattern<CustomOp> {
  private:
    const llvm::StringMap<func::FuncOp> &decompositionRegistry;
    const llvm::StringSet<llvm::MallocAllocator> &targetGateSet;

  public:
    DLCustomOpPattern(MLIRContext *context, const llvm::StringMap<func::FuncOp> &registry,
                      const llvm::StringSet<llvm::MallocAllocator> &gateSet)
        : OpRewritePattern<CustomOp>(context), decompositionRegistry(registry),
          targetGateSet(gateSet)
    {
    }

    LogicalResult matchAndRewrite(CustomOp op, PatternRewriter &rewriter) const override
    {
        StringRef gateName = op.getGateName();

        // Only decompose the op if it is not in the target gate set
        if (targetGateSet.contains(gateName)) {
            return failure();
        }

        // Find the corresponding decomposition function for the op
        auto it = decompositionRegistry.find(gateName);
        if (it == decompositionRegistry.end()) {
            return failure();
        }
        func::FuncOp decompFunc = it->second;

        // Here is the assumption that the decomposition function must have at least one input and
        // one result
        assert(decompFunc.getFunctionType().getNumInputs() > 0 &&
               "Decomposition function must have at least one input");
        assert(decompFunc.getFunctionType().getNumResults() >= 1 &&
               "Decomposition function must have at least one result");

        rewriter.setInsertionPointAfter(op);

        auto enableQreg = isa<quantum::QuregType>(decompFunc.getFunctionType().getInput(0));
        auto analyzer = CustomOpSignatureAnalyzer(op, enableQreg, rewriter);
        assert(analyzer && "Analyzer should be valid");

        auto callOperands = analyzer.prepareCallOperands(decompFunc, rewriter, op.getLoc());
        auto callOp =
            func::CallOp::create(rewriter, op.getLoc(), decompFunc.getFunctionType().getResults(),
                                 decompFunc.getSymName(), callOperands);

        // Replace the op with the call op and adjust the insert ops for the qreg mode
        if (callOp.getNumResults() == 1 && isa<quantum::QuregType>(callOp.getResult(0).getType())) {
            auto results = analyzer.prepareCallResultForQreg(callOp, rewriter);
            rewriter.replaceOp(op, results);
        }
        else {
            rewriter.replaceOp(op, callOp->getResults());
        }

        return success();
    }
};

struct DLMultiRZOpPattern : public OpRewritePattern<MultiRZOp> {
  private:
    const llvm::StringMap<func::FuncOp> &decompositionRegistry;
    const llvm::StringSet<llvm::MallocAllocator> &targetGateSet;

  public:
    DLMultiRZOpPattern(MLIRContext *context, const llvm::StringMap<func::FuncOp> &registry,
                       const llvm::StringSet<llvm::MallocAllocator> &gateSet)
        : OpRewritePattern<MultiRZOp>(context), decompositionRegistry(registry),
          targetGateSet(gateSet)
    {
    }

    LogicalResult matchAndRewrite(MultiRZOp op, PatternRewriter &rewriter) const override
    {
        std::string gateName = "MultiRZ";

        // Only decompose the op if it is not in the target gate set
        if (targetGateSet.contains(gateName)) {
            return failure();
        }

        // Find the corresponding decomposition function for the op
        auto numQubits = op.getInQubits().size();
        auto MRZNameWithQubits = gateName + "_" + std::to_string(numQubits);

        auto it = decompositionRegistry.find(MRZNameWithQubits);
        if (it == decompositionRegistry.end()) {
            return failure();
        }

        func::FuncOp decompFunc = it->second;
        // Here is the assumption that the decomposition function must have
        // at least one input and one result
        assert(decompFunc.getFunctionType().getNumInputs() > 0 &&
               "Decomposition function must have at least one input");
        assert(decompFunc.getFunctionType().getNumResults() >= 1 &&
               "Decomposition function must have at least one result");

        rewriter.setInsertionPointAfter(op);

        auto enableQreg = isa<quantum::QuregType>(decompFunc.getFunctionType().getInput(0));
        auto numQbitsAttr = decompFunc->getAttrOfType<IntegerAttr>("num_wires");
        if (!numQbitsAttr) {
            op.emitError("Decomposition function missing 'num_wires' attribute");
            return failure();
        }
        if (numQubits != static_cast<size_t>(numQbitsAttr.getInt())) {
            op.emitError("Mismatch in number of qubits: expected ")
                << numQbitsAttr.getInt() << ", got " << numQubits;
            return failure();
        }

        auto analyzer = MultiRZOpSignatureAnalyzer(op, enableQreg, rewriter);
        assert(analyzer && "Analyzer should be valid");

        auto callOperands = analyzer.prepareCallOperands(decompFunc, rewriter, op.getLoc());
        auto callOp =
            func::CallOp::create(rewriter, op.getLoc(), decompFunc.getFunctionType().getResults(),
                                 decompFunc.getSymName(), callOperands);

        // Replace the op with the call op and adjust the insert ops for the qreg mode
        if (callOp.getNumResults() == 1 && isa<quantum::QuregType>(callOp.getResult(0).getType())) {
            auto results = analyzer.prepareCallResultForQreg(callOp, rewriter);
            rewriter.replaceOp(op, results);
        }
        else {
            rewriter.replaceOp(op, callOp->getResults());
        }

        return success();
    }
};

void populateDecomposeLoweringPatterns(RewritePatternSet &patterns,
                                       const llvm::StringMap<func::FuncOp> &decompositionRegistry,
                                       const llvm::StringSet<llvm::MallocAllocator> &targetGateSet)
{
    patterns.add<DLCustomOpPattern>(patterns.getContext(), decompositionRegistry, targetGateSet);
    patterns.add<DLMultiRZOpPattern>(patterns.getContext(), decompositionRegistry, targetGateSet);
}

} // namespace quantum
} // namespace catalyst
