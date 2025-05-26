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

#include "ParameterShift.hpp"

#include "mlir/Dialect/SCF/IR/SCF.h"

#include "Gradient/Utils/DifferentialQNode.h"
#include "Gradient/Utils/GradientShape.h"
#include "Quantum/IR/QuantumOps.h"

namespace catalyst {
namespace gradient {

LogicalResult ParameterShiftLowering::matchAndRewrite(func::FuncOp op,
                                                      PatternRewriter &rewriter) const
{
    if (!(getQNodeDiffMethod(op) == "parameter-shift" && requiresCustomGradient(op))) {
        return failure();
    }

    Location loc = op.getLoc();
    rewriter.setInsertionPointAfter(op);

    // Determine the number of parameters to shift (= to the total static number of gate
    // parameters occuring in the function) and number of selectors needed (= to the number of
    // loop nests containing quantum instructions with at least one gate parameter).
    auto [numShifts, loopDepth] = analyzeFunction(op);

    // Generate the shifted version of callee, enabling us to shift an arbitrary gate
    // parameter at runtime.
    func::FuncOp shiftFn = genShiftFunction(rewriter, loc, op, numShifts, loopDepth);

    // Generate the quantum gradient function, exploiting the structure of the original function
    // to dynamically compute the partial derivate with respect to each gate parameter.
    func::FuncOp qGradFn = genQGradFunction(rewriter, loc, op, shiftFn, numShifts, loopDepth);

    // Register the quantum gradient on the quantum-only split-out QNode.
    registerCustomGradient(op, FlatSymbolRefAttr::get(qGradFn));
    return success();
}

std::pair<int64_t, int64_t> ParameterShiftLowering::analyzeFunction(func::FuncOp callee)
{
    int64_t numShifts = 0;
    int64_t loopLevel = 0;
    int64_t maxLoopDepth = 0;

    callee.walk<WalkOrder::PreOrder>([&](Operation *op) {
        if (isa<scf::ForOp>(op)) {
            loopLevel++;
        }
        else if (auto gate = dyn_cast<quantum::DifferentiableGate>(op)) {
            if (gate.getDiffParams().empty())
                return;

            numShifts += gate.getDiffParams().size();
            maxLoopDepth = std::max(loopLevel, maxLoopDepth);
        }
        else if (isa<scf::YieldOp>(op) && isa<scf::ForOp>(op->getParentOp())) {
            loopLevel--;
        }
    });

    return {numShifts, maxLoopDepth};
}

} // namespace gradient
} // namespace catalyst
