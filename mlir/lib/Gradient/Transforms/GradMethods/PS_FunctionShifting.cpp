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

#include <string>
#include <vector>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include "Quantum/IR/QuantumOps.h"

namespace catalyst {
namespace gradient {

static Value genSelectiveShift(PatternRewriter &rewriter, Location loc, Value param, Value shift,
                               const std::vector<std::pair<Value, Value>> &selectors)
{
    if (selectors.empty()) {
        return rewriter.create<arith::AddFOp>(loc, shift, param);
    }

    // Make sure all active iteration variables match the selectors.
    Value shiftCondition = rewriter.create<arith::ConstantIntOp>(loc, true, 1);
    for (auto &[iteration, selector] : selectors) {
        Value iterationMatch =
            rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, iteration, selector);
        shiftCondition = rewriter.create<arith::AndIOp>(loc, shiftCondition, iterationMatch);
    }

    scf::IfOp ifOp = rewriter.create<scf::IfOp>(
        loc, shiftCondition,
        [&](OpBuilder &builder, Location loc) { // then
            Value shiftedParam = builder.create<arith::AddFOp>(loc, shift, param);
            builder.create<scf::YieldOp>(loc, shiftedParam);
        },
        [&](OpBuilder &builder, Location loc) { // else
            builder.create<scf::YieldOp>(loc, param);
        });

    return ifOp.getResult(0);
}

func::FuncOp ParameterShiftLowering::genShiftFunction(PatternRewriter &rewriter, Location loc,
                                                      func::FuncOp callee, const int64_t numShifts,
                                                      const int64_t loopDepth)
{
    // The shiftVector is a new function argument with 1 element for each gate parameter to be
    // shifted. For gates inside of loops, we additionally use a selector to dynamically
    // choose on which iteration of a loop to shift the gate parameter.
    Type shiftVectorType = RankedTensorType::get({numShifts}, rewriter.getF64Type());
    Type selectorVectorType = RankedTensorType::get({loopDepth}, rewriter.getIndexType());

    // Define the properties of the "shifted" version of the function to be differentiated.
    std::string fnName = callee.getSymName().str() + ".shifted";
    std::vector<Type> fnArgTypes = callee.getArgumentTypes().vec();
    fnArgTypes.push_back(shiftVectorType);
    fnArgTypes.push_back(selectorVectorType);
    FunctionType fnType = rewriter.getFunctionType(fnArgTypes, callee.getResultTypes());
    StringAttr visibility = rewriter.getStringAttr("private");

    func::FuncOp shiftedFn =
        SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(callee, rewriter.getStringAttr(fnName));
    if (!shiftedFn) {
        PatternRewriter::InsertionGuard insertGuard(rewriter);

        shiftedFn =
            rewriter.create<func::FuncOp>(loc, fnName, fnType, visibility, nullptr, nullptr);

        // First copy the entire function as is, then we can add the shifts.
        // Make sure to add the shiftVector/selectorVector parameters to the new function.
        rewriter.cloneRegionBefore(callee.getBody(), shiftedFn.getBody(), shiftedFn.end());
        Value shiftVector = shiftedFn.getBlocks().front().addArgument(shiftVectorType, loc);
        Value selectorVector = shiftedFn.getBlocks().front().addArgument(selectorVectorType, loc);
        std::vector<std::pair<Value, Value>> selectors;
        selectors.reserve(loopDepth);

        int shiftsProcessed = 0;
        shiftedFn.walk<WalkOrder::PreOrder>([&](Operation *op) {
            // TODO: Add support for other SCF (and Affine?) loops in the future.
            if (auto forOp = dyn_cast<scf::ForOp>(op)) {
                // When entering a for loop, we need to remember to compare the appropriate selector
                // against the current loop iteration variable before shifting any nested gates.
                PatternRewriter::InsertionGuard insertGuard(rewriter);
                rewriter.setInsertionPointToStart(forOp.getBody());

                Value idx = rewriter.create<arith::ConstantOp>(
                    loc, rewriter.getIndexAttr(selectors.size()));
                Value selector = rewriter.create<tensor::ExtractOp>(loc, selectorVector, idx);
                Value iteration = forOp.getInductionVar();
                selectors.push_back({iteration, selector});
            }
            else if (auto gate = dyn_cast<quantum::DifferentiableGate>(op)) {
                if (gate.getDiffParams().empty()) {
                    return;
                }

                PatternRewriter::InsertionGuard insertGuard(rewriter);
                rewriter.setInsertionPoint(gate);

                ValueRange params = gate.getDiffParams();
                std::vector<Value> shiftedParams;
                shiftedParams.reserve(params.size());

                for (size_t i = 0; i < params.size(); i++) {
                    Value idx = rewriter.create<index::ConstantOp>(loc, shiftsProcessed++);
                    Value shift = rewriter.create<tensor::ExtractOp>(loc, shiftVector, idx);
                    Value shiftedParam =
                        genSelectiveShift(rewriter, loc, params[i], shift, selectors);
                    shiftedParams.push_back(shiftedParam);
                }

                gate->setOperands(gate.getDiffOperandIdx(), shiftedParams.size(), shiftedParams);
            }
            else if (isa<scf::YieldOp>(op) && isa<scf::ForOp>(op->getParentOp())) {
                // When we reach the end of a for loop, remove its iteration variable from the list.
                selectors.pop_back();
            }
        });
    }

    return shiftedFn;
}

} // namespace gradient
} // namespace catalyst
