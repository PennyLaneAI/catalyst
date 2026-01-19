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

#include "mlir/Dialect/SCF/IR/SCF.h"

#include "QEC/IR/QECOps.h"
#include "QEC/Transforms/Patterns.h"

using namespace mlir;
using namespace catalyst;
using namespace catalyst::qec;

namespace {

// Lower qec.select.ppm to scf.if with two ppm operations.
//
// For example:
// %mres, %out = qec.select.ppm(%cond, ["X"], ["Z"]) %qubits : i1, !quantum.bit
//
// becomes:
// %mres, %out = scf.if %cond -> (i1, !quantum.bit) {
//   %m0, %out0 = qec.ppm ["X"] %qubits : i1, !quantum.bit
//   scf.yield %m0, %out0 : i1, !quantum.bit
// } else {
//   %m1, %out1 = qec.ppm ["Z"] %qubits : i1, !quantum.bit
//   scf.yield %m1, %out1 : i1, !quantum.bit
// }
struct LowerSelectPPM : public OpRewritePattern<SelectPPMeasurementOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(SelectPPMeasurementOp op,
                                  PatternRewriter &rewriter) const override
    {
        Location loc = op.getLoc();
        mlir::Value selectSwitch = op.getSelectSwitch();
        ArrayAttr pauliProduct0 = op.getPauliProduct_0();
        ArrayAttr pauliProduct1 = op.getPauliProduct_1();
        ValueRange inQubits = op.getInQubits();

        SmallVector<mlir::Type> resultTypes; // (i1, qubits)
        resultTypes.push_back(rewriter.getI1Type());
        for (auto qubit : inQubits) {
            resultTypes.push_back(qubit.getType());
        }

        auto ifOp = rewriter.create<scf::IfOp>(loc, resultTypes, selectSwitch, true);
        {
            OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());
            auto ppm0 = rewriter.create<PPMeasurementOp>(loc, pauliProduct0, inQubits);
            SmallVector<mlir::Value> yieldValues;
            yieldValues.push_back(ppm0.getMres());
            yieldValues.append(ppm0.getOutQubits().begin(), ppm0.getOutQubits().end());
            rewriter.create<scf::YieldOp>(loc, yieldValues);
        }

        {
            OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToStart(&ifOp.getElseRegion().front());
            auto ppm1 = rewriter.create<PPMeasurementOp>(loc, pauliProduct1, inQubits);
            SmallVector<mlir::Value> yieldValues;
            yieldValues.push_back(ppm1.getMres());
            yieldValues.append(ppm1.getOutQubits().begin(), ppm1.getOutQubits().end());
            rewriter.create<scf::YieldOp>(loc, yieldValues);
        }

        rewriter.replaceOp(op, ifOp.getResults());
        return success();
    }
};

// Lower qec.ppr cond(...) to scf.if with ppr operation
//
// For example:
// %out = qec.ppr ["X"](4) %qubits cond(%cond) : !quantum.bit
//
// becomes:
// %out = scf.if %cond -> (!quantum.bit) {
//   %out0 = qec.ppr ["X"](4) %qubits : !quantum.bit
//   scf.yield %out0 : !quantum.bit
// } else {
//   scf.yield %qubits : !quantum.bit
// }
struct LowerCondPPR : public OpRewritePattern<PPRotationOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(PPRotationOp op, PatternRewriter &rewriter) const override
    {
        // Only match if there's a condition
        if (!op.getCondition()) {
            return failure();
        }

        Location loc = op.getLoc();
        mlir::Value condition = op.getCondition();
        ArrayAttr pauliProduct = op.getPauliProduct();
        IntegerAttr rotationKind = op.getRotationKindAttr();
        ValueRange inQubits = op.getInQubits();

        SmallVector<mlir::Type> resultTypes;
        for (auto qubit : inQubits) {
            resultTypes.push_back(qubit.getType());
        }

        auto ifOp = rewriter.create<scf::IfOp>(loc, resultTypes, condition, true);
        {
            OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());
            auto ppr = rewriter.create<PPRotationOp>(loc, resultTypes, pauliProduct, rotationKind,
                                                     inQubits);
            rewriter.create<scf::YieldOp>(loc, ppr.getOutQubits());
        }

        {
            // Unchanged else block
            OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToStart(&ifOp.getElseRegion().front());
            rewriter.create<scf::YieldOp>(loc, inQubits);
        }

        rewriter.replaceOp(op, ifOp.getResults());
        return success();
    }
};

} // namespace

namespace catalyst {
namespace qec {

void populateUnrollConditionalPPRPPMPatterns(RewritePatternSet &patterns)
{
    patterns.add<LowerSelectPPM>(patterns.getContext());
    patterns.add<LowerCondPPR>(patterns.getContext());
}

} // namespace qec
} // namespace catalyst
