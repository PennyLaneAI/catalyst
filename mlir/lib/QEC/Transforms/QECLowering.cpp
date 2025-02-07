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

#include "llvm/Support/Casting.h"

#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"

#include "QEC/IR/QECDialect.h"
#include "QEC/Transforms/Patterns.h"
#include "Quantum/IR/QuantumOps.h"

using namespace mlir;
using namespace catalyst;
using namespace catalyst::quantum;
using namespace catalyst::qec;

namespace {

// Structure to define gate conversion rules
struct GateConversion {
    SmallVector<StringRef> pauliOperators;
    double theta;
};

// Map of gate names to their Pauli decompositions
// Ref: Fig. 5 in [A Game of Surface Codes](https://doi.org/10.22331/q-2019-03-05-128)
const llvm::StringMap<GateConversion> gateMap = {{"H", {{"Z", "X", "Z"}, M_PI / 4}},
                                                 {"Hadamard", {{"Z", "X", "Z"}, M_PI / 4}},
                                                 {"S", {{"Z"}, M_PI / 4}},
                                                 {"T", {{"Z"}, M_PI / 8}},
                                                 {"CNOT", {{"Z", "X"}, M_PI / 4}}};

// Get the Pauli operators and theta for a given gate
GateConversion getPauliOperators(CustomOp op)
{
    mlir::StringRef opName = op.getGateName();
    auto gateConversion = gateMap.find(opName);

    if (gateConversion == gateMap.end()) {
        op->emitError("Unsupported gate: ") << opName;
        return GateConversion();
    }

    return gateConversion->second;
}

// Convert a CustomOp to a PPRotationOp
PPRotationOp convertCustomOpToPPR(CustomOp op, ConversionPatternRewriter &rewriter)
{
    auto loc = op.getLoc();
    auto gateConversion = getPauliOperators(op);

    if (gateConversion.pauliOperators.empty()) {
        auto msg = "Unsupported gate: " + op.getGateName().str() + " for lowering to PPR.";
        msg += " Supported gates are: H, S, T, CNOT.";
        op->emitError(msg);
        return nullptr;
    }

    ArrayAttr pauliProduct = rewriter.getStrArrayAttr(gateConversion.pauliOperators);
    ValueRange inQubits = op.getInQubits();
    TypeRange outQubitsTypes = op.getOutQubits().getType();
    Value thetaValue = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF64Type(), rewriter.getF64FloatAttr(gateConversion.theta));

    auto pprOp =
        rewriter.create<PPRotationOp>(loc, outQubitsTypes, pauliProduct, thetaValue, inQubits);

    return pprOp;
}

// Convert a MeasureOp to a PPMeasurementOp
PPMeasurementOp convertMeasureOpToPPM(MeasureOp op, ConversionPatternRewriter &rewriter)
{
    auto loc = op.getLoc();

    // Pauli product is always Z
    ArrayAttr pauliProduct = rewriter.getStrArrayAttr({"Z"});
    ValueRange inQubits = op.getInQubit();
    TypeRange outQubitsTypes = op->getResults().getType();

    auto ppmOp = rewriter.create<PPMeasurementOp>(loc, outQubitsTypes, pauliProduct, inQubits);

    return ppmOp;
}

template <typename OriginOp, typename LoweredQECOp>
struct QECOpLowering : public ConversionPattern {
    QECOpLowering(MLIRContext *context)
        : ConversionPattern(OriginOp::getOperationName(), 1, context)
    {
    }

    LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const final
    {
        Operation *loweredOp = nullptr;

        // cast to OriginOp
        if (isa<quantum::CustomOp>(op)) {
            auto originOp = cast<quantum::CustomOp>(op);
            loweredOp = convertCustomOpToPPR(originOp, rewriter);
        }
        else if (isa<quantum::MeasureOp>(op)) {
            auto originOp = cast<quantum::MeasureOp>(op);
            loweredOp = convertMeasureOpToPPM(originOp, rewriter);
        }

        if (!loweredOp) {
            op->emitError("Failed to lower operation to QEC dialect.");
            return failure();
        }

        rewriter.replaceOp(op, loweredOp);
        return success();
    }
};

using CustomOpLowering = QECOpLowering<quantum::CustomOp, qec::PPRotationOp>;
using MeasureOpLowering = QECOpLowering<quantum::MeasureOp, qec::PPMeasurementOp>;

} // namespace

namespace catalyst {
namespace qec {

void populateQECLoweringPatterns(RewritePatternSet &patterns)
{
    patterns.add<CustomOpLowering>(patterns.getContext());
    patterns.add<MeasureOpLowering>(patterns.getContext());
}

} // namespace qec
} // namespace catalyst
