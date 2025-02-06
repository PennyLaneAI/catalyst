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
#include "mlir/Support/LogicalResult.h"

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
template <typename OriginOp> GateConversion getPauliOperators(OriginOp *op)
{
    mlir::StringRef opName = op->getGateName();
    auto gateConversion = gateMap.find(opName);

    if (gateConversion == gateMap.end()) {
        op->emitError("Unsupported gate: ") << opName;
        return GateConversion();
    }

    return gateConversion->second;
}

template <typename OriginOp, typename LoweredQECOp>
LoweredQECOp convertCustomOpToPPR(OriginOp *op, ConversionPatternRewriter &rewriter)
{
    auto loc = op->getLoc();
    auto gateConversion = getPauliOperators(op);

    if (gateConversion.pauliOperators.empty()) {
        return PPRotationOp();
    }

    ArrayAttr pauliProduct = rewriter.getStrArrayAttr(gateConversion.pauliOperators);
    ValueRange inQubits = op->getInQubits();
    TypeRange outQubitsTypes = op->getOutQubits().getTypes();
    Value thetaValue = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF64Type(), rewriter.getF64FloatAttr(gateConversion.theta));

    // Create a new PPRotationOp with the Pauli operators and theta
    auto pprOp =
        rewriter.create<LoweredQECOp>(loc, outQubitsTypes, pauliProduct, thetaValue, inQubits);

    // Replace the original operation with the new PPRotationOp
    return pprOp;
}

template <typename OriginOp, typename LoweredQECOp>
LoweredQECOp convertMeasureOpToPPM(OriginOp *op, ConversionPatternRewriter &rewriter)
{
    auto loc = op->getLoc();

    // pauli product is always Z
    ArrayAttr pauliProduct = rewriter.getStrArrayAttr({"Z"});
    ValueRange inQubits = op->getInQubit();
    TypeRange outQubitsTypes = op->getResults().getType();

    auto ppmOp = rewriter.create<LoweredQECOp>(loc, outQubitsTypes, pauliProduct, inQubits);

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
        if (llvm::isa<quantum::CustomOp>(op)) {
            auto originOp = llvm::cast<quantum::CustomOp>(op);
            loweredOp = convertCustomOpToPPR<CustomOp, PPRotationOp>(&originOp, rewriter);

        }else if (llvm::isa<quantum::MeasureOp>(op)) {
            auto originOp = llvm::cast<quantum::MeasureOp>(op);
            loweredOp = convertMeasureOpToPPM<MeasureOp, PPMeasurementOp>(&originOp, rewriter);
        }

        if (!loweredOp) {
            return failure();
        }

        rewriter.replaceOp(op, loweredOp);
        return success();

        return failure();
    }
};

// TODO: add more lowering patterns here. e.g. StaticCustomOp, UnitaryCustomOp, etc.
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
