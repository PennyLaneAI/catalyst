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

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/TypeRange.h"

#include "QEC/IR/QECDialect.h"
#include "QEC/Transforms/Patterns.h"
#include "Quantum/IR/QuantumOps.h"

using namespace mlir;
using namespace catalyst;
using namespace catalyst::quantum;
using namespace catalyst::qec;

namespace {

//===----------------------------------------------------------------------===//
//                       Helper functions
//===----------------------------------------------------------------------===//

enum class GateEnum { H, S, T, CNOT, Unknown };

// Hash gate name to GateEnum
GateEnum hashGate(CustomOp op)
{
    auto gateName = op.getGateName();
    if (gateName == "H" || gateName == "Hadamard")
        return GateEnum::H;
    else if (gateName == "S")
        return GateEnum::S;
    else if (gateName == "T")
        return GateEnum::T;
    else if (gateName == "CNOT")
        return GateEnum::CNOT;
    else
        return GateEnum::Unknown;
}

// Structure to define gate conversion rules
struct GateConversion {
    SmallVector<StringRef> pauliOperators;
    double theta;
    GateConversion(SmallVector<StringRef> pauliOperators, double theta)
        : pauliOperators(pauliOperators), theta(theta)
    {
    }
    GateConversion() : pauliOperators(), theta(0.0) {}
};

Value getTheta(Location loc, GateConversion gateConversion, ConversionPatternRewriter &rewriter)
{
    return rewriter.create<arith::ConstantOp>(loc, rewriter.getF64Type(),
                                              rewriter.getF64FloatAttr(gateConversion.theta));
}

std::pair<ValueRange, TypeRange> getInQubitsAndOutQubits(CustomOp op)
{
    ValueRange inQubits = op.getInQubits();
    TypeRange outQubitsTypes = op.getOutQubits().getType();

    return std::make_pair(inQubits, outQubitsTypes);
}

//===----------------------------------------------------------------------===//
//                       Gate conversion functions
//===----------------------------------------------------------------------===//

// C(P) = G(Angle)
PPRotationOp singleQubitConversion(CustomOp op, GateConversion gateConversion,
                                   ConversionPatternRewriter &rewriter)
{
    auto loc = op->getLoc();

    auto pauliProduct = rewriter.getStrArrayAttr(gateConversion.pauliOperators);
    auto thetaValue = getTheta(loc, gateConversion, rewriter);
    auto [inQubits, outQubitsTypes] = getInQubitsAndOutQubits(op);

    auto pprOp =
        rewriter.create<PPRotationOp>(loc, outQubitsTypes, pauliProduct, thetaValue, inQubits);

    return pprOp;
}

// Ref: Fig. 5 in [A Game of Surface Codes](https://doi.org/10.22331/q-2019-03-05-128)
// C(P1, P2) = G0 · G1 · G2
// G0 = (P1 ⊗ P2)π/4
// G1 = (P1 ⊗ 1)−π/4
// G2 = (1 ⊗ P2)−π/4
LogicalResult multiQubitConversion(CustomOp op, StringRef P1, StringRef P2,
                                   ConversionPatternRewriter &rewriter)
{
    auto loc = op->getLoc();

    auto g0 = GateConversion({P1, P2}, 4);
    auto g1 = GateConversion({P1}, -4);
    auto g2 = GateConversion({P2}, -4);

    rewriter.setInsertionPoint(op);

    // G0 = (P1 ⊗ P2)π/4
    auto pauliProduct = rewriter.getStrArrayAttr(g0.pauliOperators);
    auto thetaValue = getTheta(loc, g0, rewriter);
    auto [inQubits, outQubitsTypes] = getInQubitsAndOutQubits(op);
    auto G0 =
        rewriter.create<PPRotationOp>(loc, outQubitsTypes, pauliProduct, thetaValue, inQubits);

    // G1 = (P1 ⊗ 1)−π/4
    pauliProduct = rewriter.getStrArrayAttr(g1.pauliOperators);
    thetaValue = getTheta(loc, g1, rewriter);
    SmallVector<Value> inQubitsValues{G0.getOutQubits()[0]};
    SmallVector<Type> outQubitsTypesList{G0.getOutQubits()[0].getType()};
    auto G1 = rewriter.create<PPRotationOp>(loc, outQubitsTypesList, pauliProduct, thetaValue,
                                            inQubitsValues);

    // G2 = (1 ⊗ P2)−π/4
    pauliProduct = rewriter.getStrArrayAttr(g2.pauliOperators);
    SmallVector<Value> inQubitsValues2{G0.getOutQubits()[1]};
    SmallVector<Type> inQubitsTypesList2{G0.getOutQubits()[1].getType()};
    auto G2 = rewriter.create<PPRotationOp>(loc, inQubitsTypesList2, pauliProduct, thetaValue,
                                            inQubitsValues2);

    rewriter.replaceOp(op, {G1.getOutQubits()[0], G2.getOutQubits()[0]});
    return success();
}

// H = (Z · X · Z)π/4
LogicalResult convertHGate(CustomOp op, ConversionPatternRewriter &rewriter)
{
    auto gate = GateConversion({"Z", "X", "Z"}, 4);
    auto singleQubitOp = singleQubitConversion(op, gate, rewriter);
    rewriter.replaceOp(op, singleQubitOp);
    return success();
}

// S = (Z)π/4
LogicalResult convertSGate(CustomOp op, ConversionPatternRewriter &rewriter)
{
    auto gate = GateConversion({"Z"}, 4);
    auto singleQubitOp = singleQubitConversion(op, gate, rewriter);
    rewriter.replaceOp(op, singleQubitOp);
    return success();
}

// T = (Z)π/8
LogicalResult convertTGate(CustomOp op, ConversionPatternRewriter &rewriter)
{
    auto gate = GateConversion({"Z"}, 8);
    auto singleQubitOp = singleQubitConversion(op, gate, rewriter);
    rewriter.replaceOp(op, singleQubitOp);
    return success();
}

LogicalResult convertCNOTGate(CustomOp op, ConversionPatternRewriter &rewriter)
{
    return multiQubitConversion(op, "Z", "X", rewriter);
}

// Convert a MeasureOp to a PPMeasurementOp
LogicalResult convertMeasureOpToPPM(MeasureOp op, StringRef axis,
                                    ConversionPatternRewriter &rewriter)
{
    auto loc = op.getLoc();

    ArrayAttr pauliProduct = rewriter.getStrArrayAttr({axis});
    auto inQubits = op.getInQubit();
    TypeRange outQubitsTypes = op.getOutQubit().getType();
    Type mresType = op.getMres().getType();

    auto ppmOp =
        rewriter.create<PPMeasurementOp>(loc, mresType, outQubitsTypes, pauliProduct, inQubits);

    rewriter.replaceOp(op, ppmOp.getResults());
    return success();
}

LogicalResult convertMeasureZ(MeasureOp op, ConversionPatternRewriter &rewriter)
{
    return convertMeasureOpToPPM(op, "Z", rewriter);
}

//===----------------------------------------------------------------------===//
//                       QEC Lowering Patterns
//===----------------------------------------------------------------------===//

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
        if (auto originOp = dyn_cast_or_null<CustomOp>(op)) {
            switch (hashGate(originOp)) {
            case GateEnum::H:
                return convertHGate(originOp, rewriter);
            case GateEnum::S:
                return convertSGate(originOp, rewriter);
            case GateEnum::T:
                return convertTGate(originOp, rewriter);
            case GateEnum::CNOT:
                return convertCNOTGate(originOp, rewriter);
            case GateEnum::Unknown: {
                op->emitError("Unsupported gate. Supported gates: H, S, T, CNOT");
                return failure();
            }
            }
        }
        else if (auto originOp = dyn_cast_or_null<MeasureOp>(op)) {
            return convertMeasureZ(originOp, rewriter);
        }
        else {
            op->emitError("Unsupported operation. Supported operations: CustomOp, MeasureOp");
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

void populateCliffordTToPPRPatterns(RewritePatternSet &patterns)
{
    patterns.add<CustomOpLowering>(patterns.getContext());
    patterns.add<MeasureOpLowering>(patterns.getContext());
}

} // namespace qec
} // namespace catalyst
