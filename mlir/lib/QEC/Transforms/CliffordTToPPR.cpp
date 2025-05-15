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

enum class GateEnum { H, S, T, CNOT, X, Y, Z, Unknown };

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
    else if (gateName == "PauliX" || gateName == "X")
        return GateEnum::X;
    else if (gateName == "PauliY" || gateName == "Y")
        return GateEnum::Y;
    else if (gateName == "PauliZ" || gateName == "Z")
        return GateEnum::Z;
    else
        return GateEnum::Unknown;
}

// Structure to define gate conversion rules
struct GateConversion {
    SmallVector<StringRef> pauliOperators;
    int64_t rotationKind;
    GateConversion(SmallVector<StringRef> pauliOperators, int64_t rotationKind)
        : pauliOperators(pauliOperators), rotationKind(rotationKind)
    {
    }
    GateConversion() : pauliOperators(), rotationKind(0) {}
};

// Apply adjoint transformation to a gate conversion
// If adjoint attribute is true, invert the sign of rotationKind
void applyAdjointIfNeeded(GateConversion &gateConversion, CustomOp op)
{
    if (op.getAdjoint()) {
        gateConversion.rotationKind = -gateConversion.rotationKind;
    }
}

//===----------------------------------------------------------------------===//
//                       Gate conversion functions
//===----------------------------------------------------------------------===//

// C(P) = G(Angle)
void applySingleQubitConversion(CustomOp op, SmallVector<GateConversion> gateConversions,
                                ConversionPatternRewriter &rewriter)
{
    auto loc = op->getLoc();
    auto types = op.getOutQubits().getType();
    SmallVector<Value> inQubits = op.getInQubits();
    PPRotationOp pprOp;

    for (auto gateConversion : gateConversions) {
        applyAdjointIfNeeded(gateConversion, op);

        auto pauliProduct = rewriter.getStrArrayAttr(gateConversion.pauliOperators);
        pprOp = rewriter.create<PPRotationOp>(loc, types, pauliProduct, gateConversion.rotationKind,
                                              inQubits);
        inQubits = pprOp.getOutQubits();
        types = pprOp.getOutQubits().getType();
    }

    rewriter.replaceOp(op, pprOp.getOutQubits());
}

// Ref: Fig. 5 in [A Game of Surface Codes](https://doi.org/10.22331/q-2019-03-05-128)
// C(P1, P2) = G0 · G1 · G2
// G0 = (P1 ⊗ P2)π/4
// G1 = (P1 ⊗ 1)−π/4
// G2 = (1 ⊗ P2)−π/4
LogicalResult controlledConversion(CustomOp op, StringRef P1, StringRef P2,
                                   ConversionPatternRewriter &rewriter)
{
    auto loc = op->getLoc();

    auto g0 = GateConversion({P1, P2}, 4);
    auto g1 = GateConversion({P1}, -4);
    auto g2 = GateConversion({P2}, -4);

    applyAdjointIfNeeded(g0, op);
    applyAdjointIfNeeded(g1, op);
    applyAdjointIfNeeded(g2, op);

    rewriter.setInsertionPoint(op);

    // G0 = (P1 ⊗ P2)π/4
    auto pauliProduct = rewriter.getStrArrayAttr(g0.pauliOperators);
    auto inQubitsValues = op.getInQubits();
    auto outQubitsTypesList = op.getOutQubits().getType();

    auto G0 = rewriter.create<PPRotationOp>(loc, outQubitsTypesList, pauliProduct, g0.rotationKind,
                                            inQubitsValues);

    // G1 = (P1 ⊗ 1)−π/4
    pauliProduct = rewriter.getStrArrayAttr(g1.pauliOperators);
    SmallVector<Value> inQubitsValues1{G0.getOutQubits()[0]};
    SmallVector<Type> outQubitsTypesList1{G0.getOutQubits()[0].getType()};

    auto G1 = rewriter.create<PPRotationOp>(loc, outQubitsTypesList1, pauliProduct, g1.rotationKind,
                                            inQubitsValues1);

    // G2 = (1 ⊗ P2)−π/4
    pauliProduct = rewriter.getStrArrayAttr(g2.pauliOperators);
    SmallVector<Value> inQubitsValues2{G0.getOutQubits()[1]};
    SmallVector<Type> inQubitsTypesList2{G0.getOutQubits()[1].getType()};

    auto G2 = rewriter.create<PPRotationOp>(loc, inQubitsTypesList2, pauliProduct, g1.rotationKind,
                                            inQubitsValues2);

    rewriter.replaceOp(op, {G1.getOutQubits()[0], G2.getOutQubits()[0]});
    return success();
}

// H = (Z · X · Z)π/4
LogicalResult convertHGate(CustomOp op, ConversionPatternRewriter &rewriter)
{
    auto Z0 = GateConversion({"Z"}, 4);
    auto X1 = GateConversion({"X"}, 4);
    auto Z2 = GateConversion({"Z"}, 4);

    applySingleQubitConversion(op, {Z0, X1, Z2}, rewriter);
    return success();
}

// S = (Z)π/4
LogicalResult convertSGate(CustomOp op, ConversionPatternRewriter &rewriter)
{
    auto gate = GateConversion({"Z"}, 4);
    applySingleQubitConversion(op, {gate}, rewriter);
    return success();
}

// T = (Z)π/8
LogicalResult convertTGate(CustomOp op, ConversionPatternRewriter &rewriter)
{
    auto gate = GateConversion({"Z"}, 8);
    applySingleQubitConversion(op, {gate}, rewriter);
    return success();
}

// X = (X)π/2
LogicalResult convertXGate(CustomOp op, ConversionPatternRewriter &rewriter)
{
    auto gate = GateConversion({"X"}, 2);
    applySingleQubitConversion(op, {gate}, rewriter);
    return success();
}

// Y = (Y)π/2
LogicalResult convertYGate(CustomOp op, ConversionPatternRewriter &rewriter)
{
    auto gate = GateConversion({"Y"}, 2);
    applySingleQubitConversion(op, {gate}, rewriter);
    return success();
}

// Z = (Z)π/2
LogicalResult convertZGate(CustomOp op, ConversionPatternRewriter &rewriter)
{
    auto gate = GateConversion({"Z"}, 2);
    applySingleQubitConversion(op, {gate}, rewriter);
    return success();
}

LogicalResult convertCNOTGate(CustomOp op, ConversionPatternRewriter &rewriter)
{
    return controlledConversion(op, "Z", "X", rewriter);
}

// Convert a MeasureOp to a PPMeasurementOp
LogicalResult convertMeasureOpToPPM(MeasureOp op, StringRef axis,
                                    ConversionPatternRewriter &rewriter)
{
    auto loc = op.getLoc();

    ArrayAttr pauliProduct = rewriter.getStrArrayAttr({axis});
    auto inQubits = op.getInQubit();

    Type qubitType = op.getOutQubit().getType();
    Type mresType = op.getMres().getType();
    SmallVector<Type> outQubitTypes({qubitType});

    auto ppmOp = rewriter.create<PPMeasurementOp>(loc, mresType, outQubitTypes, pauliProduct,
                                                  nullptr, inQubits);

    rewriter.replaceOp(op, ppmOp);
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
        // cast to OriginOp
        if (auto originOp = dyn_cast_or_null<CustomOp>(op)) {
            switch (hashGate(originOp)) {
            case GateEnum::H:
                return convertHGate(originOp, rewriter);
            case GateEnum::S:
                return convertSGate(originOp, rewriter);
            case GateEnum::T:
                return convertTGate(originOp, rewriter);
            case GateEnum::X:
                return convertXGate(originOp, rewriter);
            case GateEnum::Y:
                return convertYGate(originOp, rewriter);
            case GateEnum::Z:
                return convertZGate(originOp, rewriter);
            case GateEnum::CNOT:
                return convertCNOTGate(originOp, rewriter);
            case GateEnum::Unknown: {
                op->emitError(
                    "Unsupported gate. Supported gates: H, S, T, X, Y, Z, S†, T†, and CNOT");
                return failure();
            }
            }
        }
        else if (auto originOp = dyn_cast_or_null<MeasureOp>(op)) {
            return convertMeasureZ(originOp, rewriter);
        }
        op->emitError("Unsupported operation. Supported operations: CustomOp, MeasureOp");
        return failure();
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
