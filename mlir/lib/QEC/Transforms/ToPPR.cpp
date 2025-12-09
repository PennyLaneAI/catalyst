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
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/TypeRange.h"
#include "stablehlo/dialect/StablehloOps.h"

#include "QEC/IR/QECOps.h"
#include "QEC/Transforms/Patterns.h"
#include "Quantum/IR/QuantumOps.h"
#include <cmath>
#include <llvm/Support/Debug.h>
#include <llvm/Support/LogicalResult.h>
#include <optional>

#define DEBUG_TYPE "to-ppr"

using namespace mlir;
using namespace catalyst;
using namespace catalyst::quantum;
using namespace catalyst::qec;

namespace {

//===----------------------------------------------------------------------===//
//                       Helper functions
//===----------------------------------------------------------------------===//

enum class GateEnum { H, S, T, CNOT, X, Y, Z, I, Unknown };

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
    else if (gateName == "Identity" || gateName == "I")
        return GateEnum::I;
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
void applySingleQubitConversion(CustomOp op, const ArrayRef<GateConversion> &gateConversions,
                                ConversionPatternRewriter &rewriter)
{
    auto loc = op->getLoc();
    auto types = op.getOutQubits().getType();
    ValueRange inQubits = op.getInQubits();
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

// I = I
LogicalResult convertIGate(CustomOp op, ConversionPatternRewriter &rewriter)
{
    auto gate = GateConversion({"I"}, 0);
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

// Recursively resolve the constant parameter of a value and returns std::nullopt if not a constant.
std::optional<double> resolveConstantValue(Value value)
{
    if (!value)
        return std::nullopt;

    auto *defOp = value.getDefiningOp();
    if (!defOp)
        return std::nullopt;

    // Handle Tensor Dialect
    if (auto extractOp = dyn_cast<tensor::ExtractOp>(defOp)) {
        return resolveConstantValue(extractOp.getTensor());
    }

    // Handle Stablehlo Dialect
    if (auto constOp = dyn_cast<stablehlo::ConstantOp>(defOp)) {
        auto valueAttr = constOp.getValue();
        if (auto denseFPAttr = dyn_cast<DenseFPElementsAttr>(valueAttr)) {
            if (denseFPAttr.isSplat() || denseFPAttr.getNumElements() == 1) {
                return denseFPAttr.getSplatValue<APFloat>().convertToDouble();
            }
        }
        else if (auto denseIntAttr = dyn_cast<DenseIntElementsAttr>(valueAttr)) {
            if (denseIntAttr.isSplat() || denseIntAttr.getNumElements() == 1) {
                return static_cast<double>(denseIntAttr.getSplatValue<APInt>().getSExtValue());
            }
        }
        return std::nullopt;
    }
    else if (auto convertOp = dyn_cast<stablehlo::ConvertOp>(defOp)) {
        if (convertOp->getNumOperands() > 0) {
            return resolveConstantValue(convertOp.getOperand());
        }
        return std::nullopt;
    }
    else if (auto broadcastInDimOp = dyn_cast<stablehlo::BroadcastInDimOp>(defOp)) {
        if (broadcastInDimOp->getNumOperands() > 0) {
            return resolveConstantValue(broadcastInDimOp.getOperand());
        }
        return std::nullopt;
    }

    // Handle Arith Dialect
    if (auto constOp = dyn_cast<arith::ConstantOp>(defOp)) {
        auto valueAttr = constOp.getValue();
        if (auto floatAttr = dyn_cast<FloatAttr>(valueAttr)) {
            return floatAttr.getValueAsDouble();
        }
        // Handle integer constants (convert to double)
        if (auto intAttr = dyn_cast<IntegerAttr>(valueAttr)) {
            return static_cast<double>(intAttr.getValue().getSExtValue());
        }
        // Handle DenseElementsAttr for rank-0 tensors
        if (auto denseAttr = dyn_cast<DenseFPElementsAttr>(valueAttr)) {
            if (denseAttr.isSplat() || denseAttr.getNumElements() == 1) {
                return denseAttr.getSplatValue<APFloat>().convertToDouble();
            }
        }
        if (auto denseAttr = dyn_cast<DenseIntElementsAttr>(valueAttr)) {
            if (denseAttr.isSplat() || denseAttr.getNumElements() == 1) {
                return static_cast<double>(denseAttr.getSplatValue<APInt>().getSExtValue());
            }
        }
        return std::nullopt;
    }
    else if (auto indexCastOp = dyn_cast<arith::IndexCastOp>(defOp)) {
        if (defOp->getNumOperands() > 0) {
            return resolveConstantValue(defOp->getOperand(0));
        }
        return std::nullopt;
    }
    else if (auto addOp = dyn_cast<arith::AddFOp>(defOp)) {
        double sum = 0.0;
        for (auto operand : addOp.getOperands()) {
            auto operandVal = resolveConstantValue(operand);
            if (!operandVal.has_value())
                return std::nullopt;
            sum += operandVal.value();
        }
        return sum;
    }
    return std::nullopt;
}

LogicalResult convertPauliRotGate(PauliRotOp op, ConversionPatternRewriter &rewriter)
{
    auto loc = op.getLoc();

    auto angleValue = op.getAngle();
    auto pauliProduct = op.getPauliProduct();
    auto inQubits = op.getInQubits();
    auto outQubitTypes = op.getOutQubits().getType();

    auto angleOpt = resolveConstantValue(angleValue);

    if (angleOpt.has_value()) {
        constexpr double PI = llvm::numbers::pi;
        constexpr double PPR_ANGLES[4] = {0, PI / 2, PI / 4, PI / 8};
        constexpr double TOLERANCE = 1e-9;

        double angle = angleOpt.value();
        angle = std::fmod(angle, PI);

        for (auto ppr_angle : PPR_ANGLES) {
            if (std::abs(angle - ppr_angle) < TOLERANCE) {
                auto rotationKind = static_cast<int64_t>(PI / angle);
                if (op.getAdjoint()) {
                    rotationKind = -rotationKind;
                }
                auto rotationKindAttr =
                    rewriter.getI16IntegerAttr(static_cast<int16_t>(rotationKind));
                auto pprOp = rewriter.create<PPRotationOp>(loc, outQubitTypes, pauliProduct,
                                                           rotationKindAttr, inQubits);
                rewriter.replaceOp(op, pprOp.getOutQubits());
                return success();
            }
        }
    }

    // Angle is not static or not a multiple of π/8, consider this as an arbitrary angle PPR.
    auto pprArbitraryOp = rewriter.create<PPRotationArbitraryOp>(loc, outQubitTypes, pauliProduct,
                                                                 angleValue, inQubits);
    rewriter.replaceOp(op, pprArbitraryOp.getOutQubits());

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
            case GateEnum::I:
                return convertIGate(originOp, rewriter);
            case GateEnum::Unknown: {
                op->emitError(
                    "Unsupported gate. Supported gates: H, S, T, X, Y, Z, S†, T†, I, and CNOT");
                return failure();
            }
            }
        }
        else if (auto originOp = dyn_cast_or_null<PauliRotOp>(op)) {
            return convertPauliRotGate(originOp, rewriter);
        }
        else if (auto originOp = dyn_cast_or_null<MeasureOp>(op)) {
            return convertMeasureZ(originOp, rewriter);
        }
        op->emitError("Unsupported operation. Supported operations: CustomOp, MeasureOp");
        return failure();
    }
};

using CustomOpLowering = QECOpLowering<quantum::CustomOp, qec::PPRotationOp>;
using PauliRotOpLowering = QECOpLowering<quantum::PauliRotOp, qec::PPRotationOp>;
using MeasureOpLowering = QECOpLowering<quantum::MeasureOp, qec::PPMeasurementOp>;

} // namespace

namespace catalyst {
namespace qec {

void populateToPPRPatterns(RewritePatternSet &patterns)
{
    patterns.add<CustomOpLowering>(patterns.getContext());
    patterns.add<PauliRotOpLowering>(patterns.getContext());
    patterns.add<MeasureOpLowering>(patterns.getContext());
}

} // namespace qec
} // namespace catalyst
