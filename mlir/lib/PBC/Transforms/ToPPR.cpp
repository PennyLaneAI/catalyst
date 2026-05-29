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

#include "Catalyst/Utils/ConstantResolve.h"
#include "PBC/IR/PBCOps.h"
#include "PBC/Transforms/Patterns.h"
#include "Quantum/IR/QuantumOps.h"

using namespace mlir;
using namespace catalyst;
using namespace catalyst::quantum;
using namespace catalyst::pbc;

namespace {

//===----------------------------------------------------------------------===//
//                       Helper functions
//===----------------------------------------------------------------------===//

enum class GateEnum { H, S, T, CNOT, X, Y, Z, I, RX, RY, RZ, IsingXX, IsingYY, IsingZZ, Unknown };

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
    else if (gateName == "RX")
        return GateEnum::RX;
    else if (gateName == "RY")
        return GateEnum::RY;
    else if (gateName == "RZ")
        return GateEnum::RZ;
    else if (gateName == "IsingXX")
        return GateEnum::IsingXX;
    else if (gateName == "IsingYY")
        return GateEnum::IsingYY;
    else if (gateName == "IsingZZ")
        return GateEnum::IsingZZ;
    else
        return GateEnum::Unknown;
}

// Structure to define gate conversion rules
struct GateConversion {
    SmallVector<StringRef> pauliOperators;
    int8_t rotationKind;
    GateConversion(SmallVector<StringRef> pauliOperators, int8_t rotationKind)
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

void applyGlobalPhase(Location loc, Value phaseValue, ConversionPatternRewriter &rewriter)
{
    //   static GlobalPhaseOp create(::mlir::OpBuilder &builder, ::mlir::Location location,
    //   ::mlir::TypeRange out_ctrl_qubits, ::mlir::Value params, /*optional*/bool adjoint,
    //   ::mlir::ValueRange in_ctrl_qubits, ::mlir::ValueRange in_ctrl_values);

    GlobalPhaseOp::create(rewriter, loc, /*out_ctrl_qubits=*/TypeRange{}, /*params=*/phaseValue,
                          /*adjoint=*/false, /*in_ctrl_qubits*/ ValueRange{},
                          /*in_ctrl_values*/ ValueRange{});
}

void applyGlobalPhase(Location loc, const double phase, ConversionPatternRewriter &rewriter)
{
    Value paramValue = arith::ConstantOp::create(rewriter, loc, rewriter.getF64FloatAttr(phase));
    applyGlobalPhase(loc, paramValue, rewriter);
}

//===----------------------------------------------------------------------===//
//                       Gate conversion functions
//===----------------------------------------------------------------------===//

// C(P) = G(Angle)
void applySingleQubitConversion(CustomOp op, const ArrayRef<GateConversion> &gateConversions,
                                ConversionPatternRewriter &rewriter)
{
    Location loc = op->getLoc();
    ValueRange inQubits = op.getInQubits();
    PPRotationOp pprOp;

    for (auto gateConversion : gateConversions) {
        applyAdjointIfNeeded(gateConversion, op);

        pprOp = PPRotationOp::create(rewriter, loc, gateConversion.pauliOperators,
                                     gateConversion.rotationKind, inQubits);
        inQubits = pprOp.getOutQubits();
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
    auto inQubitsValues = op.getInQubits();

    auto G0 =
        PPRotationOp::create(rewriter, loc, g0.pauliOperators, g0.rotationKind, inQubitsValues);

    // G1 = (P1 ⊗ 1)−π/4
    SmallVector<Value> inQubitsValues1{G0.getOutQubits()[0]};
    SmallVector<Type> outQubitsTypesList1{G0.getOutQubits()[0].getType()};

    auto G1 =
        PPRotationOp::create(rewriter, loc, g1.pauliOperators, g1.rotationKind, inQubitsValues1);

    // G2 = (1 ⊗ P2)−π/4
    SmallVector<Value> inQubitsValues2{G0.getOutQubits()[1]};
    SmallVector<Type> inQubitsTypesList2{G0.getOutQubits()[1].getType()};

    auto G2 =
        PPRotationOp::create(rewriter, loc, g2.pauliOperators, g1.rotationKind, inQubitsValues2);

    rewriter.replaceOp(op, {G1.getOutQubits()[0], G2.getOutQubits()[0]});
    return success();
}

// H = (Z · X · Z)π/4
LogicalResult convertHGate(CustomOp op, ConversionPatternRewriter &rewriter)
{
    applyGlobalPhase(op->getLoc(), -llvm::numbers::pi / 2, rewriter);

    auto Z0 = GateConversion({"Z"}, 4);
    auto X1 = GateConversion({"X"}, 4);
    auto Z2 = GateConversion({"Z"}, 4);
    applySingleQubitConversion(op, {Z0, X1, Z2}, rewriter);
    return success();
}

// S = (Z)π/4
LogicalResult convertSGate(CustomOp op, ConversionPatternRewriter &rewriter)
{
    applyGlobalPhase(op->getLoc(), -llvm::numbers::pi / 4, rewriter);

    auto gate = GateConversion({"Z"}, 4);
    applySingleQubitConversion(op, {gate}, rewriter);
    return success();
}

// T = (Z)π/8
LogicalResult convertTGate(CustomOp op, ConversionPatternRewriter &rewriter)
{
    applyGlobalPhase(op->getLoc(), -llvm::numbers::pi / 8, rewriter);

    auto gate = GateConversion({"Z"}, 8);
    applySingleQubitConversion(op, {gate}, rewriter);
    return success();
}

// X = (X)π/2
LogicalResult convertXGate(CustomOp op, ConversionPatternRewriter &rewriter)
{
    applyGlobalPhase(op->getLoc(), -llvm::numbers::pi / 2, rewriter);

    auto gate = GateConversion({"X"}, 2);
    applySingleQubitConversion(op, {gate}, rewriter);
    return success();
}

// Y = (Y)π/2
LogicalResult convertYGate(CustomOp op, ConversionPatternRewriter &rewriter)
{
    applyGlobalPhase(op->getLoc(), -llvm::numbers::pi / 2, rewriter);

    auto gate = GateConversion({"Y"}, 2);
    applySingleQubitConversion(op, {gate}, rewriter);
    return success();
}

// Z = (Z)π/2
LogicalResult convertZGate(CustomOp op, ConversionPatternRewriter &rewriter)
{
    applyGlobalPhase(op->getLoc(), -llvm::numbers::pi / 2, rewriter);

    auto gate = GateConversion({"Z"}, 2);
    applySingleQubitConversion(op, {gate}, rewriter);
    return success();
}

// I = I
LogicalResult convertIGate(CustomOp op, ConversionPatternRewriter &rewriter)
{
    auto gate = GateConversion({"I"}, 1);
    applySingleQubitConversion(op, {gate}, rewriter);
    return success();
}

LogicalResult convertCNOTGate(CustomOp op, ConversionPatternRewriter &rewriter)
{
    applyGlobalPhase(op->getLoc(), llvm::numbers::pi / 4, rewriter);
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

    auto ppmOp = PPMeasurementOp::create(rewriter, loc, mresType, outQubitTypes, pauliProduct,
                                         nullptr, inQubits);

    rewriter.replaceOp(op, ppmOp);
    return success();
}

LogicalResult convertRotationLikeGate(Operation *op, Value angleValue, ArrayAttr pauliProduct,
                                      ValueRange inQubits, bool isAdjoint,
                                      ConversionPatternRewriter &rewriter)
{
    Location loc = op->getLoc();
    SmallVector<Type> outQubitTypes{inQubits.size(), QubitType::get(rewriter.getContext())};

    std::optional<double> angleOpt = resolveConstant(angleValue);
    if (angleOpt.has_value()) {
        constexpr double PI = llvm::numbers::pi;
        constexpr double SPECIFIC_ANGLES[6] = {PI / 2, PI / 4, PI / 8, -PI / 8, -PI / 4, -PI / 2};
        constexpr int8_t SPECIFIC_DENOMINATORS[6] = {2, 4, 8, -8, -4, -2};
        // We are choosing a very small tolerance to accommodate floating point precision issues.
        // We choose this because it is a few bits away from the precision allowed by float 64
        // and we assume the angles have magnitudes on the order of pi.
        constexpr double TOLERANCE = 1e-12;

        double pprAngle = angleOpt.value() / 2;
        double angle = std::fmod(pprAngle, PI);

        if (std::abs(angle) < TOLERANCE || PI - std::abs(angle) < TOLERANCE) {
            // If the angle is 0 or pi, we can just erase the operation.
            rewriter.replaceOp(op, inQubits);
            return success();
        }

        for (auto [i, specificAngle] : llvm::enumerate(SPECIFIC_ANGLES)) {
            if (std::abs(angle - specificAngle) < TOLERANCE) {
                int8_t rotationKind = SPECIFIC_DENOMINATORS[i];
                if (isAdjoint) {
                    rotationKind = -rotationKind;
                }

                auto pprOp =
                    PPRotationOp::create(rewriter, loc, pauliProduct, rotationKind, inQubits);

                rewriter.replaceOp(op, pprOp.getOutQubits());
                return success();
            }
        }
    }

    // Angle is not static or not a multiple of π/8, consider this as an arbitrary angle PPR.
    Value denominator =
        arith::ConstantOp::create(rewriter, loc, rewriter.getF64FloatAttr(isAdjoint ? -2.0 : 2.0));
    auto arbitraryAngle = arith::DivFOp::create(rewriter, loc, angleValue, denominator).getResult();

    auto pprArbitraryOp = PPRotationArbitraryOp::create(rewriter, loc, outQubitTypes, pauliProduct,
                                                        arbitraryAngle, inQubits);

    rewriter.replaceOp(op, pprArbitraryOp.getOutQubits());
    return success();
}

FailureOr<Value> getSingleRotationParameter(CustomOp op)
{
    if (op.getParams().size() != 1) {
        return op->emitOpError("expected exactly one parameter on " + op.getGateName());
    }
    return op.getParams().front();
}

LogicalResult convertRXGate(CustomOp op, ConversionPatternRewriter &rewriter)
{
    auto angleOrError = getSingleRotationParameter(op);
    if (failed(angleOrError)) {
        return failure();
    }
    auto pauliProduct = rewriter.getStrArrayAttr({"X"});
    return convertRotationLikeGate(op, *angleOrError, pauliProduct, op.getInQubits(),
                                   op.getAdjoint(), rewriter);
}

LogicalResult convertRYGate(CustomOp op, ConversionPatternRewriter &rewriter)
{
    auto angleOrError = getSingleRotationParameter(op);
    if (failed(angleOrError)) {
        return failure();
    }
    auto pauliProduct = rewriter.getStrArrayAttr({"Y"});
    return convertRotationLikeGate(op, *angleOrError, pauliProduct, op.getInQubits(),
                                   op.getAdjoint(), rewriter);
}

LogicalResult convertRZGate(CustomOp op, ConversionPatternRewriter &rewriter)
{
    auto angleOrError = getSingleRotationParameter(op);
    if (failed(angleOrError)) {
        return failure();
    }
    auto pauliProduct = rewriter.getStrArrayAttr({"Z"});
    return convertRotationLikeGate(op, *angleOrError, pauliProduct, op.getInQubits(),
                                   op.getAdjoint(), rewriter);
}

LogicalResult convertIsingXXGate(CustomOp op, ConversionPatternRewriter &rewriter)
{
    auto angleOrError = getSingleRotationParameter(op);
    if (failed(angleOrError)) {
        return failure();
    }
    auto pauliProduct = rewriter.getStrArrayAttr({"X", "X"});
    return convertRotationLikeGate(op, *angleOrError, pauliProduct, op.getInQubits(),
                                   op.getAdjoint(), rewriter);
}

LogicalResult convertIsingYYGate(CustomOp op, ConversionPatternRewriter &rewriter)
{
    auto angleOrError = getSingleRotationParameter(op);
    if (failed(angleOrError)) {
        return failure();
    }
    auto pauliProduct = rewriter.getStrArrayAttr({"Y", "Y"});
    return convertRotationLikeGate(op, *angleOrError, pauliProduct, op.getInQubits(),
                                   op.getAdjoint(), rewriter);
}

LogicalResult convertIsingZZGate(CustomOp op, ConversionPatternRewriter &rewriter)
{
    auto angleOrError = getSingleRotationParameter(op);
    if (failed(angleOrError)) {
        return failure();
    }
    auto pauliProduct = rewriter.getStrArrayAttr({"Z", "Z"});
    return convertRotationLikeGate(op, *angleOrError, pauliProduct, op.getInQubits(),
                                   op.getAdjoint(), rewriter);
}

LogicalResult convertMultiRZGate(MultiRZOp op, ConversionPatternRewriter &rewriter)
{
    SmallVector<Attribute> pauliVector(op.getInQubits().size(), rewriter.getStringAttr("Z"));
    auto pauliProduct = rewriter.getArrayAttr(pauliVector);
    return convertRotationLikeGate(op, op.getTheta(), pauliProduct, op.getInQubits(),
                                   op.getAdjoint(), rewriter);
}

LogicalResult convertPauliRotGate(PauliRotOp op, ConversionPatternRewriter &rewriter)
{
    return convertRotationLikeGate(op, op.getAngle(), op.getPauliProduct(), op.getInQubits(),
                                   op.getAdjoint(), rewriter);
}

LogicalResult convertMeasureZ(MeasureOp op, ConversionPatternRewriter &rewriter)
{
    return convertMeasureOpToPPM(op, "Z", rewriter);
}

//===----------------------------------------------------------------------===//
//                       PBC Lowering Patterns
//===----------------------------------------------------------------------===//

template <typename OriginOp, typename LoweredPBCOp>
struct PBCOpLowering : public ConversionPattern {
    PBCOpLowering(MLIRContext *context)
        : ConversionPattern(OriginOp::getOperationName(), 1, context)
    {
    }

    LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const final
    {
        if (auto originOp = dyn_cast<CustomOp>(op)) {
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
            case GateEnum::RX:
                return convertRXGate(originOp, rewriter);
            case GateEnum::RY:
                return convertRYGate(originOp, rewriter);
            case GateEnum::RZ:
                return convertRZGate(originOp, rewriter);
            case GateEnum::IsingXX:
                return convertIsingXXGate(originOp, rewriter);
            case GateEnum::IsingYY:
                return convertIsingYYGate(originOp, rewriter);
            case GateEnum::IsingZZ:
                return convertIsingZZGate(originOp, rewriter);
            case GateEnum::Unknown: {
                op->emitError(
                    "Unsupported gate. Supported gates: H, S, T, X, Y, Z, S†, T†, I, CNOT, "
                    "RX, RY, RZ, IsingXX, IsingYY, IsingZZ, MultiRZ, and PauliRot.");
                return failure();
            }
            }
        }
        else if (auto originOp = dyn_cast<MultiRZOp>(op)) {
            return convertMultiRZGate(originOp, rewriter);
        }
        else if (auto originOp = dyn_cast<PauliRotOp>(op)) {
            return convertPauliRotGate(originOp, rewriter);
        }
        else if (auto originOp = dyn_cast<MeasureOp>(op)) {
            return convertMeasureZ(originOp, rewriter);
        }
        op->emitError("Unsupported operation. Supported operations: CustomOp, MeasureOp");
        return failure();
    }
};

using CustomOpLowering = PBCOpLowering<quantum::CustomOp, pbc::PPRotationOp>;
using MultiRZOpLowering = PBCOpLowering<quantum::MultiRZOp, pbc::PPRotationOp>;
using PauliRotOpLowering = PBCOpLowering<quantum::PauliRotOp, pbc::PPRotationOp>;
using MeasureOpLowering = PBCOpLowering<quantum::MeasureOp, pbc::PPMeasurementOp>;

} // namespace

namespace catalyst {
namespace pbc {

void populateToPPRPatterns(RewritePatternSet &patterns)
{
    patterns.add<CustomOpLowering>(patterns.getContext());
    patterns.add<MultiRZOpLowering>(patterns.getContext());
    patterns.add<PauliRotOpLowering>(patterns.getContext());
    patterns.add<MeasureOpLowering>(patterns.getContext());
}

} // namespace pbc
} // namespace catalyst
