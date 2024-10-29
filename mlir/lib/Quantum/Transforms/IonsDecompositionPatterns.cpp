// Copyright 2024 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#define DEBUG_TYPE "ions-decomposition"

#include "Quantum/IR/QuantumOps.h"
#include "Quantum/Transforms/Patterns.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Errc.h"

using llvm::dbgs;
using namespace mlir;
using namespace catalyst::quantum;

constexpr double PI = llvm::numbers::pi;

// Define map, name to function creating decomp

void oneQubitDecomp(catalyst::quantum::CustomOp op, mlir::PatternRewriter &rewriter, double phi,
                    std::variant<mlir::Value, double> theta, double lambda)
{
    TypeRange outQubitsTypes = op.getOutQubits().getTypes();
    TypeRange outQubitsCtrlTypes = op.getOutCtrlQubits().getTypes();

    ValueRange inQubits = op.getInQubits();
    ValueRange inCtrlQubits = op.getInCtrlQubits();
    ValueRange inCtrlValues = op.getInCtrlValues();

    TypedAttr phiAttr = rewriter.getF64FloatAttr(phi);
    mlir::Value phiValue = rewriter.create<arith::ConstantOp>(op.getLoc(), phiAttr);

    mlir::Value thetaValue;
    if (std::holds_alternative<mlir::Value>(theta)) {
        thetaValue = std::get<mlir::Value>(theta);
    }
    else if (std::holds_alternative<double>(theta)) {
        TypedAttr thetaAttr = rewriter.getF64FloatAttr(std::get<double>(theta));
        thetaValue = rewriter.create<arith::ConstantOp>(op.getLoc(), thetaAttr);
    }
    TypedAttr lambdaAttr = rewriter.getF64FloatAttr(lambda);
    mlir::Value lambdaValue = rewriter.create<arith::ConstantOp>(op.getLoc(), lambdaAttr);

    auto rxPhi =
        rewriter.create<CustomOp>(op.getLoc(), outQubitsTypes, outQubitsCtrlTypes, phiValue,
                                  inQubits, "RX", nullptr, inCtrlQubits, inCtrlValues);
    auto ryTheta = rewriter.create<CustomOp>(op.getLoc(), outQubitsTypes, outQubitsCtrlTypes,
                                             thetaValue, rxPhi.getOutQubits(), "RY", nullptr,
                                             rxPhi.getInCtrlQubits(), rxPhi.getInCtrlValues());
    auto rxLambda = rewriter.create<CustomOp>(op.getLoc(), outQubitsTypes, outQubitsCtrlTypes,
                                              lambdaValue, ryTheta.getOutQubits(), "RX", nullptr,
                                              inCtrlQubits, inCtrlValues);
    op.replaceAllUsesWith(rxLambda);
}

void tDecomp(catalyst::quantum::CustomOp op, mlir::PatternRewriter &rewriter)
{
    oneQubitDecomp(op, rewriter, -PI / 2, PI / 4, PI / 2);
}

void sDecomp(catalyst::quantum::CustomOp op, mlir::PatternRewriter &rewriter)
{
    oneQubitDecomp(op, rewriter, -PI / 2, PI / 2, PI / 2);
}

void zDecomp(catalyst::quantum::CustomOp op, mlir::PatternRewriter &rewriter)
{
    oneQubitDecomp(op, rewriter, -PI / 2, PI, PI / 2);
}

void hDecomp(catalyst::quantum::CustomOp op, mlir::PatternRewriter &rewriter)
{
    oneQubitDecomp(op, rewriter, PI / 2, PI, PI / 2);
}

void psDecomp(catalyst::quantum::CustomOp op, mlir::PatternRewriter &rewriter)
{
    oneQubitDecomp(op, rewriter, -PI / 2, op.getParams().front(), PI / 2);
}

void rzDecomp(catalyst::quantum::CustomOp op, mlir::PatternRewriter &rewriter)
{
    oneQubitDecomp(op, rewriter, -PI / 2, op.getParams().front(), PI / 2);
}

void cnotDecomp(catalyst::quantum::CustomOp op, mlir::PatternRewriter &rewriter)
{
    TypeRange outQubitsTypes = op.getOutQubits().getTypes();

    mlir::Value inQubit0 = op.getInQubits().front();
    mlir::Value inQubit1 = op.getInQubits().back();

    TypedAttr piOver2Attr = rewriter.getF64FloatAttr(PI / 2);
    mlir::Value piOver2 = rewriter.create<arith::ConstantOp>(op.getLoc(), piOver2Attr);
    auto ryPiOver2 =
        rewriter.create<CustomOp>(op.getLoc(), outQubitsTypes.front(), ValueRange{}, piOver2,
                                  inQubit0, "RY", nullptr, ValueRange{}, ValueRange{});
    SmallVector<mlir::Value> qubitsAfterRy;
    qubitsAfterRy.push_back(ryPiOver2.getOutQubits().front());
    qubitsAfterRy.push_back(inQubit1);
    auto ms = rewriter.create<CustomOp>(op.getLoc(), outQubitsTypes, ValueRange{}, piOver2,
                                        qubitsAfterRy, "MS", nullptr, ValueRange{}, ValueRange{});
    mlir::Value qubit0AfterMs = ms.getOutQubits().front();
    mlir::Value qubit1AfterMs = ms.getOutQubits().back();

    TypedAttr minusPiOver2Attr = rewriter.getF64FloatAttr(-PI / 2);
    mlir::Value minusPiOver2 = rewriter.create<arith::ConstantOp>(op.getLoc(), minusPiOver2Attr);
    auto rxMinusPiOver2 =
        rewriter.create<CustomOp>(op.getLoc(), outQubitsTypes.front(), ValueRange{}, minusPiOver2,
                                  qubit0AfterMs, "RX", nullptr, ValueRange{}, ValueRange{});
    auto firstRyMinusPiOver2 =
        rewriter.create<CustomOp>(op.getLoc(), outQubitsTypes.front(), ValueRange{}, minusPiOver2,
                                  qubit1AfterMs, "RY", nullptr, ValueRange{}, ValueRange{});

    mlir::Value qubit0AfterRY = rxMinusPiOver2.getOutQubits().front();
    auto secondRyMinusPiOver2 =
        rewriter.create<CustomOp>(op.getLoc(), outQubitsTypes.front(), ValueRange{}, minusPiOver2,
                                  qubit0AfterRY, "RY", nullptr, ValueRange{}, ValueRange{});

    SmallVector<mlir::Value> qubitsEnd;
    qubitsEnd.push_back(firstRyMinusPiOver2.getOutQubits().front());
    qubitsEnd.push_back(secondRyMinusPiOver2.getOutQubits().front());
    op.replaceAllUsesWith(qubitsEnd);
}

std::map<std::string, std::function<void(catalyst::quantum::CustomOp, mlir::PatternRewriter &)>>
    funcMap = {{"T", &tDecomp},        {"S", &sDecomp},   {"Z", &zDecomp},
               {"Hadamard", &hDecomp}, {"RZ", &rzDecomp}, {"PhaseShift", &psDecomp},
               {"CNOT", &cnotDecomp}};

namespace {

struct IonsDecompositionRewritePattern : public mlir::OpRewritePattern<CustomOp> {
    using mlir::OpRewritePattern<CustomOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(CustomOp op, mlir::PatternRewriter &rewriter) const override
    {
        auto it = funcMap.find(op.getGateName().str());
        if (it != funcMap.end()) {
            auto decompFunc = it->second;
            decompFunc(op, rewriter);
            return success();
        }
        else {
            return failure();
        }
    }
};
} // namespace

namespace catalyst {
namespace quantum {

void populateIonsDecompositionPatterns(RewritePatternSet &patterns)
{
    patterns.add<IonsDecompositionRewritePattern>(patterns.getContext(), 1);
}

} // namespace quantum
} // namespace catalyst
