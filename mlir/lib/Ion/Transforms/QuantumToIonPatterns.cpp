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

#include "Ion/IR/IonOps.h"
#include "Quantum/IR/QuantumOps.h"
#include "mlir/IR/PatternMatch.h"

#include "Ion/Transforms/Patterns.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include <iostream>

using namespace mlir;
using namespace catalyst::ion;
using namespace catalyst::quantum;

namespace catalyst {
namespace ion {

struct QuantumToIonRewritePattern : public mlir::OpRewritePattern<CustomOp> {
    using mlir::OpRewritePattern<CustomOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(CustomOp op, mlir::PatternRewriter &rewriter) const override
    {
        // TODO: Assumption 1 Ion are in the same funcop as the operations
        // RX case -> PP(P1, P2)
        if (op.getGateName() == "RX") {
            auto qnode = op->getParentOfType<func::FuncOp>();
            ion::IonOp ion;
            qnode.walk([&](ion::IonOp op) {
                ion = op;
                return WalkResult::interrupt();
            });
            // std::cout << "::::::::::::::::::::::HELLO" << std::endl;
            // // Get the qubit number
            // auto inQubit = op.getInQubits().front();
            // auto extract = inQubit.getParentOfType<quantum::ExtractOp>();
            // std::cout << "::::::::::::::::::::::HELLO" << std::endl;
            // extract.dump();
            // std::cout << "::::::::::::::::::::::HELLO" << std::endl;
            // std::cout << index.value() << std::endl;
            
            // auto loc = op.getLoc();
            // auto qubits = op.getOutQubits();
            // auto ctx = op.getContext();

            // Value ppOp = rewriter
            //                  .create<ion::ParallelProtocolOp>(
            //                      loc, qubits,
            //                      [&](OpBuilder &builder, Location loc, ValueRange qubits) {
            //                          //  BeamAttr::get(ctx, transition0, rabi, detuning, phase);
            //                          //  rewriter.create<ion::PulseOp>(loc, qubits.front());
            //                          //  rewriter.create<ion::PulseOp>(loc, qubits.front());
            //                          builder.create<ion::YieldOp>(loc, qubits);
            //                      })
            //                  .getResult(0);
            // ppOp.dump();
            // Create PP
            // Create P1
            // Create P2
            // PP replace RXF
            return failure();
        }
        // RY case -> PP(P1, P2)
        else if (op.getGateName() == "RY") {
            return failure();
        }
        // MS case -> PP(P1, P2, P3, P4, P5, P6)
        else if (op.getGateName() == "MS") {
            return failure();
        }
        // Else fail
        return failure();
    }
};

void populateQuantumToIonPatterns(RewritePatternSet &patterns)
{
    patterns.add<QuantumToIonRewritePattern>(patterns.getContext());
}

} // namespace ion
} // namespace catalyst
