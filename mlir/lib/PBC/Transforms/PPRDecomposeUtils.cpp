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

#include "PBC/Transforms/PPRDecomposeUtils.h"

#include "llvm/Support/MathExtras.h"
#include "mlir/IR/IRMapping.h"

#include "PBC/IR/PBCOps.h"         // for FabricateOp, GadgetOp
#include "Quantum/IR/QuantumOps.h" // for quantum::AllocQubitOp

namespace catalyst {
namespace pbc {

std::pair<mlir::StringRef, bool> determinePauliAndSignOfMeasurement(bool avoidPauliYMeasure) {
    if (avoidPauliYMeasure) {
        return std::make_pair("Z", 0);
    }
    return std::make_pair("Y", 1);
}

mlir::OpResult initializeZeroOrPlusI(bool avoidPauliYMeasure, mlir::Location loc,
                                     mlir::PatternRewriter &rewriter) {
    if (avoidPauliYMeasure) {
        // Fabricate |Y⟩
        auto plusIOp = FabricateOp::create(rewriter, loc, LogicalInitKind::plus_i);
        return plusIOp.getOutQubits().back();
    }

    // Initialize |0⟩
    auto allocatedQubit = quantum::AllocQubitOp::create(rewriter, loc);
    return allocatedQubit.getOutQubit();
}

PPRotationOp wrapInGadget(PPRotationOp op, mlir::PatternRewriter &rewriter) {
    assert(!op.getCondition() && "a gadget implements an unconditional rotation");
    mlir::Location loc = op.getLoc();

    // pbc.ppr P(k) is exp(-i pi/k P), that is, exp(-i theta/2 P) with theta = 2 pi / k.
    const double theta = 2 * llvm::numbers::pi / static_cast<double>(op.getRotationKind());
    auto gadget = GadgetOp::create(rewriter, loc, op.getOutQubits().getTypes(),
                                   op.getPauliProduct(), rewriter.getF64FloatAttr(theta),
                                   /*verdict=*/nullptr, op.getInQubits());

    mlir::ValueRange inQubits = op.getInQubits();
    mlir::Block *body =
        rewriter.createBlock(&gadget.getRegion(), {}, inQubits.getTypes(),
                             llvm::SmallVector<mlir::Location>(inQubits.size(), loc));
    mlir::IRMapping mapping;
    mapping.map(inQubits, body->getArguments());
    auto inner = llvm::cast<PPRotationOp>(rewriter.clone(*op, mapping));
    YieldOp::create(rewriter, loc, inner.getOutQubits());

    rewriter.replaceOp(op, gadget.getOutQubits());
    rewriter.setInsertionPoint(inner);
    return inner;
}

bool needsGadget(PPRotationOp op, bool emitGadgets) {
    return emitGadgets && !op->getParentOfType<GadgetOp>();
}

} // namespace pbc
} // namespace catalyst
