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

#include "Quantum/IR/QuantumOps.h" // for quantum::AllocQubitOp

#include "QEC/IR/QECDialect.h" // for FabricateOp
#include "QEC/Transforms/PPRDecomposeUtils.h"

namespace catalyst {
namespace qec {

std::pair<mlir::StringRef, uint16_t>
determinePauliAndRotationSignOfMeasurement(bool avoidPauliYMeasure)
{
    if (avoidPauliYMeasure) {
        return std::make_pair("Z", -1);
    }
    return std::make_pair("Y", 1);
}

mlir::OpResult initializeZeroOrPlusI(bool avoidPauliYMeasure, mlir::Location loc,
                                     mlir::PatternRewriter &rewriter)
{
    if (avoidPauliYMeasure) {
        // Fabricate |Y⟩
        auto plusIOp = rewriter.create<FabricateOp>(loc, LogicalInitKind::plus_i);
        return plusIOp.getOutQubits().back();
    }

    // Initialize |0⟩
    auto allocatedQubit = rewriter.create<quantum::AllocQubitOp>(loc);
    return allocatedQubit.getOutQubit();
}

} // namespace qec
} // namespace catalyst
