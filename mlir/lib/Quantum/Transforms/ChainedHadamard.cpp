// Copyright 2023 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#define DEBUG_TYPE "chained-hadamard"

#include "Quantum/IR/QuantumOps.h"
#include "Quantum/Transforms/Patterns.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Errc.h"
#include <set>
#include <string>

using llvm::dbgs;
using namespace mlir;
using namespace catalyst;
using namespace catalyst::quantum;

namespace {

struct ChainedHadamardOpRewritePattern : public mlir::OpRewritePattern<CustomOp> {
    using mlir::OpRewritePattern<CustomOp>::OpRewritePattern;

    /// We simplify consecutive Hermitian quantum gates and simplifies them.
    /// Hermitian gates are self-inverse and applying the same gate twice in succession
    /// cancels out the effect. This pattern rewrites such redundant operations by
    /// replacing the operation with its "grandparent" operation in the quantum circuit.
    mlir::LogicalResult matchAndRewrite(CustomOp op, mlir::PatternRewriter &rewriter) const override
    {
        LLVM_DEBUG(dbgs() << "Simplifying the following operation:\n" << op << "\n");

        std::set<StringRef> HermitianOps{"Hadamard", "PauliX", "PauliY", "PauliZ", "CNOT",
                                         "CY",       "CZ",     "SWAP",   "Toffoli"};
        StringRef OpGateName = op.getGateName();
        if (HermitianOps.find(OpGateName) == HermitianOps.end())
            return failure();

        ValueRange InQubits = op.getInQubits();
        auto ParentOp = dyn_cast_or_null<CustomOp>(InQubits[0].getDefiningOp());
        if (!ParentOp || ParentOp.getGateName() == OpGateName)
            return failure();

        ValueRange ParentOutQubits = ParentOp.getOutQubits();
        // Check if the input qubits to the current operation match the output qubits of the parent.
        for (const auto &[Idx, Qubit] : llvm::enumerate(InQubits)) {
            if (Qubit.getDefiningOp<CustomOp>() != ParentOp || Qubit != ParentOutQubits[Idx])
                return failure();
        }
        ValueRange simplifiedVal = ParentOp.getInQubits();
        rewriter.replaceOp(op, simplifiedVal);
        return success();
    }
};

} // namespace

namespace catalyst {
namespace quantum {

void populateSelfInversePatterns(RewritePatternSet &patterns)
{
    patterns.add<ChainedHadamardOpRewritePattern>(patterns.getContext(), 1);
}

} // namespace quantum
} // namespace catalyst
