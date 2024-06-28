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

#define DEBUG_TYPE "chained-self-inverse"

#include "llvm/Support/Debug.h"
#include "llvm/Support/Errc.h"

#include "Quantum/IR/QuantumOps.h"
#include "Quantum/Transforms/Patterns.h"

using llvm::dbgs;
using namespace mlir;
using namespace catalyst;
using namespace catalyst::quantum;

namespace {

llvm::SmallVector<StringRef> SelfInverseGates{
    "PauliX", "PauliY", "PauliZ", "Hadamard", "CNOT", "CY", "CZ", "SWAP", "Toffoli",
};

// We check if the operation and its parent are a pair of self-inverse operations.
// If so, replace op with its "grandparent".
// e.g.
//    %1 = (a qubit value produced from somewhere)
//    %out_qubits_1 = quantum.custom "PauliZ"() %1 : !quantum.bit
//    %out_qubits_2 = quantum.custom "PauliZ"() %out_qubits_1 : !quantum.bit
//    %2 = (a user operation that uses %out_qubit_2)
// The Value %out_qubits_2 can be replaced by the Value %1 (which is its grandparent) in all uses

mlir::LogicalResult SelfInverseGatesMatchAndRewriteImpl(CustomOp op,
                                                        mlir::PatternRewriter &rewriter,
                                                        SmallVector<StringRef> SelfInverseGates)
{
    for (StringRef OpName : SelfInverseGates) {
        LLVM_DEBUG(dbgs() << "Simplifying the following " << OpName << " operation:\n"
                          << op << "\n");
        if (op.getGateName().str() != OpName)
            continue;

        ValueRange qbs = op.getInQubits();
        auto parentOp = dyn_cast<CustomOp>(qbs[0].getDefiningOp());

        if (parentOp == nullptr) {
            continue;
        }

        if (parentOp.getGateName().str() != OpName) {
            continue;
        }

        // for multiple qubit gates, the wires need to be in order
        // since the cancelled inverses need to have matching control/target wires
        // e.g. The following pair of neighbouring CNOTs should NOT be eliminated
        //    %out_qubits:2 = quantum.custom "CNOT"() %1, %2 : !quantum.bit, !quantum.bit
        //    %out_qubits_0:2 = quantum.custom "CNOT"() %out_qubits#1, %out_qubits#0 : !quantum.bit,
        //    !quantum.bit

        bool failed = false; // we want to continue to the next gate to check for in the outer loop,
                             // not this small loop
        for (size_t i = 0; i < qbs.size(); i++) {
            if ((parentOp.getOutQubits())[i] != qbs[i]) {
                failed = true;
            }
        }
        if (failed) {
            continue;
        }

        ValueRange simplifiedVal = parentOp.getInQubits();
        rewriter.replaceOp(op, simplifiedVal);
        return success();
    }
    return failure();
}

struct ChainedSelfInverseOpRewritePattern : public mlir::OpRewritePattern<CustomOp> {
    using mlir::OpRewritePattern<CustomOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(CustomOp op, mlir::PatternRewriter &rewriter) const override
    {
        return SelfInverseGatesMatchAndRewriteImpl(op, rewriter, SelfInverseGates);
    }
};

} // namespace

namespace catalyst {
namespace quantum {

void populateSelfInversePatterns(RewritePatternSet &patterns)
{
    patterns.add<ChainedSelfInverseOpRewritePattern>(patterns.getContext(), 1);
}

} // namespace quantum
} // namespace catalyst
