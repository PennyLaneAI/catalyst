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

#define DEBUG_TYPE "merge-rotations"

#include "Quantum/IR/QuantumOps.h"
#include "Quantum/Transforms/Patterns.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Errc.h"
using llvm::dbgs;
using namespace mlir;
using namespace catalyst;
using namespace catalyst::quantum;

static const mlir::StringSet<> rotationsSet = {"RX", "RY", "RZ", "CRX", "CRY", "CRZ"};

namespace {

struct MergeRotationsRewritePattern : public mlir::OpRewritePattern<CustomOp> {
    using mlir::OpRewritePattern<CustomOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(CustomOp op, mlir::PatternRewriter &rewriter) const override
    {
        LLVM_DEBUG(dbgs() << "Simplifying the following operation:\n" << op << "\n");
        auto loc = op.getLoc();
        StringRef OpGateName = op.getGateName();
        if (!rotationsSet.contains(OpGateName))
            return failure();
        ValueRange InQubits = op.getInQubits();
        auto parentOp = dyn_cast_or_null<CustomOp>(InQubits[0].getDefiningOp());
        if (!parentOp || parentOp.getGateName() != OpGateName)
            return failure();

        ValueRange ParentOutQubits = parentOp.getOutQubits();
        // Check if the input qubits to the current operation match the output qubits of the parent.
        for (const auto &[Idx, Qubit] : llvm::enumerate(InQubits)) {
            if (Qubit.getDefiningOp<CustomOp>() != parentOp || Qubit != ParentOutQubits[Idx])
                return failure();
        }


        static const mlir::StringSet<> rotationsSetCase1 = {"RX", "RY", "RZ", "CRX", "CRY", "CRZ"};

        if (rotationsSetCase1.find(OpGateName) != rotationsSetCase1.end()) {
            TypeRange OutQubitsTypes = op.getOutQubits().getTypes();
            TypeRange OutQubitsCtrlTypes = op.getOutCtrlQubits().getTypes();
            auto parentParams = parentOp.getParams().front();
            auto params = op.getParams().front();
            Value sumParams = rewriter.create<arith::AddFOp>(loc, parentParams, params).getResult();
            ValueRange parentInQubits = parentOp.getInQubits();
            ValueRange parentInCtrlQubits = parentOp.getInCtrlQubits();
            ValueRange parentInCtrlValues = parentOp.getInCtrlValues();
            auto mergeOp = rewriter.create<CustomOp>(loc, OutQubitsTypes, OutQubitsCtrlTypes,
                                                     sumParams, parentInQubits, OpGateName, nullptr,
                                                     parentInCtrlQubits, parentInCtrlValues);
            op.replaceAllUsesWith(mergeOp);
            op.erase();
            parentOp.erase();
        }
        return success();
    }
};

} // namespace

namespace catalyst {
namespace quantum {

void populateMergeRotationsPatterns(RewritePatternSet &patterns)
{
    patterns.add<MergeRotationsRewritePattern>(patterns.getContext(), 1);
}

} // namespace quantum
} // namespace catalyst
