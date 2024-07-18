// Copyright 2022-2023 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <deque>

#include "llvm/ADT/SmallPtrSet.h"

#include "Quantum/IR/QuantumInterfaces.h"
#include "Quantum/IR/QuantumOps.h"
#include "Quantum/Utils/RemoveQuantum.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;

namespace catalyst {
namespace quantum {

void removeQuantumMeasurements(func::FuncOp &function, mlir::PatternRewriter &rewriter)
{
    // Delete measurement operations.
    std::deque<Operation *> opsToDelete;
    function.walk([&](MeasurementProcess op) { opsToDelete.push_back(op); });
    SmallPtrSet<Operation *, 4> visited{opsToDelete.begin(), opsToDelete.end()};

    // Measurement operations are not allowed inside scf dialects.
    // This means that we can remove them.
    // But the question then becomes, can we have arbitrary control flow
    // inside after measurements?
    //
    // This will remove the operation in opsToDelete as long as it doesn't have any other uses.
    function.emitRemark() << " function";
    while (!opsToDelete.empty()) {
        Operation *currentOp = opsToDelete.front();
        currentOp->emitRemark() << " current";
        opsToDelete.pop_front();

        rewriter.modifyOpInPlace(currentOp, [&] {currentOp->dropAllReferences();});
        for (Operation *user : currentOp->getUsers()) {
            if (!visited.contains(user)) {
                visited.insert(user);
                opsToDelete.push_back(user);
            }
        }
        if (currentOp && currentOp->use_empty()) {
            currentOp->emitRemark() << " deleting";
            rewriter.eraseOp(currentOp);
        }
        else {
            opsToDelete.push_back(currentOp);
        }
    }
}

LogicalResult verifyQuantumFree(func::FuncOp function)
{
    assert(function->hasAttr("QuantumFree") &&
           "verifying function that doesn't have QuantumFree attribute");

    WalkResult result = function.walk([&](Operation *op) {
        if (isa<QuantumDialect>(op->getDialect()))
            return WalkResult::interrupt();
        return WalkResult::advance();
    });

    if (result.wasInterrupted())
        return failure();

    function->removeAttr("QuantumFree");
    return success();
}

} // namespace quantum
} // namespace catalyst
