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
#include "Quantum/Utils/RemoveQuantumMeasurements.h"

using namespace mlir;

namespace catalyst {
namespace quantum {

void removeQuantumMeasurements(func::FuncOp &function)
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
    // This will remove the operation in opsToDelete as long as any other uses.
    while (!opsToDelete.empty()) {
        Operation *currentOp = opsToDelete.front();
        opsToDelete.pop_front();
        currentOp->dropAllReferences();
        for (Operation *user : currentOp->getUsers()) {
            if (!visited.contains(user)) {
                visited.insert(user);
                opsToDelete.push_back(user);
            }
        }
        if (currentOp->use_empty()) {
            currentOp->erase();
        }
        else {
            opsToDelete.push_back(currentOp);
        }
    }
}

} // namespace quantum
} // namespace catalyst
