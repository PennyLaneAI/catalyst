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

#include <cmath> // std::ceil()

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Operation.h"

using namespace mlir;

namespace catalyst {

static int64_t getNumIterations(int64_t lowerBound, int64_t upperBound, int64_t step)
{
    return std::ceil((upperBound - lowerBound) / step);
}

static int64_t getIntFromArithConstantOp(arith::ConstantOp op)
{
    // The magical incantation to get a cpp integer from an arith.constant op
    assert(isa<IntegerAttr>(op.getValue()));
    return cast<IntegerAttr>(op.getValue()).getValue().getSExtValue();
}

// Given an op in a for loop body with a static number of start, end and step,
// compute the number of iterations that will be executed by the for loop.
// Returns -1 if any of the above for loop information is not static, or if the
// input operation is not inside any for loop operations.
int64_t countStaicForloopIterations(Operation *op)
{
    assert(!isa<scf::ForOp>(op));

    int64_t count = 1;
    bool changed = false;

    Operation *parent = op->getParentOp();
    while (parent) {
        if (isa<scf::ForOp>(parent)) {
            scf::ForOp forOp = cast<scf::ForOp>(parent);

            Operation *lowerBoundOp = forOp.getLowerBound().getDefiningOp();
            if (!lowerBoundOp || !isa<arith::ConstantOp>(lowerBoundOp)) {
                // Dynamic
                return -1;
            }
            int64_t l = getIntFromArithConstantOp(cast<arith::ConstantOp>(lowerBoundOp));

            Operation *upperBoundOp = forOp.getUpperBound().getDefiningOp();
            if (!upperBoundOp || !isa<arith::ConstantOp>(upperBoundOp)) {
                // Dynamic
                return -1;
            }
            int64_t u = getIntFromArithConstantOp(cast<arith::ConstantOp>(upperBoundOp));

            Operation *stepOp = forOp.getStep().getDefiningOp();
            if (!stepOp || !isa<arith::ConstantOp>(stepOp)) {
                // Dynamic
                return -1;
            }
            int64_t s = getIntFromArithConstantOp(cast<arith::ConstantOp>(stepOp));

            int64_t numIters = getNumIterations(l, u, s);
            count *= numIters;
            changed = true;
        }
        parent = parent->getParentOp();
    }

    if (!changed) {
        // Never encountered a for loop
        return -1;
    }
    return count;
}

} // namespace catalyst
