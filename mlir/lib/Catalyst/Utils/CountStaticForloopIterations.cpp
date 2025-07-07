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

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Attributes.h"

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
            if (!isa<arith::ConstantOp>(lowerBoundOp)) {
                // Dynamic, nothing to do at this for loop
                continue;
            }
            int64_t l = getIntFromArithConstantOp(cast<arith::ConstantOp>(lowerBoundOp));

            Operation *upperBoundOp = forOp.getUpperBound().getDefiningOp();
            if (!isa<arith::ConstantOp>(upperBoundOp)) {
                // Dynamic, nothing to do at this for loop
                continue;
            }
            int64_t u = getIntFromArithConstantOp(cast<arith::ConstantOp>(upperBoundOp));

            Operation *stepOp = forOp.getStep().getDefiningOp();
            if (!isa<arith::ConstantOp>(stepOp)) {
                // Dynamic, nothing to do at this for loop
                continue;
            }
            int64_t s = getIntFromArithConstantOp(cast<arith::ConstantOp>(stepOp));

            count *= getNumIterations(l, u, s);
            changed = true;
        }
        parent = parent->getParentOp();
    }

    if (!changed) {
        return -1;
    }

    return count;
}

} // namespace catalyst
