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

#include <cstdint>
#include <cmath> // std::ceil()
#include <optional>

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Operation.h"

#include "Catalyst/Utils/ConstantResolve.h"

using namespace mlir;

namespace catalyst {

static int64_t getNumIterations(double lowerBound, double upperBound, double step)
{
    assert(upperBound >= lowerBound && step > 0);
    return std::ceil((upperBound - lowerBound) / step);
}

static int64_t getIntFromArithConstantOp(arith::ConstantOp op)
{
    // The magical incantation to get a cpp integer from an arith.constant op
    assert(isa<IntegerAttr>(op.getValue()));
    return cast<IntegerAttr>(op.getValue()).getValue().getSExtValue();
}

template <typename OpTy> static bool hasAncestorOfType(Operation *op)
{
    return op->getParentOfType<OpTy>() != nullptr;
}

// Returns true if an operation is nested in a scf.if operation at any depth.
bool isOpInIfOp(Operation *op) { return hasAncestorOfType<scf::IfOp>(op); }

// Returns true if an operation is nested in a scf.while operation at any depth.
bool isOpInWhileOp(Operation *op) { return hasAncestorOfType<scf::WhileOp>(op); }

// Returns the static trip count of `forOp` if all three bounds are
// arith.constant ops, or -1 if any bound is dynamic.
int64_t countStaticForOpIterations(scf::ForOp forOp)
{
    Operation *lowerBoundOp = forOp.getLowerBound().getDefiningOp();
    if (!lowerBoundOp || !isa<arith::ConstantOp>(lowerBoundOp)) {
        return -1;
    }
    int64_t l = getIntFromArithConstantOp(cast<arith::ConstantOp>(lowerBoundOp));

    Operation *upperBoundOp = forOp.getUpperBound().getDefiningOp();
    if (!upperBoundOp || !isa<arith::ConstantOp>(upperBoundOp)) {
        return -1;
    }
    int64_t u = getIntFromArithConstantOp(cast<arith::ConstantOp>(upperBoundOp));

    Operation *stepOp = forOp.getStep().getDefiningOp();
    if (!stepOp || !isa<arith::ConstantOp>(stepOp)) {
        return -1;
    }
    int64_t s = getIntFromArithConstantOp(cast<arith::ConstantOp>(stepOp));

    return getNumIterations(l, u, s);
}

std::optional<int64_t> resolveForLoopTripCount(scf::ForOp forOp)
{
    if (auto estAttr = forOp->getAttrOfType<IntegerAttr>("estimated_iterations")) {
        return estAttr.getValue().getSExtValue();
    }
    if (auto staticTrip = forOp.getStaticTripCount()) {
        return staticTrip->getSExtValue();
    }
    auto lb = resolveConstantInt(forOp.getLowerBound());
    auto ub = resolveConstantInt(forOp.getUpperBound());
    auto step = resolveConstantInt(forOp.getStep());
    if (lb && ub && step && *step != 0 && *ub > *lb) {
        return (*ub - *lb + *step - 1) / *step;
    }
    return std::nullopt;
}

// Given an op in a for loop body with a static number of start, end and step,
// compute the number of iterations that will be executed by the for loop.
// Returns -1 if any of the above for loop information is not static.
//
// Note: if the input op is not inside any for loop operations,
// this method returns 1, since there would be just one "iteration".
int64_t countStaticForloopIterations(Operation *op)
{
    assert(!isa<scf::ForOp>(op));

    int64_t count = 1;

    Operation *parent = op->getParentOp();
    while (parent) {
        if (auto forOp = dyn_cast<scf::ForOp>(parent)) {
            int64_t iterations = countStaticForOpIterations(forOp);
            if (iterations == -1) {
                return -1;
            }
            count *= iterations;
        }
        parent = parent->getParentOp();
    }

    return count;
}

} // namespace catalyst
