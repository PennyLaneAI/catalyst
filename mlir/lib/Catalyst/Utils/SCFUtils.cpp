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

#include "Catalyst/Utils/SCFUtils.h"

#include <algorithm>
#include <cstdint>
#include <optional>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"

#include "Catalyst/IR/CatalystDialect.h"
#include "Catalyst/Utils/ConstantResolve.h"

using namespace mlir;

namespace catalyst {

static int64_t computeTripCount(int64_t lowerBound, int64_t upperBound, int64_t step) {
    assert(step > 0);
    if (upperBound <= lowerBound) {
        return 0;
    }
    return (upperBound - lowerBound + step - 1) / step;
}

static int64_t getIntFromArithConstantOp(arith::ConstantOp op) {
    // The magical incantation to get a cpp integer from an arith.constant op
    assert(isa<IntegerAttr>(op.getValue()));
    return cast<IntegerAttr>(op.getValue()).getValue().getSExtValue();
}

template <typename OpTy> static bool hasAncestorOfType(Operation *op) {
    return op->getParentOfType<OpTy>() != nullptr;
}

// Returns true if an operation is nested in a scf.if operation at any depth.
bool isOpInIfOp(Operation *op) { return hasAncestorOfType<scf::IfOp>(op); }

// Returns true if an operation is nested in a scf.while operation at any depth.
bool isOpInWhileOp(Operation *op) { return hasAncestorOfType<scf::WhileOp>(op); }

// Returns the static trip count of `forOp` if all three bounds are
// arith.constant ops, or -1 if any bound is dynamic.
int64_t countStaticForOpIterations(scf::ForOp forOp) {
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

    return computeTripCount(l, u, s);
}

std::optional<double> getEstimatedIterationsHint(Operation *op) {
    Attribute attr = op->getAttr(EstimatedIterationsAttrName);
    if (!attr) {
        return std::nullopt;
    }
    if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
        return static_cast<double>(intAttr.getValue().getSExtValue());
    }
    if (auto floatAttr = dyn_cast<FloatAttr>(attr)) {
        return floatAttr.getValueAsDouble();
    }
    return std::nullopt;
}

std::optional<double> resolveForLoopTripCount(scf::ForOp forOp) {
    if (auto iters = getEstimatedIterationsHint(forOp)) {
        return *iters;
    }
    if (auto staticTrip = forOp.getStaticTripCount()) {
        return static_cast<double>(staticTrip->getSExtValue());
    }
    auto lb = resolveConstantInt(forOp.getLowerBound());
    auto ub = resolveConstantInt(forOp.getUpperBound());
    auto step = resolveConstantInt(forOp.getStep());
    if (lb && ub && step && *step > 0 && *ub > *lb) {
        return static_cast<double>(computeTripCount(*lb, *ub, *step));
    }
    return std::nullopt;
}

struct LoopRange {
    int64_t lower;
    int64_t step;
};

struct TripCountSummary {
    int64_t total = 0;
    uint64_t invocations = 0; // how many times the innermost loop is reached
};

static TripCountSummary accumulateTripCounts(llvm::ArrayRef<LoopRange> ranges, size_t loopIndex,
                                             int64_t upperBound) {
    const LoopRange &range = ranges[loopIndex];
    if (loopIndex + 1 == ranges.size()) {
        return {computeTripCount(range.lower, upperBound, range.step), 1};
    }

    TripCountSummary summary;
    for (int64_t inductionValue = range.lower; inductionValue < upperBound;
         inductionValue += range.step) {
        TripCountSummary nested = accumulateTripCounts(ranges, loopIndex + 1, inductionValue);
        summary.total += nested.total;
        summary.invocations += nested.invocations;
    }
    return summary;
}

static double computeAverageTripCountByEnumeration(llvm::ArrayRef<LoopRange> ranges,
                                                   int64_t outerUpper) {
    TripCountSummary summary = accumulateTripCounts(ranges, 0, outerUpper);

    if (summary.invocations == 0) {
        return 0.0;
    }
    return summary.total / static_cast<double>(summary.invocations);
}

std::optional<double> resolveDirectNestedForLoopAverageTripCount(scf::ForOp forOp) {
    llvm::SmallVector<LoopRange> ranges;
    int64_t outerUpper = 0;
    scf::ForOp currentLoop = forOp;
    while (currentLoop) {
        auto lower = getConstantIntValue(currentLoop.getLowerBound());
        auto step = getConstantIntValue(currentLoop.getStep());

        if (!lower || !step || *step <= 0) {
            return std::nullopt;
        }
        ranges.push_back({*lower, *step});

        if (Attribute attr = currentLoop->getAttr(EstimatedIterationsAttrName)) {
            auto intAttr = dyn_cast<IntegerAttr>(attr);
            if (!intAttr) {
                return std::nullopt;
            }
            auto iterationCount = intAttr.getValue().trySExtValue();
            if (!iterationCount || *iterationCount < 0) {
                return std::nullopt;
            }

            int64_t span;
            if (__builtin_mul_overflow(*iterationCount, *step, &span) ||
                __builtin_add_overflow(*lower, span, &outerUpper)) {
                return std::nullopt;
            }
            break;
        }

        if (auto upper = getConstantIntValue(currentLoop.getUpperBound())) {
            outerUpper = *upper;
            break;
        }

        auto parent = dyn_cast_or_null<scf::ForOp>(currentLoop->getParentOp());
        if (!parent || currentLoop.getUpperBound() != parent.getInductionVar()) {
            return std::nullopt;
        }
        currentLoop = parent;
    }
    std::reverse(ranges.begin(), ranges.end());

    // Loops that start at 0 and step by 1 use this average: (outerUpper - depth + 1) / depth.
    // Example:
    // for i in 0..8
    //     for j in 0..i
    //         for k in 0..j (average trip count is 2)
    //             ...
    // The average trip count is (8 - 3 + 1) / 3 = 2
    bool isCanonical = true;
    for (const LoopRange &range : ranges) {
        if (range.lower != 0 || range.step != 1) {
            isCanonical = false;
            break;
        }
    }
    if (isCanonical) {
        int64_t depth = static_cast<int64_t>(ranges.size());
        double average = static_cast<double>(outerUpper - depth + 1) / static_cast<double>(depth);
        return std::max(0.0, average);
    }

    // Each loop is resolved separately, so nested non-canonical loops may revisit outer loops.
    return computeAverageTripCountByEnumeration(ranges, outerUpper);
}

// Given an op in a for loop body with a static number of start, end and step,
// compute the number of iterations that will be executed by the for loop.
// Returns -1 if any of the above for loop information is not static.
//
// Note: if the input op is not inside any for loop operations,
// this method returns 1, since there would be just one "iteration".
int64_t countStaticForloopIterations(Operation *op) {
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
