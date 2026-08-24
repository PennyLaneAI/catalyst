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
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CheckedArithmetic.h"
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
    if (auto staticTrip = forOp.getStaticTripCount()) {
        return static_cast<double>(staticTrip->getSExtValue());
    }
    if (auto iters = getEstimatedIterationsHint(forOp)) {
        return *iters;
    }
    auto lb = resolveConstantInt(forOp.getLowerBound());
    auto ub = resolveConstantInt(forOp.getUpperBound());
    auto step = resolveConstantInt(forOp.getStep());
    if (lb && ub && step && *step > 0 && *ub > *lb) {
        return static_cast<double>(computeTripCount(*lb, *ub, *step));
    }
    return std::nullopt;
}

// Store each enclosing loop's current induction value.
// Example: while evaluating `%i=3`, the map contain `%i` -> `3`.
using InductionValues = llvm::DenseMap<Value, int64_t>;

// Resolve `loop`'s own upper bound as a concrete integer so its induction values can be
// enumerated: a resolved constant, the recorded value of an already-enumerated enclosing loop,
// or, for an integer `catalyst.estimated_iterations = K` hint, the first K values starting at
// `lower`. A fractional hint has no discrete domain.
static std::optional<int64_t> resolveEnumerableUpperBound(scf::ForOp loop, int64_t lower,
                                                          int64_t step,
                                                          const InductionValues &inductionValues) {
    if (auto upper = resolveConstantInt(loop.getUpperBound())) {
        return upper;
    }
    auto it = inductionValues.find(loop.getUpperBound());
    if (it != inductionValues.end()) {
        return it->second;
    }
    auto intAttr = dyn_cast_or_null<IntegerAttr>(loop->getAttr(EstimatedIterationsAttrName));
    if (!intAttr) {
        return std::nullopt;
    }
    auto iterationCount = intAttr.getValue().trySExtValue();
    if (!iterationCount || *iterationCount < 0) {
        return std::nullopt;
    }
    auto upper = llvm::checkedMulAdd(*iterationCount, step, lower);
    if (!upper) {
        loop.emitWarning("Cannot resolve estimated loop domain: integer overflow");
    }
    return upper;
}

// Resolve `loop`'s own scalar trip count: an estimated-iterations hint (integer or fractional),
// or lower/step/upper bounds where the upper bound is a resolved constant or the recorded value
// of an already-enumerated enclosing loop.
static std::optional<double> resolveOwnTripCount(scf::ForOp loop,
                                                 const InductionValues &inductionValues) {
    if (auto iters = getEstimatedIterationsHint(loop)) {
        return *iters;
    }
    auto lower = resolveConstantInt(loop.getLowerBound());
    auto step = resolveConstantInt(loop.getStep());
    if (!lower || !step || *step <= 0) {
        return std::nullopt;
    }
    auto upper = resolveConstantInt(loop.getUpperBound());
    if (!upper) {
        auto it = inductionValues.find(loop.getUpperBound());
        if (it == inductionValues.end()) {
            return std::nullopt;
        }
        upper = it->second;
    }
    return static_cast<double>(computeTripCount(*lower, *upper, *step));
}

// True if `loop`'s own upper bound is already explained without searching further ancestors:
// either an estimated-iterations hint (of any kind) or a resolved constant.
static bool closesOwnUpperBound(scf::ForOp loop) {
    return getEstimatedIterationsHint(loop).has_value() ||
           resolveConstantInt(loop.getUpperBound()).has_value();
}

// Collect the minimal chain of enclosing scf.for loops, outer to inner, needed to resolve
// `target`'s upper-bound dependency. Walking upward, a loop's induction variable resolves any
// not-yet-explained upper bound that uses it directly; std::nullopt means some dependency
// reaches an unresolved bound, a non-scf.for parent (e.g. scf.if), or the top of the function.
static std::optional<llvm::SmallVector<scf::ForOp>> collectLoopChain(scf::ForOp target) {
    llvm::SmallVector<scf::ForOp> chain;
    llvm::SmallPtrSet<Value, 4> unresolved;

    scf::ForOp currentLoop = target;
    while (true) {
        chain.push_back(currentLoop);
        unresolved.erase(currentLoop.getInductionVar());
        if (!closesOwnUpperBound(currentLoop)) {
            unresolved.insert(currentLoop.getUpperBound());
        }
        if (unresolved.empty()) {
            std::reverse(chain.begin(), chain.end());
            return chain;
        }

        auto parent = dyn_cast_or_null<scf::ForOp>(currentLoop->getParentOp());
        if (!parent) {
            return std::nullopt;
        }
        currentLoop = parent;
    }
}

// Fast path for the most common case, where every loop shares a common lower bound,
// has a step of 1 and uses only its immediate predecessor's loop variable as its upper bound:
// for i in 0..8
//     for j in 0..i
//         for k in 0..j (average trip count is 2)
//             ...
// The average trip count is (8 - 3 + 1) / 3 = 2 for this immediate-parent chain.
static std::optional<double> tryClosedFormAverage(llvm::ArrayRef<scf::ForOp> chain) {
    scf::ForOp outer = chain.front();
    auto outerLowerBound = resolveConstantInt(outer.getLowerBound());
    if (!outerLowerBound) {
        return std::nullopt;
    }
    for (size_t i = 0; i < chain.size(); i++) {
        scf::ForOp loop = chain[i];
        if (resolveConstantInt(loop.getLowerBound()) != outerLowerBound ||
            resolveConstantInt(loop.getStep()) != 1) {
            return std::nullopt;
        }
        if (i > 0 && loop.getUpperBound() != scf::ForOp(chain[i - 1]).getInductionVar()) {
            return std::nullopt;
        }
    }
    InductionValues noInductionValues;
    auto outerUpperBound =
        resolveEnumerableUpperBound(outer, *outerLowerBound, 1, noInductionValues);
    if (!outerUpperBound) {
        return std::nullopt;
    }
    int64_t depth = static_cast<int64_t>(chain.size());
    double average = static_cast<double>(*outerUpperBound - *outerLowerBound - depth + 1) /
                     static_cast<double>(depth);
    return std::max(0.0, average);
}

// Store the target loop's total iterations and how many times it is reached across every
// enumerated context. Dividing the total by the invocation count gives its average trip count.
struct TripCountSummary {
    double totalIterations = 0.0; // The total number of times the loop body is executed
    double entryCount = 0.0; // The total number of times this loop is reached by the outer caller
};

// Evaluate chain[position..], given the induction values recorded for already-enumerated
// enclosing loops and the scalar weight contributed by loops folded so far. A loop is enumerated
// (its induction values tracked) only when a later loop's upper bound directly uses it;
// otherwise its own trip count is folded into `pathWeight` without enumeration.
static std::optional<TripCountSummary> evaluateChain(llvm::ArrayRef<scf::ForOp> chain,
                                                     size_t position, double pathWeight,
                                                     InductionValues &inductionValues) {
    scf::ForOp loop = chain[position];
    llvm::ArrayRef<scf::ForOp> descendants = chain.drop_front(position + 1);
    // Whether or not *any* subsequent loop's trip count depends on the current induction variable
    bool isReferencedLater =
        std::any_of(descendants.begin(), descendants.end(), [&](scf::ForOp descendant) {
            return descendant.getUpperBound() == loop.getInductionVar();
        });

    // Current loop's induction variable does not affect upper bound of any subsequent loop.
    // Can just get this loop's trip count and continue
    if (!isReferencedLater) {
        auto tripCount = resolveOwnTripCount(loop, inductionValues);
        if (!tripCount) {
            return std::nullopt;
        }
        if (descendants.empty()) {
            return TripCountSummary{pathWeight * *tripCount, pathWeight};
        }
        return evaluateChain(chain, position + 1, pathWeight * *tripCount, inductionValues);
    }

    auto lower = resolveConstantInt(loop.getLowerBound());
    auto step = resolveConstantInt(loop.getStep());
    if (!lower || !step || *step <= 0) {
        return std::nullopt;
    }
    auto upper = resolveEnumerableUpperBound(loop, *lower, *step, inductionValues);
    if (!upper) {
        return std::nullopt;
    }

    TripCountSummary summary;
    for (int64_t iv = *lower; iv < *upper; iv += *step) {
        inductionValues[loop.getInductionVar()] = iv;
        auto nested = evaluateChain(chain, position + 1, pathWeight, inductionValues);
        if (!nested) {
            inductionValues.erase(loop.getInductionVar());
            return std::nullopt;
        }
        summary.totalIterations += nested->totalIterations;
        summary.entryCount += nested->entryCount;
    }
    inductionValues.erase(loop.getInductionVar());
    return summary;
}

std::optional<double> resolveDirectNestedForLoopAverageTripCount(scf::ForOp forOp) {
    auto chain = collectLoopChain(forOp);
    if (!chain) {
        return std::nullopt;
    }
    if (auto closedForm = tryClosedFormAverage(*chain)) {
        return closedForm;
    }

    InductionValues inductionValues;
    auto summary = evaluateChain(*chain, 0, /*pathWeight=*/1.0, inductionValues);
    if (!summary) {
        return std::nullopt;
    }
    if (summary->entryCount == 0.0) {
        return 0.0;
    }
    return summary->totalIterations / summary->entryCount;
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
