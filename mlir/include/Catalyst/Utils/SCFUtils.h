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

#pragma once
#include <optional>

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Operation.h"

using namespace mlir;

namespace catalyst {

// Returns true if an operation is nested in a scf.if operation at any depth.
bool isOpInIfOp(Operation *op);

// Returns true if an operation is nested in a scf.while operation at any depth.
bool isOpInWhileOp(Operation *op);

// Given an op in a for loop body with a static number of start, end and step,
// compute the number of iterations that will be executed by the for loop.
// Returns -1 if any of the above for loop information is not static.
//
// Note: if the input op is not inside any for loop operations,
// this method returns 1, since there would be just one "iteration".
int64_t countStaticForloopIterations(Operation *op);

// Returns the static trip count of a single for loop if all three bounds are
// arith.constant ops, or -1 if any bound is dynamic.
int64_t countStaticForOpIterations(scf::ForOp forOp);

// Reads the `catalyst.estimated_iterations` resource-estimation hint from an op, if present.
// The hint may be an integer or a float. The value is then returned as an optional double.
std::optional<double> getEstimatedIterationsHint(Operation *op);

// Resolve a for loop's expected trip count using, in order of preference:
//   1. a `catalyst.estimated_iterations` integer/float attribute,
//   2. scf::ForOp::getStaticTripCount(), then
//   3. recursively-resolved constant lower/upper bounds and step.
// Returns std::nullopt when the trip count cannot be determined statically.
std::optional<double> resolveForLoopTripCount(scf::ForOp forOp);

// Resolve the average trip count of a loop whose upper bound is either a fixed constant or the
// induction variable of an enclosing scf.for loop. Chains of nested loops are supported to any
// depth, and a loop may depend on any enclosing loop rather than only its immediate parent.
//
// Enclosing loops contribute in one of two ways:
//   - If a later loop uses its induction variable, it is enumerated. Its domain comes from its
//     own resolved bounds, or from an *integral* `catalyst.estimated_iterations = K` hint, which
//     stands in for the first K induction values starting at its lower bound.
//   - Otherwise it only scales how often the target loop is reached, so it contributes a scalar
//     multiplicity taken from its `catalyst.estimated_iterations` hint (integer or fractional) or
//     its static trip count.
//
// Returns std::nullopt when any bound in the chain stays unresolved: an arithmetic expression, an
// unrelated dynamic loop, a non-scf.for ancestor such as `scf.if`, or an enclosing loop that must
// be enumerated but offers only a fractional estimate.
//
// Note that the result is an *average* over every enumerated context, so it may be fractional
// even when all bounds are static.
std::optional<double> resolveDirectNestedForLoopAverageTripCount(scf::ForOp forOp);

} // namespace catalyst
