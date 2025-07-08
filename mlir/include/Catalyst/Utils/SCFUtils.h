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

} // namespace catalyst
