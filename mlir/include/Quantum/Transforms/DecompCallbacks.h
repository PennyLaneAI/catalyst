// Copyright 2026 Xanadu Quantum Technologies Inc.

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

#include <string>

#include "llvm/ADT/ArrayRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"

namespace catalyst::quantum {

// One free function per quantum op whose decomposition requires runtime data.
// The plugin registers a concrete implementation; quantum-transform passes
// invoke it via the getter.
// Note: nullptr = no implementation registered yet.

using LowerPauliRotFn = mlir::OwningOpRef<mlir::func::FuncOp> (*)(mlir::MLIRContext *ctx,
                                                                  double theta,
                                                                  const std::string &pauliWord,
                                                                  llvm::ArrayRef<int> wires);

void registerLowerPauliRot(LowerPauliRotFn fn);
LowerPauliRotFn getLowerPauliRot();

// TODO: User-provided stopping_condition predicate.
// The frontend captures the
// callable in a Python-side registry keyed by conditionId and
// embeds the id in the pass-pipeline options.
// The pass invokes this function
// pointer per op-instance encountered during graph decomposition.
using StoppingConditionFn = bool (*)(const char *conditionId, const char *opName,
                                     llvm::ArrayRef<double> params, llvm::ArrayRef<int> wires);

void registerStoppingCondition(StoppingConditionFn fn);
StoppingConditionFn getStoppingCondition();

} // namespace catalyst::quantum
