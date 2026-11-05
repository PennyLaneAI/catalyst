// Copyright 2026 Xanadu Quantum Technologies Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "llvm/ADT/STLFunctionalExtras.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Value.h"

namespace catalyst::pbc {

// Recognize only the deterministic, PPR-basis runtime ABI emitted by gridsynth.
// The gate-buffer writer is accepted only for a private, immutable allocation.
bool isPBCSynthesisCall(mlir::Operation *op);

// Resolve deterministic synthesis sizes, phases, and loads from its immutable
// gate buffer using the installed Catalyst runtime. No random or user-defined
// runtime function is executed. Missing libraries or unknown inputs stay unknown.
mlir::Attribute evaluatePBCSynthesisValue(mlir::Value value,
                                          llvm::function_ref<mlir::Attribute(mlir::Value)> resolve);

} // namespace catalyst::pbc
