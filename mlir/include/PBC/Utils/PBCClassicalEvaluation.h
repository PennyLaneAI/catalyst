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

// Evaluate a pure tensor operation using the optional Python compilation plugin.
// The callback resolves its operands and captures; unresolved inputs, unavailable
// plugins, unsupported operations and backend errors leave the value unknown.
mlir::Attribute evaluatePBCClassicalValue(mlir::Value value,
                                          llvm::function_ref<mlir::Attribute(mlir::Value)> resolve);

} // namespace catalyst::pbc
