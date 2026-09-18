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

#include "mlir/IR/OpDefinition.h"

namespace catalyst {
namespace quantum {
std::string defaultGetGraphOpId(mlir::Operation *op);
bool hasModifiers(mlir::Operation *op);
} // namespace quantum
} // namespace catalyst

#include "Quantum/IR/DecompositionInterfaces.h.inc"
