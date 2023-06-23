// Copyright 2023 Xanadu Quantum Technologies Inc.

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

#include <vector>

#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/Optional.h"

namespace catalyst {

// Calculate the vector of effective gradient argument indices based on the user
// settings.
std::vector<size_t> compDiffArgIndices(llvm::Optional<mlir::DenseIntElementsAttr> indices);

}; // namespace catalyst
