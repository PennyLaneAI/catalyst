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

#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/FunctionInterfaces.h"

namespace catalyst {
namespace gradient {

class ActivityAnalyzer {
  public:
    ActivityAnalyzer(mlir::FunctionOpInterface callee, mlir::ArrayRef<size_t> diffArgIndices);

    bool isActive(mlir::Value value) const;

  private:
    mlir::DataFlowSolver solver;
};

} // namespace gradient
} // namespace catalyst
