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
#include <deque>

#include "mlir/Dialect/Func/IR/FuncOps.h"

using namespace mlir;

namespace catalyst {

void traverseCallGraph(func::FuncOp start, SymbolTableCollection &symbolTable,
                       function_ref<void(func::FuncOp)> processFunc)
{
    DenseSet<Operation *> visited{start};
    std::deque<Operation *> frontier{start};

    while (!frontier.empty()) {
        auto callable = cast<func::FuncOp>(frontier.front());
        frontier.pop_front();

        processFunc(callable);
        callable.walk([&](CallOpInterface callOp) {
            if (auto nextFunc =
                    dyn_cast_or_null<func::FuncOp>(callOp.resolveCallable(&symbolTable))) {
                if (!visited.contains(nextFunc)) {
                    visited.insert(nextFunc);
                    frontier.push_back(nextFunc);
                }
            }
        });
    }
}

} // namespace catalyst
