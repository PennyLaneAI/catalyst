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

#include "DecompUtils.hpp"

#include <cstdint>

#include "mlir/IR/BuiltinAttributes.h"

#include "Quantum/IR/QuantumInterfaces.h"

using namespace mlir;

namespace catalyst {
namespace quantum {
namespace DecompUtils {

bool isDecompositionFunction(func::FuncOp func) { return func->hasAttr(target_gate_attr_name); }

StringRef getTargetGateName(func::FuncOp func) {
    if (auto target_op_attr = func->getAttrOfType<StringAttr>(target_gate_attr_name)) {
        return target_op_attr.getValue();
    }
    return StringRef{};
}

uint64_t getNumWires(func::FuncOp func) {
    if (auto num_wires_attr = func->getAttrOfType<IntegerAttr>("num_wires")) {
        return num_wires_attr.getValue().getZExtValue();
    }
    return 0;
}

bool isInDecompRule(Operation *op) {
    while (auto parentOp = op->getParentOp()) {
        if (auto funcOp = dyn_cast<func::FuncOp>(parentOp)) {
            if (funcOp->hasAttr(target_gate_attr_name)) {
                return true;
            }
        }
        op = parentOp;
    }
    return false;
}

llvm::SmallVector<Operation *> getDecompositionRoots(ModuleOp module) {
    llvm::SmallVector<Operation *> roots;

    // Any decomposable gate that does not live in a function
    // can only be reached by rewriting the module itself.
    bool gateOutsideFunc = false;
    module.walk<WalkOrder::PreOrder>([&](Operation *op) {
        if (auto func = dyn_cast<func::FuncOp>(op)) {
            if (!isDecompositionFunction(func)) {
                roots.push_back(func);
            }
            // Whatever is inside a function is covered by the function itself.
            return WalkResult::skip();
        }
        return WalkResult::advance();
    });
    return roots;
}

} // namespace DecompUtils
} // namespace quantum
} // namespace catalyst
