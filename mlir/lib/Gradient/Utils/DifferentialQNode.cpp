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

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/SymbolTable.h"

#include "Gradient/Utils/DifferentialQNode.h"

using namespace mlir;

constexpr const char *diffMethodKey = "diff_method";
constexpr const char *pureQuantumKey = "purequantum";

bool catalyst::gradient::isQNode(func::FuncOp funcOp)
{
    return funcOp->hasAttrOfType<UnitAttr>("qnode");
}

StringRef catalyst::gradient::getQNodeDiffMethod(func::FuncOp funcOp)
{
    bool hasDiffMethod = isQNode(funcOp) && funcOp->hasAttrOfType<StringAttr>(diffMethodKey);
    if (hasDiffMethod) {
        return funcOp->getAttrOfType<StringAttr>(diffMethodKey).strref();
    }
    return "";
}

void catalyst::gradient::setRequiresCustomGradient(func::FuncOp funcOp,
                                                   FlatSymbolRefAttr pureQuantumFunc)
{
    funcOp->setAttr(pureQuantumKey, pureQuantumFunc);
}

bool catalyst::gradient::requiresCustomGradient(func::FuncOp funcOp)
{
    return funcOp->hasAttrOfType<FlatSymbolRefAttr>(pureQuantumKey);
}

void catalyst::gradient::registerCustomGradient(func::FuncOp qnode, FlatSymbolRefAttr qgradFn)
{
    Operation *pureQuantumFunc = SymbolTable::lookupNearestSymbolFrom(
        qnode, qnode->getAttrOfType<FlatSymbolRefAttr>(pureQuantumKey));
    pureQuantumFunc->setAttr("gradient.qgrad", qgradFn);

    // Mark this op as processed so it doesn't get processed again.
    qnode->removeAttr(pureQuantumKey);
}
