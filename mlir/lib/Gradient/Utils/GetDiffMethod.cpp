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

#include "Gradient/Utils/GetDiffMethod.h"

using namespace mlir;

StringRef catalyst::gradient::getQNodeDiffMethod(catalyst::gradient::GradOp gradOp) {
    const char *diffMethodKey = "diff_method";

    func::FuncOp callee =
        SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(gradOp, gradOp.getCalleeAttr());
    bool isQNode = callee->hasAttr("qnode") && callee->hasAttrOfType<StringAttr>(diffMethodKey);
    if (gradOp.getMethod() == "defer" && isQNode) {
        return callee->getAttrOfType<StringAttr>(diffMethodKey).strref();
    }
    return "";
}
