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

#ifndef GRADIENT_ATTR_DEFS
#define GRADIENT_ATTR_DEFS

include "mlir/IR/EnumAttr.td"

include "Gradient/IR/GradientDialect.td"

//===----------------------------------------------------------------------===//
// Gradient dialect enums and attributes.
//===----------------------------------------------------------------------===//

def DiffMethod : I32EnumAttr<"DiffMethod",
    "Known gradient differentiation methods",
    [
        I32EnumAttrCase<"FiniteDifference", 0, "fd">,
        I32EnumAttrCase<"Auto", 1, "auto">,
    ]> {
    let cppNamespace = "::catalyst::gradient";
    let genSpecializedAttr = 0;
}

def DiffMethodAttr : EnumAttr<Gradient_Dialect, DiffMethod, "diff_method">;

#endif // GRADIENT_ATTR_DEFS

