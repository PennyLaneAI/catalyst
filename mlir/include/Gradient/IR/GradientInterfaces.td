// Copyright 2024 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef GRADIENT_INTERFACES
#define GRADIENT_INTERFACES

include "mlir/IR/OpBase.td"

def GradientOpInterface : OpInterface<"GradientOpInterface">{
    let description = [{
        This interface provides a generic way to interact with instructions that 
        are considered gradient operations. Currently these include GradOP, JVPOp, 
        and VJPOp.
    }];
    
    let cppNamespace = "::catalyst::gradient";

    let methods = [
        InterfaceMethod<
        "Return the gradient method, which can be one of [\"auto\", \"fd\"].",
        "llvm::StringRef", "getMethod"
        >,
        InterfaceMethod<
        "Return the name of the callee, aka the function being differentiated by the gradient operation.",
        "mlir::SymbolRefAttr", "getCallee"
        >,
        InterfaceMethod<
        "Return the indices of the arguments being differentiated.",
        "std::optional<mlir::DenseIntElementsAttr>", "getDiffArgIndices"
        >,
        InterfaceMethod<
        "Return the indices of the arguments being differentiated, as an MLIR attribute.",
        "mlir::DenseIntElementsAttr", "getDiffArgIndicesAttr"
        >,
        InterfaceMethod<
        "Return the finite difference parameter.",
        "std::optional<mlir::FloatAttr>", "getFiniteDiffParam"
        >,
        InterfaceMethod<
        "Return the finite difference parameter, as an MLIR attribute.",
        "mlir::FloatAttr", "getFiniteDiffParamAttr"
        >,
    ];
}

#endif // GRADIENT_INTERFACES

