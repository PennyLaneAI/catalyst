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
        "Return the gradient method, which can be one of [\"auto\", \"fd\", \"ps\", \"adj\"].",
        "llvm::StringRef", "getMethod"
        >
    ];
}

#endif // GRADIENT_INTERFACES

