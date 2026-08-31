// Copyright 2025 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "PBC/IR/PBCOpInterfaces.h"

using namespace mlir;
using namespace catalyst::pbc;
//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

namespace { 


}

//===----------------------------------------------------------------------===//
// PBC interface definitions.
//===----------------------------------------------------------------------===//

namespace catalyst {
namespace pbc {
    return name;
}

std::string defaultGetGraphOpId(Operation *op) {
    std::string out;
    llvm::raw_string_ostream ss(out);

    DecomposableGate gate = cast<DecomposableGate>(op);

    ss << gate.getOperatorName();

    
    return op->getName().getStringRef().str();
}

}
} // namespace catalyst
#include "PBC/IR/PBCOpInterfaces.cpp.inc"
