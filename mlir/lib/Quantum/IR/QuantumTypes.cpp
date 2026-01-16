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

#include "mlir/IR/Diagnostics.h"
#include "mlir/Support/LLVM.h"

#include "Quantum/IR/QuantumAttrDefs.h"
#include "Quantum/IR/QuantumTypes.h"

using namespace mlir;
using namespace catalyst::quantum;

LogicalResult QubitType::verify(function_ref<InFlightDiagnostic()> emitError, QubitLevel level,
                                QubitRole role)
{
    // If qubit level is not QEC or Physical, role must be Null
    // In other words, abstract and logical qubits cannot specify a role
    if ((level != QubitLevel::QEC && level != QubitLevel::Physical) && role != QubitRole::Null) {
        return emitError() << "qubit role '" << stringifyQubitRole(role)
                           << "' is only allowed for qec or physical qubits; "
                           << "found level '" << stringifyQubitLevel(level) << "'";
    }
    return success();
}
