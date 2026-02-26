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

#ifndef QEC_PHYSICAL_ATTR_DEFS
#define QEC_PHYSICAL_ATTR_DEFS

include "mlir/IR/EnumAttr.td"

include "QecPhysical/IR/QecPhysicalDialect.td"

//===----------------------------------------------------------------------===//
// QecPhysical dialect enums and attributes.
//===----------------------------------------------------------------------===//

def QecPhysicalQubitRole : I32EnumAttr<"QecPhysicalQubitRole",
    "role specialization of QEC physical qubits",
    [
        I32EnumAttrCase<"Data",      0, "data">,
        I32EnumAttrCase<"Auxiliary", 1, "aux">,
    ]> {
    let cppNamespace = "catalyst::qecp";
    let genSpecializedAttr = 0;
}

def QecPhysicalQubitRoleAttr : EnumAttr<
    QecPhysical_Dialect, QecPhysicalQubitRole, "qec_physical_qubit_role">;

#endif // QEC_PHYSICAL_ATTR_DEFS
