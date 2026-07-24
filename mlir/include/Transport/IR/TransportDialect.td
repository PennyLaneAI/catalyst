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

#ifndef TRANSPORT_DIALECT
#define TRANSPORT_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/IR/DialectBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Transport dialect definition.
//===----------------------------------------------------------------------===//

def Transport_Dialect : Dialect {
    let summary = "Typed ops for setting up and driving a transport session.";
    let description = [{
        The transport dialect models a connection-oriented data-movement session
        between two endpoints: creating a session, bringing up the connection,
        exchanging memory handles, establishing a data path, and running rounds
        of request/reply traffic until teardown.
    }];

    let name = "transport";
    let cppNamespace = "::catalyst::transport";
    let useDefaultTypePrinterParser = 1;
    let usePropertiesForAttributes = 1;
}

//===----------------------------------------------------------------------===//
// Enums.
//===----------------------------------------------------------------------===//

def Transport_Role : I32EnumAttr<"Role", "transport session role", [
        I32EnumAttrCase<"Controller", 0, "controller">,
        I32EnumAttrCase<"Coprocessor", 1, "coprocessor">
    ]> {
    let cppNamespace = "::catalyst::transport";
}


//===----------------------------------------------------------------------===//
// Types.
//===----------------------------------------------------------------------===//

class Transport_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Transport_Dialect, name, traits> {
    let mnemonic = typeMnemonic;
}

// Opaque session handle, parameterized by its role. The role is a compile-time tag
// that selects which role-specific ops the session may take part in.
def Transport_SessionType : Transport_Type<"Session", "session"> {
    let summary = "An opaque transport session handle, tagged with its role.";
    let parameters = (ins EnumParameter<Transport_Role>:$role);
    let assemblyFormat = "`<` $role `>`";
}

def Transport_TokenType : Transport_Type<"Token", "token"> {
    let summary = "A handle to an in-flight asynchronous step, awaited with transport.barrier.";
}

// Role-constrained session types used by the role-specific ops.
def Transport_ControllerSession : Type<
    CPred<"::llvm::isa<::catalyst::transport::SessionType>($_self) && "
          "::llvm::cast<::catalyst::transport::SessionType>($_self).getRole() == "
          "::catalyst::transport::Role::Controller">,
    "controller transport session">;

def Transport_CoprocessorSession : Type<
    CPred<"::llvm::isa<::catalyst::transport::SessionType>($_self) && "
          "::llvm::cast<::catalyst::transport::SessionType>($_self).getRole() == "
          "::catalyst::transport::Role::Coprocessor">,
    "coprocessor transport session">;

//===----------------------------------------------------------------------===//
// Operation base.
//===----------------------------------------------------------------------===//

// All transport ops perform side-effecting I/O
class Transport_Op<string mnemonic, list<Trait> traits = []> :
    Op<Transport_Dialect, mnemonic, !listconcat(traits, [MemoryEffects<[MemWrite]>])>;

#endif // TRANSPORT_DIALECT
