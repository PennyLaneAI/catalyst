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

        A session's endpoints are a controller and a coprocessor, described by the
        `#transport.backline` attribute a module carries as `catalyst.backline`.
        Every op here lowers to a `__catalyst__transport__*` runtime call.
    }];

    let name = "transport";
    let cppNamespace = "::catalyst::transport";
    let useDefaultTypePrinterParser = 1;
    let useDefaultAttributePrinterParser = 1;
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
    let summary = "A handle to an in-flight asynchronous step, awaited with transport.await.";
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
// Attributes.
//===----------------------------------------------------------------------===//

class Transport_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<Transport_Dialect, name, traits> {
    let mnemonic = attrMnemonic;
}

// A string field that reads as empty when omitted, so consumers need no null checks.
class Transport_StrField<string desc = ""> : DefaultValuedParameter<"mlir::StringAttr",
                                                "mlir::StringAttr::get($_ctxt, \"\")", desc>;

def Transport_NodeAttr : Transport_Attr<"Node", "node"> {
    let summary = "A backline participant: a controller or a coprocessor.";

    let parameters = (ins
        Transport_StrField<"this node's name; the session registry key when non-empty">:$name,
        Transport_StrField<"the peer's address for the out-of-band handshake">:$peer,
        OptionalParameter<"mlir::IntegerAttr",
                          "TCP port for the out-of-band handshake">:$oob_port,
        Transport_StrField<"backend plugin the runtime dlopens for this node">:$backend_lib,
        Transport_StrField<"backend configuration string, passed through verbatim">:$config,
        Transport_StrField<"target triple this node's code is compiled for">:$triple,
        Transport_StrField<"executor address when this node runs out of process">:$address,
        Transport_StrField<"coprocessor function symbol the backend binds">:$symbol,
        OptionalParameter<"mlir::BoolAttr",
                          "whether this node's code is dispatched to an executor rather than "
                          "run in the compiling process">:$out_of_process,
        OptionalParameter<"mlir::IntegerAttr", "request size in bytes for one round">:$in_bytes,
        OptionalParameter<"mlir::IntegerAttr", "reply size in bytes for one round">:$out_bytes,
        OptionalParameter<"mlir::IntegerAttr",
                          "index of this node's work item within the round">:$work_item_idx
    );

    let assemblyFormat = "`<` struct(params) `>`";

    let extraClassDeclaration = [{
        /// Session registry key: `name` when set and non-empty, else `fallback`.
        mlir::StringAttr keyOr(llvm::StringRef fallback) const;

        /// Whether this node's code is dispatched to an executor rather than run in the
        /// compiling process.
        bool isOutOfProcess() const;

        /// Integer field accessors.
        int64_t oobPort() const;
        int64_t inBytes() const;
        int64_t outBytes() const;
        int64_t workItemIdx() const;
    }];
}

def Transport_BacklineAttr : Transport_Attr<"Backline", "backline"> {
    let summary = "A backline placement: one controller and the coprocessors it drives.";

    let parameters = (ins
        AttrParameter<"mlir::StringAttr",
                      "transport carrying the traffic, selecting the compiled backend">:$transport,
        AttrParameter<"::catalyst::transport::NodeAttr",
                      "the controller node that drives the session">:$controller,
        OptionalArrayRefParameter<"::catalyst::transport::NodeAttr",
                                  "the coprocessor nodes the controller drives">:$coprocessors
    );

    let assemblyFormat = [{ `<` `transport` `=` $transport `,` `controller` `=` $controller
                            (`,` `coprocessors` `=` `[` $coprocessors^ `]`)? `>` }];
    let genVerifyDecl = 1;
}

//===----------------------------------------------------------------------===//
// Operation base.
//===----------------------------------------------------------------------===//

// All transport ops perform side-effecting I/O
class Transport_Op<string mnemonic, list<Trait> traits = []> :
    Op<Transport_Dialect, mnemonic, !listconcat(traits, [MemoryEffects<[MemWrite]>])>;

#endif // TRANSPORT_DIALECT
