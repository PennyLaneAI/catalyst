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

        This is what a Backline compiles to. A [PennyLane](https://docs.pennylane.ai/en/stable/)
        `Placement` is serialized into the `catalyst.backline` module attribute, naming one
        controller and the coprocessors it drives; `inject-transport-session` reads that attribute
        and emits the session bring-up and teardown in terms of these ops,
        `lower-decode-to-transport` turns a decode into a request/reply round over the session, and
        `convert-transport-to-llvm` lowers the result to `__catalyst__transport__*` runtime calls.
        A node here is a Backline participant -- a controller or a coprocessor -- and the
        placement's transport selects which compiled backend carries the traffic.

        A node the placement dispatches to another machine is shipped and launched through the
        `executor` dialect, so a placement that mixes local and dispatched nodes uses both.

        > [!IMPORTANT]
        > The transport dialect is experimental and will not maintain API stability between
        > releases. Use at your own risk.
    }];

    let name = "transport";
    let cppNamespace = "::catalyst::transport";
    let useDefaultTypePrinterParser = 1;
    let useDefaultAttributePrinterParser = 1;
    let usePropertiesForAttributes = 1;
}

//===----------------------------------------------------------------------===//
// Type and attribute base classes.
//===----------------------------------------------------------------------===//
//
// Kept here so TransportTypes.td and TransportAttrDefs.td share them. The defs themselves live in
// those files rather than here: `-gen-dialect-doc` documents whatever attributes, types and enums
// it can reach, and this file is its entry point, so anything left here lands in
// TransportDialect.md ahead of the Types and Attributes sections instead of inside them.

class Transport_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Transport_Dialect, name, traits> {
    let mnemonic = typeMnemonic;
}

class Transport_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<Transport_Dialect, name, traits> {
    let mnemonic = attrMnemonic;
}

//===----------------------------------------------------------------------===//
// Operation base.
//===----------------------------------------------------------------------===//

// All transport ops perform side-effecting I/O
class Transport_Op<string mnemonic, list<Trait> traits = []> :
    Op<Transport_Dialect, mnemonic, !listconcat(traits, [MemoryEffects<[MemWrite]>])>;

#endif // TRANSPORT_DIALECT
