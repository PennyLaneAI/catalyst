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

//===----------------------------------------------------------------------===//
// Transport dialect definition.
//===----------------------------------------------------------------------===//

def Transport_Dialect : Dialect {
    let summary = "Runtime transport-layer ops that lower to the __catalyst__transport__* CAPI.";
    let description = [{
        A thin, typed MLIR representation of the Catalyst runtime transport CAPI
        (runtime/include/TransportCAPI.h). One op per controller-side CAPI entry
        point; `convert-transport-to-llvm` lowers each to an `llvm.call` on the
        matching `__catalyst__transport__*` symbol. The coprocessor role is an
        out-of-tree backend loaded by the runtime, so only the controller side is
        emitted here.
    }];

    let name = "transport";
    let cppNamespace = "::catalyst::transport";
    let useDefaultTypePrinterParser = 1;
    let usePropertiesForAttributes = 1;
}

//===----------------------------------------------------------------------===//
// Transport dialect types.
//===----------------------------------------------------------------------===//

class Transport_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Transport_Dialect, name, traits> {
    let mnemonic = typeMnemonic;
}

def Transport_SessionType : Transport_Type<"Session", "session"> {
    let summary = "Opaque transport controller session handle (CatalystTransportSession*).";
}

def Transport_PeerType : Transport_Type<"Peer", "peer"> {
    let summary = "Opaque peer-region descriptor handle (CatalystTransportPeerRef*).";
}

//===----------------------------------------------------------------------===//
// Transport operation base.
//===----------------------------------------------------------------------===//

class Transport_Op<string mnemonic, list<Trait> traits = []> :
    Op<Transport_Dialect, mnemonic, traits>;

#endif // TRANSPORT_DIALECT
