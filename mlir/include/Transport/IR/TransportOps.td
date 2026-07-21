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

#ifndef TRANSPORT_OPS
#define TRANSPORT_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "Transport/IR/TransportDialect.td"

//===----------------------------------------------------------------------===//
// Bring-up ops
//===----------------------------------------------------------------------===//

def Transport_ControllerCreateOp : Transport_Op<"controller_create"> {
    let summary = "Create a controller session from a named backend plugin .so.";
    let arguments = (ins StrAttr:$backend_lib, StrAttr:$config);
    let results = (outs Transport_SessionType:$session);
    let assemblyFormat = "attr-dict `->` type($session)";
}

def Transport_ConnectOp : Transport_Op<"connect"> {
    let summary = "Bring up the connection to the peer.";
    let arguments = (ins Transport_SessionType:$session, StrAttr:$peer, I16Attr:$oob_port);
    let results = (outs I32:$status);
    let assemblyFormat = "$session attr-dict `:` functional-type($session, $status)";
}

def Transport_ExchangeKeysOp : Transport_Op<"exchange_keys"> {
    let summary = "Exchange local and peer region handles; yields the peer descriptor.";
    let arguments = (ins Transport_SessionType:$session);
    let results = (outs I32:$status, Transport_PeerType:$peer);
    let assemblyFormat = "$session attr-dict `:` type($session) `->` type($peer)";
}

def Transport_EstablishChannelOp : Transport_Op<"establish_channel"> {
    let summary = "Program the data-movement channel from the local + peer regions.";
    let arguments = (ins Transport_SessionType:$session, Transport_PeerType:$peer,
                         I32Attr:$data_path);
    let results = (outs I32:$status);
    let assemblyFormat = "$session `,` $peer attr-dict `:` type($session) `,` type($peer)";
}

def Transport_CommitWorkItemOp : Transport_Op<"commit_work_item"> {
    let summary = "Build a work item (I/O sizes) in a slot before kicking rounds.";
    let arguments = (ins Transport_SessionType:$session, I32Attr:$work_item_idx,
                         I64Attr:$in_bytes, I64Attr:$out_bytes);
    let results = (outs I32:$status);
    let assemblyFormat = "$session attr-dict `:` type($session)";
}

def Transport_StartOp : Transport_Op<"start"> {
    let summary = "Start the session (non-blocking; runs until stop()).";
    let arguments = (ins Transport_SessionType:$session);
    let assemblyFormat = "$session attr-dict `:` type($session)";
}

//===----------------------------------------------------------------------===//
// Per-round ops
//===----------------------------------------------------------------------===//

def Transport_KickOp : Transport_Op<"kick"> {
    let summary = "Write the payload into the outbound slot and fire one round.";
    let arguments = (ins Transport_SessionType:$session, I64:$payload, I32Attr:$work_item_idx);
    let results = (outs I32:$status);
    let assemblyFormat = "$session `,` $payload attr-dict `:` type($session) `,` type($payload)";
}

def Transport_CollectOp : Transport_Op<"collect"> {
    let summary = "Wait for this round's reply and return it as a value.";
    let arguments = (ins Transport_SessionType:$session, I64Attr:$bytes);
    let results = (outs I64:$result);
    let assemblyFormat = "$session attr-dict `:` type($session) `->` type($result)";
}

def Transport_LastRttNsOp : Transport_Op<"last_rtt_ns"> {
    let summary = "Last round-trip time in nanoseconds.";
    let arguments = (ins Transport_SessionType:$session);
    let results = (outs I64:$rtt_ns);
    let assemblyFormat = "$session attr-dict `:` type($session) `->` type($rtt_ns)";
}

//===----------------------------------------------------------------------===//
// Teardown ops
//===----------------------------------------------------------------------===//

def Transport_StopOp : Transport_Op<"stop"> {
    let summary = "Stop the session. Idempotent.";
    let arguments = (ins Transport_SessionType:$session);
    let assemblyFormat = "$session attr-dict `:` type($session)";
}

def Transport_CloseOp : Transport_Op<"close"> {
    let summary = "Close the transport (releases the channel).";
    let arguments = (ins Transport_SessionType:$session);
    let assemblyFormat = "$session attr-dict `:` type($session)";
}

def Transport_DestroyOp : Transport_Op<"destroy"> {
    let summary = "Destroy the session and free it.";
    let arguments = (ins Transport_SessionType:$session);
    let assemblyFormat = "$session attr-dict `:` type($session)";
}

#endif // TRANSPORT_OPS
