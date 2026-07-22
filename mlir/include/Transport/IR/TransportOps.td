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
// Session creation
//===----------------------------------------------------------------------===//

def Transport_CreateOp : Transport_Op<"create"> {
    let summary = "Create a session from a backend plugin .so with a certain role";
    let description = [{
        Loads the backend `.so` and builds a session. The result type's role
        (`!transport.session<controller|coprocessor>`) selects which factory the
        runtime looks up and constrains the role-specific ops downstream.
    }];
    let arguments = (ins StrAttr:$backend_lib, StrAttr:$config);
    let results = (outs Transport_SessionType:$session);
    let assemblyFormat = "attr-dict `->` type($session)";
}

//===----------------------------------------------------------------------===//
// Bring-up ops
//===----------------------------------------------------------------------===//

def Transport_ConnectOp : Transport_Op<"connect"> {
    let summary = "Bring up the connection to the peer (blocking).";
    let arguments = (ins Transport_SessionType:$session, StrAttr:$peer, I16Attr:$oob_port);
    let assemblyFormat = "$session attr-dict `:` type($session)";
}

def Transport_ConnectAsyncOp : Transport_Op<"connect_async"> {
    let summary = "connect() on a worker; await with transport.barrier.";
    let arguments = (ins Transport_SessionType:$session, StrAttr:$peer, I16Attr:$oob_port);
    let results = (outs Transport_TokenType:$token);
    let assemblyFormat = "$session attr-dict `:` type($session) `->` type($token)";
}

def Transport_ExchangeKeysOp : Transport_Op<"exchange_keys"> {
    let summary = "Exchange region handles with the peer (blocking); result kept in the session.";
    let arguments = (ins Transport_SessionType:$session);
    let assemblyFormat = "$session attr-dict `:` type($session)";
}

def Transport_ExchangeKeysAsyncOp : Transport_Op<"exchange_keys_async"> {
    let summary = "exchange_keys() on a worker; await with transport.barrier.";
    let arguments = (ins Transport_SessionType:$session);
    let results = (outs Transport_TokenType:$token);
    let assemblyFormat = "$session attr-dict `:` type($session) `->` type($token)";
}

def Transport_BarrierOp : Transport_Op<"barrier"> {
    let summary = "Await an async step (connect_async / exchange_keys_async).";
    let arguments = (ins Transport_TokenType:$token);
    let assemblyFormat = "$token attr-dict `:` type($token)";
}

def Transport_EstablishChannelOp : Transport_Op<"establish_channel"> {
    let summary = "Set up the data channel used to transfer payloads each round.";
    let description = [{
        Arms the data channel for the given `data_path`, using this side's
        registered memory region together with the peer's region that
        `exchange_keys` learned earlier (both stored in the session). After this
        the channel is ready for `kick`/`collect`.
    }];
    let arguments = (ins Transport_SessionType:$session, Transport_DataPath:$data_path);
    let assemblyFormat = "$session $data_path attr-dict `:` type($session)";
}

//===----------------------------------------------------------------------===//
// Controller-only ops
//===----------------------------------------------------------------------===//

def Transport_CommitWorkItemOp : Transport_Op<"commit_work_item"> {
    let summary = "Build a work item (I/O sizes) in a slot before kicking rounds.";
    let arguments = (ins Transport_ControllerSession:$session, I32Attr:$work_item_idx,
                         I64Attr:$in_bytes, I64Attr:$out_bytes);
    let assemblyFormat = "$session attr-dict `:` type($session)";
}

def Transport_KickOp : Transport_Op<"kick"> {
    let summary = "Write the payload into the outbound slot and fire one round.";
    let arguments = (ins Transport_ControllerSession:$session, I64:$payload, I32Attr:$work_item_idx);
    let assemblyFormat = "$session `,` $payload attr-dict `:` type($session) `,` type($payload)";
}

//===----------------------------------------------------------------------===//
// Coprocessor-only: bind the coprocessor function (requires a coprocessor session)
//===----------------------------------------------------------------------===//

def Transport_SetCoprocessorFnOp : Transport_Op<"set_coprocessor_fn"> {
    let summary = "Bind the built-in coprocessor function (echo / on-device kernel).";
    let arguments = (ins Transport_CoprocessorSession:$session);
    let assemblyFormat = "$session attr-dict `:` type($session)";
}

//===----------------------------------------------------------------------===//
// Run / collect / teardown
//===----------------------------------------------------------------------===//

def Transport_StartOp : Transport_Op<"start"> {
    let summary = "Start the session (non-blocking; runs until stop()).";
    let arguments = (ins Transport_SessionType:$session);
    let assemblyFormat = "$session attr-dict `:` type($session)";
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
