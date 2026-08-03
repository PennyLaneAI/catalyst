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
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "Transport/IR/TransportDialect.td"

// A round's payload or reply buffer: a 1-D memref (buffer form) or tensor (value form).
def Transport_Buffer : MemRefRankOf<[I1, I8, I16, I32, I64, Index], [1]>;
def Transport_TensorBuffer : 1DTensorOf<[I1, I8, I16, I32, I64, Index]>;
def Transport_AnyBuffer : AnyTypeOf<[Transport_Buffer, Transport_TensorBuffer]>;

//===----------------------------------------------------------------------===//
// Session creation
//===----------------------------------------------------------------------===//

def Transport_CreateOp : Transport_Op<"create"> {
    let summary = "Create a transport session for a role.";
    let description = [{
        Creates a session on the backend selected by `backend_lib` and `config`; the
        result type's role picks controller or coprocessor. When `key` is set, the
        session is registered under it so `transport.get_session` can resolve it from
        another function.
    }];
    let arguments = (ins StrAttr:$backend_lib, StrAttr:$config,
                         DefaultValuedStrAttr<StrAttr, "">:$key);
    let results = (outs Transport_SessionType:$session);
    let assemblyFormat = "attr-dict `->` qualified(type($session))";
}

//===----------------------------------------------------------------------===//
// Connection bring-up
//===----------------------------------------------------------------------===//

def Transport_ConnectOp : Transport_Op<"connect"> {
    let summary = "Connect to the peer (blocking).";
    let arguments = (ins Transport_SessionType:$session, StrAttr:$peer, I16Attr:$oob_port);
    let assemblyFormat = "$session attr-dict `:` qualified(type($session))";
}

def Transport_ConnectAsyncOp : Transport_Op<"connect_async"> {
    let summary = "Non-blocking connect; await the token with transport.barrier.";
    let arguments = (ins Transport_SessionType:$session, StrAttr:$peer, I16Attr:$oob_port);
    let results = (outs Transport_TokenType:$token);
    let assemblyFormat = "$session attr-dict `:` qualified(type($session)) `->` type($token)";
}

def Transport_ExchangeKeysOp : Transport_Op<"exchange_keys"> {
    let summary = "Exchange memory-region handles with the peer (blocking).";
    let arguments = (ins Transport_SessionType:$session);
    let assemblyFormat = "$session attr-dict `:` qualified(type($session))";
}

def Transport_ExchangeKeysAsyncOp : Transport_Op<"exchange_keys_async"> {
    let summary = "Non-blocking exchange_keys; await the token with transport.barrier.";
    let arguments = (ins Transport_SessionType:$session);
    let results = (outs Transport_TokenType:$token);
    let assemblyFormat = "$session attr-dict `:` qualified(type($session)) `->` type($token)";
}

def Transport_BarrierOp : Transport_Op<"barrier"> {
    let summary = "Await an asynchronous bring-up step.";
    let arguments = (ins Transport_TokenType:$token);
    let assemblyFormat = "$token attr-dict `:` type($token)";
}

def Transport_EstablishChannelOp : Transport_Op<"establish_channel"> {
    let summary = "Arm the data channel for the given data path.";
    let description = [{
        Prepares the data channel selected by `data_path` for the rounds that follow.
        Runs after `exchange_keys` and before `kick`/`collect`.
    }];
    let arguments = (ins Transport_SessionType:$session, StrAttr:$data_path);
    let assemblyFormat = "$session $data_path attr-dict `:` qualified(type($session))";
}

//===----------------------------------------------------------------------===//
// Round setup
//===----------------------------------------------------------------------===//

def Transport_CommitWorkItemOp : Transport_Op<"commit_work_item"> {
    let summary = "Declare a round's request/reply byte sizes (controller).";
    let arguments = (ins Transport_ControllerSession:$session, I32Attr:$work_item_idx,
                         I64Attr:$in_bytes, I64Attr:$out_bytes);
    let assemblyFormat = "$session attr-dict `:` qualified(type($session))";
}

def Transport_SetCoprocessorFnOp : Transport_Op<"set_coprocessor_fn"> {
    let summary = "Bind the function the coprocessor runs.";
    let description = [{
        Binds the function the coprocessor applies to received messages before
        replying. `symbol` is the name the coprocessor resolves and invokes; an
        empty symbol selects a default.

        `convention` says which calling convention `symbol` follows:

        * `0` (default) - per-message: a host function invoked once per received
          message.
        * `1` - launch-once: invoked once at start to launch a persistent engine
          that consumes messages itself (e.g. a device kernel).
    }];
    let arguments = (ins Transport_CoprocessorSession:$session,
                         DefaultValuedStrAttr<StrAttr, "">:$symbol,
                         DefaultValuedOptionalAttr<I32Attr, "0">:$convention);
    let assemblyFormat = "$session attr-dict `:` qualified(type($session))";
}

//===----------------------------------------------------------------------===//
// Execution
//===----------------------------------------------------------------------===//

def Transport_StartOp : Transport_Op<"start"> {
    let summary = "Start the session; runs until stop (non-blocking).";
    let arguments = (ins Transport_SessionType:$session);
    let assemblyFormat = "$session attr-dict `:` qualified(type($session))";
}

def Transport_KickOp : Transport_Op<"kick"> {
    let summary = "Send a payload as one round's request (controller).";
    let arguments = (ins Transport_ControllerSession:$session, Transport_AnyBuffer:$payload,
                         I32Attr:$work_item_idx,
                         DefaultValuedOptionalAttr<I32Attr, "0">:$decoder_id);
    let assemblyFormat = "$session `,` $payload attr-dict `:` qualified(type($session)) `,` type($payload)";
}

def Transport_CollectOp : Transport_Op<"collect"> {
    let summary = "Receive the current round's reply.";
    let description = [{
        Receives the current round's reply in one of two forms: a value form that
        returns it as a tensor `$result`, or a destination-passing form that writes
        it into the `$dest` buffer. The buffer shape gives the expected reply size.
    }];
    let arguments = (ins Transport_SessionType:$session, Optional<Transport_Buffer>:$dest);
    let results = (outs Optional<Transport_TensorBuffer>:$result);
    let assemblyFormat = [{
        $session (`,` $dest^)? attr-dict `:` qualified(type($session))
        (`,` type($dest)^)? (`->` type($result)^)?
    }];
}

def Transport_LastRttNsOp : Transport_Op<"last_rtt_ns"> {
    let summary = "Round-trip time of the last round, in nanoseconds.";
    let arguments = (ins Transport_SessionType:$session);
    let results = (outs I64:$rtt_ns);
    let assemblyFormat = "$session attr-dict `:` qualified(type($session)) `->` type($rtt_ns)";
}

//===----------------------------------------------------------------------===//
// Teardown
//===----------------------------------------------------------------------===//

def Transport_StopOp : Transport_Op<"stop"> {
    let summary = "Stop the session (idempotent).";
    let arguments = (ins Transport_SessionType:$session);
    let assemblyFormat = "$session attr-dict `:` qualified(type($session))";
}

def Transport_DestroyOp : Transport_Op<"destroy"> {
    let summary = "Destroy the session and release it.";
    let arguments = (ins Transport_SessionType:$session);
    let assemblyFormat = "$session attr-dict `:` qualified(type($session))";
}

//===----------------------------------------------------------------------===//
// Session resolution
//===----------------------------------------------------------------------===//

def Transport_GetSessionOp : Transport_Op<"get_session"> {
    let summary = "Resolve a session by role and key.";
    let description = [{
        Returns the session that `transport.create` registered under (role, `key`),
        where the role is taken from the result type. Lets a session created in one
        function be used in another without passing it as a value.
    }];
    let arguments = (ins DefaultValuedStrAttr<StrAttr, "">:$key);
    let results = (outs Transport_SessionType:$session);
    let assemblyFormat = "attr-dict `:` qualified(type($session))";
}

#endif // TRANSPORT_OPS
