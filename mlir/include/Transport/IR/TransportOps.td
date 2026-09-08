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
    // peer / oob_port are only needed by transports that run an OOB TCP handshake (RDMA).
    // In-process transports (memcpy) pair on the session key, so both attrs are optional.
    let arguments = (ins Transport_SessionType:$session,
                        OptionalAttr<StrAttr>:$peer,
                        OptionalAttr<UI16Attr>:$oob_port);
    let assemblyFormat = "$session attr-dict `:` qualified(type($session))";
}

def Transport_ConnectAsyncOp : Transport_Op<"connect_async"> {
    let summary = "Non-blocking connect; await the token with transport.await.";
    let arguments = (ins Transport_SessionType:$session,
                        OptionalAttr<StrAttr>:$peer,
                        OptionalAttr<UI16Attr>:$oob_port);
    let results = (outs Transport_TokenType:$token);
    let assemblyFormat = "$session attr-dict `:` qualified(type($session)) `->` type($token)";
}

def Transport_ExchangeKeysOp : Transport_Op<"exchange_keys"> {
    let summary = "Exchange memory-region handles with the peer (blocking).";
    let arguments = (ins Transport_SessionType:$session);
    let assemblyFormat = "$session attr-dict `:` qualified(type($session))";
}

def Transport_ExchangeKeysAsyncOp : Transport_Op<"exchange_keys_async"> {
    let summary = "Non-blocking exchange_keys; await the token with transport.await.";
    let arguments = (ins Transport_SessionType:$session);
    let results = (outs Transport_TokenType:$token);
    let assemblyFormat = "$session attr-dict `:` qualified(type($session)) `->` type($token)";
}

def Transport_AwaitOp : Transport_Op<"await"> {
    let summary = "Await an asynchronous bring-up step.";
    let arguments = (ins Transport_TokenType:$token);
    let assemblyFormat = "$token attr-dict `:` type($token)";
}

def Transport_EstablishChannelOp : Transport_Op<"establish_channel"> {
    let summary = "Arm the channel for the given transport.";
    let description = [{
        Prepares the channel selected by `transport` for the rounds that follow.
        Runs after `exchange_keys` and before `post`/`collect`.
    }];
    let arguments = (ins Transport_SessionType:$session, StrAttr:$transport);
    let assemblyFormat = "$session $transport attr-dict `:` qualified(type($session))";
}

//===----------------------------------------------------------------------===//
// Round setup
//===----------------------------------------------------------------------===//

def Transport_SetMessageSizesOp : Transport_Op<"set_message_sizes"> {
    let summary = "Declare how big a round's request and reply are (controller).";
    let description = [{
        Called once at bring-up. `work_item_idx` selects which of the backend's pre-programmed
        transfer descriptors this describes
    }];
    let arguments = (ins Transport_ControllerSession:$session,
                         DefaultValuedOptionalAttr<I32Attr, "0">:$work_item_idx,
                         I64Attr:$in_bytes, I64Attr:$out_bytes);
    let assemblyFormat = "$session attr-dict `:` qualified(type($session))";
}

def Transport_SetCoprocessorFnOp : Transport_Op<"set_coprocessor_fn"> {
    let summary = "Bind the function the coprocessor runs.";
    let description = [{
        Binds the function the coprocessor applies to received messages before
        replying. `symbol` is the name the coprocessor resolves and invokes; an
        empty symbol selects a default.

        Whether that function is invoked per message or once at start to launch a
        persistent engine belongs to the backend. The runtime
        binds through whichever of the two the backend implements.
    }];
    let arguments = (ins Transport_CoprocessorSession:$session,
                         DefaultValuedStrAttr<StrAttr, "">:$symbol);
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

def Transport_ReplySlotOp : Transport_Op<"reply_slot"> {
    let summary = "This round's slot in the transport-owned reply ring (controller).";
    let description = [{
        The peer writes its reply here. Handing this to `transport.collect`.
        The slot is reused every ring wrap, so a consumer has to read it before the next round.
    }];
    let arguments = (ins Transport_ControllerSession:$session);
    let results = (outs Transport_Buffer:$slot);
    let assemblyFormat = "$session attr-dict `:` qualified(type($session)) `->` type($slot)";
}

def Transport_StagePayloadOp : Transport_Op<"stage_payload"> {
    let summary = "Stage a payload into this round's request slot (controller).";
    let description = [{
        Writes through the backend, which knows how its slot memory has to be written.

        `decoder_id` addresses the round at one of the peer's decoders. It travels in the frame
        beside the payload, so it is staged with the payload rather than given to the send.
    }];
    let arguments = (ins Transport_ControllerSession:$session, Transport_AnyBuffer:$payload,
                         DefaultValuedOptionalAttr<I32Attr, "0">:$decoder_id);
    let assemblyFormat = "$session `,` $payload attr-dict `:` qualified(type($session)) `,` type($payload)";
}

def Transport_PostOp : Transport_Op<"post"> {
    let summary = "Post this round's payload (controller).";
    let description = [{
        Transmits the payload `transport.stage_payload` put in the request slot.
    }];
    let arguments = (ins Transport_ControllerSession:$session,
                         DefaultValuedOptionalAttr<I32Attr, "0">:$work_item_idx);
    let assemblyFormat = "$session attr-dict `:` qualified(type($session))";
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
