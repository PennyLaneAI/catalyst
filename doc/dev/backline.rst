Backline: heterogeneous compilation and remote execution
########################################################

.. note::

    This page describes how Backline is built inside Catalyst: the compiler passes, dialects, and
    runtime that implement it. If you are looking for how to *use* Backline, start with the
    `Backline demo <https://pennylane.ai/demos/backline>`_ and the
    `Backline module documentation`_, which cover the user-facing API. The
    `Backline repository <https://github.com/PennyLaneAI/backline>`_ holds runnable demos,
    benchmarks, and the cross-build system for remote hardware.

.. warning::

    Backline is experimental and under active development. Interfaces described here can change
    between releases. See `Current limitations`_.

Summary
=======

Catalyst normally compiles a :func:`~.qjit` decorated QNode and the functions around it into a
single object that runs in the originating Python process. Backline expands on this, allowing
selected kernels to be cross-compiled into standalone object files for other target systems and
shipped to a separate executor process, so that they run on different hosts, accelerators, or
custom devices. Arguments and results are marshalled between the compiler host and the
participating executors, with the entire interaction represented explicitly in the compiler IR.

This capability builds a platform to target next-generation, fault-tolerant workflows such as
real-time quantum error correction, where a controller and one or more coprocessors (a CPU, a GPU,
or an FPGA-based decoder) cooperate over a low-latency transport.

Installation
============

The Backline components are off by default in a Catalyst build. To build them, enable the
following build flags:

``ENABLE_TRANSPORT``
    Builds ``rt_transport``, the backend-agnostic transport loader, along with the in-tree
    transport backends. ``ENABLE_TRANSPORT_FPGA=ON`` adds the FPGA controller backends and
    requires ``ENABLE_TRANSPORT=ON``.

``ENABLE_EXECUTOR``
    Builds ``rt_executor`` and the ``catalyst-executor`` server binary, which receives, maps, and
    invokes cross-compiled objects.

A source build with both enabled:

.. code-block:: bash

    git clone --recurse-submodules --shallow-submodules \
        https://github.com/PennyLaneAI/catalyst.git
    cd catalyst
    python3 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    export ENABLE_TRANSPORT=ON ENABLE_EXECUTOR=ON
    export LLVM_TARGETS_TO_BUILD="host;AArch64;X86"
    make all

``LLVM_TARGETS_TO_BUILD`` has to name every architecture you cross-compile a kernel for, since
:ref:`cross-compile-targets <backline-pipeline>` emits object files through the same LLVM build.
``host`` alone is enough when every node is the same architecture as the machine you compile on.

The RDMA transport backends need ``libibverbs-dev`` and a verbs device that ``ibv_devices``
lists. Soft-RoCE counts, so an RDMA NIC is not required to run them. GPU coprocessor backends need
ROCm on the machine that builds them and the GPU on the machine that runs them.

Transport backends are not shipped in the Catalyst wheel. An out-of-tree backend's libraries are
found through ``CATALYST_TRANSPORT_PATH``, a ``:``-separated list of directories searched ahead of
the installation's own library directory. The backend itself is registered either by a
``catalyst_transport_backends.json`` manifest in one of those directories or by a
``catalyst.transport_backends`` entry point of an installed package. Both give a list of
:func:`catalyst.backline.register_backend` arguments, for example:

.. code-block:: json

    [{"name": "example_backend", "transport": "example", "hardware": "cpu",
      "roles": ["controller", "coprocessor"], "max_payload_bytes": 1024}]

Running against remote hardware also needs a deployed bundle on each remote machine, holding the
``catalyst-executor`` binary and the runtime libraries the dispatched code loads. The
`Backline repository <https://github.com/PennyLaneAI/backline>`_ documents that in full:
`INSTALL.md <https://github.com/PennyLaneAI/backline/blob/main/INSTALL.md>`_ covers system
packages, ROCm, and the four hardware tiers, and ``config/xbuild`` provides the cross-build
makefile that produces the bundles.

Overview
========

The stack is organized into three interacting layers, mirroring the overall Catalyst architecture:

**Frontend**
    A PennyLane-facing API (`pennylane.backline`_) for declaring where each part of a workload
    runs. A `Placement`_ names a controller, its coprocessors, and the transport between them.
    ``catalyst.backline`` serializes that placement onto the root module as the
    ``catalyst.backline`` attribute and inserts the passes that lower it.

**Compiler core**
    MLIR passes that cross-compile annotated kernels into standalone objects, together with two
    dedicated dialects. The :doc:`transport dialect <../code/dialects/transport>` models the
    data-movement session between a controller and a coprocessor, and the
    :doc:`executor dialect <../code/dialects/executor>` models host-to-executor kernel dispatch.
    Both lower to concrete runtime calls.

**Runtime**
    Runtime components that establish a session with a given executor process, ship compiled
    kernels, marshal arguments and results, and manage session lifetime.

The end-to-end flow, from a user program to remote execution, proceeds as follows:

#. A workload is declared along with a placement (a controller, its coprocessors, and a
   transport) and executed under :func:`~.qjit` on a `Backline`_ device.
#. Before compilation, each node's executor is settled on an address, and the placement is
   serialized onto the root module as ``catalyst.backline``.
#. ``inject-transport-session`` reads ``catalyst.backline`` and emits the transport session's
   bring-up and teardown into the host entry function.
#. Kernels tagged for a separate target are preserved as nested modules instead of being inlined,
   and bufferization lowers their tensor operands and results to memrefs. ``decode`` operations
   become transport rounds over those buffers.
#. ``cross-compile-targets`` emits a standalone object file per target and reduces each nested
   module to external declarations of its entry functions.
#. ``dispatch-executor-targets`` rewrites the host-side calls to remote kernels into executor
   dialect operations.
#. The executor and transport dialects are lowered to LLVM IR as calls into their runtimes.
#. At execution time, the runtime opens a session with each executor, ships each object, and
   invokes the kernels. Results return to the host, meaning the originating Python process.

A node that runs in the originating process does not require dispatch, since its code is called
directly. A node with a given executor is dispatched: its kernel is cross-compiled, shipped to the
``catalyst-executor`` process at that node's address, and invoked there. A node is dispatched when
it carries either an ``executor`` or ``executor_options``, and a node marked ``remote`` has to
carry one, since there is no other way to reach another machine.

The ``catalyst.backline`` attribute
===================================

A placement reaches the compiler as a single attribute on the root module. Every Backline pass
reads it, so it is the contract between the PennyLane frontend and the compiler core.
``catalyst.backline`` serializes a placement into a ``#transport.backline`` attribute holding the
transport name, one controller node, and zero or more coprocessor nodes:

.. code-block:: mlir

    module attributes {catalyst.backline = #transport.backline<
        transport = "rdma",
        controller = #transport.node<peer = "127.0.0.1", oob_port = 18590 : i16,
                                     in_bytes = 8 : i64, out_bytes = 8 : i64>>} {
      ...
    }

Every field on ``#transport.node`` is optional, and a node carries only what its role and
placement imply. A controller carries the message sizes, a coprocessor carries the decode symbol,
and a node running in the compiling process carries neither an address nor a triple.

.. list-table::
    :widths: 22 78
    :header-rows: 1

    * - Field
      - Meaning
    * - ``name``
      - This node's name. When non-empty it is the session registry key.
    * - ``peer``
      - The peer's address for the out-of-band handshake.
    * - ``oob_port``
      - TCP port for the out-of-band handshake.
    * - ``backend_lib``
      - The transport backend plugin the runtime opens for this node.
    * - ``config``
      - Backend configuration string, passed through verbatim.
    * - ``triple``
      - Target triple this node's code is cross-compiled for.
    * - ``address``
      - Executor address, set when the node runs out of process.
    * - ``symbol``
      - The coprocessor function symbol the backend binds.
    * - ``out_of_process``
      - Whether this node's code is dispatched to an executor rather than run in the compiling
        process.
    * - ``in_bytes`` / ``out_bytes``
      - Request and reply size in bytes for one round. Controller only.
    * - ``work_item_idx``
      - Index of this node's work item within the round.

Only ``peer`` and ``oob_port`` describe a network handshake, and transports that pair in process,
such as ``memcpy``, leave both unset and match on the session key instead.

Transport layer
===============

The transport layer defines the contract a Backline backend implements to allow data movement
between a controller and a coprocessor. It is deliberately abstract, so that a range of data
planes and memory hierarchies, from plain memory copies to low-latency RDMA engines, work behind
the same frontend.

A transport is characterized by:

- a **data path**, selecting which engine issues the transfer, such as CPU-posted verbs, hardware
  NIC engines on an FPGA, or a GPU kernel posting work just in time,
- a **memory kind**, selecting the allocation and registration path, such as host DRAM, GPU
  memory, or FPGA memory, and
- a **role**, distinguishing the controller which drives requests from the coprocessor which
  handles them, for instance by running a decoder kernel.

Catalyst picks a node's backend from a registry. A backend qualifies when it implements the
placement's transport on the node's hardware, provides the node's role and carries the placement's
message sizes. A default backend is preferred, and a node can name another with
``backend=name``. The controller's and coprocessors' backends must share a frame
layout. The backend's library is named ``libcatalyst_transport_<backend>_<role>.so``.

The built-in backends are:

* ``rdma``: ``cpu_verbs`` on ``cpu``, ``gpu_verbs`` on ``gpu``, and ``hwhs`` on ``fpga``.
  ``fpga_verbs`` is also available on ``fpga`` when named.
* ``memcpy``: ``memcpy`` on ``cpu`` and ``memcpy_gpu`` on ``gpu``.

A backend is a shared library implementing ``ControllerSession`` or ``CoprocessorSession`` from
``runtime/include/Transport.hpp`` and exporting the matching factory symbol declared in
``runtime/include/TransportBackend.h``. It also reports the transport ABI version it was built for
and its capabilities (roles, memory kinds, frame modes, message sizes and coprocessor conventions)
through ``CatalystTransportAbiVersion`` and ``CatalystTransportQueryCaps``, which the
``CATALYST_TRANSPORT_BACKEND_CAPS`` macro defines. The loader refuses a backend built for a newer
ABI, or asked for a role it does not report. A backend exporting neither symbol is loaded as
version 1, limited to 16 B frames. The session interface only grows, so a backend keeps working
with later runtimes. A session follows a strict lifecycle:

``connect``
    Bring up the connection and the out-of-band channel used to arrange direct data transfers.
``alloc_memory``
    Allocate and register a memory region of a given memory kind on this node.
``exchange_keys``
    Swap region handles with the peer, so each side can see into the other's registered memory.
    The local reply region is provisioned here on first use.
``establish_channel``
    Program the data movement for the session over the exchanged regions.
``set_message_sizes``
    Declare how large a round's request and reply are. Controller only.
``set_coprocessor_fn``
    Register the per-round function the coprocessor runs, such as a decoder. Coprocessor only.
``start`` / ``kick`` / ``collect`` / ``stop``
    Run the engine, fire each round, gather its reply, and shut down.

Each round's request and reply travel as a **frame**, written into a slot of a ring on the
receiving side: over RDMA in the receiver's registered memory, over ``memcpy`` in the
coprocessor's own memory. A ring has ``frame_slots`` slots, 256 unless a node's backend config sets
``frame_slots=<n>``, a power of two, which the compiler then passes to every node; FPGA controllers
are only tested with 256. Round ``n`` uses slot ``n mod frame_slots``. Over RDMA, ends whose frame
configs differ fail to connect. Every transport uses the same two frame layouts, and the compiler picks one per placement from its
message sizes: the 16 B frame, BL16, when requests and replies fit in 8 B, and BLV otherwise.

.. list-table:: BL16, in 64 B slots
    :header-rows: 1
    :widths: 15 85

    * - Bytes
      - Field
    * - 0–7
      - The payload, zero-padded: the staged syndrome, or the coprocessor's reply.
    * - 8–11
      - ``fn_id``: which coprocessor function handles the round, the ``decoder_id`` staged with
        the payload (see `The transport dialect`_). A reply echoes the request's.
    * - 12–15
      - ``seq``: the round's number plus one.

.. list-table:: BLV, filling a slot of ``stride`` bytes
    :header-rows: 1
    :widths: 15 85

    * - Bytes
      - Field
    * - 0–31
      - Header: ``magic`` (``"BLV1"``), ``version`` (1), ``flags`` (0), ``fn_id``, ``seq``,
        ``round`` (0), ``len`` (payload bytes) and ``t_ptp_ns`` (0), as u32 fields apart from
        the u16 ``version`` and ``flags`` and the u64 ``t_ptp_ns``.
    * - 32–
      - The payload, ``len`` bytes. A reply's payload area is zero-filled past what the
        coprocessor function wrote.
    * - last 4
      - The completion word: ``seq`` again.

A BLV slot's ``stride`` is the smallest power of two of at least 64 B that holds the 32 B header,
the larger of the placement's request and reply, and the 4 B completion word. The compiler writes
``frame=blv;frame_payload=<bytes>`` into both ends' backend configs, and every backend derives the
same ring from them; these keys are reserved, so a node's own config cannot set them. A placement
whose messages fit in 8 B gets no keys and uses BL16.

A sender writes the whole frame, or the whole slot for BLV, in one in-order transfer, and the word
a receiver polls (BL16's ``seq``, BLV's completion word) sits at a fixed offset and lands last. A
receiver reads the payload only once that word matches the round it expects.

A per-message coprocessor function, ``CatalystCoprocessorFn`` in ``runtime/include/TransportABI.h``,
is called with the payload, its length and the ``fn_id``, whatever the transport and frame layout,
so one function can serve several decoders without reading frames. In-tree backends build and
read frames through ``runtime/lib/transport/common/interface/Frame.hpp``. An out-of-tree backend
lays them out as above, since that header is not installed.

A GPU coprocessor running a persistent kernel hands it both rings through ``CoprocLaunchDesc``
(``runtime/include/Transport.hpp``): the request ring and a host-mapped reply ring in the same
layout, with the frame mode and slot stride. The kernel answers each request with a reply frame in
the same slot, carrying the request's ``fn_id``, which the host side passes on: gpu_verbs sends it
to the controller, and ``memcpy_gpu`` hands it to the controller's thread directly. A launcher
built for this layout exports ``<symbol>_info`` reporting
``CATALYST_COPROCESSOR_LAUNCHER_MIN_ABI_VERSION`` or later; the runtime refuses any other, whose
kernel would never publish a reply it recognises. A launcher also refuses, when the session starts,
a layout its kernel cannot read. Catalyst's own GPU kernels take messages of any size, their
decoders taking the payload, its length and the ``fn_id`` like a per-message function's.
PennyLane's Triton decoders decode one 8 B word, so their launcher refuses anything but the 16 B
frame.

``cpu_verbs``, ``gpu_verbs`` and the ``memcpy`` backends set no limit on message size. Over RDMA,
BLV has been tested up to 1024 B each way on Soft-RoCE, and up to 256 KiB on a ConnectX-7. A frame
larger than the path MTU spans several packets (a 1024 B payload already fills a 2048 B slot), and
its completion word landing last then relies on the NIC placing an RDMA write's bytes in order,
which common NICs do but the RDMA specification does not guarantee. The slot count does not change
with the stride, so memory grows with message size: a 1 MiB message makes each ring 512 MiB at the
default 256 slots.

The transport dialect
=====================

The :doc:`transport dialect <../code/dialects/transport>` exposes this lifecycle as typed
operations, so a session's bring-up, its rounds, and its teardown are all visible in the IR. Every
operation lowers to a ``__catalyst__transport__*`` runtime call through
``convert-transport-to-llvm``.

A session is an opaque ``!transport.session<role>`` handle, where the role is a compile-time tag
that is part of the type. Role-specific operations constrain their operand to the matching role,
so the verifier rejects a ``transport.stage_payload`` on a coprocessor session rather than leaving
it to fail at runtime.

The operations fall into four groups:

**Bring-up**
    ``transport.create`` instantiates a session on the backend named by ``backend_lib``, with the
    result type's role selecting controller or coprocessor. ``transport.connect`` reaches the
    peer, ``transport.exchange_keys`` swaps region handles, and ``transport.establish_channel``
    programs the data movement. ``transport.set_message_sizes`` (controller) and
    ``transport.set_coprocessor_fn`` (coprocessor) settle the per-round contract.
    ``connect`` and ``exchange_keys`` each have an ``_async`` variant returning a
    ``!transport.token`` that ``transport.await`` waits on, so bring-up can overlap with other
    work.

**Rounds**
    ``transport.start`` runs the engine. Each round is ``transport.stage_payload`` to write the
    request slot, ``transport.post`` to transmit it, and ``transport.collect`` to receive the
    reply. ``transport.reply_slot`` hands back this round's slot in the transport-owned reply
    ring, and ``transport.last_rtt_ns`` reports the previous round's round-trip time.

**Teardown**
    ``transport.stop`` halts the session and is idempotent. ``transport.destroy`` releases it.

**Resolution**
    ``transport.get_session`` returns the session that ``transport.create`` registered under a
    ``(role, key)`` pair. Bring-up is emitted into ``setup()`` and teardown into ``teardown()``,
    and the rounds sit in the compiled kernel, so the session is resolved by key in each rather
    than threaded between them as a value.

Bring-up for an RDMA controller, as ``inject-transport-session`` emits it:

.. code-block:: mlir

    %s = transport.create {backend_lib = "libcatalyst_transport_cpu_verbs_controller.so",
                           config = "", key = "controller"} -> !transport.session<controller>
    transport.connect %s {peer = "127.0.0.1", oob_port = 18590 : ui16}
        : !transport.session<controller>
    transport.exchange_keys %s : !transport.session<controller>
    transport.establish_channel %s "rdma" : !transport.session<controller>
    transport.set_message_sizes %s {in_bytes = 8 : i64, out_bytes = 8 : i64}
        : !transport.session<controller>
    transport.start %s : !transport.session<controller>

And one round, resolving the session by key:

.. code-block:: mlir

    %s = transport.get_session {key = "controller"} : !transport.session<controller>
    transport.stage_payload %s, %syndrome {decoder_id = 0 : i32}
        : !transport.session<controller>, memref<2xi1>
    transport.post %s : !transport.session<controller>
    transport.collect %s, %correction : !transport.session<controller>, memref<1xindex>

``transport.collect`` has two forms. The one above is destination-passing, writing into a caller
supplied buffer whose shape gives the expected reply size. The value form instead returns the
reply as a tensor.

Encoding and the decode path
============================

A placement naming a ``qec_code`` asks for its circuits to run encoded in that code, which is what
turns a quantum circuit into transport traffic. ``catalyst.backline`` maps the code name onto an
encoding chain that runs per QNode at trace time. For ``steane`` that is
``convert-quantum-to-qecl``, ``symbol-dce``, ``inject-noise-to-qecl``, ``convert-qecl-to-qecp``,
and ``convert-qecp-to-quantum``.

Encoding expands each logical gate into its physical circuit with a round of error correction
around it: extract the stabilizers, decode the syndrome, apply the correction. The decode in each
of those rounds becomes a ``qecp.decode_esm_css`` operation, which takes a syndrome measurement
over a CSS code's Tanner graph and returns the index in the codeblock where the error occurred, or
``-1`` when no correctable error was detected.

After bufferization, ``lower-decode-to-transport`` replaces each ``qecp.decode_esm_css`` with a
transport round over its buffers, which is where the decode leaves the process. The pass is a
no-op unless the module carries a ``catalyst.backline`` attribute declaring at least one
coprocessor, so the same encoded program compiles without Backline by keeping its decode local.

A CSS code checks X and Z parity separately, and the two can be decoded by different peer-side
decoders. ``qecp.decode_esm_css`` records which family a syndrome came from in its ``check_type``
attribute, and the pass maps that onto the ``decoder_id`` carried by ``transport.stage_payload``:
``"x"`` becomes ``0`` and ``"z"`` becomes ``1``. The id travels in the frame beside the payload,
which is why it is settled when the payload is staged rather than passed to the post.

The executor dialect
====================

Host-to-executor interactions are represented explicitly in the IR by the
:doc:`executor dialect <../code/dialects/executor>`, rather than being hidden inside the runtime.
This keeps the communication visible to compiler passes and analyses. The dialect operates on an
``!executor.session`` handle produced by ``executor.open`` and threaded through the operations
that use it:

.. list-table::
    :widths: 25 75
    :header-rows: 1

    * - Operation
      - Description
    * - ``executor.open``
      - Establish a session with the executor at a given address and return a handle to it.
        Subsequent operations targeting the same address reuse this session.
    * - ``executor.send_binary``
      - Send a compiled kernel object file over the session. The executor loads it and exposes its
        symbols.
    * - ``executor.launch``
      - Invoke a sent kernel. The operands and results mirror the host-side call to the kernel.
    * - ``executor.launch_async``
      - Start a kernel that takes no operands and returns no results, without waiting for it to
        finish. Returns an ``!executor.token`` naming the launch.
    * - ``executor.await``
      - Block until the ``executor.launch_async`` that produced a token has finished. Other
        launches on the same session keep running.
    * - ``executor.call``
      - Invoke an arbitrary symbol in an already-loaded shared library. ``num_input_args`` marks
        where the input operands end and the output buffers begin.
    * - ``executor.close``
      - Release the session and its executor-side resources.

An illustrative fragment, with operands as memrefs after bufferization:

.. code-block:: mlir

    %session = executor.open("decoder-host:9000") : !executor.session
    executor.send_binary %session("/tmp/workspace/decode.o") : !executor.session
    %out = executor.launch %session("decode", "/tmp/workspace/decode.o")(%in)
        : !executor.session, (memref<8xi8>) -> memref<8xi8>
    executor.close %session : !executor.session

The dialect is lowered to LLVM IR by the ``convert-executor-to-llvm`` pass, which rewrites each
operation into a ``__catalyst__executor__*`` C-ABI call into the executor runtime, materializing
the string globals, memref descriptors, and per-argument metadata the runtime expects.

.. _backline-pipeline:

Compilation pipeline
====================

Two passes drive the target workflow. The compiler driver runs them as their own pipeline after
bufferization, and only when a compilation workspace is set, since the object files are written
into it.

``cross-compile-targets``
    Turns each nested module tagged ``catalyst.target`` into a standalone object file, records
    the path on the module as ``catalyst.object_file``, and reduces the module to external
    declarations of its entry functions.

``dispatch-executor-targets``
    Ships each dispatched object to its executor and rewrites the host-side
    ``catalyst.launch_kernel`` calls into ``executor.launch``.

A placement adds three further passes: ``inject-transport-session``,
``lower-decode-to-transport``, and ``convert-transport-to-llvm``. A placement naming a
``qec_code`` also registers the QEC encoding passes.

Runtime and execution model
===========================

At execution time, the runtime backs the executor operations emitted by the compiler. The host
process, meaning the ``qjit`` compiled program, and the executor processes communicate over a TCP
socket using LLVM's ORC execution process control (EPC) as the wire format, by way of
``SimpleRemoteEPC``. Each session owns its channel and an isolated linking layer, and each shipped
kernel object is loaded into its own ``JITDylib`` so that kernels reusing the same entry-point
symbol names do not collide.

.. note::

    The EPC protocol itself has no standalone specification. LLVM's
    `ORC design document <https://llvm.org/docs/ORCv2.html>`_ describes ``ExecutorProcessControl``
    and the remote JIT architecture at a high level, and the details live in the LLVM source tree
    under ``llvm/include/llvm/ExecutionEngine/Orc/`` (``SimpleRemoteEPC.h``,
    ``Shared/SimpleRemoteEPCUtils.h``) and in the ``llvm-jitlink`` and ``llvm-jitlink-executor``
    tools.

The runtime walks through the stages the dialect exposes: open the session, ship the binaries,
launch or call the kernels, collect the results, and close the session. Robustness measures ensure
that a session does not block indefinitely when the peer is not a live executor, and that closing
a session always terminates cleanly.

An executor is deployed and managed from Python by ``catalyst.Executor``, which runs a
``catalyst-executor`` process either as a local subprocess or on another host over SSH, and
reports the ``host:port`` address the compiled program dispatches to. An executor loads the plugin
libraries the dispatched code needs, which always include ``librt_transport.so`` and
``librt_capi.so``, plus the coprocessor's decode library and the controller's device runtime.

Inspecting a compiled program
=============================

Compilation writes its intermediate files into a workspace directory named after the compiled
function. By default this is a temporary directory that is discarded afterwards, but
``keep_intermediate=True`` puts it in the current working directory and keeps it. Each target
module's own lowering is written to a subdirectory there, which is the only way to see what a
target module became: it is lowered by its own nested pipeline rather than the one the host
program runs through.

``catalyst.Executor`` writes a host-side log per launch, named
``catalyst-executor[-<name>]-<host>-<timestamp>.log``. This is the only place a failed plugin
load is reported, so it is the first thing to read when a dispatched kernel cannot resolve a
symbol.

Transport backend errors are written to the standard error of whichever process runs the node,
prefixed with ``[transport]``. For a node running in the compiling process that is the calling
program's own output, and for a dispatched node it is the executor log above.

Current limitations
===================

As an experimental feature, several components are provisional. Each of these can change as the
platform is built out.

**Single remote memory slab.**
    A session reserves a single slab on the remote for the sections of every object it loads
    (``.text``, ``.rodata``, and so on), released when the session is destroyed. It is 1 GB unless
    the ``CATALYST_REMOTE_SLAB_BYTES`` environment variable of the process opening the session sets
    another size in bytes.

**Transport reply region.**
    A transport session asks its backend for a 16 KB local reply region at ``exchange_keys``. A
    backend whose peer writes a ring of frames into that region, as ``cpu_verbs`` does, enlarges
    it to the ring.

**Memref-only data path.**
    Only kernels whose operands and results bufferize cleanly to memrefs are eligible for the
    executor path, and the transport operations narrow this further to rank-1 memrefs of integer
    or index element type. This keeps the wire format a flat buffer with no descriptor
    marshalling, at the cost of ruling out kernels with other signatures.

**Two ways to call coprocessor code.**
    The supported way is a placement with ``qp.backline.decode`` (or ``qec_code=``).
    ``qp.runtime_declare`` and ``qp.runtime_call`` are an advanced, lower-level alternative: they
    call external symbols directly by C symbol name, and can themselves be dispatched to an
    executor.

.. _Backline module documentation: https://docs.pennylane.ai/en/latest/code/qp_backline.html
.. _pennylane.backline: https://docs.pennylane.ai/en/latest/code/qp_backline.html
.. _Placement: https://docs.pennylane.ai/en/latest/code/qp_backline.html
.. _Backline: https://docs.pennylane.ai/en/latest/code/qp_backline.html
