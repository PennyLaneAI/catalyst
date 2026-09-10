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

The Backline components are off by default in a Catalyst build. To build them, enable the following build flags:

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

Transport backends are not shipped in the Catalyst wheel. An out-of-tree backend is found through
``CATALYST_TRANSPORT_PATH``, a ``:``-separated list of directories searched ahead of the
installation's own library directory.

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

#. A workload is declared as a placement (a controller, its coprocessors, and a transport) and
   executed under :func:`~.qjit` on a `Backline`_ device.
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

A node that runs in the originating process carries no dispatch, since its code is called
directly. A node given an executor is dispatched: its kernel is cross-compiled, shipped to the
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

Catalyst resolves a placement's transport name and a node's hardware to a concrete backend
library, named ``libcatalyst_transport_<backend>_<role>.so``. The pairs currently mapped are
``rdma`` with ``cpu``, ``gpu``, or ``fpga``, and ``memcpy`` with ``cpu`` or ``gpu``.

A backend is a shared library implementing ``ControllerSession`` or ``CoprocessorSession`` from
``runtime/include/Transport.hpp`` and exporting the matching factory symbol declared in
``runtime/include/TransportBackend.h``. A session follows a strict lifecycle:

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
      - Start a ``() -> ()`` entry without waiting for it, returning an ``!executor.token``.
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

Two compiler passes drive the target workflow. The compiler driver runs them as their own
pipeline immediately after the bufferization stage, and only when a compilation workspace is set,
since cross-compilation writes object files into it.

``cross-compile-targets``
    For every nested ``builtin.module`` carrying a ``catalyst.target`` attribute, extracts the
    module body into a standalone root module, runs the default lowering pipeline on it,
    translates to LLVM IR, and emits an object file. The emitted path is recorded on the module as
    the ``catalyst.object_file`` attribute, and the module is reduced to external declarations of
    its entry functions. The target triple comes from the optional ``triple`` key on
    ``catalyst.target``, falling back to the host triple. Objects destined for local execution are
    recorded for static linking.

``dispatch-executor-targets``
    For every nested module carrying both ``catalyst.dispatch`` and ``catalyst.object_file``,
    injects one ``executor.open`` per unique address and one ``executor.send_binary`` per module
    into ``setup()``, rewrites every host-side ``catalyst.launch_kernel`` targeting that module
    into an ``executor.launch``, and erases the nested module from the host. A
    ``catalyst.custom_call`` carrying a ``dispatch`` entry in its ``backend_config`` becomes an
    ``executor.call``. No close operation is emitted, since the runtime closes every open session
    at process exit.

Two supporting behaviours in the core pipeline make this possible:

- The module-inlining pass skips modules annotated as separate targets, so a kernel destined for a
  different target is preserved as its own module rather than being flattened into the host
  program.
- ``catalyst.launch_kernel`` is bufferizable, so a kernel launched from the host has its tensor
  operands and results lowered to the memrefs the executor operations accept.

A placement adds three passes on top of this: ``inject-transport-session`` in the quantum
compilation stage, ``lower-decode-to-transport`` in the bufferization stage, and
``convert-transport-to-llvm`` before ``convert-catalyst-to-llvm``. A placement naming a
``qec_code`` also registers the QEC encoding passes and adds ``convert-qecp-to-llvm``.

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

Compiling with ``keep_intermediate=True`` keeps the workspace the driver writes into, which is
where the Backline stages leave their output. Cross-compilation needs that workspace regardless,
since it writes object files into it, and ``cross-compile-targets`` and
``dispatch-executor-targets`` are skipped entirely when no workspace is set.

The driver turns ``keep_intermediate`` into the ``save-ir-after-each`` option on
``cross-compile-targets``, which controls how much of each target module's own lowering is
written into its workspace subdirectory. Empty writes nothing, ``pipeline`` writes the extracted
MLIR and the translated LLVM IR as ``extracted.mlir`` and ``<name>.ll``, and ``changed`` or
``pass`` add the IR after each pass that altered it or after every pass. This is the way to see
what a target module became, since it is lowered by its own nested pipeline rather than the one
the host program runs through.

``cross-compile-targets`` also records the objects of statically linked targets on the root
module, and the driver writes them to a ``<module>.objects`` manifest in the workspace that the
frontend hands to the linker.

On the runtime side, ``catalyst.Executor`` writes a host-side log per launch, named
``catalyst-executor[-<name>]-<host>-<timestamp>.log``. This is the only place a failed plugin load
is reported, so it is the first thing to read when a dispatched kernel cannot resolve a symbol.
The ``verbose`` argument controls launcher narration, where ``0`` is quiet, ``1`` is normal, and
``2`` reports each command.

A transport backend that fails to load reports through the same channel, since the runtime opens
it with ``dlopen`` on the node it belongs to. Backend errors are logged with a ``[transport]``
prefix, and a round that never receives a reply surfaces as a collect error naming the cause,
which is one of memory, timeout, or stuck.

Current limitations
===================

As an experimental feature, several components are provisional. Each of these can change as the
platform is built out.

**Fixed remote memory slab.**
    A session reserves a single 1 GB slab on the remote for the sections of every object it loads
    (``.text``, ``.rodata``, and so on), released when the session is destroyed. The size is a
    compile-time constant in ``runtime/lib/executor/ExecutorSession.cpp`` and is not currently
    tunable at runtime.

**Fixed transport reply region.**
    A transport session provisions a 16 KB local reply region at ``exchange_keys``. The size is a
    compile-time constant in ``runtime/lib/transport/TransportCAPI.cpp``. Making it configurable
    per session would need a runtime entry point to override it.

**Memref-only data path.**
    Only kernels whose operands and results bufferize cleanly to memrefs are eligible for the
    executor path, and the transport operations narrow this further to rank-1 memrefs of integer
    or index element type. This keeps the wire format a flat buffer with no descriptor
    marshalling, at the cost of ruling out kernels with other signatures.

**Overlapping interfaces.**
    ``qp.runtime_declare`` and ``qp.runtime_call`` declare and invoke external symbols directly
    by C symbol name, and can themselves be dispatched to an executor. These lower-level entry
    points coexist with the Backline placement frontend and are expected to be consolidated.

**Transport backends are build-time.**
    A transport backend is selected by resolving the placement's transport name and the node's
    hardware against a fixed table in ``catalyst.backline``. Adding a backend to that table
    requires a Catalyst change, even though the backend itself loads as an out-of-tree shared
    library.

.. _Backline module documentation: https://docs.pennylane.ai/en/latest/code/qp_backline.html
.. _pennylane.backline: https://docs.pennylane.ai/en/latest/code/qp_backline.html
.. _Placement: https://docs.pennylane.ai/en/latest/code/qp_backline.html
.. _Backline: https://docs.pennylane.ai/en/latest/code/qp_backline.html
