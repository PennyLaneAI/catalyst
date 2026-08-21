# `fpga_verbs`: the software-handshake FPGA controller

A controller that runs on a Xilinx VPK120 board and drives an RDMA round trip against a
coprocessor, with the CPU posting the syndrome through ibverbs and polling for the reply. It
implements Catalyst's `catalyst::transport::ControllerSession` and ships as a loadable transport
backend (`libcatalyst_transport_fpga_verbs_controller.so`), so Catalyst's `librt_transport` can
`dlopen` it.

Its counterpart is [`../fpga_hwhs`](../fpga_hwhs), where the board's handshake engine drives the
round itself and no CPU is in the round trip. Both reach the fabric through ibverbs, so `hwhs` and
`swhs` name where the handshake runs, which is what separates them. [`../cpu_verbs`](../cpu_verbs)
and [`../gpu_verbs`](../gpu_verbs) are its siblings.

| | |
|---|---|
| Role | Controller: writes requests, waits for replies |
| Interface | `ControllerSession` (`commit_work_item` / `kick` / `data_slot` / `collect`) |
| Memory | on-board allocator (`libumm.so.1`), `dlopen`'d at runtime |
| Target | aarch64 (VPK120) |

The sources still spell themselves `fpga_verbs`: the namespace is `rdma::devices::fpga_verbs`, the
self-test is `fpga_verbs_selftest.cpp`, and the built library keeps that name because it is the
filename the deployed bundles carry, and the one the backline repository's
`benchmarks/placement.py` asks for.

## Build

Two builds reach these sources, for different reasons.

**Cross-compiled for the board**, which is what the demos deploy: `config/xbuild` in the
[backline](https://github.com/PennyLaneAI/backline) repository, driven by its `CROSSBUILD.md`. Read
that first — it covers host packages, the target's sysroot and the Catalyst checkout, none of which
this page repeats.

One component builds this device. Its session is compiled into the library rather than shipped
beside it, which is why there is no separate session component the way `fpga_hwhs` has one:

```bash
cd config/xbuild                 # in the backline repository
make build TARGET=vpk120 COMPONENT=swhs-capi-backend CATALYST=~/catalyst
```

The artifact lands at
`build/vpk120/components/swhs-capi-backend/libcatalyst_transport_fpga_verbs_controller.so`. To get
the whole controller stack the board needs, build the bundle instead, which carries both handshake
arrangements and the transport runtime:

```bash
make bundle TARGET=vpk120 BUNDLE=vpk-controller CATALYST=~/catalyst
make deploy TARGET=vpk120 BUNDLE=vpk-controller SSH=petalinux@<board>
```

**Natively, from this tree**, for compiling and testing without a board. It is off by default,
because the device needs the patched headers below and, at run time, hardware this build cannot
check for:

```bash
cmake -B build runtime -DENABLE_TRANSPORT=ON -DENABLE_TRANSPORT_FPGA=ON
cmake --build build --target catalyst_transport_fpga_verbs_controller
```

That path also builds `fpga_verbs_selftest` and, with `BUILD_TESTING`, the Catch2 tests under
[`test/`](test/) — none of which the cross-build produces.

### The vendored headers are not optional

This device compiles against the **vendored** board headers in
[`../vendor/infiniband/`](../vendor/) rather than the sysroot's. The board's libibverbs has an extra
`verbs_context` slot, and that struct is dispatched by offset, so a mismatched layout sends verbs
calls through the wrong slots. The compile is clean and the failure is at run time. Both components
carry an `-I` at that directory for this reason, and the CMake targets add it with `BEFORE` so it
is searched ahead of the sysroot's copy.

## Using it as a Catalyst transport backend

`libcatalyst_transport_fpga_verbs_controller.so` exports `CatalystTransportControllerFactory` (via
`GENERATE_TRANSPORT_CONTROLLER_FACTORY` from `TransportBackend.h`). Catalyst's `librt_transport`
`dlopen`s it and passes a `;`-separated config string. These are the keys
[`FpgaBackendConfig.hpp`](FpgaBackendConfig.hpp) reads, and anything else in the string is ignored
rather than rejected:

| key | default | meaning |
|---|---|---|
| `dev` | `xib_0` | RDMA device name |
| `gid` | `1` | GID index |
| `ring` | `K_RING_SLOTS` (256) | ring slots |
| `stride_log2` | `6` | slot stride, log2 (6 = 64 B) |
| `data_mem` | *(auto)* | request-ring placement: `ps` \| `pl` \| `bram` |
| `reply_mem` | *(auto)* | reply-ring placement: `ps` \| `pl` \| `bram` |

For example: `dev=xib_0;gid=3;ring=256;data_mem=pl;reply_mem=ps`.

**Leave `ring` alone unless you are changing both ends.** The handshake exchanges queue-pair
numbers, a memory key and an MTU, and nothing about ring geometry, so the two sides agree only
because they are compiled against the same `K_RING_SLOTS` from Catalyst's `WireProtocol.hpp`. That
is why the default here is that constant rather than a number. Setting `ring` to anything else makes
the controller index slots as `cursor % ring` while `cpu_verbs` and `gpu_verbs` keep indexing as
`cursor & (K_RING_SLOTS - 1)`, and the round trip stalls at the first cursor where those differ.

This is the path the backline repository's `benchmarks/` drives. The config string comes from
`config_swhs` in its `config/machines.toml`, and `benchmarks/placement.py` names this library in
`init_args["backend_lib"]`, the compiler's hardware and transport mapping having no name for the
pair. The benchmark's `sw-handshake` cell is this controller:

```bash
./benchmarks/bench.py --ctrl sw-handshake --coproc gpu-steane -o rtt.csv
```

## Layout

| file | purpose |
|---|---|
| `FpgaControllerSession.{hpp,cpp}` | the `ControllerSession` implementation |
| `FpgaControllerFactory.cpp` | plugin entry point (the exported factory symbol) |
| `FpgaBackendConfig.hpp` | config-string parsing |
| `UmmLib.hpp` | `dlopen` wrapper for the on-board allocator |
| `fpga_verbs_selftest.cpp` | standalone RTT self-test and benchmark, not built here |
| `CMakeLists.txt` | how the origin tree built the above, not wired into this repository |
| `test/` | Catch2 unit tests, not built here |
