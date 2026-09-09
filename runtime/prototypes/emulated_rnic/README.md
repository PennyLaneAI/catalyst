# Emulated RDMA NIC spike

Can we exercise the RDMA transport without RDMA hardware, and without Soft-RoCE mutating a
runner's RDMA links the way `.github/actions/setup-soft-roce` has to?

[rocm-ernic](https://github.com/ROCm/rocm-ernic) (AMD ROCm Storage, MIT, early-access) is a
libvfio-user server that emulates an RDMA NIC as a PCI device for a VM guest. Backends: `loopback`
for self-test, `tcp` for two instances talking to each other, `verbs` for passthrough to real
hardware.

## What this directory does

Builds the emulator and runs its own test suite in a container -- no VM, no privileges, no RDMA
hardware:

```
./run.sh test      # build + upstream ctest
./run.sh serve     # run the emulator, loopback backend, socket on a Docker volume
./run.sh shell     # poke around
```

Upstream is pinned by commit in the `Dockerfile`, because "early-access" means a red ctest would
otherwise be ambiguous between our setup and an upstream change.

## Status: builds and self-tests on arm64

Verified natively on Apple Silicon (M4, colima's Ubuntu 24.04 / kernel 6.8 VM):

```
1/5 pci-config-test ................ Passed
2/5 data-transfer-test ............. Skipped
3/5 rdma-cm-test ................... Skipped
4/5 rdma-backend-query-port-unit ... Passed
5/5 ernic_dc_uapi .................. Passed
```

The two skips are the data path, and upstream says why itself: `No RDMA devices found` /
`This test requires a VM with loopback backend`. `serve` reaches
`Device realized, waiting for client connection...`, so the server side is sound.

### Two upstream gaps found on the way

* `INSTALL.md`'s package list omits `libcmocka-dev`, which libvfio-user's `meson.build` takes as a
  hard dependency -- `meson setup` aborts without it.
* libvfio-user installs no pkg-config file, so rocm-ernic falls back to `find_library` with
  `NO_DEFAULT_PATH` over hints that name only `x86_64-linux-gnu` (its `CMakeLists.txt:71-77`). On
  arm64 the library lands in `/usr/lib/aarch64-linux-gnu` and configure fails. The `Dockerfile`
  sidesteps this with `meson --libdir=lib`; both are worth reporting upstream.

## What a container cannot do, and why

The emulator presents a *PCI device to a guest*, and the ernic driver has to load into a guest
kernel to produce a verbs device. Containers share the host kernel, so no `/dev/infiniband/uverbs0`
appears and our transport has nothing to open. A container is the server half only.

The client half is a VM, and it is no longer exotic: QEMU 10.1 shipped an upstream vfio-user
client, `-device vfio-user-pci,socket=...`. Note that it is not built on macOS hosts -- Homebrew's
QEMU 11.0.0 lists no vfio devices at all -- so the client belongs inside the Linux VM, alongside
the emulator.

## Stage two: what a real device costs

Upstream tiers its own CI the same way this spike splits:

| Tier | Scope | Needs KVM |
|------|-------|-----------|
| 1 | build, ctest, loopback backend | no |
| 2 | two-VM RDMA functional | yes |
| 3 | performance sweeps | yes |

Tier 1 is what `./run.sh test` does, and it is green here. A verbs device our transport can open is
Tier 2 by upstream's own reckoning, so no amount of container flags gets there.

What Tier 2 needs, with the costs as measured rather than guessed:

* **Nested virt on the Mac** -- `colima start -z --cpu 8 --memory 8` (M3+; M4 here, 16 GB, so 8 GB
  to the VM). Currently colima runs 2 vCPU / 2 GB with no `/dev/kvm`. Unverified until restarted.
* **QEMU with the vfio-user client** -- no source build required after all: `debian:sid` packages
  QEMU 11.1.1 which lists `vfio-user-pci`. Run it in a container with `--device /dev/kvm`, sharing
  the emulator's socket volume.
* **A guest driver and a custom verbs provider** -- the emulated device is *not* PVRDMA-compatible
  despite the log line: `driver/` is an out-of-tree module (DKMS, `driver/setup-rocm-ernic-dkms.sh`)
  and `rdma-core/` is a custom provider that has to be built inside the guest
  (`scripts/build-rdma-core.sh`). Upstream automates both in an Ansible `guest-setup` role.
* **Two instances on the `tcp` backend** for two eRNICs to reach each other; `loopback` is
  single-device.

Upstream's `scripts/local-vm-test.sh` is not a portable recipe -- it hard-codes
`/opt/qemu-v10.1.2`, a personal `~/Projects/qemu-minimal` checkout, `qemu-system-x86_64`, and an
`ssh` user. The Ansible route wants a `sbates130272.batesste` Galaxy collection. Either way the
guest plumbing is ours to write.

## The binding constraint is not the NIC

Running `frontend/test/pytest/test_backline.py` against an emulated device needs Catalyst built
*inside the guest*, and this checkout has no build at all: `frontend/mlir_quantum/` is absent and
`frontend/catalyst/lib` and `bin` are empty, so `import catalyst` fails on `mlir_quantum` before
any transport is chosen. LLVM itself is built (`mlir/llvm-project/build/bin` is populated), but the
dialects build and the `make frontend` copy step are not. That work is on the critical path for any
local pytest, RDMA or otherwise, and dwarfs the emulator setup.

Cheaper alternative if the goal is just a local verbs device rather than an emulated RNIC
specifically: Soft-RoCE in the colima VM (`linux-modules-extra-6.8.0-100-generic` is installed
there now), then a privileged container with `/dev/infiniband` -- the same path CI takes.
