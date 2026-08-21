# Vendored `infiniband/` headers

`infiniband/` here is a **modified copy** of the system verbs headers. The FPGA controller backends
compile against it rather than the sysroot's copy: `fpga_hwhs` through `HwhsControllerSession.cpp`,
and `fpga_verbs` through `FpgaControllerSession.cpp` and `UmmLib.hpp`.

Nothing else in the tree uses it. `cpu_verbs` and `gpu_verbs` compile against the system headers.

## What is patched

One line — an extra member in `struct ibv_context_ops`:

```c
void *(*_compat_reg_mr_ex)(void);
```

That struct is a function-pointer table **embedded in `struct ibv_context` by value**, so the added
member shifts the offset of every slot after it, and of every `ibv_context` field after `ops`.
Measured against libibverbs 50.0:

| | system | vendored |
|---|---|---|
| `sizeof(struct ibv_context_ops)` | 256 | 264 |
| `sizeof(struct ibv_context)` | 328 | 336 |
| `offsetof(struct ibv_context, cmd_fd)` | 264 | 272 |

Build the board's code against the copy that lacks it and each of those calls dispatches through the
wrong slot: the compile is clean, the failure is at run time, and it does not look like a header
problem.

## How it wins, and how to break it

The include is added with `SYSTEM BEFORE`, so it is searched ahead of `/usr/include` and ahead of a
sysroot's copy. `SYSTEM` rather than a plain `-I` because this is third-party code and the runtime
builds with `-Wall -Werror`, which otherwise fails inside `verbs.h` on its own deprecated enum
conversion.

Two things would break it. Dropping the include is the obvious one. The subtler one is linking a
target that was compiled against the *system* header into an artifact built against this one — the
two layouts then coexist in one binary. That is why `rdma/common` is compiled a second time as
`transport_common_fpga` when `ENABLE_TRANSPORT_FPGA` is on, rather than reusing `transport_common`.

## A native build is a compile gate, not a runnable artifact

With `ENABLE_TRANSPORT_FPGA=ON` on an x86-64 host these backends compile and link, and that is
worth having as a check. They are not runnable there. The libraries link the host's stock
`libibverbs.so.1`, whose `ibv_context` is 328 bytes, while their own translation units were compiled
against the 336-byte layout above. The mismatch is exactly the one this directory exists to create
deliberately for the board, where the runtime `libibverbs` is the patched one.

So read a passing native build as "it compiles and links", never as "it works here".

## Provenance

Carried over from `rdma_dev/devices/vendor` when that tree was frozen, and lived in the backline
repository's `devices/vendor/` until the device sources moved here. That tree is not public, so the
citation records where the code came from rather than somewhere to go and look.
