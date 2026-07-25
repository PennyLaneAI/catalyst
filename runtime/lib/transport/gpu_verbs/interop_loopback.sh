#!/usr/bin/env bash
#
# Cross-device interop loopback: gpu_verbs coprocessor (this device's GPU, over
# RDMA/GPUDirect) + cpu_verbs controller (already-migrated CPU transport),
# talking over one real mlx5 port via the shared common/WireProtocol framing:
#   - gpu coprocessor: background; listens on the OOB port, runs the built-in
#     on-device echo kernel against a GpuHbm memory region
#   - cpu controller:  foreground; connects, sends one syndrome, collects the
#     reply
# Both sides agree on data_path="cpu_verbs", so the echo bounces the syndrome
# back and both roles observe DEMO_SYNDROME and exit 0; the script prints
# "INTEROP: PASS".
#
# Requires a working RDMA NIC with GPUDirect RDMA support (dma-buf MR) on the
# chosen device, e.g. mlx5_1 / gid 3 (RoCEv2/IPv4).
# Env overrides: DEV (mlx5_1), GID (3), PORT (18560).
set -u
DEV=${DEV:-mlx5_1}; GID=${GID:-3}; PORT=${PORT:-18560}
B=$(cd "$(dirname "$0")" && pwd)/../../../build
GPU="$B/lib/transport/gpu_verbs/gpu_verbs_selftest"
CPU="$B/lib/transport/cpu_verbs/cpu_verbs_selftest"
[ -x "$GPU" ] && [ -x "$CPU" ] || { echo "build gpu_verbs_selftest + cpu_verbs_selftest first"; exit 2; }

timeout 30 "$GPU" --dev "$DEV" --gid "$GID" --port "$PORT" \
  >/tmp/il_gpu.log 2>&1 &
SPID=$!
sleep 1
timeout 30 "$CPU" --role controller --dev "$DEV" --gid "$GID" --peer 127.0.0.1 --port "$PORT" \
  >/tmp/il_cpu.log 2>&1
CRC=$?
wait $SPID; SRC=$?
echo "=== gpu coproc ==="; cat /tmp/il_gpu.log
echo "=== cpu ctrl ==="; cat /tmp/il_cpu.log
echo "gpu coproc rc=$SRC cpu ctrl rc=$CRC"
[ "$SRC" = 0 ] && [ "$CRC" = 0 ] && echo "INTEROP: PASS" || { echo "INTEROP: FAIL"; exit 1; }
