#!/usr/bin/env bash
#
# Cross-device loopback test over a real RDMA port. A gpu_verbs coprocessor
# runs the echo decoder in the background, and a cpu_verbs
# controller in the foreground that sends one syndrome and collects the reply.
# Both observe DEMO_SYNDROME and exit 0, then the script prints "INTEROP: PASS".
#
# Requires a NIC with GPUDirect RDMA (dma-buf MR) support.
set -u
DEV=${DEV:-}; GID=${GID:-}; PORT=${PORT:-18560}
if [ -z "$DEV" ] || [ -z "$GID" ]; then
  echo "usage: DEV=<rdma_device> GID=<gid_index> [PORT=<n>] $0"
  echo "devices on this host:"
  ibv_devices 2>/dev/null || echo "  (ibv_devices not found; install rdma-core tools)"
  echo "GID indices for a device: show_gids <dev>, or /sys/class/infiniband/<dev>/ports/1/gids"
  exit 2
fi
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
