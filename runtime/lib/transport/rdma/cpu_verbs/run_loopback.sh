#!/usr/bin/env bash
#
# Manual two-process loopback test for the CPU-verbs transport. Launches
# the cpu_verbs_selftest binary twice on this host, talking to each other over a
# local RDMA device (default rxe0 / SoftRoCE):
#   - coprocessor: background; listens on the OOB port and runs the built-in echo
#   - controller:  foreground; connects, sends one syndrome, collects the reply
# The echo bounces the syndrome back, so both roles observe DEMO_SYNDROME and exit
# 0, and the script prints "LOOPBACK: PASS".
#
# Requires a working RDMA device (e.g. rxe0 SoftRoCE).
# Env overrides: DEV (rxe0), GID (1), PORT (18560), BIN (built selftest path).
set -u
DEV=${DEV:-rxe0}; GID=${GID:-1}; PORT=${PORT:-18560}
BIN=${BIN:-"$(cd "$(dirname "$0")" && pwd)/../../../build/lib/transport/cpu_verbs/cpu_verbs_selftest"}
[ -x "$BIN" ] || { echo "binary not found: $BIN (build the cpu_verbs_selftest target first)"; exit 2; }

timeout 30 "$BIN" --role coprocessor --dev "$DEV" --gid "$GID" --port "$PORT" \
  >/tmp/cvl_coproc.log 2>&1 &
SPID=$!
sleep 1
timeout 30 "$BIN" --role controller --dev "$DEV" --gid "$GID" --peer 127.0.0.1 --port "$PORT" \
  >/tmp/cvl_ctrl.log 2>&1
CRC=$?
wait $SPID; SRC=$?
echo "=== coprocessor ==="; cat /tmp/cvl_coproc.log
echo "=== controller ==="; cat /tmp/cvl_ctrl.log
echo "coprocessor rc=$SRC controller rc=$CRC"
[ "$SRC" = 0 ] && [ "$CRC" = 0 ] && echo "LOOPBACK: PASS" || { echo "LOOPBACK: FAIL"; exit 1; }
