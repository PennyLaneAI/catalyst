#!/usr/bin/env bash
# Two-process SoftRoCE round-trip on rxe0/lo: coprocessor (listens, built-in echo)
# + controller (connects, kicks one syndrome). Both echo -> both observe DEMO_SYNDROME.
set -u
DEV=${DEV:-rxe0}; GID=${GID:-1}; PORT=${PORT:-18560}
BIN=${BIN:-"$(cd "$(dirname "$0")" && pwd)/../build/cpu_libibverbs/cpu_libibverbs_main"}
[ -x "$BIN" ] || { echo "binary not found: $BIN (build the cpu_libibverbs_main target first)"; exit 2; }

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
