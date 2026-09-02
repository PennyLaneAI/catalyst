#!/usr/bin/env bash
# Build the emulated-RNIC image and run one of: the upstream test suite, the loopback server, or a
# shell. See README.md for what each buys us.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
image="${ERNIC_IMAGE:-catalyst/emulated-rnic}"
socket_volume="${ERNIC_SOCKET_VOLUME:-ernic-sock}"

usage() {
    cat <<'USAGE'
usage: run.sh [test|serve|shell|build]

  build   build the image only
  test    build, then run rocm-ernic's ctest suite (default)
  serve   build, then run the emulator with the loopback backend on a shared volume
  shell   build, then drop into the container

environment:
  ERNIC_IMAGE          image tag to build/run       (default catalyst/emulated-rnic)
  ERNIC_SOCKET_VOLUME  volume holding the socket    (default ernic-sock, `serve` only)
USAGE
}

build() {
    # No --platform: the host architecture is the point, so an arm64 Mac builds arm64.
    docker build -t "$image" "$here"
}

case "${1:-test}" in
    build) build ;;
    test)  build; docker run --rm "$image" ;;
    serve)
        build
        # A Docker volume, not a bind mount from macOS: a client has to connect(2) to this socket,
        # and over virtiofs it never materializes host-side -- the server creates and chmods it,
        # then macOS sees an empty directory. A volume keeps the socket inside the Linux VM, which
        # is where a QEMU vfio-user client has to run anyway.
        docker volume create "$socket_volume" >/dev/null
        docker run --rm -it -v "$socket_volume:/run/ernic" "$image" \
            rocm-ernic --socket /run/ernic/vfio-user-rocm-ernic.sock --backend loopback --verbose
        ;;
    shell) build; docker run --rm -it "$image" bash ;;
    -h|--help|help) usage ;;
    *) usage >&2; exit 2 ;;
esac
