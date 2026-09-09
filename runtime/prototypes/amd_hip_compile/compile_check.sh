#!/usr/bin/env bash
# Compile the transport's .hip sources with the AMD toolchain, for each CDNA target.
#
# CI otherwise only builds these with nvcc plus fetched hip-dev headers (runtime/lib/transport/
# CMakeLists.txt:55-65), so an AMD-only compile error ships unnoticed. Device-only: no GPU, no
# driver, no linking.
#
# Runs amdclang++ directly when it is on PATH (a CI runner with ROCm installed, or inside the
# image). Otherwise it re-executes itself inside catalyst/amd-hip-compile, which has one -- so the
# compile loop below is the single definition of what "the check" means.
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
archs="${OFFLOAD_ARCHES:-gfx950 gfx90a}"
sources=(common/coproc/gpu/GpuLaunchers.hip common/coproc/gpu/GpuRuntime.hip)

if ! command -v amdclang++ >/dev/null 2>&1; then
    image="${AMD_HIP_IMAGE:-catalyst/amd-hip-compile}"
    if ! docker image inspect "$image" >/dev/null 2>&1; then
        echo "no amdclang++ on PATH and no $image; run ./build_image.sh first" >&2
        exit 2
    fi
    # --platform: the image is x86_64 because ROCm publishes no arm64 packages. Native on an
    # x86_64 runner, qemu-x86_64 on Apple Silicon.
    exec docker run --rm --platform linux/amd64 -v "$repo:/catalyst" \
        -e "OFFLOAD_ARCHES=$archs" "$image" \
        bash /catalyst/runtime/prototypes/amd_hip_compile/compile_check.sh
fi

cd "$repo/runtime/lib/transport"
status=0
for arch in $archs; do
    for src in "${sources[@]}"; do
        if amdclang++ -x hip --offload-arch="$arch" --offload-device-only -std=c++20 -O2 \
            --rocm-path=/opt/rocm -I /opt/rocm/include -I "$repo/runtime/include" \
            -I common/coproc/gpu -I common/interface -I common/coproc -I . -I rdma/common \
            -c "$src" -o /tmp/hip-check.o; then
            printf 'ok      %-40s %-8s %s bytes\n' "$src" "$arch" "$(stat -c %s /tmp/hip-check.o)"
        else
            printf 'FAILED  %-40s %s\n' "$src" "$arch"
            status=1
        fi
        rm -f /tmp/hip-check.o
    done
done
exit $status
