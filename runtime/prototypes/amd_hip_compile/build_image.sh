#!/usr/bin/env bash
# Build the amdclang++ image without buildx.
#
# `docker build` needs the buildx plugin to honour --platform; the legacy builder ignores it and
# builds for the host, which on arm64 points apt at ports.ubuntu.com and makes every ROCm package
# "unable to locate". `docker run --platform` does work, so provision a container and commit it.
# Where buildx exists, the Dockerfile beside this script is equivalent and preferable.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
image="${AMD_HIP_IMAGE:-catalyst/amd-hip-compile}"
container="amd-hip-setup-$$"

docker rm -f "$container" >/dev/null 2>&1 || true
docker run --platform linux/amd64 -i --name "$container" \
    -e "ROCM_VERSION=${ROCM_VERSION:-7.1.1}" ubuntu:24.04 bash -s < "$here/provision.sh"

# Guard against committing a container whose provisioning silently no-oped.
docker logs "$container" 2>&1 | grep -q '^PROVISIONED amdclang++' || {
    echo "provisioning did not report success; not committing" >&2
    exit 1
}
docker commit --change 'WORKDIR /catalyst' "$container" "$image" >/dev/null
docker rm -f "$container" >/dev/null
echo "committed $image"
