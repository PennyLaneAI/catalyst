#!/usr/bin/env bash
# Runs *inside* an amd64 ubuntu:24.04 container: install the AMD compiler and HIP headers.
# Kept in its own file and fed to `bash -s` so no host shell ever parses it -- interpolating this
# into a double-quoted `docker run bash -c "..."` invites backtick substitution and quoting bugs.
set -euo pipefail

rocm_version="${ROCM_VERSION:-7.1.1}"
export DEBIAN_FRONTEND=noninteractive

apt-get update
apt-get install -y --no-install-recommends ca-certificates curl gnupg

# Key to a file rather than piping curl into gpg: under qemu-x86_64 a piped curl can short-read,
# leaving a truncated keyring that only surfaces later as an unresolvable package.
curl -fsSL --retry 3 https://repo.radeon.com/rocm/rocm.gpg.key -o /tmp/rocm.gpg.key
gpg --dearmor < /tmp/rocm.gpg.key > /usr/share/keyrings/rocm.gpg
rm /tmp/rocm.gpg.key
test -s /usr/share/keyrings/rocm.gpg

printf 'deb [signed-by=/usr/share/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/%s noble main\n' \
  "$rocm_version" > /etc/apt/sources.list.d/rocm.list
apt-get update

# Redirect to a file instead of piping into grep -q: grep exits at the first match, apt-cache takes
# SIGPIPE, and pipefail turns that into a fatal 141.
apt-cache policy rocm-llvm > /tmp/rocm-policy
grep -q 'Candidate: [0-9]' /tmp/rocm-policy

# Pin the ROCm repo above Ubuntu universe. Without this apt satisfies HIP dependencies with
# universe's libamdhip64-dev 5.7.1, whose /usr/include/hip headers shadow ROCm's and predate
# hipMemRangeHandleTypeDmaBufFd -- the compile then fails on an enum that does exist in 7.1.1.
printf 'Package: *\nPin: origin repo.radeon.com\nPin-Priority: 1001\n' > /etc/apt/preferences.d/rocm

# rocm-device-libs is separate from rocm-llvm and carries the amdgcn bitcode; without it
# amdclang++ refuses device compilation with "cannot find ROCm device library". hip-runtime-amd
# brings the AMD-side HIP headers that hip-dev alone does not guarantee.
apt-get install -y --no-install-recommends \
  rocm-llvm hip-runtime-amd hip-dev rocm-device-libs
rm -rf /var/lib/apt/lists/*

# No -type f: rocm-llvm ships amdclang++ as a symlink to amdllvm, so a file-only filter finds
# nothing. -print -quit rather than a pipe into head, same SIGPIPE reason as above.
amdclang=$(find /opt -name 'amdclang++' -print -quit)
test -n "$amdclang"
ln -s "$amdclang" /usr/local/bin/amdclang++
echo "PROVISIONED amdclang++ at $amdclang"
