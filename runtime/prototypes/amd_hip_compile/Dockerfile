# The AMD compiler, so the transport's .hip sources get built by the toolchain they are written
# for. CI compiles them with nvcc plus fetched hip-dev headers (runtime/lib/transport/
# CMakeLists.txt:55-65), so an AMD-only compile error ships undetected today.
#
# --platform=linux/amd64 is deliberate and load-bearing: ROCm publishes x86_64 packages only. On an
# arm64 host this runs under qemu-x86_64 binfmt -- slow, but adequate for compiling a handful of
# kernels. On an x86_64 CI runner it is native.
FROM --platform=linux/amd64 ubuntu:24.04

ARG ROCM_VERSION=7.1.1
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl gnupg \
    && rm -rf /var/lib/apt/lists/*

# Key to a file before dearmouring, not through a pipe: under qemu-x86_64 a piped curl can deliver
# a short read, leaving a truncated keyring that turns every ROCm package into "unable to locate"
# several steps later. Verified here instead, where the error names the cause.
RUN curl -fsSL --retry 3 https://repo.radeon.com/rocm/rocm.gpg.key -o /tmp/rocm.gpg.key \
    && gpg --dearmor < /tmp/rocm.gpg.key > /usr/share/keyrings/rocm.gpg \
    && rm /tmp/rocm.gpg.key \
    && test -s /usr/share/keyrings/rocm.gpg \
    && printf 'deb [signed-by=/usr/share/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/%s noble main\n' "${ROCM_VERSION}" > /etc/apt/sources.list.d/rocm.list \
    && apt-get update \
    && apt-cache policy rocm-llvm > /tmp/rocm-policy \
    && grep -q 'Candidate: [0-9]' /tmp/rocm-policy

# rocm-device-libs is not separately packaged in 7.1.1; rocm-llvm carries the amdgcn bitcode.
RUN apt-get install -y --no-install-recommends rocm-llvm hip-dev \
    && rm -rf /var/lib/apt/lists/*

# The prefix is versioned (/opt/rocm-7.1.1, with /opt/rocm a symlink), so resolve the binary once
# rather than hardcoding either spelling.
RUN amdclang=$(find /opt  -name 'amdclang++' -print -quit) \
    && test -n "$amdclang" \
    && echo "amdclang++ at $amdclang" \
    && ln -s "$amdclang" /usr/local/bin/amdclang++
WORKDIR /catalyst
