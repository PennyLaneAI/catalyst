# Builds the AMD emulated RDMA NIC (rocm-ernic) and runs its own test suite.
#
# Stage one of the emulated-RNIC spike: proves the emulator builds and self-tests on this
# architecture, needing no VM, no privileges and no RDMA hardware. The emulator is a userspace
# libvfio-user server, so a container is enough to exercise it -- but only by itself. Handing the
# device to our transport needs a guest kernel to load the ernic driver, which a container cannot
# provide; see README.md.
#
# Multi-arch: ubuntu:24.04 resolves to the host architecture, so this is a native arm64 build on
# Apple Silicon. Ubuntu 24.04 is the platform rocm-ernic documents as tested.
FROM ubuntu:24.04

# Pinned rather than tracking main: rocm-ernic is labelled early-access, so a moving target would
# make a red ctest ambiguous between "our setup" and "upstream changed". Bump deliberately.
ARG ERNIC_REF=8a76d2ca24557408508d672a25a6a13580b671cf
ARG VFIO_USER_REF=323f4cb6cddc3713fb7aebe44436f28b28b5413a

ENV DEBIAN_FRONTEND=noninteractive

# The package list is rocm-ernic's INSTALL.md, plus git/ca-certificates to fetch sources and
# python3 for libvfio-user's meson build. libcmocka-dev is missing from that list: libvfio-user's
# meson.build takes it as a hard dependency, so `meson setup` aborts without it.
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential ca-certificates git python3 \
      cmake meson ninja-build pkg-config libcmocka-dev \
      libibverbs-dev librdmacm-dev libglib2.0-dev libjson-c-dev \
    && rm -rf /var/lib/apt/lists/*

# libvfio-user first: rocm-ernic links against it and expects it installed system-wide.
#
# --libdir=lib, not meson's Debian default of lib/aarch64-linux-gnu: libvfio-user installs no
# pkg-config file, so rocm-ernic falls back to find_library with NO_DEFAULT_PATH over hints that
# only spell out x86_64-linux-gnu (CMakeLists.txt:71-77). Installing into plain /usr/lib lands on
# a hint that holds for every architecture. Worth reporting upstream.
WORKDIR /src/libvfio-user
RUN git clone https://github.com/nutanix/libvfio-user.git . \
    && git checkout --quiet ${VFIO_USER_REF} \
    && meson setup build --prefix=/usr --libdir=lib \
    && ninja -C build \
    && ninja -C build install \
    && ldconfig

WORKDIR /src/rocm-ernic
RUN git clone https://github.com/ROCm/rocm-ernic.git . \
    && git checkout --quiet ${ERNIC_REF} \
    && cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug \
    && cmake --build build

# Default: the upstream suite. `docker run ... rocm-ernic --socket ... --backend loopback` instead
# starts the emulator, which is what a QEMU vfio-user client would connect to.
ENV PATH="/src/rocm-ernic/build:${PATH}"
CMD ["ctest", "--test-dir", "build", "--output-on-failure"]
