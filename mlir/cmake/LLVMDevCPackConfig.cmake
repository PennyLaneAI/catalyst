# CPack configuration for the catalyst-llvm-dev package.
#
# The other two packages (catalyst-dev, catalyst-runtime-dev) are produced by
# our own CMake projects, which include(CPack) directly. This one is different:
# its payload is the LLVM/MLIR distribution staged by
#
#     cmake --build <llvm build dir> --target install-distribution
#
# which is the only supported way to honour LLVM_DISTRIBUTION_COMPONENTS, and it
# belongs to LLVM's CMake project rather than ours. So instead of hooking into a
# CMakeLists.txt, this is a standalone config driven by mlir/Makefile as
#
#     CATALYST_LLVM_STAGING_DIR=<staged install prefix> \
#     CATALYST_LLVM_VERSION=<llvm version>-<llvm commit> \
#         cpack --config <this file> -B <output dir>
#
# The two inputs come through the environment rather than `cpack -D`, because
# cpack applies -D definitions only *after* evaluating this file, so they would
# not be visible to the checks and the file name below.
#
# Everything else is kept deliberately in step with the CPack blocks in
# mlir/CMakeLists.txt and runtime/CMakeLists.txt so all three archives share a
# naming scheme and layout.

set(CATALYST_LLVM_STAGING_DIR "$ENV{CATALYST_LLVM_STAGING_DIR}")
set(CATALYST_LLVM_VERSION "$ENV{CATALYST_LLVM_VERSION}")

if(NOT CATALYST_LLVM_STAGING_DIR)
    message(FATAL_ERROR
        "CATALYST_LLVM_STAGING_DIR is not set. Run this via 'make pack-llvm-dev'.")
endif()
if(NOT EXISTS "${CATALYST_LLVM_STAGING_DIR}")
    message(FATAL_ERROR
        "Staging directory ${CATALYST_LLVM_STAGING_DIR} does not exist. "
        "The install-distribution target must run before cpack.")
endif()
if(NOT CATALYST_LLVM_VERSION)
    message(FATAL_ERROR
        "CATALYST_LLVM_VERSION is not set. Run this via 'make pack-mlir-dev'.")
endif()
# Guard against a half-resolved version.
if(NOT CATALYST_LLVM_VERSION MATCHES "^.+-[0-9a-f]+$")
    message(FATAL_ERROR
        "CATALYST_LLVM_VERSION ('${CATALYST_LLVM_VERSION}') is malformed; "
        "expected <llvm version>-<llvm commit>, e.g. 22.0.0git-8f264586.")
endif()

set(CPACK_PACKAGE_NAME "catalyst-llvm-dev")
set(CPACK_PACKAGE_VERSION "${CATALYST_LLVM_VERSION}")
set(CPACK_PACKAGE_VENDOR "Xanadu Quantum Technologies Inc.")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY
    "LLVM/MLIR libraries, headers and tools required to build Catalyst")
set(CPACK_PACKAGE_DESCRIPTION
    "The subset of the LLVM/MLIR distribution that Catalyst builds against, as "
    "selected by LLVM_DISTRIBUTION_COMPONENTS in mlir/Makefile.")

# A standalone cpack run has no CMAKE_SYSTEM_NAME/CMAKE_SYSTEM_PROCESSOR (there
# is no project being configured), but the host equivalents are available and
# agree with them for the native builds this package is produced from.
set(CPACK_GENERATOR "TGZ")
set(CPACK_PACKAGE_FILE_NAME
    "${CPACK_PACKAGE_NAME}-${CMAKE_HOST_SYSTEM_NAME}-${CMAKE_HOST_SYSTEM_PROCESSOR}-${CPACK_PACKAGE_VERSION}")

# Package the already-staged tree as-is. The trailing "." is the subdirectory to
# place it under inside the archive, i.e. the staged bin/, lib/ and include/ land
# at the archive root rather than nested.
set(CPACK_INSTALLED_DIRECTORIES "${CATALYST_LLVM_STAGING_DIR};.")

# extracts into catalyst-llvm-dev-<sys>-<arch>-<ver>/ rather than spilling bin/,
# lib/ and include/ into the current directory.
set(CPACK_INCLUDE_TOPLEVEL_DIRECTORY ON)
