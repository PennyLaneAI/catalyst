cmake_minimum_required(VERSION 3.26)

project(catalyst_runtime)
include(FetchContent)
include(ExternalProject)

set(CMAKE_CXX_STANDARD  20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler options
option(ENABLE_CODE_COVERAGE "Enable code coverage" OFF)
option(ENABLE_ADDRESS_SANITIZER "Enable address sanitizer" OFF)
option(RUNTIME_CLANG_TIDY "Enable Clang Tidy" OFF)
option(RUNTIME_ENABLE_WARNINGS "Enable -Wall and -Werror" ON)

option(ENABLE_OPENQASM "Build OpenQasm backend device" OFF)
option(ENABLE_OQD "Build OQD backend device" OFF)
option(ENABLE_TRANSPORT "Build the backend-agnostic transport loader (rt_transport)" OFF)
option(ENABLE_TRANSPORT_FPGA "Build the FPGA controller backends" OFF)
set(LIGHTNING_CLIFFISH_LIB "" CACHE FILEPATH
    "lightning.cliffish device library built in pennylane-lightning; when set, its runtime tests are built")
if(ENABLE_TRANSPORT_FPGA AND NOT ENABLE_TRANSPORT)
    message(FATAL_ERROR
        "ENABLE_TRANSPORT_FPGA requires ENABLE_TRANSPORT=ON.")
endif()

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(runtime_includes ${PROJECT_SOURCE_DIR}/include)
set(capi_utils_includes ${PROJECT_SOURCE_DIR}/lib/capi)
set(backend_utils_includes
    ${PROJECT_SOURCE_DIR}/lib/backend/common
    ${MLIR_INCLUDE_DIRS}                      # RunnerUtils.h
)

# Get LLVM hash to target from source tree.
file(READ ../.dep-versions DEPENDENCY_VERSIONS)
string(REGEX MATCH "llvm=([0-9a-f]+)" _ ${DEPENDENCY_VERSIONS})
set(LLVM_HASH ${CMAKE_MATCH_1})
message(STATUS "Detected LLVM version - ${LLVM_HASH}")

# Note on pybind11 vs python discovery order:
# If Python is looked for first, then we have to look for all the components needed by pybind11.
# In particular, if pybind11::embed is used, then we need to find both headers (Development.Module)
# and the shared library (Development.Embed) before pybind11 is discovered.
# With the other order PyBind will discover everything it needs.
# Note on flags:
# - PYTHON_EXECUTABLE is a pybind11 specific flag used by its own (legacy) Python discovery process,
#   it will not affect find_package(Python) calls.
# - Python_EXECUTABLE is a cmake flag used in find_package(Python) to guide the discovery.
# Note that pybind11 can be made to use find_python (instead of its legacy discovery), and thus
# respect Python_EXECUTABLE), via the PYBIND11_FINDPYTHON flag.

# Here, we look for the desired Python version early to avoid any problems with mismatched packages.
# The desired Python environment should be specified ahead of time via -DPython_EXECUTABLE=...
# The optional component is only used for the C++ test suite (to spin up its own interpreter),
# and requires libpython.so to be available on the system.
find_package(Python REQUIRED
    COMPONENTS Interpreter Development.Module
    OPTIONAL_COMPONENTS Development.Embed Development.SABIModule
)

message(STATUS "ENABLE_OPENQASM is ${ENABLE_OPENQASM}.")
message(STATUS "ENABLE_OQD is ${ENABLE_OQD}.")
message(STATUS "LIGHTNING_CLIFFISH_LIB is '${LIGHTNING_CLIFFISH_LIB}'.")

message(STATUS "ENABLE_CODE_COVERAGE is ${ENABLE_CODE_COVERAGE}.")
message(STATUS "ENABLE_ADDRESS_SANITIZER is ${ENABLE_ADDRESS_SANITIZER}.")
message(STATUS "RUNTIME_ENABLE_WARNINGS is ${RUNTIME_ENABLE_WARNINGS}.")

if(ENABLE_ADDRESS_SANITIZER)
    add_compile_options(-fsanitize=address)
    add_link_options(-fsanitize=address)
endif()

if(RUNTIME_ENABLE_WARNINGS)
    set(CMAKE_COMPILE_WARNING_AS_ERROR ON)
    add_compile_options(-Wall)
endif()

######################
## Packaging set-up ##
######################

include(GNUInstallDirs)

# Adds <targets> to the catalyst-runtime-dev component. Call it from the CMakeLists.txt that defines
# the target; install() attaches the rule to the calling directory, so the usual directory scoping
# applies.
#
# Defined ahead of the libraries, because each one registers its own install rule via
# catalyst_runtime_install() as it is defined. The package therefore follows the ENABLE_* options
# for free: a target that was never configured cannot register a rule, and there is no central list
# to keep in sync.
function(catalyst_runtime_install)
    install(TARGETS ${ARGN}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT catalyst-runtime-dev
        RUNTIME DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT catalyst-runtime-dev)

    # Nothing reads this back to decide what to install; it only exists so the configure log can
    # report what a given set of options produced.
    set_property(GLOBAL APPEND PROPERTY CATALYST_RUNTIME_INSTALL_TARGETS ${ARGN})
endfunction(catalyst_runtime_install)

# rt_capi dlopen()s catalyst_callback_registry.so, and the transport/executor libs load their
# siblings the same way. CMake strips the build-tree RPATH on install, so the installed libs need
# their own $ORIGIN lookup to stay loadable once unpacked somewhere else. Every target created below
# inherits this as its INSTALL_RPATH; tests/ picks it up too, harmlessly, since nothing there is
# installed.
if(APPLE)
    set(CMAKE_INSTALL_RPATH "@loader_path")
else()
    set(CMAKE_INSTALL_RPATH "$ORIGIN")
endif()

add_subdirectory(lib)
add_subdirectory(tests)

if(APPLE AND (${CMAKE_SYSTEM_PROCESSOR} STREQUAL arm64))
    if(CMAKE_OSX_SYSROOT)
        get_filename_component(OSX_SDK_PATH ${CMAKE_OSX_SYSROOT} REALPATH)
    else()
        message(STATUS "CMAKE_OSX_SYSROOT is not set, using xcrun to get the official SDK path")

        # Use xcrun to get the official SDK path
        execute_process(
            COMMAND xcrun --sdk macosx --show-sdk-path
            OUTPUT_VARIABLE XCRUN_SDK_PATH
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE XCRUN_RESULT
        )

        if(XCRUN_RESULT EQUAL 0)
            # get the realpath of the SDK path
            get_filename_component(OSX_SDK_PATH ${XCRUN_SDK_PATH} REALPATH)
        else()
            message(FATAL_ERROR "Failed to get the official SDK path since xcrun failed, try to run "
                    "`xcrun --sdk macosx --show-sdk-path` for more information")
        endif()
    endif()

    # macOS 26+ SDKs ship LAPACK 3.12.x. The 32-bit Accelerate preset does not
    # provide the Index-64 symbols that build still links.
    set(LAPACKE_INDEX64_ARG)
    if(EXISTS "${OSX_SDK_PATH}/SDKSettings.json")
        file(READ "${OSX_SDK_PATH}/SDKSettings.json" _osx_sdk_settings)
        string(JSON OSX_SDK_VERSION GET "${_osx_sdk_settings}" Version)
        if(OSX_SDK_VERSION VERSION_GREATER_EQUAL "26.0")
            set(LAPACKE_INDEX64_ARG "-DBUILD_INDEX64_EXT_API=OFF")
        endif()
    endif()

    # Let CMake figure out, if rebuilding the external project is necessary
    ExternalProject_Add(lapacke-accelerate
        GIT_REPOSITORY https://github.com/lepus2589/accelerate-lapacke.git
        GIT_TAG v2.0.0
        PREFIX _lapacke-accelerate
        INSTALL_DIR _lapacke-accelerate/install
        CMAKE_ARGS "--preset" "accelerate-lapacke32"
                   "-DCMAKE_STAGING_PREFIX=<INSTALL_DIR>"
                   "-DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/lib"
                   "-DCMAKE_OSX_SYSROOT=${OSX_SDK_PATH}"
                   ${LAPACKE_INDEX64_ARG}
        INSTALL_COMMAND "${CMAKE_COMMAND}" --build <BINARY_DIR> --target install
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different "<INSTALL_DIR>/lib/liblapacke.3.dylib" "${CMAKE_BINARY_DIR}/lib"
        LOG_CONFIGURE ON
        LOG_INSTALL ON
        LOG_BUILD ON
    )
    add_dependencies(rt_capi lapacke-accelerate)  # automatically build with the runtime
endif()  # if(APPLE AND (${CMAKE_SYSTEM_PROCESSOR} STREQUAL arm64))


##############################
## Catalyst Runtime Version ##
##############################

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/../cmake")  # shared with mlir/
include(ReadCatalystVersion)

require_catalyst_version(CATALYST_PACKAGE_VERSION)


###########################################################################
## Generate a Catalyst Runtime Development Package for External Projects ##
###########################################################################

# The shared libraries and Python modules are installed by the CMakeLists.txt that defines them,
# through catalyst_runtime_install() (see the packaging set-up above). What follows covers the
# artifacts that have no target behind them.

get_property(CATALYST_RUNTIME_INSTALL_TARGETS
    GLOBAL PROPERTY CATALYST_RUNTIME_INSTALL_TARGETS)
list(SORT CATALYST_RUNTIME_INSTALL_TARGETS)
message(STATUS "Catalyst runtime package targets: ${CATALYST_RUNTIME_INSTALL_TARGETS}")

# Public runtime headers (QuantumDevice.hpp, RuntimeCAPI.h, Types.h, ...). These only include each
# other, so the directory is self-contained.
install(DIRECTORY ${runtime_includes}/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    COMPONENT catalyst-runtime-dev
    FILES_MATCHING
      PATTERN "*.h"
      PATTERN "*.hpp")

# Device capability configs. runtime/lib/backend/CMakeLists.txt configure_file()s one per enabled
# backend into ${CMAKE_BINARY_DIR}/lib/backend, so globbing picks up exactly the backends this build
# enabled. That runs during add_subdirectory(lib) above, i.e. before this glob is evaluated.
#
# The destination is load-bearing: _load_device_capabilities() in
# frontend/catalyst/device/qjit_device.py resolves a config as
# <RUNTIME_LIB_DIR>/../lib/backend/<device_name>.toml, so these must sit in lib/backend/ beside the
# libs, and not, for example, in share/.
file(GLOB CATALYST_DEVICE_CONFIG_FILES "${CMAKE_BINARY_DIR}/lib/backend/*.toml")
if(CATALYST_DEVICE_CONFIG_FILES)
    install(FILES ${CATALYST_DEVICE_CONFIG_FILES}
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/backend
        COMPONENT catalyst-runtime-dev)
else()
    message(WARNING "No device config (*.toml) files found in ${CMAKE_BINARY_DIR}/lib/backend")
endif()

# On macOS arm64 the LAPACKE-over-Accelerate shim is built by ExternalProject and copied next to our
# libs; rt_capi needs it at load time. FILES_MATCHING makes this a no-op on platforms/configurations
# where it was never built.
if(APPLE AND (${CMAKE_SYSTEM_PROCESSOR} STREQUAL arm64))
    install(DIRECTORY "${CMAKE_BINARY_DIR}/lib/"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT catalyst-runtime-dev
        FILES_MATCHING PATTERN "liblapacke*")
endif()

# Install license to share/doc/catalyst_runtime/LICENSE
install(FILES "${PROJECT_SOURCE_DIR}/../LICENSE"
    DESTINATION "${CMAKE_INSTALL_DOCDIR}"
    COMPONENT catalyst-runtime-dev)

set(CPACK_PACKAGE_NAME "catalyst-runtime-dev")
set(CPACK_PACKAGE_VERSION "${CATALYST_PACKAGE_VERSION}")
set(CPACK_PACKAGE_VENDOR "Xanadu Quantum Technologies Inc.")
set(CPACK_RESOURCE_FILE_LICENSE "${PROJECT_SOURCE_DIR}/../LICENSE")
set(CPACK_GENERATOR "TGZ")
set(CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_NAME}-${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}-${CATALYST_PACKAGE_VERSION}")
set(CPACK_ARCHIVE_COMPONENT_INSTALL ON)
set(CPACK_INCLUDE_TOPLEVEL_DIRECTORY ON)                   # extracts into catalyst-runtime-dev-<ver>
set(CPACK_COMPONENT_INCLUDE_TOPLEVEL_DIRECTORY ON)         # as above
set(CPACK_COMPONENTS_ALL catalyst-runtime-dev)             # only catalyst-runtime-dev component
set(CPACK_COMPONENTS_GROUPING ALL_COMPONENTS_IN_ONE)       # single archive, no per-component split

message(STATUS "Preparing CPack package ${CPACK_PACKAGE_FILE_NAME}")
include(CPack)
