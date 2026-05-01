################################
# catalyst-runtime-driver (host)
################################

find_package(LLVM CONFIG REQUIRED)
message(STATUS "remote_driver: Found LLVM ${LLVM_PACKAGE_VERSION} (${LLVM_DIR})")

set(_llvm_cxx_match_flags)
if(NOT LLVM_ENABLE_RTTI)
    list(APPEND _llvm_cxx_match_flags -fno-rtti)
endif()
if(DEFINED LLVM_ENABLE_EH AND NOT LLVM_ENABLE_EH)
    list(APPEND _llvm_cxx_match_flags -fno-exceptions)
endif()

set(_driver_target_libs)
set(_driver_target_defs)
list(FIND LLVM_TARGETS_TO_BUILD "X86" _has_x86)
list(FIND LLVM_TARGETS_TO_BUILD "AArch64" _has_aarch64)
if(_has_x86 GREATER -1)
    list(APPEND _driver_target_libs
        X86CodeGen X86AsmParser X86Desc X86Info)
    list(APPEND _driver_target_defs CATALYST_REMOTE_HAS_X86=1)
endif()
if(_has_aarch64 GREATER -1)
    list(APPEND _driver_target_libs
        AArch64CodeGen AArch64AsmParser AArch64Desc AArch64Info)
    list(APPEND _driver_target_defs CATALYST_REMOTE_HAS_AARCH64=1)
endif()
if(NOT _driver_target_libs)
    message(FATAL_ERROR
        "LLVM at ${LLVM_DIR} has neither X86 nor AArch64 in "
        "LLVM_TARGETS_TO_BUILD; catalyst_remote_driver needs at least "
        "one to drive a remote executor.")
endif()

llvm_map_components_to_libnames(_driver_llvm_libs
    core support orcjit jitlink object passes
    ${_driver_target_libs}
)

add_library(catalyst_remote_driver SHARED CatalystRemoteDriver.cpp)

target_include_directories(catalyst_remote_driver PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_include_directories(catalyst_remote_driver SYSTEM PRIVATE
    ${LLVM_INCLUDE_DIRS}
)

separate_arguments(_driver_llvm_defs NATIVE_COMMAND "${LLVM_DEFINITIONS}")
target_compile_definitions(catalyst_remote_driver PRIVATE
    ${_driver_llvm_defs}
    ${_driver_target_defs}
)
target_compile_options(catalyst_remote_driver PRIVATE
    -Wall
    $<$<COMPILE_LANGUAGE:CXX>:${_llvm_cxx_match_flags}>
)

target_link_libraries(catalyst_remote_driver PRIVATE
    ${_driver_llvm_libs}
    pthread
    ${CMAKE_DL_LIBS}
)

set_target_properties(catalyst_remote_driver PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

install(TARGETS catalyst_remote_driver LIBRARY DESTINATION lib)
