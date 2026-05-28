set(LIBRARY_NAME quantum-transforms)

add_subdirectory(DecompGraphSolver)
# add_subdirectory(QuantumPythonCallbacks)

file(GLOB SRC
    AdjointPatterns.cpp
    BufferizableOpInterfaceImpl.cpp
    CancelInversesPatterns.cpp
    ConversionPatterns.cpp
    DecomposeLoweringPatterns.cpp
    DecompCallbacks.cpp
    DecompCallbacksLoader.cpp
    DisentangleCNOT.cpp
    DisentangleSWAP.cpp
    GridsynthPatterns.cpp
    IonsDecompositionPatterns.cpp
    LoopBoundaryOptimizationPatterns.cpp
    MergeRotationsPatterns.cpp
    InstantiateDecompRules.cpp
    SplitMultipleTapes.cpp
    adjoint_lowering.cpp
    cancel_inverses.cpp
    cp_global_buffers.cpp
    decompose_lowering.cpp
    dynamic_one_shot.cpp
    emit_catalyst_pyface.cpp
    graph_decomposition.cpp
    gridsynth.cpp
    ions_decompositions.cpp
    loop_boundary_optimization.cpp
    merge_global_phase.cpp
    merge_rotation.cpp
    quantum_to_llvm.cpp
    split_non_commuting.cpp
    split_to_single_terms.cpp
)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
    ${dialect_libs}
    ${conversion_libs}
    MLIRQuantum
    ExternalStablehloLib
)

set(DEPENDS
    MLIRQuantumPassIncGen
)

add_mlir_library(${LIBRARY_NAME} STATIC
    ${SRC}
    LINK_LIBS
    PRIVATE
    ${LIBS}
    $<TARGET_OBJECTS:decompsolver>
    DEPENDS
    ${DEPENDS}
)
target_compile_features(${LIBRARY_NAME} PUBLIC cxx_std_20)
target_include_directories(${LIBRARY_NAME} PUBLIC
                           .
                           ${PROJECT_SOURCE_DIR}/include
                           ${CMAKE_BINARY_DIR}/include
                           DecompGraphSolver/
                       )

# Discover the libpython SHARED-library filename the active Python uses at
# runtime so the decomp-callbacks loader can dlopen it before opening the
# plugin .so.
#
# Note that we need the SONAME of the shared library, NOT the link-time name.
if (CATALYST_ENABLE_PYTHON_CALLBACKS)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    # Resolve both the bare SONAME and the absolute path to libpython.
    #   - SONAME is the portable fallback used by wheel installs
    #   - Absolute path is also needed for dev builds, to ensure that dlopen
    #     doesn't accidentally pick up a same-version system libpython whose
    #     sys.prefix points at.
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "
import os, sys, sysconfig
inst = sysconfig.get_config_var('INSTSONAME') or ''
ld   = sysconfig.get_config_var('LDLIBRARY')  or ''
soname = ''
for cand in (inst, ld):
    if cand and ('.so' in cand or '.dylib' in cand):
        soname = cand; break
if not soname:
    ver = '%d.%d' % (sys.version_info.major, sys.version_info.minor)
    soname = 'libpython' + ver + ('.dylib' if sys.platform == 'darwin' else '.so.1.0')
libdir = sysconfig.get_config_var('LIBDIR') or ''
full   = os.path.join(libdir, soname) if libdir else ''
if full and not os.path.exists(full):
    full = ''
print(soname)
print(full)
"
        OUTPUT_VARIABLE _libpython_query_out
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _libpython_query_rc
    )
    if (NOT _libpython_query_rc EQUAL 0 OR _libpython_query_out STREQUAL "")
        message(FATAL_ERROR
            "Could not determine libpython shared-library filename. "
            "The decomp-callbacks loader needs this to dlopen libpython at runtime.")
    endif()
    # Split the two-line output into SONAME (line 1) and absolute path (line 2).
    string(REPLACE "\n" ";" _libpython_query_lines "${_libpython_query_out}")
    list(LENGTH _libpython_query_lines _libpython_query_n)
    list(GET _libpython_query_lines 0 _libpython_soname)
    set(_libpython_path "")
    if (_libpython_query_n GREATER_EQUAL 2)
        list(GET _libpython_query_lines 1 _libpython_path)
    endif()

    message(STATUS "Catalyst decomp-callbacks loader libpython SONAME: ${_libpython_soname}")
    if (_libpython_path)
        message(STATUS "Catalyst decomp-callbacks loader libpython path:   ${_libpython_path}")
    endif()

    # Build the set of -D defines for the loader TU.
    set(_loader_defines "CATALYST_LIBPYTHON_SONAME=\"${_libpython_soname}\"")
    if (_libpython_path)
        list(APPEND _loader_defines "CATALYST_LIBPYTHON_PATH=\"${_libpython_path}\"")
    endif()
    set_source_files_properties(DecompCallbacksLoader.cpp PROPERTIES
        COMPILE_DEFINITIONS "${_loader_defines}"
        COMPILE_OPTIONS "-fexceptions"
    )
endif()
