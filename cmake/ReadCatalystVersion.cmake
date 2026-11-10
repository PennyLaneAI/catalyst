# Reads the Catalyst version from frontend/catalyst/_version.py.
#
# frontend/catalyst/_version.py is the single source of truth for the Catalyst
# version. mlir/ and runtime/ are configured as independent CMake projects, so
# neither can inherit the version from the other; this module is shared by both
# instead of each carrying its own copy of the parsing logic.
#
# Provides:
#   CATALYST_VERSION_PY_FILE     path to frontend/catalyst/_version.py
#   read_catalyst_version()      lenient read, yields "unknown" on failure
#   require_catalyst_version()   strict read, hard-fails on failure

# This module lives in <repo root>/cmake, so _version.py is one level up.
cmake_path(SET CATALYST_VERSION_PY_FILE NORMALIZE
    "${CMAKE_CURRENT_LIST_DIR}/../frontend/catalyst/_version.py")

# Extracts __version__ from ${input_file} into ${out_var} in the caller's scope,
# or "unknown" (with a warning) if it cannot be read. Callers that cannot
# proceed without a real version should use require_catalyst_version() instead.
function(read_catalyst_version input_file out_var)
    set(version_number "unknown")
    if (NOT EXISTS ${input_file})
        message(WARNING "Could not find ${input_file}")
    else()
        file(READ ${input_file} input_content)
        if (NOT input_content MATCHES "__version__ = \"([^\"]+)\"")
            message(WARNING "Could not find Catalyst version in ${input_file}")
        else()
            set(version_number "${CMAKE_MATCH_1}")
        endif()
    endif()
    set(${out_var} "${version_number}" PARENT_SCOPE)
endfunction(read_catalyst_version)

# As above, but aborts the configure rather than yielding "unknown". Use this
# for anything that would otherwise bake a bogus version into a build artifact,
# such as the CPack package name.
function(require_catalyst_version out_var)
    read_catalyst_version("${CATALYST_VERSION_PY_FILE}" version_number)
    if (version_number STREQUAL "unknown")
        message(FATAL_ERROR
            "Failed to read Catalyst package version from ${CATALYST_VERSION_PY_FILE}")
    endif()
    message(STATUS "Found Catalyst package version: ${version_number}")
    set(${out_var} "${version_number}" PARENT_SCOPE)
endfunction(require_catalyst_version)
