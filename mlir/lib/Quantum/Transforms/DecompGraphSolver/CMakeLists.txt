# CMake 3.30 removed the bundled FindBoost module (CMP0167).
# Set to NEW so find_package uses Boost's own BoostConfig.cmake,
# or falls through to FetchContent if not found on the system.
if(POLICY CMP0167)
    cmake_policy(SET CMP0167 NEW)
endif()

# This is the version tested and shipped with Ubuntu 24 apt.
# Using older versions (e.g. 1.66 on rhel8 or 1.74 on Ubuntu 22)
# can cause failure, or will require linking the Boost random library
# explicitly. When using 1.83 or newer, the decompsolver target
# will not require external linking to Boost libraries.
set(MIN_BOOST_VERSION 1.83.0)

include(FetchContent)

# Locally we don't need to check for Boost::multiprecision explicitly.
# Linking Boost::random automatically includes multiprecision headers
# Explicitly adding it here and target_link_libraries will cause failure.
find_package(Boost ${MIN_BOOST_VERSION} COMPONENTS graph QUIET)

if (Boost_FOUND)
    message(STATUS "Found System Boost ${Boost_VERSION}")
    set(CATALYST_BOOST_FETCHED FALSE CACHE BOOL "Boost was found on the system" FORCE)
else()
    message(STATUS "Boost >= ${MIN_BOOST_VERSION} not found on system. Fetching...")

    # set(BOOST_INCLUDE_LIBRARIES graph)
    # set(BOOST_ENABLE_CMAKE ON)

    # Ensure BOOST_ENABLE_CMAKE is set BEFORE MakeAvailable
    set(BOOST_ENABLE_CMAKE ON CACHE BOOL "")

    FetchContent_Declare(
        Boost
        URL https://github.com/boostorg/boost/releases/download/boost-1.89.0/boost-1.89.0-cmake.tar.xz
        URL_HASH SHA256=67acec02d0d118b5de9eb441f5fb707b3a1cdd884be00ca24b9a73c995511f74
        DOWNLOAD_EXTRACT_TIMESTAMP ON
        SYSTEM # required to prevent deprecation warnings->errors on macOS
    )

    FetchContent_MakeAvailable(Boost)


    set(BOOST_TARGETS boost_container boost_graph boost_core boost_assert boost_throw_exception boost_config)

    foreach(t_name IN LISTS BOOST_TARGETS)
        if(TARGET ${t_name})
            # Check if the target is an INTERFACE library
            get_target_property(target_type ${t_name} TYPE)

            if (target_type STREQUAL "INTERFACE_LIBRARY")
                # For header-only targets, we set the options as INTERFACE
                # so they apply to anyone who includes them (like your code)
                target_compile_options(${t_name} INTERFACE -Wno-error -w)
            else()
                # For compiled targets (like boost_container), we use PRIVATE
                # so Boost's own .c/.cpp files compile without -Werror
                target_compile_options(${t_name} PRIVATE -Wno-error -w)
            endif()
        endif()
    endforeach()

endif()


add_library(decompsolver OBJECT DGBuilder.cpp DGSolver.cpp)

target_link_libraries(decompsolver
    PRIVATE Boost::graph
)

target_include_directories(decompsolver PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${Boost_INCLUDE_DIRS}
)

set_property(TARGET decompsolver PROPERTY POSITION_INDEPENDENT_CODE ON)
