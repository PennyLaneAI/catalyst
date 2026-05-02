# CMake 3.30 removed the bundled FindBoost module (CMP0167).
# Set to NEW so find_package uses Boost's own BoostConfig.cmake,
# or falls through to FetchContent if not found on the system.
if(POLICY CMP0167)
    cmake_policy(SET CMP0167 NEW)
endif()

# This is the version tested and shipped with Ubuntu 24 apt.
# Using older versions (e.g. 1.66 on rhel8 or 1.74 on Ubuntu 22)
# can cause failure, or will require linking the Boost random library
# explicitly.
set(MIN_BOOST_VERSION 1.83.0)

include(FetchContent)

# Locally we don't need to check for Boost::multiprecision explicitly.
# Linking Boost::random automatically includes multiprecision headers
# Explicitly adding it here and target_link_libraries will cause failure.
find_package(Boost ${MIN_BOOST_VERSION} COMPONENTS random graph QUIET)

if (Boost_FOUND)
    message(STATUS "Found System Boost ${Boost_VERSION}")
    set(CATALYST_BOOST_FETCHED FALSE CACHE BOOL "Boost was found on the system" FORCE)
else()
    message(STATUS "Boost >= ${MIN_BOOST_VERSION} not found on system. Fetching...")

    set(BOOST_INCLUDE_LIBRARIES multiprecision random graph)
    set(BOOST_ENABLE_CMAKE ON)

    FetchContent_Declare(
        Boost
        URL https://github.com/boostorg/boost/releases/download/boost-1.89.0/boost-1.89.0-cmake.tar.xz
        URL_HASH SHA256=67acec02d0d118b5de9eb441f5fb707b3a1cdd884be00ca24b9a73c995511f74
        DOWNLOAD_EXTRACT_TIMESTAMP ON
        SYSTEM # required to prevent deprecation warnings->errors on macOS
    )

    FetchContent_MakeAvailable(Boost)

    set(CATALYST_BOOST_FETCHED TRUE CACHE BOOL "Boost was fetched via FetchContent" FORCE)
endif()


add_subdirectory(capi)
add_subdirectory(backend)
add_subdirectory(registry)
add_subdirectory(RSDecompRuntime)

if(ENABLE_OQD)
add_subdirectory(OQDcapi)
endif()
