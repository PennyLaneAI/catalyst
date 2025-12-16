set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(rt_rsdecomp SHARED RSDecomp.cpp Rings.cpp)

# This is the version tested and shipped with Ubuntu 24 apt.
# Using older versions (e.g. 1.66 on rhel8 or 1.74 on Ubuntu 22)
# can cause failure, or will require linking the Boost random library
# explicitly. When using 1.83 or newer, the rt_rsdecomp target
# will not require external linking to Boost libraries.
set(MIN_BOOST_VERSION 1.83.0)

include(FetchContent)

# Locally we don't need to check for Boost::multiprecision explicitly.
# Linking Boost::random automatically includes multiprecision headers
# Explicitly adding it here and target_link_libraries will cause failure.
find_package(Boost ${MIN_BOOST_VERSION} COMPONENTS random QUIET)

if (Boost_FOUND)
    message(STATUS "Found System Boost ${Boost_VERSION}")

    target_link_libraries(rt_rsdecomp PUBLIC Boost::random)

else()
    message(STATUS "Boost not found. Fetching...")

    set(BOOST_INCLUDE_LIBRARIES multiprecision random)
    set(BOOST_ENABLE_CMAKE ON)

    # Note: Need to use .xz version from github (or for newer require -cmake.tar.gz), as
    # normal release from boost website do not include required cmake files.
    FetchContent_Declare(
        Boost
        URL https://github.com/boostorg/boost/releases/download/boost-1.84.0/boost-1.84.0.tar.xz
        URL_MD5 893b5203b862eb9bbd08553e24ff146a
        DOWNLOAD_EXTRACT_TIMESTAMP ON
    )
    
    FetchContent_MakeAvailable(Boost)

    target_link_libraries(rt_rsdecomp PUBLIC Boost::random Boost::multiprecision)
endif()

target_include_directories(rt_rsdecomp
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

set_property(TARGET rt_rsdecomp PROPERTY POSITION_INDEPENDENT_CODE ON)
