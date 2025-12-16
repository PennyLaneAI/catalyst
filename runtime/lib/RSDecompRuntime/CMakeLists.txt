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

    FetchContent_Declare(
        Boost
        URL https://archives.boost.io/release/1.89.0/source/boost_1_89_0.tar.bz2
        URL_HASH SHA256=85a33fa22621b4f314f8e85e1a5e2a9363d22e4f4992925d4bb3bc631b5a0c7a
        DOWNLOAD_EXTRACT_TIMESTAMP ON
    )

    FetchContent_MakeAvailable(Boost)

endif()

target_include_directories(rt_rsdecomp
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

set_property(TARGET rt_rsdecomp PROPERTY POSITION_INDEPENDENT_CODE ON)
