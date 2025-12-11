set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(MIN_BOOST_VERSION 1.74.0)

find_package(Boost ${MIN_BOOST_VERSION} QUIET)

if(Boost_FOUND)
    message(STATUS "Found system Boost: ${Boost_VERSION}")
else()
    message(STATUS "Boost >= ${MIN_BOOST_VERSION} not found on system. Fetching Boost 1.89.0...")
    
    include(FetchContent)
    
    set(BOOST_INCLUDE_LIBRARIES multiprecision random)
    set(BOOST_ENABLE_CMAKE ON)

    FetchContent_Declare(
        Boost
        URL https://github.com/boostorg/boost/releases/download/boost-1.89.0/boost-1.89.0-cmake.tar.xz
        URL_HASH SHA256=67acec02d0d118b5de9eb441f5fb707b3a1cdd884be00ca24b9a73c995511f74
        DOWNLOAD_EXTRACT_TIMESTAMP ON
    )
    
    FetchContent_MakeAvailable(Boost)
endif()

add_library(rt_rsdecomp SHARED RSDecomp.cpp Rings.cpp CliffordData.cpp)

target_include_directories(rt_rsdecomp
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

target_link_libraries(rt_rsdecomp PUBLIC Boost::headers)

set_property(TARGET rt_rsdecomp PROPERTY POSITION_INDEPENDENT_CODE ON)
