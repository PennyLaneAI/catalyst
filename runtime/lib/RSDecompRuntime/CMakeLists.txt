set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(rt_rsdecomp SHARED RSDecomp.cpp Rings.cpp)

# Boost targets are already defined by the parent runtime/lib/CMakeLists.txt.
# rt_rsdecomp is a SHARED library, so PUBLIC linking ensures Boost symbols
# are resolved at load time.
if(CATALYST_BOOST_FETCHED)
    target_link_libraries(rt_rsdecomp
        PUBLIC Boost::multiprecision Boost::random
    )
else()
    target_link_libraries(rt_rsdecomp
        PUBLIC Boost::random
    )
endif()

target_include_directories(rt_rsdecomp
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

set_property(TARGET rt_rsdecomp PROPERTY POSITION_INDEPENDENT_CODE ON)
