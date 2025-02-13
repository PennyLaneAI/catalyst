find_package(CapnProto REQUIRED)

add_library(qpr-proto STATIC)
target_sources(qpr-proto PRIVATE qpr.capnp.c++)
target_include_directories(qpr-proto PUBLIC . ${CAPNP_INCLUDE_DIRS})
target_link_libraries(qpr-proto ${CAPNP_LIBRARIES})

# -----

set(LIBRARY_NAME qpr-transforms)

file(GLOB SRC
    conversion.cpp
)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
set(LIBS
    ${dialect_libs}
    qpr-proto
)

add_mlir_library(${LIBRARY_NAME} STATIC ${SRC} LINK_LIBS PRIVATE ${LIBS} DEPENDS qpr-proto)
target_include_directories(${LIBRARY_NAME} PUBLIC
                           .
                           ${PROJECT_SOURCE_DIR}/include
                           ${CMAKE_BINARY_DIR}/include)
