set(LIBRARY_NAME PBCAnalysis)

set(SRC
    PBCDepthAnalysis.cpp
)

add_library(${LIBRARY_NAME} STATIC ${SRC})
target_compile_features(${LIBRARY_NAME} PUBLIC cxx_std_20)
target_include_directories(${LIBRARY_NAME} PUBLIC
                           ${PROJECT_SOURCE_DIR}/include
                           ${CMAKE_BINARY_DIR}/include)

# PBCUtils: PBCLayer depth computation.
# catalyst-analysis: ResourceAnalysisRegistry for self-registration.
# No reverse link from those targets into PBCAnalysis.
target_link_libraries(${LIBRARY_NAME} PRIVATE PBCUtils catalyst-analysis)
