add_mlir_library(MLIRCatalystTest
    TestPass.cpp

    DEPENDS
    MLIRCatalystTestIncGen
)

target_include_directories(MLIRCatalystTest PUBLIC
                           .
                           ${PROJECT_SOURCE_DIR}/include
                           ${CMAKE_BINARY_DIR}/include)
