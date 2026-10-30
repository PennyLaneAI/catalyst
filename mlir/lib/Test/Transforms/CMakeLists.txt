add_mlir_library(MLIRCatalystTest
    TestPass.cpp

    DEPENDS
    MLIRCatalystTestIncGen
)

target_include_directories(MLIRCatalystTest PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
