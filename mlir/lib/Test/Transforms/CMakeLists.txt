add_mlir_library(MLIRCatalystTest
    TestPass.cpp
    PythonCallbackTest.cpp

    DEPENDS
    MLIRCatalystTestIncGen

    LINK_LIBS PRIVATE
    pybind11::embed

    LINK_LIBS PUBLIC
    TestUtils
)

target_include_directories(MLIRCatalystTest PUBLIC
    .
    ${PROJECT_SOURCE_DIR}/include
    ${CMAKE_BINARY_DIR}/include
)
