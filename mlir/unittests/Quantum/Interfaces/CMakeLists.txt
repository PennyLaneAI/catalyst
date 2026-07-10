add_catalyst_unittest(CatalystQuantumInterfaceTests
    TestDecomposableGateInterface.cpp
)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

target_link_libraries(CatalystQuantumInterfaceTests PRIVATE
    ${dialect_libs}
    MLIRQuantum
    MLIRQRef
)
