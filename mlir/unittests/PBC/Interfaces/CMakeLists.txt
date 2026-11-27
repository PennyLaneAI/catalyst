add_catalyst_unittest(CatalystPBCInterfaceTests
    TestDecomposableGateInterface.cpp
)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

target_link_libraries(CatalystPBCInterfaceTests PRIVATE
    ${dialect_libs}
    MLIRQuantum
    MLIRQRef
    MLIRPBC
    MLIRMBQC
)
