add_catalyst_unittest(CatalystQRefUnitTests
  InterfaceTest.cpp
  TestDecomposableGateInterface.cpp
)

target_link_libraries(CatalystQRefUnitTests PRIVATE
  MLIRArithDialect
  MLIRFuncDialect
  MLIRMBQC
  MLIRQRef
  MLIRTestDialect
  MLIRQuantum
  MLIRIR
  MLIRParser
  MLIRPBC
  MLIRMBQC
)
