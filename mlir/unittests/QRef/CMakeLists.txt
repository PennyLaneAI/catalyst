add_catalyst_unittest(CatalystQRefUnitTests
  InterfaceTest.cpp
  TestDecomposableGateInterface.cpp
  TestMeasurement.cpp
)

target_link_libraries(CatalystQRefUnitTests PRIVATE
  catalyst-analysis
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
