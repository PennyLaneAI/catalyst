add_catalyst_unittest(CatalystQRefUnitTests
  InterfaceTest.cpp
  TestDecomposableOpInterface.cpp
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
