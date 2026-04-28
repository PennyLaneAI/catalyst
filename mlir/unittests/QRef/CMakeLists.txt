add_catalyst_unittest(CatalystQRefUnitTests
  InterfaceTest.cpp
)

target_link_libraries(CatalystQRefUnitTests PRIVATE
  MLIRArithDialect
  MLIRFuncDialect
  MLIRMBQC
  MLIRQRef
  MLIRQuantum
  MLIRIR
  MLIRParser
  MLIRPBC
)
