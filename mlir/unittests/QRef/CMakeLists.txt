add_catalyst_unittest(CatalystQRefUnitTests
  InterfaceTest.cpp
)

target_link_libraries(CatalystQRefUnitTests PRIVATE
  MLIRArithDialect
  MLIRFuncDialect
  MLIRQRef
  MLIRQuantum
  MLIRIR
  MLIRParser
)
