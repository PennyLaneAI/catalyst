add_catalyst_unittest(CatalystRefQuantumUnitTests
  InterfaceTest.cpp
)

target_link_libraries(CatalystRefQuantumUnitTests PRIVATE
  MLIRArithDialect
  MLIRFuncDialect
  MLIRRefQuantum
  MLIRIR
  MLIRParser
)
