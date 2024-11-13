add_catalyst_unittest(CatalystExampleTests
  Example.cpp
)

target_link_libraries(CatalystExampleTests PRIVATE
  MLIRArithDialect
  MLIRFuncDialect
  MLIRQuantum
  MLIRIR
  MLIRParser
)
