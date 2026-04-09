add_catalyst_unittest(CatalystTraceQubitUnitTests
  TraceQubitTest.cpp
)

target_link_libraries(CatalystTraceQubitUnitTests PRIVATE
  MLIRArithDialect
  MLIRFuncDialect
  MLIRSCFDialect
  MLIRQuantum
  MLIRIR
  MLIRParser
)
