add_subdirectory(Interfaces)

add_catalyst_unittest(CatalystQuantumUnitTests
  TestMeasurement.cpp
)

target_link_libraries(CatalystQuantumUnitTests PRIVATE
  catalyst-analysis
  MLIRFuncDialect
  MLIRQuantum
  MLIRIR
  MLIRParser
)
