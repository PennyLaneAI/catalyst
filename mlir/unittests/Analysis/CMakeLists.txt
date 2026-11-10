add_catalyst_unittest(CatalystAnalysisTests
    ResourceAnalysisTest.cpp
)

target_link_libraries(CatalystAnalysisTests PRIVATE
    catalyst-analysis
    MLIRArithDialect
    MLIRCatalystUtils
    MLIRFuncDialect
    MLIRIR
    MLIRParser
    MLIRPass
    MLIRQuantum
    MLIRSCFDialect
)
