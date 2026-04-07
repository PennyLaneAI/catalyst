add_catalyst_unittest(CatalystUtilsTests
    PrintVersion.cpp
)

target_link_libraries(CatalystUtilsTests PRIVATE
    MLIRCatalystUtils
)
