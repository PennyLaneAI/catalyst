add_llvm_library(StandalonePlugin
        MODULE 
        standalone-plugin.cpp

        DEPENDS
        MLIRStandalone
        PLUGIN_TOOL
        catalyst-cli

        LINK_LIBS
        MLIRStandalone
        )
target_include_directories(
        StandalonePlugin
        PRIVATE
        "${STANDALONE_BINARY_DIR}/include"
)
