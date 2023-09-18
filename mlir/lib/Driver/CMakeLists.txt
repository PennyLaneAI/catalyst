if(WIN32)
    set(EXTERNAL_LIB Version)
elseif(APPLE)
    set(EXTERNAL_LIB z ncurses)
elseif(UNIX)
    set(EXTERNAL_LIB z tinfo)
endif()

set(ENZYME_STATIC_LIB ON)
add_subdirectory(${ENZYME_SRC_DIR}/enzyme ${CMAKE_CURRENT_BINARY_DIR}/enzyme)

include_directories(${ENZYME_SRC_DIR}/enzyme/Enzyme)

# Taken from llvm/tools/llc/CMakeLists.txt
# TODO: May be possible to simplify linked libs
set(LLVM_LINK_COMPONENTS
  AllTargetsAsmParsers
  AllTargetsCodeGens
  AllTargetsDescs
  AllTargetsInfos
  CodeGen
  Core
  )

add_mlir_library(CatalystCompilerDriver
    CompilerDriver.cpp
    CatalystLLVMTarget.cpp

    LINK_LIBS PRIVATE
    MhloRegisterDialects
    StablehloRegister
    ${ALL_MHLO_PASSES}
    ${EXTERNAL_LIB}

    EnzymeStatic-${LLVM_VERSION_MAJOR}
    DEPENDS
    EnzymeStatic-${LLVM_VERSION_MAJOR}
    )
