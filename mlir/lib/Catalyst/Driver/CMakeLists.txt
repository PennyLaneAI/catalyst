if(WIN32)
    set(EXTERNAL_LIB Version)
elseif(UNIX AND NOT APPLE)
    set(EXTERNAL_LIB z tinfo)
elseif(APPLE)
    set(EXTERNAL_LIB z ncurses)
endif()

add_subdirectory(./../../../Enzyme/enzyme ${CMAKE_CURRENT_BINARY_DIR}/Enzyme/enzyme)

include_directories(../../../Enzyme/enzyme/Enzyme)

# Taken from llvm/tools/llc/CMakeLists.txt
# TODO: May be possible to simplify linked libs
set(LLVM_LINK_COMPONENTS
  AllTargetsAsmParsers
  AllTargetsCodeGens
  AllTargetsDescs
  AllTargetsInfos
  Analysis
  AsmPrinter
  CodeGen
  CodeGenTypes
  Core
  IRReader
  MC
  MIRParser
  Remarks
  ScalarOpts
  SelectionDAG
  Support
  Target
  TargetParser
  TransformUtils
  Vectorize
  )

add_mlir_library(CatalystCompilerDriver
    CompilerDriver.cpp
    CatalystLLVMTarget.cpp

    LINK_LIBS PRIVATE
    MhloRegisterDialects
    StablehloRegister
    ${ALL_MHLO_PASSES}
    ${EXTERNAL_LIB}

    LLVMEnzyme-${LLVM_VERSION_MAJOR}
)
