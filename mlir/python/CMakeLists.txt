include(AddMLIRPython)
# top level package (the API has been embedded in a relocatable way).
# TODO: Add an upstream cmake param for this vs having a global here.
add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=mlir_quantum.")


################################################################################
# Sources
################################################################################

declare_mlir_python_sources(QuantumPythonSources)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT QuantumPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/"
  TD_FILE dialects/QuantumOps.td
  SOURCES
  dialects/quantum.py
  DIALECT_NAME quantum)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT QuantumPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/"
  TD_FILE dialects/GradientOps.td
  SOURCES
  dialects/gradient.py
  DIALECT_NAME gradient)

declare_mlir_python_extension(QuantumPythonSources.Extension
  MODULE_NAME _quantumDialects
  ADD_TO_PARENT QuantumPythonSources
  SOURCES
    QuantumExtension.cpp
  EMBED_CAPI_LINK_LIBS
    QuantumCAPI
)

#declare_mlir_python_extension(CatalystPythonDriver
#  MODULE_NAME _catalystDriver
#  ADD_TO_PARENT QuantumPythonSources
#  SOURCES
#    PyCompilerDriver.cpp
#  PRIVATE_LINK_LIBS
#    QuantumCAPI
#)

################################################################################
# Common CAPI
################################################################################

add_library(CatalystPythonDriver MODULE PyCompilerDriver.cpp)

add_mlir_python_common_capi_library(QuantumPythonCAPI
  INSTALL_COMPONENT QuantumPythonModules
  INSTALL_DESTINATION python_packages/quantum/mlir_quantum/compiler_driver
  OUTPUT_DIRECTORY "${MLIR_BINARY_DIR}/python_packages/quantum/mlir_quantum/compiler_driver"
  RELATIVE_INSTALL_ROOT "../../../.."
  DECLARED_SOURCES
    QuantumPythonSources
    MLIRPythonExtension.RegisterEverything
    MLIRPythonSources.Core
)

set(LOCAL_LIBS MLIRPresburger;MLIRAnalysis;MLIRAsmParser;MLIRBytecodeReader;MLIRBytecodeWriter;MLIRBytecodeOpInterface;MLIRAffineToStandard;MLIRAMDGPUToROCDL;MLIRArithAttrToLLVMConversion;MLIRArithToLLVM;MLIRArithToSPIRV;MLIRArmNeon2dToIntr;MLIRAsyncToLLVM;MLIRBufferizationToMemRef;MLIRComplexToLLVM;MLIRComplexToLibm;MLIRComplexToSPIRV;MLIRComplexToStandard;MLIRControlFlowToLLVM;MLIRControlFlowToSPIRV;MLIRFuncToLLVM;MLIRFuncToSPIRV;MLIRGPUToGPURuntimeTransforms;MLIRGPUToNVVMTransforms;MLIRGPUToROCDLTransforms;MLIRGPUToSPIRV;MLIRGPUToVulkanTransforms;MLIRIndexToLLVM;MLIRLinalgToLLVM;MLIRLinalgToStandard;MLIRLLVMCommonConversion;MLIRMathToFuncs;MLIRMathToLibm;MLIRMathToLLVM;MLIRMathToSPIRV;MLIRMemRefToLLVM;MLIRMemRefToSPIRV;MLIRNVGPUToNVVM;MLIROpenACCToSCF;MLIROpenMPToLLVM;MLIRPDLToPDLInterp;MLIRReconcileUnrealizedCasts;MLIRSCFToControlFlow;MLIRSCFToGPU;MLIRSCFToOpenMP;MLIRSCFToSPIRV;MLIRShapeToStandard;MLIRSPIRVToLLVM;MLIRTensorToLinalg;MLIRTensorToSPIRV;MLIRTosaToArith;MLIRTosaToLinalg;MLIRTosaToSCF;MLIRTosaToTensor;MLIRVectorToLLVM;MLIRVectorToGPU;MLIRVectorToSCF;MLIRVectorToSPIRV;MLIRObservers;MLIRDebug;MLIRAffineAnalysis;MLIRAffineDialect;MLIRAffineTransforms;MLIRAffineTransformOps;MLIRAffineUtils;MLIRAMDGPUDialect;MLIRAMDGPUTransforms;MLIRAMDGPUUtils;MLIRArithDialect;MLIRArithValueBoundsOpInterfaceImpl;MLIRArithTransforms;MLIRArithUtils;MLIRArmNeonDialect;MLIRArmSMEDialect;MLIRArmSMETransforms;MLIRArmSVEDialect;MLIRArmSVETransforms;MLIRAsyncDialect;MLIRAsyncTransforms;MLIRAMXDialect;MLIRAMXTransforms;MLIRBufferizationDialect;MLIRBufferizationTransformOps;MLIRBufferizationTransforms;MLIRComplexDialect;MLIRControlFlowDialect;MLIRDLTIDialect;MLIREmitCDialect;MLIRFuncInlinerExtension;MLIRFuncAllExtensions;MLIRFuncDialect;MLIRFuncTransforms;MLIRGPUDialect;MLIRGPUTransforms;MLIRGPUTransformOps;MLIRIndexDialect;MLIRIRDL;MLIRLinalgDialect;MLIRLinalgTransformOps;MLIRLinalgTransforms;MLIRLinalgUtils;MLIRLLVMIRTransforms;MLIRLLVMDialect;MLIRNVVMDialect;MLIRROCDLDialect;MLIRMathDialect;MLIRMathTransforms;MLIRMemRefDialect;MLIRMemRefTransformOps;MLIRMemRefTransforms;MLIRMemRefUtils;MLIRMLProgramDialect;MLIRNVGPUDialect;MLIRNVGPUUtils;MLIRNVGPUTransforms;MLIROpenACCDialect;MLIROpenMPDialect;MLIRPDLDialect;MLIRPDLInterpDialect;MLIRQuantDialect;MLIRQuantUtils;MLIRSCFDialect;MLIRSCFTransformOps;MLIRSCFTransforms;MLIRSCFUtils;MLIRShapeDialect;MLIRShapeOpsTransforms;MLIRSparseTensorEnums;MLIRSparseTensorDialect;MLIRSparseTensorTransforms;MLIRSparseTensorPipelines;MLIRSparseTensorUtils;MLIRSPIRVDialect;MLIRSPIRVModuleCombiner;MLIRSPIRVConversion;MLIRSPIRVTransforms;MLIRSPIRVUtils;MLIRTensorDialect;MLIRTensorInferTypeOpInterfaceImpl;MLIRTensorTilingInterfaceImpl;MLIRTensorTransforms;MLIRTensorTransformOps;MLIRTensorUtils;MLIRTosaDialect;MLIRTosaTransforms;MLIRTransformDialect;MLIRTransformPDLExtension;MLIRTransformDialectTransforms;MLIRTransformDialectUtils;MLIRDialectUtils;MLIRVectorDialect;MLIRMaskableOpInterface;MLIRMaskingOpInterface;MLIRVectorTransforms;MLIRVectorTransformOps;MLIRVectorUtils;MLIRX86VectorDialect;MLIRX86VectorTransforms;MLIRDialect;MLIRIR;MLIRCallInterfaces;MLIRCastInterfaces;MLIRControlFlowInterfaces;MLIRCopyOpInterface;MLIRDataLayoutInterfaces;MLIRDerivedAttributeOpInterface;MLIRDestinationStyleOpInterface;MLIRInferIntRangeInterface;MLIRInferTypeOpInterface;MLIRLoopLikeInterface;MLIRMemorySlotInterfaces;MLIRParallelCombiningOpInterface;MLIRRuntimeVerifiableOpInterface;MLIRShapedOpInterfaces;MLIRSideEffectInterfaces;MLIRTilingInterface;MLIRVectorInterfaces;MLIRViewLikeInterface;MLIRValueBoundsOpInterface;MLIRInferIntRangeCommon;MLIRParser;MLIRPass;MLIRReduce;MLIRRewrite;MLIRSupport;MLIRTableGen;MLIRTargetCpp;MLIRSPIRVDeserialization;MLIRSPIRVSerialization;MLIRSPIRVBinaryUtils;MLIRSPIRVTranslateRegistration;MLIRArmNeonToLLVMIRTranslation;MLIRArmSMEToLLVMIRTranslation;MLIRArmSVEToLLVMIRTranslation;MLIRAMXToLLVMIRTranslation;MLIRBuiltinToLLVMIRTranslation;MLIRGPUToLLVMIRTranslation;MLIRLLVMIRToLLVMTranslation;MLIRLLVMToLLVMIRTranslation;MLIRNVVMToLLVMIRTranslation;MLIROpenACCToLLVMIRTranslation;MLIROpenMPToLLVMIRTranslation;MLIRROCDLToLLVMIRTranslation;MLIRX86VectorToLLVMIRTranslation;MLIRTargetLLVMIRExport;MLIRToLLVMIRTranslationRegistration;MLIRTargetLLVMIRImport;MLIRFromLLVMIRTranslationRegistration;MLIRLspServerSupportLib;MLIRLspServerLib;MLIROptLib;MLIRReduceLib;MLIRTblgenLib;MLIRTranslateLib;MLIRPDLLAST;MLIRPDLLCodeGen;MLIRPDLLODS;MLIRPluginsLib;MLIRTransformUtils;MLIRTransforms;MLIRExecutionEngineUtils;MLIRExecutionEngine;MLIRJitRunner;mlir_float16_utils;MLIRSparseTensorRuntime;mlir_c_runner_utils;mlir_runner_utils;mlir_async_runtime)
target_link_libraries(CatalystPythonDriver PRIVATE pybind11::headers pybind11::module QuantumCAPI QuantumPythonCAPI ${LLVM_LIBS} ${LOCAL_LIBS} LLVMSupport CatalystCompilerDriver MLIRCatalyst MLIRQuantum MLIRGradient)

set_target_properties(CatalystPythonDriver PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${MLIR_BINARY_DIR}/python_packages/quantum/mlir_quantum/compiler_driver)
set_target_properties(CatalystPythonDriver PROPERTIES BUILD_WITH_INSTALL_RPATH OFF BUILD_RPATH $ORIGIN INSTALL_RPATH $ORIGIN )
set_target_properties(CatalystPythonDriver PROPERTIES BUILD_WITH_INSTALL_RPATH OFF BUILD_RPATH ${MLIR_BINARY_DIR}/python_packages/quantum/mlir_quantum/compiler_driver INSTALL_RPATH ${MLIR_BINARY_DIR}/python_packages/quantum/mlir_quantum/compiler_driver)
set_target_properties(CatalystPythonDriver PROPERTIES BUILD_WITH_INSTALL_RPATH OFF BUILD_RPATH ${MLIR_BINARY_DIR}/python_packages/quantum/mlir_quantum/compiler_driver INSTALL_RPATH ${MLIR_BINARY_DIR}/python_packages/quantum/mlir_quantum/_mlir_libs)

################################################################################
# Instantiation of all Python modules
################################################################################

add_mlir_python_modules(QuantumPythonModules
  ROOT_PREFIX "${MLIR_BINARY_DIR}/python_packages/quantum/mlir_quantum/"
  INSTALL_PREFIX "python_packages/quantum/mlir_quantum/"
  DECLARED_SOURCES
    QuantumPythonSources
    MLIRPythonExtension.RegisterEverything
    MLIRPythonSources
  COMMON_CAPI_LINK_LIBS
    QuantumPythonCAPI
  )

add_dependencies(QuantumPythonModules CatalystPythonDriver)
