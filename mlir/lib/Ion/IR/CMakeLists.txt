add_mlir_library(MLIRIon
    IonDialect.cpp
    IonInterfaces.cpp
    IonOps.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/Ion

    DEPENDS
    MLIRIonInterfacesIncGen
    MLIRIonOpsIncGen
    MLIRIonEnumsIncGen
)
