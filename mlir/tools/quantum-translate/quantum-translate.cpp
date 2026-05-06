
#include "Quantum/IR/QuantumDialect.h"
#include "mlir/InitAllTranslations.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"

namespace mlir {
void registerToQASM3Translation();
}

int main(int argc, char **argv)
{
    mlir::registerAllTranslations();
    mlir::registerToQASM3Translation();

    return failed(mlir::mlirTranslateMain(argc, argv, "Quantum Translation Tool"));
}
