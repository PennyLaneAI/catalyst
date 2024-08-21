#pragma once

using namespace mlir;

namespace catalyst {

namespace quantum {

void registerBufferizableOpInterfaceExternalModels(mlir::DialectRegistry &registry);

}

} // namespace catalyst
