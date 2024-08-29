#pragma once

using namespace mlir;

namespace catalyst {

namespace gradient {

void registerBufferizableOpInterfaceExternalModels(mlir::DialectRegistry &registry);

}

} // namespace catalyst