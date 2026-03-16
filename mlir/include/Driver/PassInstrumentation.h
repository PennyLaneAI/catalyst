#pragma once

#include "Driver/CompilerDriver.hpp"
#include "Timer.hpp"
#include "mlir/Pass/PassInstrumentation.h"

namespace catalyst {

class CatalystPassInstrumentation : public mlir::PassInstrumentation {
  public:
    CatalystPassInstrumentation(const driver::CompilerOptions &options,
                                driver::CompilerOutput &output, catalyst::utils::Timer<> &timer);

    void runBeforePass(mlir::Pass *pass, mlir::Operation *operation) override;
    void runAfterPass(mlir::Pass *pass, mlir::Operation *operation) override;
    void runAfterPassFailed(mlir::Pass *pass, mlir::Operation *operation) override;

  private:
    void dumpIRAfterPass(mlir::Pass *pass, mlir::Operation *op);

    const driver::CompilerOptions &options;
    driver::CompilerOutput &output;
    catalyst::utils::Timer<> &timer;
    // Store fingerprints before each pass to detect changes
    llvm::DenseMap<mlir::Pass *, std::optional<mlir::OperationFingerPrint>> beforePassFingerprints;
};

} // namespace catalyst