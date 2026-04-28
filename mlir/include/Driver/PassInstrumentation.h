// Copyright 2026 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "mlir/Pass/PassInstrumentation.h"

#include "Driver/CompilerDriver.h"

#include "Timer.h"

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
