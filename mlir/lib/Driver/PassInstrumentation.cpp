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

#include "Driver/PassInstrumentation.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include "Driver/CompilerDriver.h"
#include "Driver/LineUtils.h"
#include "Driver/Support.h"
#include "Driver/Timer.h"

namespace catalyst {

CatalystPassInstrumentation::CatalystPassInstrumentation(const driver::CompilerOptions &options,
                                                         driver::CompilerOutput &output,
                                                         catalyst::utils::Timer<> &timer)
    : options(options), output(output), timer(timer)
{
}

void CatalystPassInstrumentation::runBeforePass(mlir::Pass *pass, mlir::Operation *operation)
{
    if (this->options.verbosity >= driver::Verbosity::Debug && !this->timer.is_active()) {
        this->timer.start();
    }
    if (this->options.keepIntermediate == driver::SaveTemps::AfterPassChanged) {
        this->beforePassFingerprints[pass] = mlir::OperationFingerPrint(operation);
    }
}

void CatalystPassInstrumentation::runAfterPass(mlir::Pass *pass, mlir::Operation *operation)
{
    // Handle verbosity logging
    if (this->options.verbosity >= driver::Verbosity::Debug) {
        auto pipelineName = pass->getName();
        this->timer.dump(pipelineName.str(), /*add_endl */ false);
        catalyst::utils::LinesCount::call(*operation);
    }

    bool shouldDump = false;

    if (this->options.keepIntermediate == driver::SaveTemps::AfterPass) {
        shouldDump = true;
    }
    else if (this->options.keepIntermediate == driver::SaveTemps::AfterPassChanged) {
        // If change detection is enabled, only dump if IR is changed
        auto it = this->beforePassFingerprints.find(pass);
        if (it != this->beforePassFingerprints.end() && it->second.has_value()) {
            mlir::OperationFingerPrint afterFingerprint(operation);
            shouldDump = (afterFingerprint != it->second.value());
            this->beforePassFingerprints.erase(it);
        }
        else {
            // Fingerprint not found: default to dumping to be safe
            shouldDump = true;
        }
    }

    if (shouldDump) {
        this->dumpIRAfterPass(pass, operation);
    }
}

void CatalystPassInstrumentation::runAfterPassFailed(mlir::Pass *pass, mlir::Operation *operation)
{
    // Always dump on failure for debugging
    this->options.diagnosticStream << "While processing '" << pass->getName().str() << "' pass ";
    std::string tmp;
    llvm::raw_string_ostream s{tmp};
    s << *operation;
    if (this->options.keepIntermediate) {
        driver::dumpToFile(
            this->options,
            this->output.nextPipelineSummaryFilename(pass->getName().str() + "_FAILED"), tmp);
    }

    // Clean up fingerprint if present
    if (this->options.keepIntermediate == driver::SaveTemps::AfterPassChanged) {
        this->beforePassFingerprints.erase(pass);
    }
}

void CatalystPassInstrumentation::dumpIRAfterPass(mlir::Pass *pass, mlir::Operation *op)
{
    auto pipelineName = pass->getName();

    // Save IR after pass
    std::string tmp;
    llvm::raw_string_ostream s{tmp};
    if (this->options.dumpModuleScope) {
        mlir::ModuleOp mod = isa<mlir::ModuleOp>(op) ? cast<mlir::ModuleOp>(op)
                                                     : op->getParentOfType<mlir::ModuleOp>();
        s << mod;
    }
    else {
        s << *op;
    }
    std::string fileName;
    llvm::raw_string_ostream os(fileName);
    os << pipelineName;
    if (auto funcOp = dyn_cast<mlir::func::FuncOp>(op)) {
        os << '_' << funcOp.getName();
    }
    dumpToFile(this->options, this->output.nextPassDumpFilename(fileName), tmp);
}

} // namespace catalyst
