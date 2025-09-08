// Copyright 2025 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#define DEBUG_TYPE "quantum-specs-info"

#include <nlohmann/json.hpp>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"

#include "Quantum/IR/QuantumDialect.h"
#include "Quantum/IR/QuantumOps.h"
#include "Quantum/IR/QuantumInterfaces.h"
#include "QEC/IR/QECDialect.h"

using namespace llvm;
using namespace mlir;
using namespace catalyst;
using namespace catalyst::quantum;
using namespace catalyst::qec;
using json = nlohmann::json;

namespace catalyst {
namespace quantum {

#define GEN_PASS_DEF_QUANTUMSPECSINFOPASS
#define GEN_PASS_DECL_QUANTUMSPECSINFOPASS

#include "Quantum/Transforms/Passes.h.inc"

struct QuantumSpecsInfoPass : public impl::QuantumSpecsInfoPassBase<QuantumSpecsInfoPass> {
    using impl::QuantumSpecsInfoPassBase<QuantumSpecsInfoPass>::QuantumSpecsInfoPassBase;

    LogicalResult printQuantumSpecs()
    {
        llvm::BumpPtrAllocator stringAllocator;
        llvm::DenseMap<StringRef, llvm::DenseMap<StringRef, int>> PPMSpecs;

        int numQuantumGates=0;
        int numMeasureOp=0;
        int numPPRotation=0;
        int numPPMeasurement=0;
        int numTotal=0;

        // Walk over all operations in the IR (could be ModuleOp or FuncOp)
        WalkResult wr = getOperation()->walk([&](Operation *op) {

            //if (auto qOp = dyn_cast<quantum::QuantumGate>(op)) {
            if (isa<quantum::QuantumGate>(op)) {    
                auto parentFuncOp = op->getParentOfType<func::FuncOp>();
                StringRef funcName = parentFuncOp.getName();

                //llvm::errs() << funcName;
                PPMSpecs[funcName]["QuantumGate_Count"] = ++numQuantumGates;
                PPMSpecs[funcName]["Total_Count"] = ++numTotal;
                //numQuantumGates++;
                return WalkResult::advance();
            }

            if (isa<quantum::MeasureOp>(op)) {    
                auto parentFuncOp = op->getParentOfType<func::FuncOp>();
                StringRef funcName = parentFuncOp.getName();

                //llvm::errs() << funcName;
                PPMSpecs[funcName]["MeasureOp_Count"] = ++numMeasureOp;
                PPMSpecs[funcName]["Total_Count"] = ++numTotal;
                //numQuantumGates++;
                return WalkResult::advance();
            }

            // Count PPMs
            else if (isa<qec::PPMeasurementOp>(op)) {
                auto parentFuncOp = op->getParentOfType<func::FuncOp>();
                StringRef funcName = parentFuncOp.getName();
                PPMSpecs[funcName]["PPMeasurement_Count"] = ++numPPMeasurement;
                PPMSpecs[funcName]["Total_Count"] = ++numTotal;
                return WalkResult::advance();
            }

            // Count PPRs
            else if (isa<qec::PPRotationOp>(op)) {
                auto parentFuncOp = op->getParentOfType<func::FuncOp>();
                StringRef funcName = parentFuncOp.getName();
                PPMSpecs[funcName]["PPRotation_Count"] = ++numPPRotation;
                PPMSpecs[funcName]["Total_Count"] = ++numTotal;
                return WalkResult::advance();
            }

            // Skip other ops
            else {
                return WalkResult::skip();
            }

        });

        if (wr.wasInterrupted()) {
            return failure();
        }

        std::error_code EC;
        //llvm::raw_fd_ostream fileOutputStream("test.txt", EC, llvm::sys::fs::OF_None);
        llvm::raw_fd_ostream fileOutputStream("test.txt", EC, llvm::sys::fs::OF_Append);
        if (EC) {
            llvm::errs() << "Error opening file: " << EC.message() << "\n";
            return failure(); // Handle error
        }
        json PPMSpecsJson = PPMSpecs;
        llvm::outs() << PPMSpecsJson.dump(4)
                     << "\n"; // dump(4) makes an indent with 4 spaces when printing JSON
        fileOutputStream << PPMSpecsJson.dump(4)
                     << "\n"; // dump(4) makes an indent with 4 spaces when printing JSON
        fileOutputStream.flush();
        return success();
    }

    void runOnOperation() final 
    { 
        if (failed(printQuantumSpecs())) {
            signalPassFailure();
        }
    }
};

} //namespace quantum

std::unique_ptr<Pass> createQuantumSpecsInfoPass() { return std::make_unique<QuantumSpecsInfoPass>(); }

} // namespace catalyst