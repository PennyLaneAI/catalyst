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

#include <fstream>
#include <memory>
#include <string>

#include "llvm/ADT/StringMap.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "stablehlo/dialect/StablehloOps.h"

#include "Catalyst/Analysis/ResourceAnalysis.h"
#include "Catalyst/Analysis/ResourceResult.h"
#include "Quantum/Transforms/Passes.h"

#include "GraphPrep.hpp"
#include "ResourceProjection.hpp"

using namespace mlir;

namespace catalyst {
namespace quantum {

#define GEN_PASS_DEF_RESOURCEESTIMATEPASS
#include "Quantum/Transforms/Passes.h.inc"

struct ResourceEstimatePass : public impl::ResourceEstimatePassBase<ResourceEstimatePass> {
    using ResourceEstimatePassBase::ResourceEstimatePassBase;

    void runOnOperation() final {
        ModuleOp module = getOperation();

        ResourceAnalysis analysis(module, {}, /*collectDetailedOperations=*/true);
        FailureOr<llvm::StringMap<ResourceResult>> estimates =
            projectDecomposition(module, analysis,
                                 {
                                     .targetGateSetOption = targetGateSetOption,
                                     .fixedDecompsOption = fixedDecompsOption,
                                     .excludeDecompsOption = excludeDecompsOption,
                                     .bytecodeRulesFile = bytecodeRulesFile,
                                     .libQPDPath = libQPDPath,
                                     .libpythonPath = libpythonPath,
                                 },
                                 verboseOption);
        if (failed(estimates)) {
            return signalPassFailure();
        }

        if (outputJson) {
            std::string jsonStr = buildJsonString(*estimates);
            if (outputFname.empty()) {
                llvm::outs() << jsonStr;
            } else {
                writeJsonToFile(jsonStr, outputFname);
            }
        }
    }

  private:
    /// Serialize the estimates into a JSON string. The JSON borrows keys from `estimates`, so it
    /// must be printed while they are alive.
    static std::string buildJsonString(const llvm::StringMap<ResourceResult> &estimates) {
        llvm::json::Object root;
        for (const auto &funcEntry : estimates) {
            root[funcEntry.getKey()] = funcEntry.getValue().toJson();
        }

        llvm::json::Value jsonValue(std::move(root));
        return llvm::formatv("{0:2}", jsonValue).str() + "\n";
    }

    /// Write JSON to a file.
    static void writeJsonToFile(const std::string &jsonStr, const std::string &fileName) {
        std::ofstream ofile(fileName);
        if (!ofile.is_open()) {
            llvm::errs() << "Error: could not open resource output file: " << fileName << "\n";
            return;
        }
        ofile << jsonStr;
    }
};

} // namespace quantum
} // namespace catalyst
