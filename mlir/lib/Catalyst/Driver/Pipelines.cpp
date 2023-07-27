// Copyright 2023 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "Catalyst/Driver/Pipelines.h"
#include "Catalyst/Driver/CompilerDriver.h"
#include "Catalyst/Driver/Support.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "llvm/Support/FileSystem.h"

#include <filesystem>
#include <list>

using namespace mlir;
namespace fs = std::filesystem;

namespace {

std::string joinPasses(const Pipeline::PassList &passes)
{
    std::string joined;
    llvm::raw_string_ostream stream{joined};
    llvm::interleaveComma(passes, stream);
    return joined;
}

struct CatalystIRPrinterConfig : public PassManager::IRPrinterConfig {
    typedef std::function<LogicalResult(Pass*, PrintCallbackFn print)> PrintHandler;
    PrintHandler printHandler;

    CatalystIRPrinterConfig(PrintHandler printHandler) :
        IRPrinterConfig (/*printModuleScope=*/true), printHandler(printHandler)
    {
    }

    void printAfterIfEnabled(Pass *pass, Operation *operation,
                             PrintCallbackFn printCallback) final {
        if(failed(printHandler(pass, printCallback))) {
            operation->emitError("IR printing failed");
        }
    }
};

} // namespace

LogicalResult catalyst::runDefaultLowering(const CompilerSpec &spec,
                                           const CompilerOptions &options,
                                           ModuleOp moduleOp,
                                           CompilerOutput::PipelineOutputs &outputs)
{
    auto pm = PassManager::on<ModuleOp>(options.ctx, PassManager::Nesting::Implicit);

    std::unordered_map<void*, std::list<Pipeline::Name>> pipelineTailMarkers;
    for (const auto &pipeline : spec.pipelinesCfg) {
        if (failed(parsePassPipeline(joinPasses(pipeline.passes), pm, options.diagnosticStream))) {
            return failure();
        }
        PassManager::pass_iterator p = pm.end();
        void *lastPass = &(*(p-1));
        pipelineTailMarkers[lastPass].push_back(pipeline.name);
    }

    if (options.keepIntermediate) {

        {
            std::string tmp;
            { llvm::raw_string_ostream s{tmp}; s << moduleOp; }
            std::string outFile = fs::path(options.moduleName.str()).replace_extension(".mlir");
            if (failed(catalyst::dumpToFile(options, outFile, tmp))) {
                return failure();
            }
        }

        {
            size_t pipelineIdx = 0;
            auto printHandler = [&](Pass* pass, CatalystIRPrinterConfig::PrintCallbackFn print) -> LogicalResult {
                auto res = pipelineTailMarkers.find(pass);
                if(res != pipelineTailMarkers.end()) {
                    for( const auto &pn : res->second) {
                        std::string outFile = fs::path(std::to_string(pipelineIdx++) + "_" + pn).replace_extension(".mlir");
                        {llvm::raw_string_ostream s{outputs[pn]}; print(s);}
                        if(failed(catalyst::dumpToFile(options, outFile, outputs[pn]))) {
                            return failure();
                        }
                    }
                }
                return success();
            };

            pm.enableIRPrinting(
                std::unique_ptr<PassManager::IRPrinterConfig>(new CatalystIRPrinterConfig(printHandler)));
        }
    }

    if (failed(pm.run(moduleOp))) {
        return failure();
    }

    return success();
}
