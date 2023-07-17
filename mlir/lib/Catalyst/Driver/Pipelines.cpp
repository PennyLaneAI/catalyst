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

using namespace mlir;
namespace fs = std::filesystem;

namespace {
// clang-format off
const static SmallVector<const char *> mhloToCorePasses = {
    "func.func(chlo-legalize-to-hlo)",
    "stablehlo-legalize-to-hlo",
    "func.func(mhlo-legalize-control-flow)",
    "func.func(hlo-legalize-to-linalg)",
    "func.func(mhlo-legalize-to-std)",
    "convert-to-signless",
};

const static SmallVector<const char *> quantumCompilationPasses = {
    "lower-gradients",
    "convert-arraylist-to-memref",
};

const static SmallVector<const char *> bufferizationPasses = {
    "inline",
    "gradient-bufferize",
    "scf-bufferize",
    "convert-tensor-to-linalg",      // tensor.pad
    "convert-elementwise-to-linalg", // Must be run before --arith-bufferize
    "arith-bufferize",
    "empty-tensor-to-alloc-tensor",
    "func.func(bufferization-bufferize)",
    "func.func(tensor-bufferize)",
    "func.func(linalg-bufferize)",
    "func.func(tensor-bufferize)",
    "quantum-bufferize",
    "func-bufferize",
    "func.func(finalizing-bufferize)",
    // "func.func(buffer-hoisting)",
    "func.func(buffer-loop-hoisting)",
    // "func.func(buffer-deallocation)",
    "convert-bufferization-to-memref",
    "canonicalize",
    // "cse",
    "cp-global-memref",
};

const static SmallVector<const char *> lowerToLLVMPasses = {
    "func.func(convert-linalg-to-loops)",
    "convert-scf-to-cf",
    // This pass expands memref operations that modify the metadata of a memref (sizes, offsets,
    // strides) into a sequence of easier to analyze constructs. In particular, this pass
    // transforms operations into explicit sequence of operations that model the effect of this
    // operation on the different metadata. This pass uses affine constructs to materialize
    // these effects. Concretely, expanded-strided-metadata is used to decompose memref.subview
    // as it has no lowering in -finalize-memref-to-llvm.
    "expand-strided-metadata",
    "lower-affine",
    "arith-expand", // some arith ops (ceildivsi) require expansion to be lowered to llvm
    "convert-complex-to-standard", // added for complex.exp lowering
    "convert-complex-to-llvm",
    "convert-math-to-llvm",
    // Run after -convert-math-to-llvm as it marks math::powf illegal without converting it.
    "convert-math-to-libm",
    "convert-arith-to-llvm",
    "finalize-memref-to-llvm{use-generic-functions}",
    "convert-index-to-llvm",
    "convert-gradient-to-llvm",
    "convert-quantum-to-llvm",
    "emit-catalyst-py-interface",
    // Remove any dead casts as the final pass expects to remove all existing casts,
    // but only those that form a loop back to the original type.
    "canonicalize",
    "reconcile-unrealized-casts",
};
// clang-format on

std::string joinPasses(const SmallVector<const char *> &passes)
{
    std::string joined;
    llvm::raw_string_ostream stream{joined};
    llvm::interleaveComma(passes, stream);
    return joined;
}

struct Pipeline {
    const char *name;
    const SmallVector<const char *> passes;
};

/// Configure the printing of intermediate IR between pass stages.
/// By overriding the shouldPrintAfterPass hook, this function sets up both 1. after which passes
/// the IR should be printed, and 2. printing the IR to files in the workspace.
void configureIRPrinting(const CompilerOptions &options, PassManager &pm,
                         llvm::raw_ostream &outStream, std::string &outStr,
                         MutableArrayRef<Pipeline> pipelines,
                         function_ref<LogicalResult()> dumpIntermediate)
{
    auto shouldPrintAfterPass = [&](Pass *pass, Operation *) {
        Pipeline *pipeline = llvm::find_if(pipelines, [&pass](Pipeline pipeline) {
            // Print the IR after the last pass of each pipeline stage.
            return pipeline.passes.back() == pass->getArgument();
        });
        bool shouldPrint = pipeline != std::end(pipelines);
        if (shouldPrint && !outStr.empty() && failed(dumpIntermediate()) &&
            failed(dumpIntermediate())) {
            return false;
        }
        return shouldPrint;
    };

    pm.enableIRPrinting(/*shouldPrintBeforePass=*/[](Pass *, Operation *) { return false; },
                        shouldPrintAfterPass, /*printModuleScope=*/true,
                        /*printAfterOnlyOnChange=*/false, /*printAfterOnlyOnFailure=*/false,
                        /*out=*/outStream);
}
} // namespace

LogicalResult catalyst::runDefaultLowering(const CompilerOptions &options, ModuleOp moduleOp)
{
    Pipeline pipelines[] = {{.name = "no_mhlo", .passes = mhloToCorePasses},
                            {.name = "gradients_lowered", .passes = quantumCompilationPasses},
                            {.name = "bufferized", .passes = bufferizationPasses},
                            {.name = "llvm_dialect", .passes = lowerToLLVMPasses}};
    auto pm = PassManager::on<ModuleOp>(options.ctx, PassManager::Nesting::Implicit);

    // We enable printing and dumping intermediate IR by hooking into the shouldPrintAfterPass
    // method when configuring the PassManager. The PassManager prints to outStr and checks if it
    // should print *before* printing, meaning outStr will contain the IR after the *previous* pass
    // that should be printed. We thus need to keep track of a separate pipelineIdx to know which
    // pass has its output *currently* stored in outStr.
    std::string outStr;
    llvm::raw_string_ostream outStream{outStr};
    size_t pipelineIdx = 0;
    auto dumpIntermediate = [&](std::optional<std::string> outFile = std::nullopt) {
        if (!outFile) {
            outFile = fs::path(std::to_string(pipelineIdx + 1) + "_" + pipelines[pipelineIdx].name)
                          .replace_extension(".mlir");
            pipelineIdx++;
        }
        if (failed(catalyst::dumpToFile(options.workspace, outFile.value(), outStr))) {
            return failure();
        }
        outStr.clear();
        return success();
    };

    if (options.keepIntermediate) {
        // Dump the IR before running any passes
        outStream << moduleOp;
        if (failed(
                dumpIntermediate(fs::path(options.moduleName.str()).replace_extension(".mlir")))) {
            return failure();
        }
        outStr.clear();

        configureIRPrinting(options, pm, outStream, outStr, pipelines, dumpIntermediate);
    }

    for (const auto &pipeline : pipelines) {
        if (failed(parsePassPipeline(joinPasses(pipeline.passes), pm, options.diagnosticStream))) {
            return failure();
        }
    }

    if (failed(pm.run(moduleOp))) {
        return failure();
    }

    // After the last pass, outStr will need to be dumped one last time.
    if (options.keepIntermediate && !outStr.empty() && failed(dumpIntermediate())) {
        return failure();
    }

    return success();
}
