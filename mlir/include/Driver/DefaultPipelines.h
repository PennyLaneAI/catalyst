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

#pragma once

#include <format>
#include <numeric>
#include <ranges>
#include <string>
#include <vector>

namespace catalyst {
namespace driver {

using PassNames = std::vector<std::string>;
struct PipelineInfo {
    std::string name;
    PassNames passNames;
};
using PipelineNames = std::vector<std::string>;
using PipelineList = std::vector<PipelineInfo>;

const PipelineList pipelineList{
    {"quantum-compilation-pipeline",
     {// We want the invariant that transforms that generate multiple
      // tapes will generate multiple qnodes. One for each tape.
      // Split multiple tapes enforces that invariant.
      "split-multiple-tapes",
      // Run the transform sequence defined in the MLIR module
      "builtin.module(apply-transform-sequence)",
      // Nested modules are something that will be used in the future
      // for making device specific transformations.
      // Since at the moment, nothing in the runtime is using them
      // and there is no lowering for them,
      // we inline them to preserve the semantics. We may choose to
      // keep inlining modules targeting the Catalyst runtime.
      // But qnodes targeting other backends may choose to lower
      // this into something else.
      "inline-nested-module", "lower-mitigation", "adjoint-lowering", "disable-assertion"}},
    {"hlo-lowering-pipeline",
     {"canonicalize", "func.func(chlo-legalize-to-stablehlo)",
      "func.func(stablehlo-legalize-control-flow)",
      "func.func(stablehlo-aggressive-simplification)", "stablehlo-legalize-to-linalg",
      "func.func(stablehlo-legalize-to-std)", "func.func(stablehlo-legalize-sort)",
      "stablehlo-convert-to-signless", "canonicalize", "scatter-lowering",
      "hlo-custom-call-lowering", "cse", "func.func(linalg-detensorize{aggressive-mode})",
      "detensorize-scf", "detensorize-function-boundary", "canonicalize", "symbol-dce"}},
    {"gradient-lowering-pipeline", {"annotate-invalid-gradient-functions", "lower-gradients"}},
    {"bufferization-pipeline",
     {"inline",
      "convert-tensor-to-linalg",      // tensor.pad
      "convert-elementwise-to-linalg", // must be run before --one-shot-bufferize
      "gradient-preprocess",
      // Keep eliminate-empty-tensors commented out until benchmarks use more structure
      // and produce functions of reasonable size. Otherwise, eliminate-empty-tensors
      // will consume a significant amount of compile time along with one-shot-bufferize.
      //
      // "eliminate-empty-tensors",
      "one-shot-bufferize",
      "canonicalize", // remove dead memrefToTensorOp's
      "gradient-postprocess",
      // Introduced during gradient-bufferize of callbacks
      "func.func(buffer-hoisting)", "func.func(buffer-loop-hoisting)",
      // TODO: investigate re-adding this after new buffer dealloc pipeline
      //       removed due to high stack memory use in nested structures
      // "func.func(promote-buffers-to-stack)",
      // TODO: migrate to new buffer deallocation "buffer-deallocation-pipeline"
      "func.func(buffer-deallocation)", "convert-arraylist-to-memref",
      "convert-bufferization-to-memref",
      "canonicalize", // must be after convert-bufferization-to-memref
                      // otherwise there are issues in lowering of dynamic tensors.
                      // "cse",
      "cp-global-memref"}},
    {"llvm-dialect-lowering-pipeline",
     {"qnode-to-async-lowering", "async-func-to-async-runtime", "async-to-async-runtime",
      "convert-async-to-llvm", "expand-realloc", "convert-gradient-to-llvm",
      "memrefcpy-to-linalgcpy", "func.func(convert-linalg-to-loops)", "convert-scf-to-cf",
      // This pass expands memref ops that modify the metadata of a memref (sizes, offsets,
      // strides) into a sequence of easier to analyze constructs. In particular, this pass
      // transforms ops into explicit sequence of operations that model the effect of this
      // operation on the different metadata. This pass uses affine constructs to materialize
      // these effects. Concretely, expanded-strided-metadata is used to decompose
      // memref.subview as it has no lowering in -finalize-memref-to-llvm.
      "expand-strided-metadata", "lower-affine",
      "arith-expand", // some arith ops (ceildivsi) require expansion to be lowered to llvm
      "convert-complex-to-standard", // added for complex.exp lowering
      "convert-complex-to-llvm", "convert-math-to-llvm",
      // Run after -convert-math-to-llvm as it marks math::powf illegal without converting it.
      "convert-math-to-libm", "convert-arith-to-llvm",
      "memref-to-llvm-tbaa", // load and store are converted to llvm with tbaa tags
      "finalize-memref-to-llvm{use-generic-functions}", "convert-index-to-llvm",
      "convert-catalyst-to-llvm", "convert-quantum-to-llvm",
      // There should be no identical code folding
      // (`mergeIdenticalBlocks` in the MLIR source code)
      // between convert-async-to-llvm and add-exception-handling.
      // So, if there's a pass from the beginning of this list to here that does folding,
      // add-exception-handling will fail to add async.drop_ref correctly.
      // See https://github.com/PennyLaneAI/catalyst/pull/995
      "add-exception-handling", "emit-catalyst-py-interface",
      // Remove any dead casts as the final pass expects to remove all existing casts,
      // but only those that form a loop back to the original type.
      "canonicalize", "reconcile-unrealized-casts", "gep-inbounds", "register-inactive-callback"}}};

PipelineNames getPipelineNames()
{
    static std::vector<std::string> names =
        std::accumulate(driver::pipelineList.begin(), driver::pipelineList.end(),
                        std::vector<std::string>{}, [](auto acc, const auto &pipelineInfo) {
                            acc.emplace_back(pipelineInfo.name);
                            return acc;
                        });
    return names;
}

PassNames getQuantumCompilationStage(bool disableAssertion = true)
{
    auto &&ret =
        pipelineList[0].passNames | std::views::filter([&disableAssertion](const auto &passName) {
            return (!disableAssertion && (passName == "disable-assertion")) ? false : true;
        });
    return PassNames{ret.begin(), ret.end()};
}

PassNames getHLOLoweringStage() { return pipelineList[1].passNames; }

PassNames getGradientLoweringStage() { return pipelineList[2].passNames; }

PassNames getBufferizationStage(bool asyncQNodes = false)
{
    const std::string bufferizationOptions = std::format(
        "{{{} {} {} {}{}}}", "bufferize-function-boundaries", "allow-return-allocs-from-loops",
        "function-boundary-type-conversion=identity-layout-map",
        "unknown-type-conversion=identity-layout-map", (asyncQNodes ? " copy-before-write" : ""));
    auto &&ret = pipelineList[3].passNames |
                 std::views::transform([&bufferizationOptions](const auto &passName) {
                     if (passName == "one-shot-bufferize") {
                         return passName + bufferizationOptions;
                     }
                     return passName;
                 });
    return PassNames{ret.begin(), ret.end()};
}

PassNames getLLVMDialectLoweringStage(bool asyncQNodes = false)
{
    auto &&ret =
        pipelineList[4].passNames | std::views::filter([&asyncQNodes](const auto &passName) {
            return (!asyncQNodes &&
                    (passName == "qnode-to-async-lowering" ||
                     passName == "async-func-to-async-runtime" ||
                     passName == "async-to-async-runtime" || passName == "convert-async-to-llvm" ||
                     passName == "add-exception-handling"))
                       ? false
                       : true;
        });
    return PassNames{ret.begin(), ret.end()};
}

} // namespace driver
} // namespace catalyst
