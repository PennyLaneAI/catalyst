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

#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

using namespace mlir;

namespace {
LogicalResult addMhloToCorePasses(PassManager &pm)
{
    const char *mhloToCorePipeline = "func.func(chlo-legalize-to-hlo),"
                                     "stablehlo-legalize-to-hlo,"
                                     "func.func(mhlo-legalize-control-flow),"
                                     "func.func(hlo-legalize-to-linalg),"
                                     "func.func(mhlo-legalize-to-std),"
                                     "convert-to-signless";
    return parsePassPipeline(mhloToCorePipeline, pm);
}

LogicalResult addQuantumCompilationPasses(PassManager &pm)
{
    const char *quantumPipeline = "lower-gradients,"
                                  "convert-arraylist-to-memref";

    return parsePassPipeline(quantumPipeline, pm);
}

LogicalResult addBufferizationPasses(PassManager &pm)
{
    const char *bufferizationPipeline =
        "inline,"
        "gradient-bufferize,"
        "scf-bufferize,"
        "convert-tensor-to-linalg,"      // tensor.pad
        "convert-elementwise-to-linalg," // Must be run before --arith-bufferize
        "arith-bufferize,"
        "empty-tensor-to-alloc-tensor,"
        "func.func(bufferization-bufferize),"
        "func.func(tensor-bufferize),"
        "func.func(linalg-bufferize),"
        "func.func(tensor-bufferize),"
        "quantum-bufferize,"
        "func-bufferize,"
        "func.func(finalizing-bufferize),"
        // "func.func(buffer-hoisting),"
        "func.func(buffer-loop-hoisting),"
        // "func.func(buffer-deallocation),"
        "convert-bufferization-to-memref,"
        "canonicalize,"
        // "cse,"
        "cp-global-memref";
    return parsePassPipeline(bufferizationPipeline, pm);
}

LogicalResult addLowerToLLVMPasses(PassManager &pm)
{
    const char *lowerToLLVMDialectPipeline =
        "func.func(convert-linalg-to-loops),"
        "convert-scf-to-cf,"
        // This pass expands memref operations that modify the metadata of a memref (sizes, offsets,
        // strides) into a sequence of easier to analyze constructs. In particular, this pass
        // transforms operations into explicit sequence of operations that model the effect of this
        // operation on the different metadata. This pass uses affine constructs to materialize
        // these effects. Concretely, expanded-strided-metadata is used to decompose memref.subview
        // as it has no lowering in -finalize-memref-to-llvm.
        "expand-strided-metadata,"
        "lower-affine,"
        "arith-expand," // some arith ops (ceildivsi) require expansion to be lowered to llvm
        "convert-complex-to-standard," // added for complex.exp lowering
        "convert-complex-to-llvm,"
        "convert-math-to-llvm,"
        // Run after -convert-math-to-llvm as it marks math::powf illegal without converting it.
        "convert-math-to-libm,"
        "convert-arith-to-llvm,"
        "finalize-memref-to-llvm{use-generic-functions},"
        "convert-index-to-llvm,"
        "convert-gradient-to-llvm,"
        "convert-quantum-to-llvm,"
        "emit-catalyst-py-interface,"
        // Remove any dead casts as the final pass expects to remove all existing casts,
        // but only those that form a loop back to the original type.
        "canonicalize,"
        "reconcile-unrealized-casts";
    return parsePassPipeline(lowerToLLVMDialectPipeline, pm);
}
} // namespace

LogicalResult catalyst::runDefaultLowering(MLIRContext *ctx, ModuleOp moduleOp)
{
    auto pm = PassManager::on<ModuleOp>(ctx, PassManager::Nesting::Implicit);
    using PassesFunc = LogicalResult (*)(PassManager &);
    PassesFunc pfs[] = {addMhloToCorePasses, addQuantumCompilationPasses, addBufferizationPasses,
                        addLowerToLLVMPasses};

    for (const auto &pf : pfs) {
        if (failed(pf(pm))) {
            return failure();
        }
    }

    return pm.run(moduleOp);
}
