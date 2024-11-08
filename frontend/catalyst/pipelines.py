# Copyright 2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""
This module contains the pipelines that are used to compile a quantum function to LLVM.

.. note::

    for DEFAULT_PIPELINES pipeline and the pipelines in DEFAULT_ASYNC_PIPELINES, 
    any change should be reflected in the mlir/lib/Driver/Pipelines.cpp files as well.
    This is to ensure that the command line tool catalyst-cli default pipelines are 
    in sync with the pipelines defined in the Python frontend.

"""

from copy import deepcopy

ENFORCE_RUNTIME_INVARIANTS_PASS = (
    "EnforeRuntimeInvariantsPass",
    [
        # We want the invariant that transforms that generate multiple
        # tapes will generate multiple qnodes. One for each tape.
        # Split multiple tapes enforces that invariant.
        "split-multiple-tapes",
        # Run the transform sequence defined in the MLIR module
        "apply-transform-sequence",
        # Nested modules are something that will be used in the future
        # for making device specific transformations.
        # Since at the moment, nothing in the runtime is using them
        # and there is no lowering for them,
        # we inline them to preserve the semantics. We may choose to
        # keep inlining modules targetting the Catalyst runtime.
        # But qnodes targetting other backends may choose to lower
        # this into something else.
        "inline-nested-module",
    ],
)

HLO_LOWERING_PASS = (
    "HLOLoweringPass",
    [
        "canonicalize",
        "func.func(chlo-legalize-to-hlo)",
        "stablehlo-legalize-to-hlo",
        "func.func(mhlo-legalize-control-flow)",
        "func.func(hlo-legalize-to-linalg)",
        "func.func(mhlo-legalize-to-std)",
        "func.func(hlo-legalize-sort)",
        "convert-to-signless",
        "canonicalize",
        "scatter-lowering",
        "hlo-custom-call-lowering",
        "cse",
        "func.func(linalg-detensorize{aggressive-mode})",
        "detensorize-scf",
        "canonicalize",
    ],
)

QUANTUM_COMPILATION_PASS = (
    "QuantumCompilationPass",
    [
        "annotate-function",
        "lower-mitigation",
        "lower-gradients",
        "adjoint-lowering",
        "disable-assertion",
    ],
)

BUFFERIZATION_PASS = (
    "BufferizationPass",
    [
        "one-shot-bufferize{dialect-filter=memref}",
        "inline",
        "gradient-preprocess",
        "gradient-bufferize",
        "scf-bufferize",
        "convert-tensor-to-linalg",  # tensor.pad
        "convert-elementwise-to-linalg",  # Must be run before --arith-bufferize
        "arith-bufferize",
        "empty-tensor-to-alloc-tensor",
        "func.func(bufferization-bufferize)",
        "func.func(tensor-bufferize)",
        "catalyst-bufferize",  # Must be run before -- func.func(linalg-bufferize)
        "func.func(linalg-bufferize)",
        "func.func(tensor-bufferize)",
        "quantum-bufferize",
        "func-bufferize",
        "func.func(finalizing-bufferize)",
        "canonicalize",  # Remove dead memrefToTensorOp's
        "gradient-postprocess",
        # introduced during gradient-bufferize of callbacks
        "func.func(buffer-hoisting)",
        "func.func(buffer-loop-hoisting)",
        "func.func(buffer-deallocation)",
        "convert-arraylist-to-memref",
        "convert-bufferization-to-memref",
        "canonicalize",  # Must be after convert-bufferization-to-memref
        # otherwise there are issues in lowering of dynamic tensors.
        # "cse",
        "cp-global-memref",
    ],
)


MLIR_TO_LLVM_PASS = (
    "MLIRToLLVMDialect",
    [
        "expand-realloc",
        "convert-gradient-to-llvm",
        "memrefcpy-to-linalgcpy",
        "func.func(convert-linalg-to-loops)",
        "convert-scf-to-cf",
        # This pass expands memref ops that modify the metadata of a memref (sizes, offsets,
        # strides) into a sequence of easier to analyze constructs. In particular, this pass
        # transforms ops into explicit sequence of operations that model the effect of this
        # operation on the different metadata. This pass uses affine constructs to materialize
        # these effects. Concretely, expanded-strided-metadata is used to decompose
        # memref.subview as it has no lowering in -finalize-memref-to-llvm.
        "expand-strided-metadata",
        "lower-affine",
        "arith-expand",  # some arith ops (ceildivsi) require expansion to be lowered to llvm
        "convert-complex-to-standard",  # added for complex.exp lowering
        "convert-complex-to-llvm",
        "convert-math-to-llvm",
        # Run after -convert-math-to-llvm as it marks math::powf illegal without converting it.
        "convert-math-to-libm",
        "convert-arith-to-llvm",
        "memref-to-llvm-tbaa",  # load and store are converted to llvm with tbaa tags
        "finalize-memref-to-llvm{use-generic-functions}",
        "convert-index-to-llvm",
        "convert-catalyst-to-llvm",
        "convert-quantum-to-llvm",
        # There should be no identical code folding
        # (`mergeIdenticalBlocks` in the MLIR source code)
        # between convert-async-to-llvm and
        # add-exception-handling.
        # So, if there's a pass from the beginning
        # of this list to here that does folding
        # add-exception-handling will fail to add async.drop_ref
        # correctly. See https://github.com/PennyLaneAI/catalyst/pull/995
        "add-exception-handling",
        "emit-catalyst-py-interface",
        # Remove any dead casts as the final pass expects to remove all existing casts,
        # but only those that form a loop back to the original type.
        "canonicalize",
        "reconcile-unrealized-casts",
        "gep-inbounds",
        "register-inactive-callback",
    ],
)


DEFAULT_PIPELINES = [
    ENFORCE_RUNTIME_INVARIANTS_PASS,
    HLO_LOWERING_PASS,
    QUANTUM_COMPILATION_PASS,
    BUFFERIZATION_PASS,
    MLIR_TO_LLVM_PASS,
]

MLIR_TO_LLVM_ASYNC_PASS = deepcopy(MLIR_TO_LLVM_PASS)
MLIR_TO_LLVM_ASYNC_PASS[1][:0] = [
    "qnode-to-async-lowering",
    "async-func-to-async-runtime",
    "async-to-async-runtime",
    "convert-async-to-llvm",
]

DEFAULT_ASYNC_PIPELINES = [
    ENFORCE_RUNTIME_INVARIANTS_PASS,
    HLO_LOWERING_PASS,
    QUANTUM_COMPILATION_PASS,
    BUFFERIZATION_PASS,
    MLIR_TO_LLVM_ASYNC_PASS,
]
