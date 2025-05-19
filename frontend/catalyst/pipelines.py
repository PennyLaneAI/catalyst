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

    For DEFAULT_PIPELINES pipeline and the pipelines in DEFAULT_ASYNC_PIPELINES,
    any change should be reflected in the mlir/lib/Driver/Pipelines.cpp files as well.
    This is to ensure that Catalyst's command line tool default pipelines are
    in sync with the pipelines defined in the Python frontend.

"""

import sys
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from io import TextIOWrapper
from operator import is_not
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

from catalyst.utils.exceptions import CompileError


# pylint: disable=too-many-instance-attributes
@dataclass
class CompileOptions:
    """Generic compilation options, for which reasonable default values exist.

    Args:
        verbose (Optional[bool]): flag indicating whether to enable verbose output.
            Default is ``False``
        logfile (Optional[TextIOWrapper]): the logfile to write output to.
            Default is ``sys.stderr``
        keep_intermediate (Optional[bool]): flag indicating whether to keep intermediate results.
            Default is ``False``
        pipelines (Optional[List[Tuple[str,List[str]]]]): A list of tuples. The first entry of the
            tuple corresponds to the name of a pipeline. The second entry of the tuple corresponds
            to a list of MLIR passes.
        autograph (Optional[bool]): flag indicating whether experimental autograph support is to
            be enabled.
        autograph_include (Optional[Iterable[str]]): A list of (sub)modules to be allow-listed
        for autograph conversion.
        async_qnodes (Optional[bool]): flag indicating whether experimental asynchronous execution
            of QNodes support is to be enabled.
        lower_to_llvm (Optional[bool]): flag indicating whether to attempt the LLVM lowering after
            the main compilation pipeline is complete. Default is ``True``.
        static_argnums (Optional[Union[int, Iterable[int]]]): indices of static arguments.
            Default is ``None``.
        static_argnames (Optional[Union[str, Iterable[str]]]): names of static arguments.
            Default is ``None``.
        abstracted_axes (Optional[Any]): store the abstracted_axes value. Defaults to ``None``.
        disable_assertions (Optional[bool]): disables all assertions. Default is ``False``.
        seed (Optional[int]) : the seed for random operations in a qjit call.
            Default is None.
        circuit_transform_pipeline (Optional[dict[str, dict[str, str]]]):
            A dictionary that specifies the quantum circuit transformation pass pipeline order,
            and optionally arguments for each pass in the pipeline.
            Default is None.
        pass_plugins (Optional[Set[Path]]): List of paths to pass plugins.
        dialect_plugins (Optional[Set[Path]]): List of paths to dialect plugins.
    """

    verbose: Optional[bool] = False
    logfile: Optional[TextIOWrapper] = sys.stderr
    target: Optional[str] = "binary"
    keep_intermediate: Optional[bool] = False
    pipelines: Optional[List[Any]] = None
    autograph: Optional[bool] = False
    autograph_include: Optional[Iterable[str]] = ()
    async_qnodes: Optional[bool] = False
    static_argnums: Optional[Union[int, Iterable[int]]] = None
    static_argnames: Optional[Union[str, Iterable[str]]] = None
    abstracted_axes: Optional[Union[Iterable[Iterable[str]], Dict[int, str]]] = None
    lower_to_llvm: Optional[bool] = True
    checkpoint_stage: Optional[str] = ""
    disable_assertions: Optional[bool] = False
    seed: Optional[int] = None
    circuit_transform_pipeline: Optional[dict[str, dict[str, str]]] = None
    pass_plugins: Optional[Set[Path]] = None
    dialect_plugins: Optional[Set[Path]] = None

    def __post_init__(self):
        # Check that async runs must not be seeded
        if self.async_qnodes and self.seed is not None:
            raise CompileError(
                """
                Seeding has no effect on asynchronous QNodes,
                as the execution order of parallel runs is not guaranteed.
                As such, seeding an asynchronous run is not supported.
                """
            )

        # Check that seed is 32-bit unsigned int
        if (self.seed is not None) and (self.seed < 0 or self.seed > 2**32 - 1):
            raise ValueError(
                """
                Seed must be an unsigned 32-bit integer!
                """
            )

        # Make the format of static_argnums easier to handle.
        static_argnums = self.static_argnums
        if static_argnums is None:
            self.static_argnums = ()
        elif isinstance(static_argnums, int):
            self.static_argnums = (static_argnums,)
        elif isinstance(static_argnums, Iterable):
            self.static_argnums = tuple(static_argnums)
        if self.pass_plugins is None:
            self.pass_plugins = set()
        if self.dialect_plugins is None:
            self.dialect_plugins = set()

    def __deepcopy__(self, memo):
        """Make a deep copy of all fields of a CompileOptions object except the logfile, which is
        copied directly"""
        return CompileOptions(
            **{
                k: (deepcopy(v) if k != "logfile" else self.logfile)
                for k, v in self.__dict__.items()
                if k != "logfile"
            }
        )

    def get_pipelines(self) -> List[Tuple[str, List[str]]]:
        """Get effective pipelines"""
        if self.pipelines:
            return self.pipelines
        return self.get_stages()

    def get_stages(self):
        """Returns all stages in order for compilation"""
        # Dictionaries in python are ordered
        stages = {}
        stages["EnforceRuntimeInvariantsPass"] = get_enforce_runtime_invariants_stage(self)
        stages["HLOLoweringPass"] = get_hlo_lowering_stage(self)
        stages["QuantumCompilationPass"] = get_quantum_compilation_stage(self)
        stages["BufferizationPass"] = get_bufferization_stage(self)
        stages["MLIRToLLVMDialect"] = get_convert_to_llvm_stage(self)
        return list(stages.items())


def get_enforce_runtime_invariants_stage(_options: CompileOptions) -> List[str]:
    """Returns the list of passes in the enforce runtime invariant stage."""
    enforce_runtime_invariants = [
        # We want the invariant that transforms that generate multiple
        # tapes will generate multiple qnodes. One for each tape.
        # Split multiple tapes enforces that invariant.
        "split-multiple-tapes",
        # Run the transform sequence defined in the MLIR module
        "builtin.module(apply-transform-sequence)",
        # Nested modules are something that will be used in the future
        # for making device specific transformations.
        # Since at the moment, nothing in the runtime is using them
        # and there is no lowering for them,
        # we inline them to preserve the semantics. We may choose to
        # keep inlining modules targeting the Catalyst runtime.
        # But qnodes targeting other backends may choose to lower
        # this into something else.
        "inline-nested-module",
    ]
    return enforce_runtime_invariants


def get_hlo_lowering_stage(_options: CompileOptions) -> List[str]:
    """Returns the list of passes to lower StableHLO to upstream MLIR dialects."""
    hlo_lowering = [
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
    ]
    return hlo_lowering


def get_quantum_compilation_stage(options: CompileOptions) -> List[str]:
    """Returns the list of passes that performs quantum transformations"""

    quantum_compilation = [
        "annotate-function",
        "lower-mitigation",
        "lower-gradients",
        "adjoint-lowering",
        "disable-assertion" if options.disable_assertions else None,
    ]
    return list(filter(partial(is_not, None), quantum_compilation))


def get_bufferization_stage(_options: CompileOptions) -> List[str]:
    """Returns the list of passes that performs bufferization"""

    options = "bufferize-function-boundaries allow-return-allocs-from-loops function-boundary-type-conversion=identity-layout-map unknown-type-conversion=identity-layout-map"

    bufferization = [
        "inline",
        "convert-tensor-to-linalg",  # tensor.pad
        "convert-elementwise-to-linalg",  # Must be run before --arith-bufferize
        "gradient-preprocess",
        ####################
        "one-shot-bufferize{dialect-filter=gradient unknown-type-conversion=identity-layout-map}",
        "one-shot-bufferize{dialect-filter=scf " + options + "}",
        "one-shot-bufferize{dialect-filter=arith " + options + "}",
        "empty-tensor-to-alloc-tensor",
        "one-shot-bufferize{dialect-filter=bufferization " + options + "}",
        "func.func(tensor-bufferize)",  # TODO
        # "one-shot-bufferize{dialect-filter=tensor " + options + "}",
        # Catalyst dialect's bufferization must be run before --func.func(linalg-bufferize)
        "one-shot-bufferize{dialect-filter=catalyst unknown-type-conversion=identity-layout-map}",
        "one-shot-bufferize{dialect-filter=linalg " + options + "}",
        "func.func(tensor-bufferize)",  # TODO
        "one-shot-bufferize{dialect-filter=quantum}",
        "func-bufferize",  # TODO
        ####################
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
    ]

    __bufferization = [
        "inline",
        "gradient-preprocess",
        "convert-elementwise-to-linalg",
        "canonicalize",
        # Preprocessing:
        # rewrite_in_destination_passing_style
        #
        # We are not rewriting everything in DPS before -one-shot-bufferize
        # This was discussed with the main author of the -one-shot-bufferize
        # pass and he stated the following:
        #
        #     One-Shot Bufferize was designed for ops that are in DPS (destination-passing style).
        #     Ops that are not in DPS can still be bufferized,
        #     but a new buffer will be allocated for every tensor result.
        #     That’s functionally correct but inefficient.
        #
        #     I’m not sure whether it’s better to first migrate to the new bufferization,
        #     then turn the ops into DPS ops, or do it the other way around.
        #     One benefit of implementing the bufferization first is that
        #     it’s a smaller step that you can already run end-to-end.
        #     And you can think of the DPS of a performance improvement on top of it.
        #
        # https://discourse.llvm.org/t/steps-of-migrating-to-one-shot-bufferization/81062/2
        #
        # Here, please note that gradient-preprocessing is different than rewriting in DPS.
        # So, overall, we are skipping this section while we first focus on migrating to the
        # new -one-shot-bufferize
        "eliminate-empty-tensors",
        (
            # Before we enter one-shot-bufferize, here is what we expect:
            # * Given
            #
            #     One-Shot Bufferize was designed for ops that are in DPS
            #     (destination-passing style).
            #     Ops that are not in DPS can still be bufferized,
            #     but a new buffer will be allocated for every tensor result.
            #     That’s functionally correct but inefficient.
            #
            #   https://discourse.llvm.org/t/steps-of-migrating-to-one-shot-bufferization/81062/2
            #
            #   we expect that results will be (automatically?) converted into new buffers. And it
            #   is up to us to just define the bufferization for the operands.
            #
            # So what is the state of the catalyst, gradient, quantum dialects at this point?
            #
            # Let's start with quantum:
            #
            # |-------------------------|--------------------|
            # |      operation          |  has result tensor |
            # |-------------------------|--------------------|
            # | quantum.set_state       |                    |
            # | quantum.set_basis_state |                    |
            # | quantum.unitary         |                    |
            # | quantum.hermitian       |                    |
            # | quantum.hamiltonian     |                    |
            # | quantum.sample_op       |     YES            |
            # | quantum.counts_op       |     YES            |
            # | quantum.probs_op        |     YES            |
            # | quantum.state_op        |     YES            |
            # |-------------------------|--------------------|
            # | catalyst.print_op       |                    |
            # | catalyst.custom_call    |     YES            |
            # | catalyst.callback       |                    |
            # | catalyst.callback_call  |     YES            |
            # | catalyst.launch_kernel  |     YES            |
            # |-------------------------|--------------------|
            # | gradient.grad           |     YES            |
            # | gradient.value_and_grad |     YES            |
            # | gradient.adjoint        |     YES            |
            # | gradient.backprop       |     YES            |
            # | gradient.jvp            |     YES            |
            # | gradient.vjp            |     YES            |
            # | gradient.forward        |     YES            |
            # | gradient.reverse        |     YES            |
            # |-------------------------|--------------------|
            #
            # So what this means is that for the operands, all the ones that have the YES
            # means that no operands are written to. They are only read.
            "one-shot-bufferize"
            "{"
            "bufferize-function-boundaries "
            # - Bufferize function boundaries (experimental).
            #
            #     By default, function boundaries are not bufferized.
            #     This is because there are currently limitations around function graph
            #     bufferization:
            #     recursive calls are not supported.
            #     As long as there are no recursive calls, function boundary bufferization can be
            #     enabled with bufferize-function-boundaries.
            #     Each tensor function argument and tensor function result is then turned into a memref.
            #     The layout map of the memref type can be controlled with function-boundary-type-conversion.
            #
            # https://mlir.llvm.org/docs/Bufferization/#using-one-shot-bufferize
            "allow-return-allocs-from-loops "
            # - Allows returning/yielding new allocations from a loop.
            # https://github.com/llvm/llvm-project/pull/83964
            # https://github.com/llvm/llvm-project/pull/87594
            "function-boundary-type-conversion=identity-layout-map "
            "unknown-type-conversion=identity-layout-map"
            # - Controls layout maps when bufferizing function signatures.
            #     You can control the memref types at the function boundary with
            #     function-boundary-type-conversion. E.g., if you set it to identity-layout-map,
            #     you should get the same type as with --func-bufferize.
            #     By default, we put a fully dynamic layout map strided<[?, ?], offset: ?>
            #     because that works best if you don't know what layout map the buffers at
            #     the call site have -- you can always cast a buffer to a type with
            #     fully dynamic layout map. (But not the other way around. That may require a
            #     reallocation.)
            #
            #  https://discord.com/channels/636084430946959380/642426447167881246/1212338527824515102
            "}"
        ),
        # Remove dead memrefToTensorOp's
        # introduced during gradient-bufferize of callbacks
        # TODO: Figure out how to remove this.
        "gradient-postprocess",
        "func.func(buffer-hoisting)",
        "func.func(buffer-loop-hoisting)",
        # TODO: Figure out how to include the other buffer-level optimizations.
        # -buffer-results-to-out-params,
        # -drop-equivalent-buffer-results,
        # -promote-buffers-to-stack
        # Deallocation
        # The buffer deallocation pass has been deprecated in favor of the
        # ownership-based buffer deallocation pipeline.
        # The deprecated pass has some limitations that may cause memory leaks in the resulting IR.
        # TODO: Switch to one-shot-bufferization once it is merged.
        "func.func(buffer-deallocation)",
        # catalyst.list_* operations are not bufferized through
        # the bufferization interface
        # This is because they store a memref inside of a memref
        # which is incompatible with the bufferization pipeline.
        "convert-arraylist-to-memref",
        "convert-bufferization-to-memref",
        # Must be after convert-bufferization-to-memref
        # otherwise there are issues in lowering of dynamic tensors.
        "canonicalize",
        # "cse",
        "cp-global-memref",
    ]

    return bufferization


def get_convert_to_llvm_stage(options: CompileOptions) -> List[str]:
    """Returns the list of passes that lowers MLIR upstream dialects to LLVM Dialect"""

    convert_to_llvm = [
        "qnode-to-async-lowering" if options.async_qnodes else None,
        "async-func-to-async-runtime" if options.async_qnodes else None,
        "async-to-async-runtime" if options.async_qnodes else None,
        "convert-async-to-llvm" if options.async_qnodes else None,
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
        "add-exception-handling" if options.async_qnodes else None,
        "emit-catalyst-py-interface",
        # Remove any dead casts as the final pass expects to remove all existing casts,
        # but only those that form a loop back to the original type.
        "canonicalize",
        "reconcile-unrealized-casts",
        "gep-inbounds",
        "register-inactive-callback",
    ]
    return list(filter(partial(is_not, None), convert_to_llvm))


def get_stages(options):
    """Returns all stages in order for compilation"""
    # Dictionaries in python are ordered
    stages = {}
    stages["EnforceRuntimeInvariantsPass"] = get_enforce_runtime_invariants_stage(options)
    stages["HLOLoweringPass"] = get_hlo_lowering_stage(options)
    stages["QuantumCompilationPass"] = get_quantum_compilation_stage(options)
    stages["BufferizationPass"] = get_bufferization_stage(options)
    stages["MLIRToLLVMDialect"] = get_convert_to_llvm_stage(options)
    return list(stages.items())
