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
        experimental_capture (bool): If set to ``True``,
            use PennyLane's experimental program capture capabilities
            to capture the function for compilation.
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
    experimental_capture: Optional[bool] = False
    circuit_transform_pipeline: Optional[dict[str, dict[str, str]]] = None
    pass_plugins: Optional[Set[Path]] = None
    dialect_plugins: Optional[Set[Path]] = None

    def __post_init__(self):
        # Check that async runs must not be seeded
        if self.async_qnodes and self.seed != None:
            raise CompileError(
                """
                Seeding has no effect on asyncronous qnodes,
                as the execution order of parallel runs is not guaranteed.
                As such, seeding an asynchronous run is not supported.
                """
            )

        # Check that seed is 32-bit unsigned int
        if (self.seed != None) and (self.seed < 0 or self.seed > 2**32 - 1):
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
        stages["EnforeRuntimeInvariantsPass"] = get_enforce_runtime_invariants_stage(self)
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
        # Lower the static custom ops to regular custom ops with dynamic parameters.
        "static-custom-lowering",
        # Nested modules are something that will be used in the future
        # for making device specific transformations.
        # Since at the moment, nothing in the runtime is using them
        # and there is no lowering for them,
        # we inline them to preserve the semantics. We may choose to
        # keep inlining modules targetting the Catalyst runtime.
        # But qnodes targetting other backends may choose to lower
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
    bufferization = [
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
        "add-exception-handling",
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
    stages["EnforeRuntimeInvariantsPass"] = get_enforce_runtime_invariants_stage(options)
    stages["HLOLoweringPass"] = get_hlo_lowering_stage(options)
    stages["QuantumCompilationPass"] = get_quantum_compilation_stage(options)
    stages["BufferizationPass"] = get_bufferization_stage(options)
    stages["MLIRToLLVMDialect"] = get_convert_to_llvm_stage(options)
    return list(stages.items())
