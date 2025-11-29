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

import enum
import sys
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from io import TextIOWrapper
from operator import is_not
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

from catalyst.utils.exceptions import CompileError

PipelineStage = Tuple[str, List[str]]
PipelineStages = List[PipelineStage]


class KeepIntermediateLevel(enum.IntEnum):
    """Enum to control the level of intermediate file keeping."""

    NONE = 0  # No intermediate files are kept.
    PIPELINE = 1  # Intermediate files are saved after each pipeline.
    CHANGED = 2  # Intermediate files are saved after each pass (only if changed).
    PASS = 3  # Intermediate files are saved after each pass, even if unchanged.


def _parse_keep_intermediate(
    level: Union[str, int, bool, None],
) -> KeepIntermediateLevel:
    """Parse the keep_intermediate value into a KeepIntermediateLevel enum."""
    match level:
        case 0 | 1 | 2 | 3:
            return KeepIntermediateLevel(level)
        case "none" | None:
            return KeepIntermediateLevel.NONE
        case "pipeline":
            return KeepIntermediateLevel.PIPELINE
        case "changed":
            return KeepIntermediateLevel.CHANGED
        case "pass":
            return KeepIntermediateLevel.PASS
        case _:
            raise ValueError(
                f"Invalid value for keep_intermediate: {level}. "
                "Valid values are True, False, 0, 1, 2, 3, 'none', 'pipeline', 'changed', 'pass'."
            )


# pylint: disable=too-many-instance-attributes
@dataclass
class CompileOptions:
    """Generic compilation options, for which reasonable default values exist.

    Args:
        verbose (Optional[bool]): Flag indicating whether to enable verbose output.
            Default is ``False``.
        logfile (Optional[TextIOWrapper]): The logfile to write output to.
            Default is ``sys.stderr``.
        keep_intermediate (Optional[Union[str, int, bool]]): Level controlling intermediate file
            generation.

            - ``False`` or ``0`` or ``"none"`` (default): No intermediate files are kept.
            - ``True`` or ``1`` or ``"pipeline"``: Intermediate files are saved after each pipeline.
            - ``2`` or ``"changed"``: Intermediate files are saved after each pass only if changed.
            - ``3`` or ``"pass"``: Intermediate files are saved after each pass, even if unchanged.
        use_nameloc (Optional[bool]): If ``True``, add function parameter names to the IR as name
            locations.
        pipelines (Optional[List[Tuple[str,List[str]]]]): A list of tuples. The first entry of the
            tuple corresponds to the name of a pipeline. The second entry of the tuple corresponds
            to a list of MLIR passes.
        autograph (Optional[bool]): Flag indicating whether experimental autograph support is to
            be enabled.
        autograph_include (Optional[Iterable[str]]): A list of (sub)modules to be allow-listed
            for autograph conversion.
        async_qnodes (Optional[bool]): Flag indicating whether experimental asynchronous execution
            of QNodes support is to be enabled.
        lower_to_llvm (Optional[bool]): Flag indicating whether to attempt the LLVM lowering after
            the main compilation pipeline is complete. Default is ``True``.
        static_argnums (Optional[Union[int, Iterable[int]]]): Indices of static arguments.
            Default is ``None``.
        static_argnames (Optional[Union[str, Iterable[str]]]): Names of static arguments.
            Default is ``None``.
        abstracted_axes (Optional[Any]): Store the abstracted_axes value. Default is ``None``.
        disable_assertions (Optional[bool]): Disable all assertions. Default is ``False``.
        seed (Optional[int]) : the seed for random operations in a qjit call.
            Default is ``None``.
        circuit_transform_pipeline (Optional[dict[str, dict[str, str]]]):
            A dictionary that specifies the quantum circuit transformation pass pipeline order,
            and optionally arguments for each pass in the pipeline.
            Default is ``None``.
        pass_plugins (Optional[Iterable[Path]]): List of paths to pass plugins.
        dialect_plugins (Optional[Iterable[Path]]): List of paths to dialect plugins.
    """

    verbose: Optional[bool] = False
    logfile: Optional[TextIOWrapper] = sys.stderr
    target: Optional[str] = "binary"
    keep_intermediate: Optional[Union[str, int, bool, KeepIntermediateLevel]] = False
    use_nameloc: Optional[bool] = False
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
        # Convert keep_intermediate to Enum
        self.keep_intermediate = _parse_keep_intermediate(self.keep_intermediate)

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
        else:
            self.pass_plugins = set(self.pass_plugins)

        if self.dialect_plugins is None:
            self.dialect_plugins = set()
        else:
            self.dialect_plugins = set(self.dialect_plugins)

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

    def get_pipelines(self) -> PipelineStages:
        """Get effective pipelines"""
        if self.pipelines:
            return self.pipelines
        return self.get_stages()

    def get_stages(self) -> PipelineStages:
        """Returns all stages in order for compilation"""
        # Dictionaries in python are ordered
        stages = {}
        stages["QuantumCompilationStage"] = get_quantum_compilation_stage(self)
        stages["HLOLoweringStage"] = get_hlo_lowering_stage(self)
        stages["GradientLoweringStage"] = get_gradient_lowering_stage(self)
        stages["BufferizationStage"] = get_bufferization_stage(self)
        stages["MLIRToLLVMDialectConversion"] = get_convert_to_llvm_stage(self)
        return list(stages.items())


def get_quantum_compilation_stage(_options: CompileOptions) -> List[str]:
    """Returns the list of passes that performs quantum compilation"""

    user_transform_passes = [
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
        "lower-mitigation",
        "adjoint-lowering",
        "disable-assertion" if _options.disable_assertions else None,
    ]
    return list(filter(partial(is_not, None), user_transform_passes))


def get_hlo_lowering_stage(_options: CompileOptions) -> List[str]:
    """Returns the list of passes to lower StableHLO to upstream MLIR dialects."""
    hlo_lowering = [
        "canonicalize",
        "func.func(chlo-legalize-to-stablehlo)",
        "func.func(stablehlo-legalize-control-flow)",
        "func.func(stablehlo-aggressive-simplification)",
        "stablehlo-legalize-to-linalg",
        "func.func(stablehlo-legalize-to-std)",
        "func.func(stablehlo-legalize-sort)",
        "stablehlo-convert-to-signless",
        "canonicalize",
        "scatter-lowering",
        "hlo-custom-call-lowering",
        "cse",
        "func.func(linalg-detensorize{aggressive-mode})",
        "detensorize-scf",
        "detensorize-function-boundary",
        "canonicalize",
        "symbol-dce",
    ]
    return hlo_lowering


def get_gradient_lowering_stage(_options: CompileOptions) -> List[str]:
    """Returns the list of passes that performs gradient lowering"""

    gradient_lowering = [
        "annotate-invalid-gradient-functions",
        "lower-gradients",
    ]
    return gradient_lowering


def get_bufferization_stage(options: CompileOptions) -> List[str]:
    """Returns the list of passes that performs bufferization"""

    bufferization_options = """bufferize-function-boundaries
        allow-return-allocs-from-loops
        function-boundary-type-conversion=identity-layout-map
        unknown-type-conversion=identity-layout-map""".replace(
        "\n", " "
    )
    if options.async_qnodes:
        bufferization_options += " copy-before-write"

    bufferization = [
        "inline",
        "convert-tensor-to-linalg",  # tensor.pad
        "convert-elementwise-to-linalg",  # Must be run before --one-shot-bufferize
        "gradient-preprocess",
        # "eliminate-empty-tensors",
        # Keep eliminate-empty-tensors commented out until benchmarks use more structure
        # and produce functions of reasonable size. Otherwise, eliminate-empty-tensors
        # will consume a significant amount of compile time along with one-shot-bufferize.
        ####################
        "one-shot-bufferize{" + bufferization_options + "}",
        ####################
        "canonicalize",  # Remove dead memrefToTensorOp's
        "gradient-postprocess",
        # introduced during gradient-bufferize of callbacks
        "func.func(buffer-hoisting)",
        "func.func(buffer-loop-hoisting)",
        # TODO: investigate re-adding this after new buffer dealloc pipeline
        #       removed due to high stack memory use in nested structures
        # "func.func(promote-buffers-to-stack)",
        # TODO: migrate to new buffer deallocation "buffer-deallocation-pipeline"
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


def default_pipeline() -> PipelineStages:
    """Return the pipeline stages for default Catalyst workloads.

    The pipeline stages are returned as a list of tuples in the form

    .. code-block:: python

        [
            ('stage1', ['pass1a', 'pass1b', ...]),
            ('stage2', ['pass2a', ...]),
            ...
        ]

    where the first entry in the tuple is the stage name and the second entry is a list of MLIR
    passes.

    Returns:
        PipelineStages: The list of pipeline stages.

    **Example**

    The sequence of pipeline stages returned by this function can be passed directly to the
    `pipelines` argument of :func:`~.qjit`. For example,

    >>> my_pipeline = default_pipeline()
    >>> # <modify my_pipeline as needed>
    >>> @qjit(pipelines=my_pipeline)
    ... @qml.qnode(device)
    ... def circuit():
    ...     ...
    """
    options = CompileOptions()  # Use all default compile options
    return options.get_stages()


def insert_pass_after(pipeline: list[str], new_pass: str, ref_pass: str) -> None:
    """Insert a pass into an existing pass pipeline at the position after the given reference pass.

    If the reference pass appears multiple times in the pipeline, the new pass is inserted only once
    after the first occurrence of the reference pass.

    Args:
        pipeline (list[str]): An existing pass pipeline, given as a list of passes.
        new_pass (str): The name of the pass to insert.
        ref_pass (str): The name of the reference pass after which the new pass is inserted.

    Raises:
        ValueError: If `ref_pass` is not found in the pass pipeline.

    Example:
        >>> pipeline = ["pass1", "pass2"]
        >>> insert_pass_after(pipeline, "new_pass", ref_pass="pass1")
        >>> pipeline
        ['pass1', 'new_pass', 'pass2']
    """
    try:
        ref_index = pipeline.index(ref_pass)
    except ValueError as e:
        raise ValueError(
            f"Cannot insert pass '{new_pass}' into pipeline; reference pass '{ref_pass}' not found"
        ) from e

    pipeline.insert(ref_index + 1, new_pass)


def insert_pass_before(pipeline: list[str], new_pass: str, ref_pass: str) -> None:
    """Insert a pass into an existing pass pipeline at the position before the given reference pass.

    If the reference pass appears multiple times in the pipeline, the new pass is inserted only once
    before the first occurrence of the reference pass.

    Args:
        pipeline (list[str]): An existing pass pipeline, given as a list of passes.
        new_pass (str): The name of the pass to insert.
        ref_pass (str): The name of the reference pass before which the new pass is inserted.

    Raises:
        ValueError: If `ref_pass` is not found in the pass pipeline.

    Example:
        >>> pipeline = ["pass1", "pass2"]
        >>> insert_pass_before(pipeline, "new_pass", ref_pass="pass1")
        >>> pipeline
        ['new_pass', 'pass1', 'pass2']
    """
    try:
        ref_index = pipeline.index(ref_pass)
    except ValueError as e:
        raise ValueError(
            f"Cannot insert pass '{new_pass}' into pipeline; reference pass '{ref_pass}' not found"
        ) from e

    pipeline.insert(ref_index, new_pass)


def insert_stage_after(stages: PipelineStages, new_stage: PipelineStage, ref_stage: str) -> None:
    """Insert a compilation stage into an existing sequence of stages at the position after the
    given reference stage.

    If the reference stage appears multiple times in the sequence of stages, the new stage is
    inserted only once after the first occurrence of the reference pass.

    Args:
        pipeline (PipelineStages): An existing sequence of compilation stages.
        new_stage (PipelineStage): The new compilation stage, given as a tuple where the first
            element is the stage name and the second is a list of strings corresponding to
            compilation pass names.
        ref_stage (str): The name of the reference stage after which the new stage is inserted.

    Raises:
        ValueError: If `ref_stage` is not found in the pass pipeline.

    Example:
        >>> stages = [("stage1", ["s1p1", "s1p2"]), ("stage2", ["s2p1", "s2p2"])]
        >>> insert_stage_after(stages, ("new_stage", ["p0"]), ref_stage="stage1")
        >>> stages
        [('stage1', ['s1p1', 's1p2']), ('new_stage', ['p0']), ('stage2', ['s2p1', 's2p2'])]
    """
    if not hasattr(new_stage, "__len__") or len(new_stage) != 2:
        raise TypeError(
            "The stage to insert must be a tuple in the form ('stage name', ['pass', 'pass', ...])"
        )

    stage_names = [stage[0] for stage in stages]
    try:
        ref_index = stage_names.index(ref_stage)
    except ValueError as e:
        raise ValueError(
            f"Cannot insert stage '{new_stage[0]}' into sequence of stages; "
            f"reference stage '{ref_stage}' not found"
        ) from e

    stages.insert(ref_index + 1, new_stage)


def insert_stage_before(stages: PipelineStages, new_stage: PipelineStage, ref_stage: str) -> None:
    """Insert a compilation stage into an existing sequence of stages at the position before the
    given reference stage.

    If the reference stage appears multiple times in the sequence of stages, the new stage is
    inserted only once before the first occurrence of the reference pass.

    Args:
        pipeline (PipelineStages): An existing sequence of compilation stages.
        new_stage (PipelineStage): The new compilation stage, given as a tuple where the first
            element is the stage name and the second is a list of strings corresponding to
            compilation pass names.
        ref_stage (str): The name of the reference stage before which the new stage is inserted.

    Raises:
        ValueError: If `ref_stage` is not found in the pass pipeline.

    Example:
        >>> stages = [("stage1", ["s1p1", "s1p2"]), ("stage2", ["s2p1", "s2p2"])]
        >>> insert_stage_before(stages, ("new_stage", ["p0"]), ref_stage="stage1")
        >>> stages
        [('new_stage', ['p0']), ('stage1', ['s1p1', 's1p2']), ('stage2', ['s2p1', 's2p2'])]
    """
    if not hasattr(new_stage, "__len__") or len(new_stage) != 2:
        raise TypeError(
            "The stage to insert must be a tuple in the form ('stage name', ['pass', 'pass', ...])"
        )

    stage_names = [stage[0] for stage in stages]
    try:
        ref_index = stage_names.index(ref_stage)
    except ValueError as e:
        raise ValueError(
            f"Cannot insert stage '{new_stage[0]}' into sequence of stages; "
            f"reference stage '{ref_stage}' not found"
        ) from e

    stages.insert(ref_index, new_stage)
