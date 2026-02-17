# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This file contains the implementation of the `specs` function for the Unified Compiler."""

from __future__ import annotations

import warnings
from collections import defaultdict
from typing import Literal

from pennylane.workflow.qnode import QNode

from catalyst.jit import QJIT
from catalyst.passes.pass_api import PassPipelineWrapper
from catalyst.python_interface.compiler import Compiler

from .specs_collector import ResourcesResult, specs_collect
from .xdsl_conversion import get_mlir_module


class StopCompilation(Exception):
    """Custom exception to stop compilation early when the desired specs level is reached."""


def mlir_specs(
    qnode: QJIT,
    level: int | tuple[int] | list[int] | Literal["all"],
    *args,
    level_to_markers: dict[int, tuple[str]] | None = None,
    **kwargs,
) -> ResourcesResult | dict[str, ResourcesResult]:
    """Compute the specs used for a circuit at the level of an MLIR pass.

    Args:
        qnode (QJIT): The (QJIT'd) qnode to get the specs for
        level (int | tuple[int] | list[int] | "all"): The MLIR pass level to get the specs for
        *args: Positional arguments to pass to the QNode
        level_to_markers (dict[str, tuple[str]]): A dictionary that maps integer levels to marker labels.
        **kwargs: Keyword arguments to pass to the QNode

    Returns:
        ResourcesResult | dict[str, ResourcesResult]: The resources for the circuit at the
          specified level
    """

    cache: dict[int, tuple[ResourcesResult, str]] = {}
    level_to_markers: dict[int, list[str]] = level_to_markers or {}

    if not isinstance(qnode, QJIT) or (
        not isinstance(qnode.original_function, QNode)
        and not (
            isinstance(qnode.original_function, PassPipelineWrapper)
            and isinstance(qnode.original_qnode, QNode)
        )
    ):
        raise ValueError(
            "The provided `qnode` argument does not appear to be a valid QJIT compiled QNode."
        )

    if args or kwargs:
        warnings.warn(
            "The `specs` function does not yet support dynamic arguments, "
            "so the results may not reflect information provided by the arguments.",
            UserWarning,
        )

    max_level: int | None = _get_max_level(level)

    def _specs_callback(previous_pass, module, next_pass, pass_level=0):
        """Callback function for gathering circuit specs."""

        pass_instance = previous_pass if previous_pass else next_pass
        result = specs_collect(module)

        pass_name = str(pass_instance) + f" (MLIR-{pass_level})"
        # Always prioritize marker label if it exists
        if m := level_to_markers.get(pass_level):
            pass_name = ", ".join(m if not isinstance(m, str) else [m])

        cache[pass_level] = (
            result,
            pass_name if pass_level else "Before MLIR Passes (MLIR-0)",
        )

        if max_level is not None and pass_level >= max_level:
            raise StopCompilation("Stopping compilation after reaching max specs level.")

    mlir_module = get_mlir_module(qnode, args, kwargs)
    try:
        Compiler.run(mlir_module, callback=_specs_callback)
    except StopCompilation:
        # We use StopCompilation to interrupt the compilation once we reach
        # the desired level
        pass

    if max_level is None:
        return {cache_item[1]: cache_item[0] for cache_item in cache.values()}

    if isinstance(level, (tuple, list)):
        if any(lvl not in cache for lvl in level):
            missing = [str(lvl) for lvl in level if lvl not in cache]
            raise ValueError(
                f"Requested specs levels {', '.join(missing)} not found in MLIR pass list."
            )
        return {f"{cache[lvl][1]}": cache[lvl][0] for lvl in level if lvl in cache}

    if level not in cache:
        raise ValueError(f"Requested specs level {level} not found in MLIR pass list.")

    # Just one level was specified
    return cache[level][0]


def _get_max_level(level: int | tuple[int] | list[int] | Literal["all"]) -> int | None:
    """Retrieve the maximum level based on the user input.

    Args:
        level: User's level sequence.

    Returns:
        max_level (int | None): The maximum level.

    """
    if level == "all":
        return None

    if isinstance(level, int):
        return level

    if isinstance(level, (tuple, list)):
        if not all(isinstance(lvl, int) for lvl in level):
            raise ValueError("All elements in 'level' sequence must be integers.")
        return max(level)

    raise ValueError("The 'level' argument must be an int, a tuple/list of ints, or 'all'.")
