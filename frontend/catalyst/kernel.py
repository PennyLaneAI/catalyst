# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Declarations and calls for pre-compiled external kernels callable from @qjit programs."""

from __future__ import annotations

import os
from dataclasses import dataclass

import jax.numpy as jnp


@dataclass(frozen=True)
class KernelDescriptor:
    """Describes the ABI of a pre-compiled external kernel.

    Attributes:
        name:        C symbol name exported by the artifact.
        artifact:    Absolute path to the shared library (.so / .dylib).
        output_spec: Tuple of (shape_tuple, dtype_str) for each output tensor.
    """

    name: str
    artifact: str  # absolute path to .so / .dylib
    output_spec: tuple  # ((shape_tuple, dtype_str), ...)


def _to_hashable(spec):
    """Convert a ShapeDtypeStruct or tuple of them to a hashable spec tuple."""
    specs = (spec,) if hasattr(spec, "shape") else spec
    entries = []
    for s in specs:
        shape = tuple(s.shape)
        if any(d is None for d in shape):
            raise ValueError(
                f"kernel: dynamic shapes unsupported; got shape {shape}. "
                "All dimensions must be static integers."
            )
        entries.append((shape, str(s.dtype)))
    return tuple(entries)


def declare(name: str, artifact: str, outputs) -> KernelDescriptor:
    """Declare a pre-compiled external kernel for use with :func:`kernel.runtime_call`.

    Args:
        name:     C symbol exported by the artifact (e.g. ``"my_func"``).
        artifact: Path to the shared library. Resolved relative to ``os.getcwd()``
                  if not absolute. The file must exist at declare time.
        outputs:  :class:`jax.ShapeDtypeStruct` or tuple of them describing each
                  output tensor. JAX needs these shapes at trace time to infer
                  what the function returns.

    Returns:
        A :class:`KernelDescriptor`
    """
    artifact = os.path.abspath(artifact)
    if not os.path.isfile(artifact):
        raise FileNotFoundError(f"kernel.declare: artifact not found: {artifact!r}")

    return KernelDescriptor(
        name=name,
        artifact=artifact,
        output_spec=_to_hashable(outputs),
    )


def runtime_call(kernel_descriptor, *args):
    """Call a pre-compiled external kernel from inside ``@qjit``.

    Args:
        kernel_descriptor: A :class:`KernelDescriptor` returned by :func:`declare`.
        *args: Input tensors. JAX infers shapes and dtypes at trace time.

    Returns:
        tuple: Output tensors. For a single output, unpack explicitly::

            (result,) = kernel.runtime_call(my_kernel, data)

    Raises:
        NotImplementedError: On use under ``jax.grad`` or ``jax.vmap``.
    """
    from catalyst.jax_primitives import runtime_call_p  # pylint: disable=import-outside-toplevel

    jax_args = [jnp.asarray(a) for a in args]
    return tuple(runtime_call_p.bind(*jax_args, kernel_descriptor=kernel_descriptor))
