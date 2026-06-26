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
from typing import Optional

import jax.numpy as jnp


@dataclass(frozen=True)
class KernelDescriptor:
    """Describes the ABI of an external kernel callable via :func:`runtime_call`.

    One of two shapes, distinguished by ``artifact``:
      * **local kernel** — ``artifact`` is a path to the shared library; ``remote_address`` is None.
      * **remote symbol** — ``artifact`` is None (the symbol lives on the executor);
        ``remote_address`` is the bound executor (``"host:port"``) or None to inherit the program's
        single executor.

    Attributes:
        name:           C symbol name. For a local kernel it is exported by ``artifact``; for a
                        remote kernel it is a symbol already loaded on the executor.
        artifact:       Absolute path to the shared library (.so / .dylib), or ``None`` for a
                        remote symbol (which lives on the executor, not in a local artifact).
        output_spec:    Tuple of (shape_tuple, dtype_str) for each output tensor.
        remote_address: For a remote kernel, the executor address it targets, e.g.
                        ``"ADDR:PORT"``; ``None`` means "inherit the program's single
                        executor". Must be ``None`` for a local kernel.
    """

    name: str
    artifact: Optional[str]  # absolute path to .so / .dylib, or None for a remote symbol
    output_spec: tuple  # ((shape_tuple, dtype_str), ...)
    remote_address: Optional[str] = None  # bound executor for a remote symbol; None => inherit

    @property
    def remote(self) -> bool:
        """True iff this is an executor-side symbol (no local artifact)."""
        return self.artifact is None

    def __post_init__(self):
        # `remote_address` only makes sense for a remote symbol; a local kernel has no executor.
        if self.remote_address is not None and self.artifact is not None:
            raise ValueError(
                "kernel: `remote_address` is only valid for a remote kernel (artifact=None)"
            )


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


def declare(name: str, artifact: Optional[str] = None, outputs=None, *,
            remote=False) -> KernelDescriptor:
    """Declare an external kernel for use with :func:`kernel.runtime_call`.

    Args:
        name:     C symbol name. For a local kernel, exported by ``artifact``; for a remote
                  kernel, a symbol already loaded on the executor (e.g.
                  ``"fpga_trampoline_a_setup"``).
        artifact: Path to the shared library for a local kernel (resolved relative to
                  ``os.getcwd()`` if not absolute; must exist at declare time). Omit for a
                  remote kernel.
        outputs:  :class:`jax.ShapeDtypeStruct` or tuple of them describing each output tensor.
                  JAX needs these at trace time to infer what the call returns.
        remote:   Mark the symbol as executor-side. Pass the **remote device**
                  (``target(..., address=...)``) to bind this call explicitly to that
                  executor's address. ``True`` inheriting the program's single remote executor.

    Returns:
        A :class:`KernelDescriptor`
    """
    output_spec = _to_hashable(outputs) if outputs is not None else ()

    if remote:
        address = None
        if remote is not True:
            from catalyst.api_extensions.target import get_dispatch

            dispatch = get_dispatch(remote)
            if dispatch is None:
                raise ValueError(
                    "kernel.declare(remote=<device>): the device has no remote dispatch; create it "
                    "with target(..., address=...) first, or pass remote=True to inherit "
                    "the program's single executor."
                )
            address = dispatch.address
        # Remote symbol: no local artifact; `remote` is derived from artifact=None.
        return KernelDescriptor(
            name=name, artifact=None, output_spec=output_spec, remote_address=address
        )

    if artifact is None:
        raise ValueError(
            "kernel.declare: a local kernel requires `artifact`; pass remote=True for an "
            "executor-side symbol"
        )
    artifact = os.path.abspath(artifact)
    if not os.path.isfile(artifact):
        raise FileNotFoundError(f"kernel.declare: artifact not found: {artifact!r}")

    return KernelDescriptor(name=name, artifact=artifact, output_spec=output_spec)


def define(builder, *, name: Optional[str] = None, outputs):
    """Build a kernel with ``builder`` and declare it, as a single decorator.

    Args:
        builder: A backend-specific object implementing ``build(kernel_fn, *, name) -> path``, 
            where ``path`` points to a shared library exporting ``name`` with the 
            :func:`runtime_call` ABI.
        name: Symbol the artifact must export. Defaults to ``kernel_fn.__name__``;
            passed to both ``builder.build`` and :func:`declare`.
        outputs: :class:`jax.ShapeDtypeStruct` or tuple of them, forwarded to :func:`declare`.

    Returns:
        KernelDescriptor: the declared kernel.
    """

    def wrap(kernel_fn):
        sym = name or getattr(kernel_fn, "__name__", None)
        artifact = builder.build(kernel_fn, name=sym)
        return declare(sym, artifact=str(artifact), outputs=outputs)

    return wrap


def runtime_call(kernel_descriptor, *args):
    """Call an external kernel from inside ``@qjit`` — local or remote.

    If ``kernel_descriptor.remote`` is set, the symbol is dispatched on the program's remote
    executor (rewritten to a ``remote.call``); otherwise it calls the local artifact.

    Args:
        kernel_descriptor: A :class:`KernelDescriptor` returned by :func:`declare` / :func:`define`.
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
