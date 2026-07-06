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

"""Tag a PennyLane device as a separate cross-compilation target, optionally remote."""

from dataclasses import dataclass
from typing import Optional

import pennylane as qp

_TARGET_ATTR = "_catalyst_target"
_DISPATCH_ATTR = "_catalyst_dispatch"


@dataclass(frozen=True)
class Target:
    """Cross-compilation target spec attached to a device by :func:`target`.

    Args:
        backend: Optional backend name, recorded as metadata on the target module.
        pipeline: Optional name of a lowering pipeline registered with compiler.
        triple: Optional LLVM target triple. Defaults to the host triple.
    """

    backend: Optional[str] = None
    pipeline: Optional[str] = None
    triple: Optional[str] = None


@dataclass(frozen=True)
class RemoteDispatch:
    """Remote dispatch spec attached to a device by :func:`remote`.

    Args:
        address: Executor address, e.g. ``"127.0.0.1:1373"``.
    """

    address: str


def target(
    device,
    *,
    backend: Optional[str] = None,
    pipeline: Optional[str] = None,
    triple: Optional[str] = None,
):
    """Tag a PennyLane device as a separate cross-compilation target and return it.

    Any QNode wrapping the returned device is kept as a separate compilation unit which carries
    ``catalyst.target = {backend, pipeline, triple}`` and is cross-compiled to a standalone
    object file rather than being inlined into the host module.

    Args:
        device: A PennyLane device.
        backend: Optional backend name, recorded as metadata on the target module.
        pipeline: Optional lowering-pipeline name registered with the compiler.
        triple: Optional LLVM target triple. Defaults to the host triple.

    Returns:
        The same device, now tagged with target metadata.
    """
    setattr(device, _TARGET_ATTR, Target(backend=backend, pipeline=pipeline, triple=triple))
    return device


def remote(device, *, address: str):
    """Mark a :func:`target` device for remote dispatch and return it.

    Stamps ``catalyst.dispatch = {address}`` so the ``dispatch-remote-targets`` pass ships the
    compiled object to ``address`` and rewrites host-side calls to it into remote calls. Wrap the
    device with :func:`target` first to set cross-compilation options; a default target is applied
    if it has none.

    Args:
        device: A PennyLane device.
        address: Executor address the object is shipped to and called on, e.g. ``"127.0.0.1:1373"``.

    Returns:
        The same device, now also tagged for remote dispatch.
    """
    if get_target(device) is None:
        target(device)
    setattr(device, _DISPATCH_ATTR, RemoteDispatch(address=address))
    return device


def get_target(device) -> Optional[Target]:
    """Return the :class:`Target` previously attached via :func:`target`/:func:`remote`, or ``None``."""
    return getattr(device, _TARGET_ATTR, None)


def get_dispatch(device) -> Optional[RemoteDispatch]:
    """Return the :class:`RemoteDispatch` attached via :func:`remote`, or ``None``."""
    return getattr(device, _DISPATCH_ATTR, None)


def run_remote(device, endpoint, *, backend: Optional[str] = None):
    """Run an ordinary device's circuits on a remote executor and return it.

    Simple wrapper over :func:`target` + :func:`remote`: tags ``device`` so its QNodes are cross-compiled to a standalone object and dispatched to a remote host. This provides a simple syntax for running any device (e.g. ``null.qubit``) remotely::

        dev = catalyst.run_remote(qp.device("null.qubit", wires=2), "host:port")

    Args:
        device: A PennyLane device.
        endpoint: The remote host — an address string (``"host:port"``) or a ``pennylane.Endpoint``
            (its ``host`` and, from ``attrs``, an optional target ``triple`` are used). ``run_remote``
            always dispatches remotely, regardless of an Endpoint's ``local`` flag.
        backend: Optional backend name recorded as metadata on the target.

    Returns:
        The same ``device``, now tagged for remote execution.
    """
    if isinstance(endpoint, str):
        address, triple = endpoint, None
    else:
        address = endpoint.host
        triple = (getattr(endpoint, "attrs", None) or {}).get("triple")
    return remote(target(device, backend=backend, triple=triple), address=address)
