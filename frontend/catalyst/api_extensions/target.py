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

_TARGET_ATTR = "_catalyst_target"
_DISPATCH_ATTR = "_catalyst_dispatch"


@dataclass(frozen=True)
class Target:
    """Cross-compilation target spec attached to a device by :func:`target`.

    Args:
        pipeline: Optional name of a lowering pipeline registered with the compiler.
        triple: Optional LLVM target triple. Defaults to the host triple.
    """

    pipeline: Optional[str] = None
    triple: Optional[str] = None


@dataclass(frozen=True)
class RemoteDispatch:
    """Remote dispatch spec attached to a device by :func:`target` with an ``address``.

    Args:
        address: Executor address, e.g. ``"127.0.0.1:1373"``.
    """

    address: str


def target(
    device,
    *,
    executor=None,
    pipeline: Optional[str] = None,
    triple: Optional[str] = None,
    address: Optional[str] = None,
):
    """Tag a PennyLane device as a separate cross-compilation target and return it.

    Any QNode wrapping the returned device is kept as a separate compilation unit which carries
    ``catalyst.target = {pipeline, triple}`` and is cross-compiled to a standalone object rather
    than being inlined into the host module. With ``address`` set, the object is additionally
    dispatched to a remote executor (``catalyst.dispatch = {address}``); without it, the object is
    statically linked and runs in-process.

    Pass ``executor`` (a launched :class:`catalyst.Executor`, or any object exposing ``.address``
    and an optional ``.triple``) to fill ``address`` and ``triple`` from it — a single source of
    truth for *where* the target runs and *which* architecture it is cross-compiled for, so neither
    needs to be typed by hand. Explicit ``address``/``triple`` still win if given.

    Args:
        device: A PennyLane device.
        executor: Optional launched executor to source ``address`` (and ``triple``) from.
        pipeline: Optional lowering-pipeline name registered with the compiler.
        triple: Optional LLVM target triple. Defaults to the host triple.
        address: Optional executor address; when set, the target is dispatched remotely.

    Returns:
        The same device, now tagged with target (and, if ``address`` is given, dispatch) metadata.
    """
    if executor is not None:
        if address is None:
            address = executor.address
        if triple is None:
            triple = getattr(executor, "triple", None)
    setattr(device, _TARGET_ATTR, Target(pipeline=pipeline, triple=triple))
    if address is not None:
        setattr(device, _DISPATCH_ATTR, RemoteDispatch(address=address))
    return device


def get_target(device) -> Optional[Target]:
    """Return the :class:`Target` attached via :func:`target`, or ``None``."""
    return getattr(device, _TARGET_ATTR, None)


def get_dispatch(device) -> Optional[RemoteDispatch]:
    """Return the :class:`RemoteDispatch` attached via :func:`target` (``address=...``), or ``None``."""
    return getattr(device, _DISPATCH_ATTR, None)
