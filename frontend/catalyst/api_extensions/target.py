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
    pipeline: Optional[str] = None,
    triple: Optional[str] = None,
    address: Optional[str] = None,
    executor=None,
):
    """Tag a PennyLane device as a separate cross-compilation target and return it.

    Any QNode wrapping the returned device is kept as a separate compilation unit which carries
    ``catalyst.target = {pipeline, triple}`` and is cross-compiled to a standalone object rather
    than being inlined into the host module. With ``address`` set, the object is additionally
    dispatched to a remote executor (``catalyst.dispatch = {address}``); without it, the object is
    statically linked and runs in-process.

    Args:
        device: A PennyLane device.
        pipeline: Optional lowering-pipeline name registered with the compiler.
        triple: Optional LLVM target triple. Defaults to the host triple.
        address: Optional executor address; when set, the target is dispatched remotely.
        executor: Optional launched :class:`catalyst.Executor` (or any object exposing ``.address``
            and optionally ``.triple``) to source ``address`` and ``triple`` from — a single source
            of truth for where the target runs and which triple it is built for. Explicit
            ``address``/``triple`` still win.

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


_BACKLINE_EXECUTOR_ATTR = "_catalyst_backline_executor"


def attach_executor(device, executor):
    """Attach a launched :class:`catalyst.Executor` to a backline ``device``.

    A backline device whose controller is ``remote`` is cross-compiled and dispatched like a
    ``target(...)`` device; the attached executor supplies the dispatch ``address`` and ``triple``
    (see :func:`get_target`/:func:`get_dispatch`). Explicit ``target(...)`` tags still win.

    Returns the same device, for chaining.
    """
    setattr(device, _BACKLINE_EXECUTOR_ATTR, executor)
    return device


def _backline_controller(device):
    """The controller node of a backline placement attached to ``device``, or ``None``."""
    backline = getattr(device, "backline", None)
    return getattr(backline, "controller", None) if backline is not None else None


def _controller_executors(device):
    """Executors that can source the controller's dispatch coords, most-preferred first: the
    controller node's own ``executor``, then a device-attached one (:func:`attach_executor`)."""
    from catalyst.backline import _realize_executor

    ctrl = _backline_controller(device)
    node_ex = _realize_executor(ctrl) if ctrl is not None else None
    dev_ex = getattr(device, _BACKLINE_EXECUTOR_ATTR, None)
    return [ex for ex in (node_ex, dev_ex) if ex is not None]


def _backline_triple(device) -> Optional[str]:
    """Triple for a remote backline controller: an executor's triple wins, else the controller's."""
    for ex in _controller_executors(device):
        if getattr(ex, "triple", None):
            return ex.triple
    ctrl = _backline_controller(device)
    return getattr(ctrl, "triple", None) if ctrl is not None else None


def _backline_dispatch_address(device) -> Optional[str]:
    """Dispatch address for a remote backline controller: an executor's address wins, else ``addr:port``."""
    for ex in _controller_executors(device):
        if getattr(ex, "address", None):
            return ex.address
    ctrl = _backline_controller(device)
    addr = getattr(ctrl, "addr", None) if ctrl is not None else None
    if not addr:
        return None
    port = getattr(ctrl, "port", None)
    return f"{addr}:{port}" if port else addr


def get_target(device) -> Optional[Target]:
    """Return the cross-compilation :class:`Target` for ``device``, or ``None``.

    An explicit :func:`target` tag wins. Otherwise, if the device carries a backline placement whose
    controller is ``remote``, the controller QNode is cross-compiled like a target device, with the
    triple from an attached :class:`catalyst.Executor` (or the controller's ``triple`` field).
    """
    explicit = getattr(device, _TARGET_ATTR, None)
    if explicit is not None:
        return explicit
    ctrl = _backline_controller(device)
    if ctrl is not None and getattr(ctrl, "remote", False):
        return Target(pipeline=None, triple=_backline_triple(device))
    return None


def get_backline_role(device) -> Optional[str]:
    """Return the backline role ``device``'s module plays, or ``None``.

    Tagging the role lets the transport passes find a module by which node it belongs to, rather than
    matching on the triple/address copied into ``catalyst.target``/``catalyst.dispatch``.
    """
    ctrl = _backline_controller(device)
    if ctrl is not None and getattr(ctrl, "remote", False):
        return "controller"
    return None


def get_dispatch(device) -> Optional[RemoteDispatch]:
    """Return the :class:`RemoteDispatch` for ``device``, or ``None``.

    An explicit :func:`target` ``address=`` wins. Otherwise a ``remote`` backline controller is
    dispatched to the address of its attached :class:`catalyst.Executor` (or its ``addr:port``).
    """
    explicit = getattr(device, _DISPATCH_ATTR, None)
    if explicit is not None:
        return explicit
    ctrl = _backline_controller(device)
    if ctrl is not None and getattr(ctrl, "remote", False):
        address = _backline_dispatch_address(device)
        if address:
            return RemoteDispatch(address=address)
    return None
