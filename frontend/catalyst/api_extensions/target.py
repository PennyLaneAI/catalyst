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

"""Tag a PennyLane device with a separate cross-compilation target, optionally remote."""

from dataclasses import dataclass
from typing import Optional

_TARGET_ATTR = "_catalyst_target"
_DISPATCH_ATTR = "_catalyst_dispatch"


def _attr(obj, name, default=None):
    """None-safe :func:`getattr`."""
    return default if obj is None else getattr(obj, name, default)


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
    """Tag a PennyLane device with a separate cross-compilation target and return it.

    Any QNode wrapping the returned device is kept as a separate compilation unit, tagged with a
    :class:`Target`. It is cross-compiled to a standalone object rather than inlined into the host
    module.

    With ``address`` set, the object is dispatched to a remote executor and tagged with a
    :class:`RemoteDispatch`. Without ``address``, it is statically linked and runs in-process.

    Args:
        device: A PennyLane device.
        pipeline: Optional lowering-pipeline name registered with the compiler.
        triple: Optional LLVM target triple. Defaults to the host triple.
        address: Optional executor address. Defaults to local dispatch.
        executor: Optional launched :class:`catalyst.Executor` supplying ``address``/``triple``
            when not passed directly.

    Returns:
        The device with the target tag set.
    """
    if executor is not None:
        if address is None:
            address = executor.address
        if triple is None:
            triple = _attr(executor, "triple")
    setattr(device, _TARGET_ATTR, Target(pipeline=pipeline, triple=triple))
    if address is not None:
        setattr(device, _DISPATCH_ATTR, RemoteDispatch(address=address))
    return device


_BACKLINE_EXECUTOR_ATTR = "_catalyst_backline_executor"


def attach_executor(device, executor):
    """Attach a launched :class:`catalyst.Executor` to a backline ``device``.

    Args:
        device: A PennyLane device with a backline placement whose controller is ``remote``.
        executor: A launched :class:`catalyst.Executor`.

    Returns:
        The same device, tagged with the executor for chaining.
    """
    setattr(device, _BACKLINE_EXECUTOR_ATTR, executor)
    return device


def _get_backline_controller(device):
    """The controller node of a backline placement attached to ``device``, or ``None``."""
    return _attr(_attr(device, "backline"), "controller")


def _get_controller_executors(device):
    """Controller dispatch-coord sources, in preference order: the node's own ``executor``, then one attached via :func:`attach_executor`."""
    from catalyst.backline import _realize_executor

    ctrl = _get_backline_controller(device)
    node_ex = _realize_executor(ctrl) if ctrl is not None else None
    dev_ex = _attr(device, _BACKLINE_EXECUTOR_ATTR)
    return [ex for ex in (node_ex, dev_ex) if ex is not None]


def _get_backline_triple(device) -> Optional[str]:
    """Triple for a remote backline controller, sourced from an executor. Defaults to the controller's own."""
    for ex in _get_controller_executors(device):
        if _attr(ex, "triple"):
            return ex.triple
    return _attr(_get_backline_controller(device), "triple")


def _get_backline_dispatch_address(device) -> Optional[str]:
    """Dispatch address for a remote backline controller, sourced from an executor. Defaults to the controller's own ``addr:port``."""
    for ex in _get_controller_executors(device):
        if _attr(ex, "address"):
            return ex.address
    ctrl = _get_backline_controller(device)
    addr = _attr(ctrl, "addr")
    if not addr:
        return None
    port = _attr(ctrl, "port")
    return f"{addr}:{port}" if port else addr


def get_target(device) -> Optional[Target]:
    """Return the cross-compilation :class:`Target` for ``device``, or ``None``.

    Args:
        device: A PennyLane device that may carry a :class:`Target` tag or a backline placement.

    Returns:
        Target: The attached :class:`Target` tag, or one derived from a remote backline controller.
        None: If ``device`` is neither a target nor a remote backline controller.
    """
    # Attached target: return the Target tag set by target(...).
    attached = _attr(device, _TARGET_ATTR)
    if attached is not None:
        return attached

    # Derived target: build one from a remote backline controller.
    ctrl = _get_backline_controller(device)
    if not _attr(ctrl, "remote", False):
        return None
    return Target(pipeline=None, triple=_get_backline_triple(device))


def get_backline_role(device) -> Optional[str]:
    """Return the backline role ``device``'s module plays, or ``None``.

    Args:
        device: A PennyLane device that may carry a backline placement.

    Returns:
        str: ``"controller"`` for the controller of a remote backline placement.
        None: If ``device`` has no backline role.
    """
    ctrl = _get_backline_controller(device)
    if _attr(ctrl, "remote", False):
        return "controller"
    return None


def get_dispatch(device) -> Optional[RemoteDispatch]:
    """Return the :class:`RemoteDispatch` for ``device``, or ``None``.

    Args:
        device: A PennyLane device that may carry a :class:`RemoteDispatch` tag or a remote
            backline controller.

    Returns:
        RemoteDispatch: The attached :class:`RemoteDispatch` tag, or one derived from a remote backline controller.
        None: If ``device`` is not dispatched to a remote executor.
    """
    # Attached dispatch: return the RemoteDispatch tag set by target(..., address=...).
    attached = _attr(device, _DISPATCH_ATTR)
    if attached is not None:
        return attached

    # Derived dispatch: build one from a remote backline controller.
    ctrl = _get_backline_controller(device)
    if not _attr(ctrl, "remote", False):
        return None
    address = _get_backline_dispatch_address(device)
    return RemoteDispatch(address=address) if address else None
