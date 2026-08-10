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

"""Unit tests for :mod:`catalyst.api_extensions.target`."""

from catalyst.api_extensions.target import (
    _DISPATCH_ATTR,
    _TARGET_ATTR,
    RemoteDispatch,
    Target,
    _attr,
    attach_executor,
    get_backline_role,
    get_dispatch,
    get_target,
    target,
)


class _Dev:
    """Minimal stand-in for a PennyLane device (attribute-mutable)."""


class _Ctrl:
    """Stand-in for a backline controller node.

    A controller carries no address or triple of its own: the executor supplies both, and the
    transport endpoint (``comm_host``/``oob_port``) belongs to a coprocessor.
    """

    def __init__(self, *, remote=False, executor=None):
        self.remote = remote
        if executor is not None:
            self.executor = executor


class _Placement:
    """Stand-in for a backline placement carrying a controller."""

    def __init__(self, controller):
        self.controller = controller


class _Ex:
    """Stand-in for a :class:`catalyst.Executor` exposing ``address`` and ``triple``."""

    def __init__(self, address=None, triple=None):
        if address is not None:
            self.address = address
        if triple is not None:
            self.triple = triple


class TestAttr:
    """The None-safe :func:`_attr` helper."""

    def test_none_object_returns_default(self):
        """``_attr(None, ...)`` returns the default without touching the missing object."""
        assert _attr(None, "anything") is None
        assert _attr(None, "anything", default="X") == "X"

    def test_missing_attr_returns_default(self):
        """A missing attribute on a real object returns the default."""
        assert _attr(_Dev(), "missing") is None
        assert _attr(_Dev(), "missing", default=42) == 42

    def test_present_attr_returned(self):
        """A present attribute is returned as-is."""
        obj = _Dev()
        obj.x = "hello"
        assert _attr(obj, "x") == "hello"


class TestTargetFn:
    """The :func:`target` setter."""

    def test_sets_target_attr_and_returns_device(self):
        """Attaches a :class:`Target` and returns the same device; no dispatch tag without an address."""
        dev = _Dev()
        out = target(dev, pipeline="opt", triple="x86_64")
        assert out is dev
        assert getattr(dev, _TARGET_ATTR) == Target(pipeline="opt", triple="x86_64")
        assert not hasattr(dev, _DISPATCH_ATTR)

    def test_sets_dispatch_when_address_given(self):
        """Passing ``address=`` additionally attaches a :class:`RemoteDispatch`."""
        dev = _Dev()
        target(dev, address="10.0.0.1:1373")
        assert getattr(dev, _DISPATCH_ATTR) == RemoteDispatch(address="10.0.0.1:1373")

    def test_executor_supplies_address_and_triple(self):
        """An ``executor=`` fills in ``address``/``triple`` when those args aren't passed."""
        dev = _Dev()
        target(dev, executor=_Ex(address="1.2.3.4:1373", triple="aarch64"))
        assert getattr(dev, _TARGET_ATTR).triple == "aarch64"
        assert getattr(dev, _DISPATCH_ATTR).address == "1.2.3.4:1373"

    def test_direct_args_win_over_executor(self):
        """Explicit ``address``/``triple`` take precedence over the executor's values."""
        dev = _Dev()
        target(
            dev,
            address="direct-addr:2",
            triple="direct-triple",
            executor=_Ex(address="ex-addr:1", triple="ex-triple"),
        )
        assert getattr(dev, _TARGET_ATTR).triple == "direct-triple"
        assert getattr(dev, _DISPATCH_ATTR).address == "direct-addr:2"


class TestAttachExecutor:
    """The :func:`attach_executor` setter."""

    def test_stores_executor_and_returns_device(self):
        """Attaches the executor under the backline-executor attr and returns the device."""
        dev = _Dev()
        ex = _Ex(address="a:1")
        assert attach_executor(dev, ex) is dev
        assert getattr(dev, "_catalyst_backline_executor") is ex


class TestGetTarget:
    """The :func:`get_target` resolver: attached tag, else derived from a remote backline controller."""

    def test_untagged_device_returns_none(self):
        """No tag and no backline: returns ``None``."""
        assert get_target(_Dev()) is None

    def test_returns_attached_tag(self):
        """A tag attached via :func:`target` is returned as-is."""
        dev = _Dev()
        target(dev, triple="x86_64")
        assert get_target(dev) == Target(pipeline=None, triple="x86_64")

    def test_derived_from_remote_backline_has_no_triple_without_an_executor(self):
        """A remote controller derives a :class:`Target`, but the triple comes only from an executor."""
        dev = _Dev()
        dev.placement = _Placement(_Ctrl(remote=True))
        assert get_target(dev) == Target(pipeline=None, triple=None)

    def test_backline_but_non_remote_returns_none(self):
        """A non-remote backline controller does not produce a derived tag."""
        dev = _Dev()
        dev.placement = _Placement(_Ctrl(remote=False))
        assert get_target(dev) is None

    def test_attached_executor_supplies_the_triple(self):
        """A device-attached executor's triple is used for the derived target."""
        dev = _Dev()
        dev.placement = _Placement(_Ctrl(remote=True))
        attach_executor(dev, _Ex(triple="ex-triple"))
        assert get_target(dev).triple == "ex-triple"

    def test_node_executor_triple_wins_over_device_attached_executor(self):
        """The controller-node's own executor takes precedence over a device-attached one."""
        dev = _Dev()
        dev.placement = _Placement(_Ctrl(remote=True, executor=_Ex(triple="node-triple")))
        attach_executor(dev, _Ex(triple="attached-triple"))
        assert get_target(dev).triple == "node-triple"


class TestGetDispatch:
    """The :func:`get_dispatch` resolver: attached tag, else derived from a remote backline controller."""

    def test_untagged_device_returns_none(self):
        """No tag and no backline: returns ``None``."""
        assert get_dispatch(_Dev()) is None

    def test_returns_attached_dispatch_tag(self):
        """A dispatch tag attached via ``target(..., address=...)`` is returned as-is."""
        dev = _Dev()
        target(dev, address="1.2.3.4:1373")
        assert get_dispatch(dev) == RemoteDispatch(address="1.2.3.4:1373")

    def test_derived_from_attached_executor_address(self):
        """Derives ``RemoteDispatch`` from the executor's address."""
        dev = _Dev()
        dev.placement = _Placement(_Ctrl(remote=True))
        attach_executor(dev, _Ex(address="10.0.0.5:1373"))
        assert get_dispatch(dev) == RemoteDispatch(address="10.0.0.5:1373")

    def test_node_executor_address_wins_over_device_attached_executor(self):
        """The controller-node's own executor takes precedence over a device-attached one."""
        dev = _Dev()
        dev.placement = _Placement(_Ctrl(remote=True, executor=_Ex(address="node-addr:1")))
        attach_executor(dev, _Ex(address="attached-addr:2"))
        assert get_dispatch(dev) == RemoteDispatch(address="node-addr:1")

    def test_non_remote_controller_returns_none(self):
        """A non-remote backline controller does not produce a derived dispatch."""
        dev = _Dev()
        dev.placement = _Placement(_Ctrl(remote=False, executor=_Ex(address="a:1")))
        assert get_dispatch(dev) is None

    def test_remote_controller_without_an_executor_returns_none(self):
        """A remote controller with no executor has nowhere to dispatch to.

        A node's own address is the transport's out-of-band endpoint, on a different channel from
        the executor's dispatch address, so it must not be used as a fallback.
        """
        dev = _Dev()
        dev.placement = _Placement(_Ctrl(remote=True))
        assert get_dispatch(dev) is None


class TestGetBacklineRole:
    """The :func:`get_backline_role` classifier."""

    def test_plain_device_returns_none(self):
        """A device without a backline placement has no role."""
        assert get_backline_role(_Dev()) is None

    def test_remote_backline_returns_controller(self):
        """A remote backline placement identifies the device as the ``"controller"``."""
        dev = _Dev()
        dev.placement = _Placement(_Ctrl(remote=True))
        assert get_backline_role(dev) == "controller"

    def test_non_remote_backline_returns_none(self):
        """A non-remote backline placement does not confer a role."""
        dev = _Dev()
        dev.placement = _Placement(_Ctrl(remote=False))
        assert get_backline_role(dev) is None
