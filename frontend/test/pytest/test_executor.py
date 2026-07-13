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

"""Tests for :class:`catalyst.Executor` and the remote-dispatch contract it feeds.

The lifecycle tests are pure Python (no executor process). The one that actually launches a local
``catalyst-executor`` subprocess is skipped unless the binary is present, and runs in a temp cwd so
it never writes a log into the source tree.
"""
import platform
import types
from pathlib import Path

import pytest

from catalyst import Executor, kernel, target
from catalyst.api_extensions.target import RemoteDispatch, get_dispatch, get_target
from catalyst.executor import _default_executor_bin, _triple_from_uname


class _DispatchHandle:
    """A minimal stand-in for a dispatch target: any object carrying ``_catalyst_dispatch`` is
    accepted by ``kernel.declare(remote=...)`` (the same duck-typed seam a backline endpoint uses).
    """

    def __init__(self, address):
        self._catalyst_dispatch = RemoteDispatch(address=address)


# --- construction / lifecycle (no executor process) ----------------------------------------------


def test_construction_is_inert():
    """Constructing an Executor deploys nothing; .address is unavailable until launch."""
    ex = Executor("127.0.0.1:1234")
    with pytest.raises(RuntimeError):
        _ = ex.address


def test_manual_mode_passthrough():
    """Neither local nor host: attach to the given address, spawning no process."""
    ex = Executor("127.0.0.1:1234").launch()
    assert ex.address == "127.0.0.1:1234"
    ex.stop()  # no-op in manual mode; must not raise
    ex.stop()  # idempotent


def test_manual_mode_context_manager():
    with Executor("10.0.0.9:9999") as ex:
        assert ex.address == "10.0.0.9:9999"


def test_launch_is_idempotent():
    ex = Executor("127.0.0.1:1234")
    assert ex.launch() is ex
    assert ex.launch() is ex
    assert ex.address == "127.0.0.1:1234"


def test_repr_carries_name_and_host():
    r = repr(Executor(host="10.0.0.9", name="role-a"))
    assert "role-a" in r and "10.0.0.9" in r


# --- the dispatch contract (target -> get_dispatch -> kernel.declare) -----------------------------


def test_target_tags_dispatch_and_triple():
    dev = types.SimpleNamespace()
    out = target(dev, address="127.0.0.1:1234", triple="aarch64-unknown-linux-gnu")
    assert out is dev
    dispatch = get_dispatch(dev)
    tgt = get_target(dev)
    assert dispatch is not None and dispatch.address == "127.0.0.1:1234"
    assert tgt is not None and tgt.triple == "aarch64-unknown-linux-gnu"


def test_target_without_address_has_no_dispatch():
    dev = target(types.SimpleNamespace(), triple="aarch64-unknown-linux-gnu")
    tgt = get_target(dev)
    assert get_dispatch(dev) is None
    assert tgt is not None and tgt.triple == "aarch64-unknown-linux-gnu"


def test_kernel_declare_remote_resolves_address():
    dev = target(types.SimpleNamespace(), address="127.0.0.1:1234")
    kd = kernel.declare("sym", remote=dev)
    assert kd.remote and kd.remote_address == "127.0.0.1:1234"


def test_kernel_declare_remote_accepts_duck_typed_handle():
    """A bare object carrying _catalyst_dispatch resolves — the backline _EndpointHandle pattern."""
    kd = kernel.declare("sym", remote=_DispatchHandle("10.0.0.9:1"))
    assert kd.remote_address == "10.0.0.9:1"


def test_kernel_declare_remote_true_inherits_program_executor():
    kd = kernel.declare("sym", remote=True)
    assert kd.remote and kd.remote_address is None


def test_kernel_declare_remote_without_dispatch_raises():
    with pytest.raises(ValueError):
        kernel.declare("sym", remote=types.SimpleNamespace())


# --- executor triple: auto-detect + explicit + carry into target() (#2/#3) -----------------------


def test_triple_from_uname_mapping():
    assert _triple_from_uname("Linux", "x86_64") == "x86_64-unknown-linux-gnu"
    assert _triple_from_uname("Linux", "aarch64") == "aarch64-unknown-linux-gnu"
    assert _triple_from_uname("Darwin", "arm64") == "arm64-apple-darwin"
    assert _triple_from_uname("Darwin", "x86_64") == "x86_64-apple-darwin"
    assert _triple_from_uname("Plan9", "sparc") is None


def test_triple_local_autodetect_matches_host():
    """A local executor's triple is the host's, auto-detected from the platform."""
    assert Executor(local=True).triple == _triple_from_uname(platform.system(), platform.machine())


def test_triple_explicit_wins_without_probe():
    """An explicit triple= is returned verbatim (no SSH probe, even for an unreachable host)."""
    ex = Executor(host="unreachable.invalid", triple="x86_64-apple-darwin")
    assert ex.triple == "x86_64-apple-darwin"


def test_triple_attach_only_is_none():
    """Attach-only (neither local nor host, no explicit triple): nothing to detect."""
    assert Executor("127.0.0.1:1234").triple is None


def test_target_from_executor_sources_address_and_triple():
    """target(device, executor=ex) fills address + triple from the executor."""
    ex = Executor("127.0.0.1:1234", triple="aarch64-unknown-linux-gnu").launch()
    dev = target(types.SimpleNamespace(), executor=ex)
    d, t = get_dispatch(dev), get_target(dev)
    assert d is not None and d.address == "127.0.0.1:1234"
    assert t is not None and t.triple == "aarch64-unknown-linux-gnu"
    ex.stop()


def test_target_explicit_overrides_executor():
    """Explicit address=/triple= win over the executor's."""
    ex = Executor("127.0.0.1:1234", triple="aarch64-unknown-linux-gnu").launch()
    dev = target(
        types.SimpleNamespace(), executor=ex, address="10.0.0.9:5", triple="x86_64-apple-darwin"
    )
    d, t = get_dispatch(dev), get_target(dev)
    assert d is not None and d.address == "10.0.0.9:5"
    assert t is not None and t.triple == "x86_64-apple-darwin"
    ex.stop()


def test_target_executor_is_duck_typed():
    """executor= accepts any object exposing .address and .triple, not just Executor."""
    fake = types.SimpleNamespace(address="1.2.3.4:9", triple="aarch64-unknown-linux-gnu")
    dev = target(types.SimpleNamespace(), executor=fake)
    d, t = get_dispatch(dev), get_target(dev)
    assert d is not None and d.address == "1.2.3.4:9"
    assert t is not None and t.triple == "aarch64-unknown-linux-gnu"


# --- a real local launch (gated on a built binary; runs in a temp cwd) ----------------------------

_HAS_EXECUTOR = Path(_default_executor_bin()).exists()


@pytest.mark.skipif(not _HAS_EXECUTOR, reason="no built catalyst-executor binary on this machine")
def test_local_executor_lifecycle(tmp_path, monkeypatch):
    """_LocalProcess actually spawns catalyst-executor, binds, exposes its address, and stops."""
    monkeypatch.chdir(tmp_path)  # any log the launcher writes lands here, not in the source tree
    ex = Executor(local=True, plugins=[], ready_timeout=30).launch()
    try:
        host, sep, port = ex.address.partition(":")
        assert host == "127.0.0.1" and sep == ":" and port.isdigit()
    finally:
        ex.stop()
