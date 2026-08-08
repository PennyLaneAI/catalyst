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

"""Unit tests for :mod:`catalyst.executor.manager` — the public :class:`Executor`: config
defaults, mode dispatch, attach-mode launch, workspace lifecycle (``setup_workspace`` /
``remove_workspace``), and the ``_scp_bundle`` / ``_deploy_bundle`` split. Subprocess-facing
calls (``SCP.deploy``, ``RemoteOps.rmdir``) are mocked."""

from unittest.mock import MagicMock, patch

import pytest

from catalyst.executor.manager import Executor, ExecutorConfig


class TestExecutorConfigDefaults:
    """Default values of :class:`ExecutorConfig`."""

    def test_defaults(self):
        """Fresh :class:`ExecutorConfig` uses documented defaults."""
        c = ExecutorConfig()
        assert c.user == ""
        assert c.port is None
        assert c.copy is False
        assert c.sudo is True
        assert c.ready_timeout == 60.0
        assert c.verbose == 1


class TestExecutorConstruction:
    """Attribute wiring in :meth:`Executor.__init__` for each mode."""

    def test_carry_mode_defaults(self):
        """Attach-only mode (no host, no local) — inert state and default name."""
        ex = Executor("1.2.3.4:5")
        assert ex.host is None
        assert ex._local is False
        assert ex.name == "executor"
        assert not ex._launched

    def test_local_mode_flag(self):
        """``local=True`` is stored on the instance."""
        ex = Executor(local=True)
        assert ex._local is True

    def test_host_mode_stored(self):
        """``host`` and ``user`` land on ``.host`` and the config."""
        ex = Executor(host="10.0.0.9", user="me")
        assert ex.host == "10.0.0.9"
        assert ex._cfg.user == "me"


class TestExecutorAddress:
    """The :attr:`Executor.address` property guard."""

    def test_unlaunched_raises(self):
        """Reading ``address`` before ``launch()`` raises :class:`RuntimeError`."""
        ex = Executor("1.2.3.4:5")
        with pytest.raises(RuntimeError, match="not launched"):
            _ = ex.address

    def test_attach_mode_launch_exposes_address(self):
        """Attach-only mode: after ``launch()``, ``.address`` returns the given endpoint."""
        # Attach-only mode (neither local nor host) short-circuits launch(), no subprocess.
        ex = Executor("1.2.3.4:5").launch()
        assert ex.address == "1.2.3.4:5"


class TestSetupWorkspace:
    """Preconditions and behavior of :meth:`Executor.setup_workspace`."""

    def test_missing_host_raises(self):
        """Requires ``host=`` — otherwise :class:`ValueError`."""
        ex = Executor(workspace="~/ws", bundle="/tmp/b")
        with pytest.raises(ValueError, match="host="):
            ex.setup_workspace()

    def test_missing_workspace_raises(self):
        """Requires a pinned ``workspace=`` — otherwise :class:`ValueError`."""
        ex = Executor(host="h", bundle="/tmp/b")
        with pytest.raises(ValueError, match="workspace="):
            ex.setup_workspace()

    def test_missing_bundle_raises(self):
        """Requires ``bundle=`` — otherwise :class:`ValueError`."""
        ex = Executor(host="h", workspace="~/ws")
        with pytest.raises(ValueError, match="bundle="):
            ex.setup_workspace()

    def test_calls_deploy_and_disables_copy(self, tmp_path):
        """Deploys the bundle and flips ``copy=False`` so a later ``launch()`` won't re-copy."""
        # Real bundle dir so _deploy_bundle runs through until SCP.deploy is mocked.
        bundle = tmp_path / "bundle"
        bundle.mkdir()
        ex = Executor(host="h", user="me", workspace="~/ws", bundle=str(bundle), copy=True)
        with patch.object(Executor, "_deploy_bundle") as deploy:
            ex.setup_workspace()
        deploy.assert_called_once()
        assert ex._cfg.copy is False


class TestRemoveWorkspace:
    """Preconditions and behavior of :meth:`Executor.remove_workspace`."""

    def test_missing_host_raises(self):
        """Requires ``host=`` — otherwise :class:`ValueError`."""
        ex = Executor(workspace="~/ws")
        with pytest.raises(ValueError, match="host="):
            ex.remove_workspace()

    def test_missing_workspace_raises(self):
        """Requires a pinned ``workspace=`` — otherwise :class:`ValueError`."""
        ex = Executor(host="h")
        with pytest.raises(ValueError, match="workspace="):
            ex.remove_workspace()

    def test_delegates_to_remote_rmdir(self):
        """Delegates to :meth:`RemoteOps.rmdir` and threads through ``force=``."""
        ex = Executor(host="h", user="me", workspace="~/ws")
        with patch("catalyst.executor.manager.RemoteOps.rmdir") as rmdir:
            ex.remove_workspace(force=True)
        rmdir.assert_called_once()
        assert rmdir.call_args.kwargs.get("force") is True


class TestScpBundleGate:
    """The ``copy`` + ``bundle`` gate on :meth:`Executor._scp_bundle`."""

    def test_noop_when_copy_false(self):
        """No deploy when ``copy=False``."""
        ex = Executor(host="h", bundle="/tmp/b", copy=False)
        with patch.object(Executor, "_deploy_bundle") as deploy:
            ex._scp_bundle("me", "h", "ws")
        deploy.assert_not_called()

    def test_noop_when_bundle_none(self):
        """No deploy when ``bundle`` is unset."""
        ex = Executor(host="h", copy=True)
        with patch.object(Executor, "_deploy_bundle") as deploy:
            ex._scp_bundle("me", "h", "ws")
        deploy.assert_not_called()

    def test_delegates_when_gates_open(self):
        """Both flags set: delegates to :meth:`_deploy_bundle`."""
        ex = Executor(host="h", bundle="/tmp/b", copy=True)
        with patch.object(Executor, "_deploy_bundle") as deploy:
            ex._scp_bundle("me", "h", "ws")
        deploy.assert_called_once_with("me", "h", "ws")


class TestDeployBundle:
    """The build + scp composition inside :meth:`Executor._deploy_bundle`."""

    def test_calls_build_then_scp_deploy(self, tmp_path):
        """Calls ``build(triple, bundle)`` then :meth:`SCP.deploy` with the same bundle."""
        bundle = tmp_path / "b"
        bundle.mkdir()
        build = MagicMock()
        ex = Executor(host="h", user="me", bundle=str(bundle), build=build, triple="x86_64")
        with patch("catalyst.executor.manager.SCP.deploy") as scp_deploy:
            ex._deploy_bundle("me", "h", "ws")
        build.assert_called_once_with("x86_64", bundle)
        scp_deploy.assert_called_once_with("me", "h", bundle, "ws")

    def test_no_build_still_deploys(self, tmp_path):
        """No ``build=`` recipe: still deploys the bundle as-is."""
        bundle = tmp_path / "b"
        bundle.mkdir()
        ex = Executor(host="h", bundle=str(bundle))
        with patch("catalyst.executor.manager.SCP.deploy") as scp_deploy:
            ex._deploy_bundle("me", "h", "ws")
        scp_deploy.assert_called_once()


class TestLaunchAttachMode:
    """Attach-mode short-circuit and idempotence of :meth:`Executor.launch`."""

    def test_short_circuits_no_subprocess(self):
        """Attach mode spawns no process; ``address`` remains the constructor value."""
        # Attach mode: neither local nor host, no _proc created, address preserved.
        ex = Executor("1.2.3.4:5").launch()
        assert ex._launched is True
        assert ex._proc is None
        assert ex.address == "1.2.3.4:5"

    def test_idempotent(self):
        """Second ``launch()`` is a no-op."""
        ex = Executor("1.2.3.4:5").launch()
        ex.launch()
        assert ex._launched is True
        assert ex._proc is None


class TestStopIdempotent:
    """Safety of :meth:`Executor.stop` under repeated / no-launch calls."""

    def test_stop_before_launch_ok(self):
        """Calling ``stop()`` before ``launch()`` is a no-op, not an error."""
        ex = Executor("1.2.3.4:5")
        ex.stop()
        assert ex._launched is False

    def test_stop_after_attach_launch_ok(self):
        """``stop()`` after an attach-mode ``launch()`` clears the launched flag."""
        ex = Executor("1.2.3.4:5").launch()
        ex.stop()
        assert ex._launched is False


class TestContextManager:
    """``with Executor(...) as ex`` lifecycle."""

    def test_enter_launches_exit_stops(self):
        """``__enter__`` launches; ``__exit__`` stops."""
        with Executor("1.2.3.4:5") as ex:
            assert ex._launched is True
        assert ex._launched is False


class TestRepr:
    """The :meth:`Executor.__repr__` shape."""

    def test_shape(self):
        """``repr()`` includes name, host, and launched state."""
        r = repr(Executor("1.2.3.4:5", host="h", local=False))
        assert "name='executor'" in r
        assert "host='h'" in r
        assert "launched=False" in r
