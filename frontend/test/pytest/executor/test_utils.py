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

"""Unit tests for :mod:`catalyst.executor.utils` — regex classifiers, shell-fragment builders,
path/log resolvers, verbosity state, and small free helpers (:func:`random_port`,
:func:`triple_from_uname`, :func:`log_cmd`)."""

from unittest.mock import patch

import pytest

from catalyst.executor.utils import (
    ExecutorCli,
    Paths,
    Patterns,
    PortInUse,
    Raw,
    ShellCommand,
    log_cmd,
    random_port,
    set_verbose,
    triple_from_uname,
    verbose_level,
)


class TestPortInUse:
    """The :class:`PortInUse` exception type."""

    def test_is_exception(self):
        """Subclasses :class:`Exception`."""
        assert issubclass(PortInUse, Exception)


class TestRaw:
    """The :class:`Raw` string marker used for no-quote shell values."""

    def test_is_str_subclass(self):
        """``Raw`` is a :class:`str` subclass whose value equals the wrapped text."""
        assert issubclass(Raw, str)
        assert Raw("$HOME") == "$HOME"

    def test_preserves_content(self):
        """A ``Raw`` instance is both :class:`Raw` and :class:`str`."""
        r = Raw("$LIBDIR/x.so")
        assert isinstance(r, Raw)
        assert isinstance(r, str)


class TestPatterns:
    """Regex classifiers on :class:`Patterns` (``is_ready`` / ``is_port_conflict`` / ``is_ssh_prompt`` / ``is_sudo_fail``)."""

    @pytest.mark.parametrize(
        "line",
        [
            "Listening on 127.0.0.1:1373",
            "some prefix Listening on host:1234 more",
            "executor ready, waiting for next connection",
        ],
    )
    def test_is_ready_true(self, line):
        """Lines announcing the bound socket are classified as ready."""
        assert Patterns.is_ready(line)

    @pytest.mark.parametrize("line", ["", "just some log", "listening on nowhere"])
    def test_is_ready_false(self, line):
        """Ordinary log lines are not classified as ready."""
        assert not Patterns.is_ready(line)

    @pytest.mark.parametrize(
        "line",
        ["Address already in use", "Could not request local forwarding"],
    )
    def test_is_port_conflict_true(self, line):
        """Remote-bind and local-forward failures are classified as port conflicts."""
        assert Patterns.is_port_conflict(line)

    def test_is_port_conflict_false(self):
        """Ordinary lines are not classified as port conflicts."""
        assert not Patterns.is_port_conflict("no problem here")

    @pytest.mark.parametrize(
        "line",
        ["user@host's password:", "Enter passphrase for key '/home/x/.ssh/id_rsa':"],
    )
    def test_is_ssh_prompt_true(self, line):
        """SSH password and passphrase prompts are classified as ssh-prompt."""
        assert Patterns.is_ssh_prompt(line)

    def test_is_ssh_prompt_false(self):
        """Ordinary lines are not classified as ssh-prompt."""
        assert not Patterns.is_ssh_prompt("Welcome to Ubuntu")

    @pytest.mark.parametrize(
        "line",
        [
            "Sorry, try again.",
            "sudo: incorrect password",
            "authentication failure",
            "sudo: 3 incorrect password attempts",
        ],
    )
    def test_is_sudo_fail_true(self, line):
        """Sudo password-rejection lines are classified as sudo-fail."""
        assert Patterns.is_sudo_fail(line)

    def test_is_sudo_fail_false(self):
        """Ordinary lines are not classified as sudo-fail."""
        assert not Patterns.is_sudo_fail("all good")


class TestShellCommand:
    """Shell-fragment builders on :class:`ShellCommand`."""

    def test_sudo_probe(self):
        """``sudo_probe()`` is the exact non-interactive check string."""
        assert ShellCommand.sudo_probe() == "sudo -n true 2>/dev/null"

    def test_sudo_pw_wraps_cmd(self):
        """``sudo_pw`` prefixes with ``sudo -S -p ''``."""
        assert ShellCommand.sudo_pw("pkill -f x") == "sudo -S -p '' pkill -f x"

    def test_sudo_np_wraps_cmd(self):
        """``sudo_np`` prefixes with ``sudo -n`` (NOPASSWD only)."""
        assert ShellCommand.sudo_np("pkill -f x") == "sudo -n pkill -f x"

    def test_pkill_shell_quotes_pattern(self):
        """``pkill`` shell-quotes the pattern so metacharacters can't escape."""
        # Metacharacters in the pattern must not break out.
        cmd = ShellCommand.pkill("evil; rm -rf /")
        assert cmd.startswith("pkill -f ")
        assert "'evil; rm -rf /'" in cmd

    def test_rm_rf_quotes_path(self):
        """``rm_rf`` shell-quotes paths with spaces."""
        cmd = ShellCommand.rm_rf("/tmp/space dir")
        assert cmd.startswith("rm -rf ")
        assert "'/tmp/space dir'" in cmd

    def test_mkdir_p_quotes_path(self):
        """``mkdir_p`` shell-quotes paths with spaces."""
        cmd = ShellCommand.mkdir_p("/tmp/x y")
        assert cmd.startswith("mkdir -p ")
        assert "'/tmp/x y'" in cmd

    def test_path_bare_tilde(self):
        """A bare ``~`` maps to ``"$HOME"``."""
        assert ShellCommand.path("~") == '"$HOME"'

    def test_path_tilde_slash(self):
        """``~/foo`` expands to ``"$HOME"/foo`` (no quotes needed on the tail)."""
        # ~/foo → "$HOME"/'foo'
        assert ShellCommand.path("~/foo") == '"$HOME"/foo'

    def test_path_tilde_slash_with_space(self):
        """The tail after ``~/`` is quoted if it needs it."""
        assert ShellCommand.path("~/my dir") == '"$HOME"/\'my dir\''

    def test_path_absolute_quoted(self):
        """A plain absolute path without whitespace passes through unquoted."""
        assert ShellCommand.path("/tmp/x") == "/tmp/x"

    def test_path_absolute_with_space_quoted(self):
        """An absolute path with whitespace is shell-quoted."""
        assert ShellCommand.path("/tmp/x y") == "'/tmp/x y'"


class TestExecutorCli:
    """CLI flag constants for the ``catalyst-executor`` binary."""

    def test_flag_constants(self):
        """``PLUGIN_FLAG`` and ``BIND_FLAG`` are the exact expected strings."""
        assert ExecutorCli.PLUGIN_FLAG == "--plugin="
        assert ExecutorCli.BIND_FLAG == "--bind="


class TestVerbosity:
    """The :func:`set_verbose` / :func:`verbose_level` global-state pair."""

    def teardown_method(self):
        """Reset verbosity so tests are order-independent."""
        set_verbose(1)

    def test_default(self):
        """Level 1 (INFO) is the default verbosity."""
        set_verbose(1)
        assert verbose_level() == 1

    def test_zero(self):
        """Level 0 is honored."""
        set_verbose(0)
        assert verbose_level() == 0

    def test_three(self):
        """Level 3+ (trace) is honored."""
        set_verbose(3)
        assert verbose_level() == 3


class TestPaths:
    """Path resolution helpers on :class:`Paths`."""

    def test_workspace_prefix(self):
        """The workspace safety prefix is a stable constant."""
        assert Paths.WORKSPACE_PREFIX == "catalyst-exec-"

    def test_executor_bin(self):
        """The executor-binary name is a stable constant."""
        assert Paths.EXECUTOR_BIN == "catalyst-executor"

    def test_default_workspace_shape(self):
        """A default workspace is ``~/``-rooted and contains the safety prefix."""
        ws = Paths.default_workspace()
        assert ws.startswith("~/")
        assert Paths.WORKSPACE_PREFIX in ws

    def test_random_suffix_hex_len(self):
        """The random suffix is a 3-char hex tag."""
        s = Paths._random_suffix()
        assert len(s) == 3
        int(s, 16)  # must parse as hex

    def test_resolve_log_disabled(self):
        """``disabled=True`` disables log-file resolution entirely."""
        assert Paths.resolve_log("h", disabled=True) is None

    def test_resolve_log_explicit(self):
        """An explicit path pins the log file."""
        assert Paths.resolve_log("h", explicit="/tmp/my.log") == "/tmp/my.log"

    def test_resolve_log_default_shape(self):
        """A default log filename includes the binary name, host, and a ``.log`` suffix."""
        log = Paths.resolve_log("myhost")
        assert log.startswith("catalyst-executor-")
        assert "myhost" in log
        assert log.endswith(".log")

    def test_resolve_log_named_tag(self):
        """A non-default ``name`` is embedded in the log filename."""
        log = Paths.resolve_log("myhost", name="worker")
        assert "-worker-" in log

    def test_timestamp_format(self):
        """``_timestamp()`` follows the ``YYYY-MM-DD_HH-MM-SS`` layout."""
        ts = Paths._timestamp()
        # Format: YYYY-MM-DD_HH-MM-SS
        assert len(ts) == 19
        assert ts[4] == ts[7] == "-"
        assert ts[10] == "_"
        assert ts[13] == ts[16] == "-"

    def test_default_executor_bin_prefers_first_candidate(self, tmp_path):
        """Primary candidate wins: ``<lib>/catalyst-executor`` if it exists."""
        rt = tmp_path / "rt"
        rt.mkdir()
        (rt / Paths.EXECUTOR_BIN).write_text("bin")
        with patch("catalyst.executor.utils.get_lib_path", return_value=str(rt)):
            assert Paths.default_executor_bin() == str(rt / Paths.EXECUTOR_BIN)

    def test_default_executor_bin_falls_through_to_remote_subdir(self, tmp_path):
        """Primary missing: falls through to ``<lib>/remote/catalyst-executor``."""
        rt = tmp_path / "rt"
        (rt / "remote").mkdir(parents=True)
        (rt / "remote" / Paths.EXECUTOR_BIN).write_text("bin")
        with patch("catalyst.executor.utils.get_lib_path", return_value=str(rt)):
            assert Paths.default_executor_bin() == str(rt / "remote" / Paths.EXECUTOR_BIN)

    def test_default_executor_bin_falls_back_to_name_on_path(self, tmp_path):
        """Nothing found on disk: returns the bare binary name (assume it's on ``$PATH``)."""
        # Neither candidate exists → fall back to the bare binary name (assume it's on $PATH).
        with patch(
            "catalyst.executor.utils.get_lib_path", return_value=str(tmp_path / "nope")
        ):
            assert Paths.default_executor_bin() == Paths.EXECUTOR_BIN


class TestRandomPort:
    """The :func:`random_port` ephemeral-port generator."""

    def test_in_expected_range(self):
        """Every draw lands in ``[20000, 59999]``."""
        for _ in range(50):
            p = random_port()
            assert 20000 <= p <= 59999


class TestTripleFromUname:
    """The :func:`triple_from_uname` LLVM-triple mapper."""

    @pytest.mark.parametrize(
        "system,machine,expected",
        [
            ("Linux", "x86_64", "x86_64-unknown-linux-gnu"),
            ("Linux", "amd64", "x86_64-unknown-linux-gnu"),
            ("Linux", "aarch64", "aarch64-unknown-linux-gnu"),
            ("Linux", "arm64", "aarch64-unknown-linux-gnu"),
            ("Darwin", "x86_64", "x86_64-apple-darwin"),
            ("Darwin", "arm64", "arm64-apple-darwin"),
            ("Darwin", "aarch64", "arm64-apple-darwin"),
        ],
    )
    def test_common(self, system, machine, expected):
        """Common Linux/Darwin ``(system, machine)`` combos map to the right triple."""
        assert triple_from_uname(system, machine) == expected

    def test_case_insensitive_and_stripped(self):
        """Input is lowercased and stripped before matching."""
        assert triple_from_uname("  LINUX ", " X86_64 ") == "x86_64-unknown-linux-gnu"

    def test_unknown_arch_returns_none(self):
        """An unknown architecture yields ``None``."""
        assert triple_from_uname("Linux", "sparc") is None

    def test_unknown_system_returns_none(self):
        """An unknown operating system yields ``None``."""
        assert triple_from_uname("Plan9", "x86_64") is None


class TestLogCmd:
    """The :func:`log_cmd` DEBUG-level command echo."""

    def test_debug_emit(self):
        """Calls ``logger.debug`` with a shell-quoted rendering of the argv."""
        with patch("catalyst.executor.utils.logger.debug") as debug:
            log_cmd(["ssh", "me@h", "cmd with space"])
        debug.assert_called_once()
        # Verify the argv was shell-quoted.
        joined = debug.call_args.args[1]
        assert "me@h" in joined
        assert "'cmd with space'" in joined
