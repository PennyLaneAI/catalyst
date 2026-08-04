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

"""Unit tests for setup.py."""

import builtins
import io
import runpy
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import setuptools
from setuptools._distutils import sysconfig, util


def test_macos_setup_uses_local_arch_for_custom_calls_and_wheel_tag(monkeypatch):
    """Test macOS setup does not inherit universal2 arch flags from Python."""
    repo_root = Path(__file__).resolve().parents[3]
    setup_mock = Mock()
    variables = {
        "CFLAGS": "-O3 -arch arm64 -arch x86_64",
        "LDFLAGS": "-arch arm64 -arch x86_64",
        "LDSHARED": "clang -bundle -arch arm64 -arch x86_64",
        "LDCXXSHARED": "clang++ -bundle -arch arm64 -arch x86_64",
    }
    real_open = builtins.open

    def get_config_var(name):
        if name == "MACOSX_DEPLOYMENT_TARGET":
            return "11.0"
        return variables.get(name)

    def open_without_revision_write(file, *args, **kwargs):
        mode = args[0] if args else kwargs.get("mode", "r")
        if Path(file).as_posix() == "frontend/catalyst/_revision.py" and "w" in mode:
            return io.StringIO()
        return real_open(file, *args, **kwargs)

    monkeypatch.chdir(repo_root)
    monkeypatch.setattr(builtins, "open", open_without_revision_write)
    monkeypatch.setattr(setuptools, "setup", setup_mock)
    monkeypatch.setattr(subprocess, "check_output", lambda *_, **__: b"revision")
    monkeypatch.setattr("platform.system", lambda: "Darwin")
    monkeypatch.setattr("platform.machine", lambda: "arm64")
    monkeypatch.setattr(sysconfig, "get_config_vars", lambda: variables)
    monkeypatch.setattr(sysconfig, "get_config_var", get_config_var)
    monkeypatch.setattr(util, "get_platform", lambda: "macosx-11.0-universal2")
    monkeypatch.setattr(sys, "argv", ["setup.py"])

    runpy.run_path(str(repo_root / "setup.py"), run_name="__main__")

    setup_kwargs = setup_mock.call_args.kwargs
    custom_calls_extension = next(
        ext
        for ext in setup_kwargs["ext_modules"]
        if ext.name == "catalyst.utils.libcustom_calls"
    )

    assert custom_calls_extension.extra_compile_args == ["-std=c++20", "-arch", "arm64"]
    assert custom_calls_extension.extra_link_args == ["-arch", "arm64"]
    assert setup_kwargs["options"]["bdist_wheel"]["plat_name"] == "macosx_11_0_arm64"
    assert variables["CFLAGS"] == "-O3 -arch arm64"
    assert variables["LDFLAGS"] == "-arch arm64"
    assert variables["LDSHARED"] == "clang -dynamiclib -arch arm64"
    assert variables["LDCXXSHARED"] == "clang++ -dynamiclib -arch arm64"
