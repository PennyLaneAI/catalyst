"""
Unit tests for CompilerDriver class
"""

import os
import warnings
import pytest

from catalyst.compiler import CompilerDriver
from temp_env import TempEnv


# pylint: disable=too-few-public-methods
class TestTempEnv:
    """Test the temporary environment class"""

    # pylint: disable=missing-function-docstring
    def test_temp_env(self):
        old_values = os.environ
        with TempEnv(somevar="hello"):
            assert os.environ["somevar"]
            for key, value in old_values.items():
                assert os.environ[key] == value
        assert old_values == os.environ


class TestCompilerDriver:
    """Unit test for CompilerDriver class."""

    def test_catalyst_cc_available(self):
        """Test that the compiler resolution order contains the preferred compiler and no warnings
        are emitted"""
        compiler = "c99"
        with TempEnv(CATALYST_CC=compiler):
            # If a warning is emitted, raise an error.
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                # pylint: disable=protected-access
                compilers = CompilerDriver._get_compiler_fallback_order([])
                assert compiler in compilers

    def test_catalyst_cc_unavailable_warning(self):
        """Test that a warning is emitted when the preferred compiler is not in PATH."""
        with TempEnv(CATALYST_CC="this-binary-does-not-exist"):
            with pytest.warns(UserWarning, match="User defined compiler.* is not in PATH."):
                # pylint: disable=protected-access
                CompilerDriver._get_compiler_fallback_order([])

    def test_compiler_failed_warning(self):
        """Test that a warning is emitted when a compiler failed."""
        compiler = "cc"
        with pytest.warns(UserWarning, match="Compiler .* failed .*"):
            # pylint: disable=protected-access
            CompilerDriver._attempt_link(compiler, [""], "in.o", "out.so")

    def test_link_fail_exception(self):
        """Test that an exception is raised when all compiler possibilities are exhausted."""
        with pytest.raises(EnvironmentError, match="Unable to link .*"):
            CompilerDriver.link("in.o", "out.so", fallback_compilers=["this-binary-does-not-exist"])
