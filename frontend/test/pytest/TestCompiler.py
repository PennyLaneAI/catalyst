"""
Unit tests for AbstractLinker class
"""

import os
import warnings
import pytest

from catalyst.compiler import AbstractLinker
from catalyst.utils.temp_env import TempEnv


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


class TestAbstractLinker:
    """Unit test for AbstractLinker class."""

    def test_catalyst_cc_available(self):
        """Test that the linker resolution order contains the preferred linker and no warnings
        are emitted"""
        linker = "c99"
        with TempEnv(CATALYST_CC=linker):
            # If a warning is emitted, raise an error.
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                # pylint: disable=protected-access
                linkers = AbstractLinker._lro([])
                assert linker in linkers

    def test_catalyst_cc_unavailable_warning(self):
        """Test that a warning is emitted when the preferred linker is not in PATH."""
        with TempEnv(CATALYST_CC="this-binary-does-not-exist"):
            with pytest.warns(UserWarning, match="User defined linker.* is not in PATH."):
                # pylint: disable=protected-access
                AbstractLinker._lro([])

    def test_linker_failed_warning(self):
        """Test that a warning is emitted when a linker failed."""
        linker = "cc"
        with pytest.warns(UserWarning, match="Linker .* failed .*"):
            # pylint: disable=protected-access
            AbstractLinker._attempt_link(linker, [""], "in.o", "out.so")

    def test_link_fail_exception(self):
        """Test that an exception is raised when all linker possibilities are exhausted."""
        with pytest.raises(EnvironmentError, match="Unable to link .*"):
            AbstractLinker.link("in.o", "out.so", fallback_linkers=["this-binary-does-not-exist"])
