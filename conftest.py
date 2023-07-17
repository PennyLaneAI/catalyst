"""Setup pytest configurations."""
import pytest


def pytest_addoption(parser):
    """Add pytest custom options."""

    parser.addoption(
        "--backend",
        action="store",
        default="lightning.qubit",
        help="Name of the backend device",
    )

    parser.addoption(
        "--runbraket",
        action="store",
        default="NONE",
        help="Run AWS Braket ['ALL', 'LOCAL', 'REMOTE'] tests;"
        " runtime must be compiled with `ENABLE_OPENQASM=ON`",
    )


def pytest_generate_tests(metafunc):
    """A pytest fixture to define custom parametrization"""

    if "backend" in metafunc.fixturenames:
        metafunc.parametrize("backend", [metafunc.config.getoption("--backend")])


def pytest_configure(config):
    """A pytest configure helper method"""

    config.addinivalue_line(
        "markers",
        "braketlocal: run on local aws-braket devices backed by `OpenQasmDevice` in the runtime",
    )

    config.addinivalue_line(
        "markers",
        "braketremote: run on remote aws-braket devices backed by `OpenQasmDevice` in the runtime",
    )


def pytest_collection_modifyitems(config, items):
    """A pytest items modifier method"""

    braket_val = config.getoption("--runbraket")
    if braket_val in ["ALL", "LOCAL", "REMOTE"]:
        # only runs test with the braket marker
        braket_tests = []
        for item in items:
            if (braket_val in ["ALL", "LOCAL"] and item.get_closest_marker("braketlocal")) or (
                braket_val in ["ALL", "REMOTE"] and item.get_closest_marker("braketremote")
            ):
                braket_tests.append(item)
        items[:] = braket_tests
    else:
        # skip braket tests
        skipper = pytest.mark.skip()
        for item in items:
            if "braketlocal" in item.keywords or "braketremote" in item.keywords:
                item.add_marker(skipper)
