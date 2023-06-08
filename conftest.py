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
        default="OFF",
        help="Run AWS Braket tests; runtime must be compiled with `ENABLE_OPENQASM=ON`",
    )


def pytest_generate_tests(metafunc):
    """A pytest fixture to define custom parametrization"""

    if "backend" in metafunc.fixturenames:
        metafunc.parametrize("backend", [metafunc.config.getoption("--backend")])


def pytest_configure(config):
    """A pytest configure helper method"""

    config.addinivalue_line(
        "markers", "braket: run on aws-braket devices using `OpenQasmDevice` in the runtime"
    )


def pytest_collection_modifyitems(config, items):
    """A pytest items modifier method"""

    if config.getoption("--runbraket") == "ON":
        # only runs test with the braket marker
        braket_tests = []
        for item in items:
            if item.get_closest_marker("braket"):
                braket_tests.append(item)
        items[:] = braket_tests
    else:
        # skip braket tests
        skipper = pytest.mark.skip()
        for item in items:
            if "braket" in item.keywords:
                item.add_marker(skipper)
