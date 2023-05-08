def pytest_addoption(parser):
    """Add pytest custom options."""

    parser.addoption(
        "--backend",
        action="store",
        default="lightning.qubit",
        help="Name of the backend device",
    )


def pytest_generate_tests(metafunc):
    """A pytest fixture to define custom parametrization"""

    if "backend" in metafunc.fixturenames:
        metafunc.parametrize("backend", [metafunc.config.getoption("--backend")])
