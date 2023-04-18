def pytest_addoption(parser):
    parser.addoption(
        "--backend",
        action="store",
        default="lightning.qubit",
        help="Name of the backend device",
    )


def pytest_generate_tests(metafunc):
    if "backend" in metafunc.fixturenames:
        metafunc.parametrize("backend", [metafunc.config.getoption("--backend")])
