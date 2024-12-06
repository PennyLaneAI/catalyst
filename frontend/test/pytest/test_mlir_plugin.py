from pathlib import Path

import jax
import pennylane as qml

from catalyst.utils.runtime_environment import get_bin_path
from catalyst.passes import apply_pass

plugin_path = get_bin_path("cli", "CATALYST_BIN_DIR") + "/../StandalonePlugin.so"
plugin = Path(plugin_path)

def test_standalone_plugin():
    """Generate MLIR for the standalone plugin. Do not execute code.
    The code execution test is in the lit test. See that test
    for more information as to why that is the case."""

    @apply_pass("standalone-switch-bar-foo")
    @qml.qnode(qml.device("lightning.qubit", wires=0))
    def qnode():
        return qml.state()


    @qml.qjit(pass_plugins=[plugin], dialect_plugins=[plugin], target="mlir")
    def module():
        return qnode()

    # It would be nice if we were able to convine lit tests with
    # pytest
    assert "standalone-switch-bar-foo" in module.mlir
