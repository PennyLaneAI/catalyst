import pennylane as qml
from standalone_plugin import SwitchBarToFoo


@SwitchBarToFoo()
@qml.qnode(qml.device("lightning.qubit", wires=0))
def qnode():
    return qml.state()


@qml.qjit(target="mlir", keep_intermediate=True)
def module():
    return qnode()


assert "standalone-switch-bar-foo" in module.mlir
