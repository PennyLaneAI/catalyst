from catalyst import qjit_ast
import pennylane as qml

n_qubits = 2

dev = qml.device("lightning.qubit", wires=n_qubits)


def test_basic():
    @qjit_ast
    @qml.qnode(dev)
    def simple():
        qml.CNOT((0, 1))
        qml.PauliX(0)
        qml.Rot(3.43, 44, 44.3, wires=(0,))
        return qml.expval(qml.PauliZ(0))

    simple()

    expected_snapshot = """
module {
  func.func @simple() -> f64 {
    %cst = arith.constant 4.430000e+01 : f64
    %cst_0 = arith.constant 4.400000e+01 : f64
    %cst_1 = arith.constant 3.430000e+00 : f64
    %0 = quantum.alloc( 2) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %2 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit
    %3:2 = quantum.custom "CNOT"() %1, %2 : !quantum.bit, !quantum.bit
    %4 = quantum.insert %0[ 0], %3#0 : !quantum.reg, !quantum.bit
    %5 = quantum.insert %4[ 1], %3#1 : !quantum.reg, !quantum.bit
    %6 = quantum.extract %5[ 0] : !quantum.reg -> !quantum.bit
    %7 = quantum.custom "PauliX"() %6 : !quantum.bit
    %8 = quantum.insert %5[ 0], %7 : !quantum.reg, !quantum.bit
    %9 = quantum.extract %8[ 0] : !quantum.reg -> !quantum.bit
    %10 = quantum.custom "Rot"(%cst_1, %cst_0, %cst) %9 : !quantum.bit
    %11 = quantum.insert %8[ 0], %10 : !quantum.reg, !quantum.bit
    %12 = quantum.extract %11[ 0] : !quantum.reg -> !quantum.bit
    %13 = quantum.namedobs %12[ PauliZ] : !quantum.obs
    %14 = quantum.expval %13 : f64
    return %14 : f64
  }
}
"""

    assert simple.mlir.strip() == expected_snapshot.strip()


def test_scalar_param():
    @qjit_ast
    @qml.qnode(dev)
    def scalar_param(x: float):
        qml.RX(x, wires=0)
        return qml.expval(qml.PauliZ(0))

    scalar_param(0.3)
    print(scalar_param.mlir)
