from catalyst import qjit_ast
import pennylane as qml
import pennylane.numpy as pnp

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
  func.func @jit_simple() -> f64 {
    %0 = call @simple() : () -> f64
    return %0 : f64
  }
  func.func private @simple() -> f64 {
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
  func.func @setup() {
    quantum.init
    return
  }
  func.func @teardown() {
    quantum.finalize
    return
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

    expected_snapshot = """
module {
  func.func @jit_scalar_param(%arg0: f64) -> f64 {
    %0 = call @scalar_param(%arg0) : (f64) -> f64
    return %0 : f64
  }
  func.func private @scalar_param(%arg0: f64) -> f64 {
    %0 = quantum.alloc( 2) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %2 = quantum.custom "RX"(%arg0) %1 : !quantum.bit
    %3 = quantum.insert %0[ 0], %2 : !quantum.reg, !quantum.bit
    %4 = quantum.extract %3[ 0] : !quantum.reg -> !quantum.bit
    %5 = quantum.namedobs %4[ PauliZ] : !quantum.obs
    %6 = quantum.expval %5 : f64
    return %6 : f64
  }
  func.func @setup() {
    quantum.init
    return
  }
  func.func @teardown() {
    quantum.finalize
    return
  }
}
    """

    scalar_param(0.3)
    assert scalar_param.mlir.strip() == expected_snapshot.strip()


def test_if_else():
    @qjit_ast
    @qml.qnode(dev)
    def ifelse(x: float, n: int):
        if n % 2 == 0:
            qml.RX(x, wires=0)
        elif x > 4:
            qml.RZ(x - 2.3, 1)
        else:
            qml.RY(x * 2, 0)
        return qml.expval(qml.PauliZ(0))

    ifelse(5.4, 4)

    expected_snapshot = """
module {
  func.func @jit_ifelse(%arg0: f64, %arg1: i64) -> f64 {
    %0 = call @ifelse(%arg0, %arg1) : (f64, i64) -> f64
    return %0 : f64
  }
  func.func private @ifelse(%arg0: f64, %arg1: i64) -> f64 {
    %cst = arith.constant 2.000000e+00 : f64
    %cst_0 = arith.constant 4.000000e+00 : f64
    %cst_1 = arith.constant 2.300000e+00 : f64
    %c0_i64 = arith.constant 0 : i64
    %c2_i64 = arith.constant 2 : i64
    %0 = quantum.alloc( 2) : !quantum.reg
    %1 = arith.remsi %arg1, %c2_i64 : i64
    %2 = arith.cmpi eq, %1, %c0_i64 : i64
    %3 = scf.if %2 -> (!quantum.reg) {
      %7 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
      %8 = quantum.custom "RX"(%arg0) %7 : !quantum.bit
      %9 = quantum.insert %0[ 0], %8 : !quantum.reg, !quantum.bit
      scf.yield %9 : !quantum.reg
    } else {
      %7 = arith.cmpf ogt, %arg0, %cst_0 : f64
      %8 = scf.if %7 -> (!quantum.reg) {
        %9 = arith.subf %arg0, %cst_1 : f64
        %10 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit
        %11 = quantum.custom "RZ"(%9) %10 : !quantum.bit
        %12 = quantum.insert %0[ 1], %11 : !quantum.reg, !quantum.bit
        scf.yield %12 : !quantum.reg
      } else {
        %9 = arith.mulf %arg0, %cst : f64
        %10 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
        %11 = quantum.custom "RY"(%9) %10 : !quantum.bit
        %12 = quantum.insert %0[ 0], %11 : !quantum.reg, !quantum.bit
        scf.yield %12 : !quantum.reg
      }
      scf.yield %8 : !quantum.reg
    }
    %4 = quantum.extract %3[ 0] : !quantum.reg -> !quantum.bit
    %5 = quantum.namedobs %4[ PauliZ] : !quantum.obs
    %6 = quantum.expval %5 : f64
    return %6 : f64
  }
  func.func @setup() {
    quantum.init
    return
  }
  func.func @teardown() {
    quantum.finalize
    return
  }
}
    """
    assert ifelse.mlir.strip() == expected_snapshot.strip()


def test_range_for_loop():
    @qjit_ast
    @qml.qnode(dev)
    def range_for(x: float, n: int):
        for i in range(n):
            qml.RX(x * i, wires=i)
        for i in range(-4, n):
            qml.Hadamard(i)
        for i in range(-4, n, 2):
            qml.PauliX(i + 1)
        return qml.expval(qml.PauliZ(0))

    range_for(5.4, 2)
    expected_snapshot = """
module {
  func.func @jit_range_for(%arg0: f64, %arg1: i64) -> f64 {
    %0 = call @range_for(%arg0, %arg1) : (f64, i64) -> f64
    return %0 : f64
  }
  func.func private @range_for(%arg0: f64, %arg1: i64) -> f64 {
    %c2 = arith.constant 2 : index
    %c-4 = arith.constant -4 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c1_i64 = arith.constant 1 : i64
    %0 = quantum.alloc( 2) : !quantum.reg
    %1 = arith.index_cast %arg1 : i64 to index
    %2 = scf.for %arg2 = %c0 to %1 step %c1 iter_args(%arg3 = %0) -> (!quantum.reg) {
      %10 = arith.index_cast %arg2 : index to i64
      %11 = arith.sitofp %10 : i64 to f64
      %12 = arith.mulf %arg0, %11 : f64
      %13 = arith.index_cast %arg2 : index to i64
      %14 = quantum.extract %arg3[%13] : !quantum.reg -> !quantum.bit
      %15 = quantum.custom "RX"(%12) %14 : !quantum.bit
      %16 = quantum.insert %arg3[%13], %15 : !quantum.reg, !quantum.bit
      scf.yield %16 : !quantum.reg
    }
    %3 = arith.index_cast %arg1 : i64 to index
    %4 = scf.for %arg2 = %c-4 to %3 step %c1 iter_args(%arg3 = %2) -> (!quantum.reg) {
      %10 = arith.index_cast %arg2 : index to i64
      %11 = quantum.extract %arg3[%10] : !quantum.reg -> !quantum.bit
      %12 = quantum.custom "Hadamard"() %11 : !quantum.bit
      %13 = quantum.insert %arg3[%10], %12 : !quantum.reg, !quantum.bit
      scf.yield %13 : !quantum.reg
    }
    %5 = arith.index_cast %arg1 : i64 to index
    %6 = scf.for %arg2 = %c-4 to %5 step %c2 iter_args(%arg3 = %4) -> (!quantum.reg) {
      %10 = arith.index_cast %arg2 : index to i64
      %11 = arith.addi %10, %c1_i64 : i64
      %12 = quantum.extract %arg3[%11] : !quantum.reg -> !quantum.bit
      %13 = quantum.custom "PauliX"() %12 : !quantum.bit
      %14 = quantum.insert %arg3[%11], %13 : !quantum.reg, !quantum.bit
      scf.yield %14 : !quantum.reg
    }
    %7 = quantum.extract %6[ 0] : !quantum.reg -> !quantum.bit
    %8 = quantum.namedobs %7[ PauliZ] : !quantum.obs
    %9 = quantum.expval %8 : f64
    return %9 : f64
  }
  func.func @setup() {
    quantum.init
    return
  }
  func.func @teardown() {
    quantum.finalize
    return
  }
}
    """
    assert range_for.mlir.strip() == expected_snapshot.strip()


def test_assign():
    @qjit_ast
    @qml.qnode(dev)
    def assign(n):
        x = n + 5
        y = x * 2
        return y

    assign(3)

    expected_snapshot = """
module {
  func.func @jit_assign(%arg0: i64) -> i64 {
    %0 = call @assign(%arg0) : (i64) -> i64
    return %0 : i64
  }
  func.func private @assign(%arg0: i64) -> i64 {
    %c2_i64 = arith.constant 2 : i64
    %c5_i64 = arith.constant 5 : i64
    %0 = arith.addi %arg0, %c5_i64 : i64
    %1 = arith.muli %0, %c2_i64 : i64
    return %1 : i64
  }
  func.func @setup() {
    quantum.init
    return
  }
  func.func @teardown() {
    quantum.finalize
    return
  }
}
    """
    assert assign.mlir.strip() == expected_snapshot.strip()


def test_list_comprehension():
    @qjit_ast
    @qml.qnode(dev)
    def list_comp(n):
        exp_vals = [qml.expval(qml.PauliZ(position)) for position in range(n)]
        return exp_vals

    list_comp(2)
    expected_snapshot = """
module {
  func.func @jit_list_comp(%arg0: i64) -> tensor<?xf64> {
    %0 = call @list_comp(%arg0) : (i64) -> tensor<?xf64>
    return %0 : tensor<?xf64>
  }
  func.func private @list_comp(%arg0: i64) -> tensor<?xf64> {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = quantum.alloc( 2) : !quantum.reg
    %1 = arith.index_cast %arg0 : i64 to index
    %2 = tensor.empty(%1) : tensor<?xf64>
    %3 = scf.for %arg1 = %c0 to %1 step %c1 iter_args(%arg2 = %2) -> (tensor<?xf64>) {
      %4 = arith.index_cast %arg1 : index to i64
      %5 = quantum.extract %0[%4] : !quantum.reg -> !quantum.bit
      %6 = quantum.namedobs %5[ PauliZ] : !quantum.obs
      %7 = quantum.expval %6 : f64
      %inserted = tensor.insert %7 into %arg2[%arg1] : tensor<?xf64>
      scf.yield %inserted : tensor<?xf64>
    }
    return %3 : tensor<?xf64>
  }
  func.func @setup() {
    quantum.init
    return
  }
  func.func @teardown() {
    quantum.finalize
    return
  }
}
    """
    assert list_comp.mlir.strip() == expected_snapshot.strip()


def test_tensor_slice():
    @qjit_ast
    @qml.qnode(dev)
    def tensor_slice(x, n):
        return x[n]

    tensor_slice(pnp.array([[4, 5], [6, 7]]), 1)
    expected_snapshot = """
module {
  func.func @jit_tensor_slice(%arg0: tensor<2x2xi64>, %arg1: i64) -> tensor<2xi64> {
    %0 = call @tensor_slice(%arg0, %arg1) : (tensor<2x2xi64>, i64) -> tensor<2xi64>
    return %0 : tensor<2xi64>
  }
  func.func private @tensor_slice(%arg0: tensor<2x2xi64>, %arg1: i64) -> tensor<2xi64> {
    %0 = arith.index_cast %arg1 : i64 to index
    %extracted_slice = tensor.extract_slice %arg0[%0, 0] [1, 2] [1, 1] : tensor<2x2xi64> to tensor<2xi64>
    return %extracted_slice : tensor<2xi64>
  }
  func.func @setup() {
    quantum.init
    return
  }
  func.func @teardown() {
    quantum.finalize
    return
  }
}
    """
    assert tensor_slice.mlir.strip() == expected_snapshot.strip()


def test_call_twice():
    def subfunc(param):
        qml.RX(param, wires=0)

    @qjit_ast
    @qml.qnode(dev)
    def call_twice(x: float, n: int):
        if n < 0:
            subfunc(x)
        else:
            subfunc(x * 3)
        return qml.expval(qml.PauliZ(0))

    call_twice(4.4, 2)

    expected_snapshot = """
module {
  func.func @jit_call_twice(%arg0: f64, %arg1: i64) -> f64 {
    %0 = call @call_twice(%arg0, %arg1) : (f64, i64) -> f64
    return %0 : f64
  }
  func.func private @call_twice(%arg0: f64, %arg1: i64) -> f64 {
    %cst = arith.constant 3.000000e+00 : f64
    %c0_i64 = arith.constant 0 : i64
    %0 = quantum.alloc( 2) : !quantum.reg
    %1 = arith.cmpi slt, %arg1, %c0_i64 : i64
    %2 = scf.if %1 -> (!quantum.reg) {
      %6 = func.call @subfunc(%arg0, %0) : (f64, !quantum.reg) -> !quantum.reg
      scf.yield %6 : !quantum.reg
    } else {
      %6 = arith.mulf %arg0, %cst : f64
      %7 = func.call @subfunc(%6, %0) : (f64, !quantum.reg) -> !quantum.reg
      scf.yield %7 : !quantum.reg
    }
    %3 = quantum.extract %2[ 0] : !quantum.reg -> !quantum.bit
    %4 = quantum.namedobs %3[ PauliZ] : !quantum.obs
    %5 = quantum.expval %4 : f64
    return %5 : f64
  }
  func.func private @subfunc(%arg0: f64, %arg1: !quantum.reg) -> !quantum.reg {
    %0 = quantum.extract %arg1[ 0] : !quantum.reg -> !quantum.bit
    %1 = quantum.custom "RX"(%arg0) %0 : !quantum.bit
    %2 = quantum.insert %arg1[ 0], %1 : !quantum.reg, !quantum.bit
    return %2 : !quantum.reg
  }
  func.func @setup() {
    quantum.init
    return
  }
  func.func @teardown() {
    quantum.finalize
    return
  }
}
    """
    assert call_twice.mlir.strip() == expected_snapshot.strip()
