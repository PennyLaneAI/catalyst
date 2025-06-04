// To run the merge unitary gates pass on this mlir file, do:
// quantum-opt --merge-unitary-gates MergeUnitaryGatesTest.mlir

func.func @circuit(%arg0: complex<f64>) -> i1 {
    %c00 = complex.constant [0.0, 0.0] : complex<f64>
    %c10 = complex.constant [1.0, 0.0] : complex<f64>
    %c20 = complex.constant [2.0, 0.0] : complex<f64>

    %0 = complex.exp %arg0 : complex<f64>
    %A = tensor.from_elements %c10, %c00, %c00, %0 : tensor<2x2xcomplex<f64>>

    %1 = complex.mul %arg0, %c20 : complex<f64>
    %2 = complex.exp %1 : complex<f64>
    %B = tensor.from_elements %c10, %c00, %c00, %2 : tensor<2x2xcomplex<f64>>

    %reg = quantum.alloc(1) : !quantum.reg
    %q0 = quantum.extract %reg[ 0] : !quantum.reg -> !quantum.bit
    %q1 = quantum.custom "Hadamard"() %q0 : !quantum.bit
    %q2 = quantum.unitary(%A : tensor<2x2xcomplex<f64>>) %q1 : !quantum.bit
    %q3 = quantum.unitary(%B : tensor<2x2xcomplex<f64>>) %q2 : !quantum.bit

    %m, %q4 = quantum.measure %q3 : i1, !quantum.bit

    return %m : i1
}

// The output of the pass is
//  func.func @circuit(%arg0: complex<f64>) -> i1 {
//    %cst = complex.constant [0.000000e+00, 0.000000e+00] : complex<f64>
//    %cst_0 = complex.constant [1.000000e+00, 0.000000e+00] : complex<f64>
//    %cst_1 = complex.constant [2.000000e+00, 0.000000e+00] : complex<f64>
//    %0 = complex.exp %arg0 : complex<f64>
//    %from_elements = tensor.from_elements %cst_0, %cst, %cst, %0 : tensor<2x2xcomplex<f64>>
//    %1 = complex.mul %arg0, %cst_1 : complex<f64>
//    %2 = complex.exp %1 : complex<f64>
//    %from_elements_2 = tensor.from_elements %cst_0, %cst, %cst, %2 : tensor<2x2xcomplex<f64>>
//    %3 = quantum.alloc( 1) : !quantum.reg
//    %4 = quantum.extract %3[ 0] : !quantum.reg -> !quantum.bit
//    %out_qubits = quantum.custom "Hadamard"() %4 : !quantum.bit
//    %5 = tensor.empty() : tensor<2x2xcomplex<f64>>
//    %6 = linalg.matmul ins(%from_elements_2, %from_elements : tensor<2x2xcomplex<f64>>, tensor<2x2xcomplex<f64>>) outs(%5 : tensor<2x2xcomplex<f64>>) -> tensor<2x2xcomplex<f64>>
//    %out_qubits_3 = quantum.unitary(%6 : tensor<2x2xcomplex<f64>>) %out_qubits : !quantum.bit
//    %mres, %out_qubit = quantum.measure %out_qubits_3 : i1, !quantum.bit
//    return %mres : i1
//  }
//
// which has the two unitaries merged into one!
