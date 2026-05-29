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