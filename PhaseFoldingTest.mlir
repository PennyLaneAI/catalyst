// func.func @doc_circuit(%arg0: complex<f64>) -> i1 {
//     %c00 = complex.constant [0.0, 0.0] : complex<f64>
//     %c10 = complex.constant [1.0, 0.0] : complex<f64>
//     %c20 = complex.constant [2.0, 0.0] : complex<f64>

//     %0 = complex.exp %arg0 : complex<f64>
//     %A = tensor.from_elements %c10, %c00, %c00, %0 : tensor<2x2xcomplex<f64>>

//     %1 = complex.mul %arg0, %c20 : complex<f64>
//     %2 = complex.exp %1 : complex<f64>
//     %B = tensor.from_elements %c10, %c00, %c00, %2 : tensor<2x2xcomplex<f64>>

//     %reg = quantum.alloc(1) : !quantum.reg
//     %q0 = quantum.extract %reg[0] : !quantum.reg -> !quantum.bit
//     %q1 = quantum.custom "Hadamard"() %q0 : !quantum.bit
//     %q2 = quantum.unitary(%A : tensor<2x2xcomplex<f64>>) %q1 : !quantum.bit
//     %q3 = quantum.unitary(%B : tensor<2x2xcomplex<f64>>) %q2 : !quantum.bit

//     %m, %q4 = quantum.measure %q3 : i1, !quantum.bit

//     return %m : i1
// }

// func.func @test() {
//     %reg = quantum.alloc( 2) : !quantum.reg
//     %q0 = quantum.extract %reg[ 0] : !quantum.reg -> !quantum.bit
//     %q1 = quantum.extract %reg[ 1] : !quantum.reg -> !quantum.bit

//     %c0 = arith.constant 0 : index
//     %c4 = arith.constant 4 : index
//     %c1 = arith.constant 1 : index

//     %c = arith.constant 1.1212 : f64
    
//     %q2:2 = quantum.custom "CNOT"() %q0, %q1 : !quantum.bit, !quantum.bit
//     %q3 = quantum.custom "RZ"(%c) %q2#0 : !quantum.bit
//     %q4 = quantum.custom "T"() %q2#1 : !quantum.bit
//     %q5 = quantum.custom "Hadamard"() %q3 : !quantum.bit
//     %q6 = quantum.custom "PauliX"() %q4 : !quantum.bit
//     %q7 = quantum.custom "PauliY"() %q5 : !quantum.bit
//     %q8 = quantum.custom "PauliZ"() %q6 : !quantum.bit
    

//     // %r = scf.for %i = %c0 to %c4 step %c1 iter_args(%x = %q1) -> !quantum.bit {
//     //     %q1_1 = quantum.custom "PauliX"() %x : !quantum.bit
//     //     scf.yield %q1_1 : !quantum.bit
//     // }


//     func.return
// }

func.func @ex_424() -> (!quantum.bit, !quantum.bit) {
    %reg = quantum.alloc( 2) : !quantum.reg
    // %i = arith.constant 1 : index
    %q0 = quantum.extract %reg[ 0] : !quantum.reg -> !quantum.bit
    %q1 = quantum.extract %reg[ 1] : !quantum.reg -> !quantum.bit
    
    %q2 = quantum.custom "T"() %q0 : !quantum.bit
    %q3 = quantum.custom "T"() %q1 : !quantum.bit
    %q4:2 = quantum.custom "CNOT"() %q2, %q3 : !quantum.bit, !quantum.bit
    %q5 = quantum.custom "T"() %q4#1 {adjoint}: !quantum.bit
    %q6:2 = quantum.custom "CNOT"() %q4#0, %q5 : !quantum.bit, !quantum.bit

    %q7 = quantum.custom "Hadamard"() %q6#1 : !quantum.bit

    %q8 = quantum.custom "T"() %q6#0 : !quantum.bit
    %q9 = quantum.custom "T"() %q7 : !quantum.bit
    %q10:2 = quantum.custom "CNOT"() %q9, %q8 : !quantum.bit, !quantum.bit
    %q11 = quantum.custom "T"() %q10#1 {adjoint}: !quantum.bit
    %q12:2 = quantum.custom "CNOT"() %q10#0, %q11 : !quantum.bit, !quantum.bit

    func.return %q12#1, %q12#0 : !quantum.bit, !quantum.bit
}