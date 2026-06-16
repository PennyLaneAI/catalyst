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

func.func @ex_425(%arg0: tensor<2xi64>) -> (!quantum.bit, !quantum.bit) {
    %reg = quantum.alloc( 2) : !quantum.reg
    %q0 = quantum.extract %reg[ 0] : !quantum.reg -> !quantum.bit
    %q1 = quantum.extract %reg[ 1] : !quantum.reg -> !quantum.bit

    // %2 = stablehlo.convert %arg0 : (tensor<2xi64>) -> tensor<2xcomplex<f64>>
    // %3 = quantum.set_state(%2) %q1 : (tensor<2xcomplex<f64>>, !quantum.bit) -> !quantum.bit
    // %q3 = quantum.custom "T"() %3 : !quantum.bit


    %tens01 = arith.constant dense<[false]> : tensor<1xi1>
    %q2 = quantum.set_basis_state(%tens01) %q1 : (tensor<1xi1>, !quantum.bit) -> !quantum.bit

    %q3 = quantum.custom "T"() %q2 : !quantum.bit   // l1
    %q4:2 = quantum.custom "CNOT"() %q0, %q3 : !quantum.bit, !quantum.bit
    %q5 = quantum.custom "T"() %q4#0 : !quantum.bit   // l2
    %65 = quantum.custom "T"() %q4#1 : !quantum.bit   // l3

    func.return %q0, %q3 : !quantum.bit, !quantum.bit
}

func.func @ex_1() -> (!quantum.bit, !quantum.bit) {
    %reg = quantum.alloc( 2) : !quantum.reg
    // %i = arith.constant 1 : index
    %q0 = quantum.extract %reg[ 0] : !quantum.reg -> !quantum.bit
    %q1 = quantum.extract %reg[ 1] : !quantum.reg -> !quantum.bit    
    
    // %c0 = arith.constant 3.141592 : f64

    %tens01 = arith.constant dense<[0]> : tensor<1xi1>
    %q18 = quantum.set_basis_state(%tens01) %q1 : (tensor<1xi1>, !quantum.bit) -> !quantum.bit

    %q2 = quantum.custom "T"() %q0 : !quantum.bit   // l1
    // %q2 = quantum.custom "RZ"(%c0) %q0 : !quantum.bit
    %q3 = quantum.custom "T"() %q18 : !quantum.bit   // l2
    %q4:2 = quantum.custom "CNOT"() %q2, %q3 : !quantum.bit, !quantum.bit
    %q5 = quantum.custom "T"() %q4#1 {adjoint}: !quantum.bit    // l3
    %q6:2 = quantum.custom "CNOT"() %q4#0, %q5 : !quantum.bit, !quantum.bit

    %q7 = quantum.custom "Hadamard"() %q6#1 : !quantum.bit

    %q21 = quantum.custom "PauliX"() %q6#0 : !quantum.bit

    %q8 = quantum.custom "T"() %q21 : !quantum.bit  // l4
    %q9 = quantum.custom "T"() %q7 : !quantum.bit   // l5
    %q10:2 = quantum.custom "CNOT"() %q9, %q8 : !quantum.bit, !quantum.bit
    %q11 = quantum.custom "T"() %q10#1 {adjoint}: !quantum.bit  // l6
    %q12:2 = quantum.custom "CNOT"() %q10#0, %q11 : !quantum.bit, !quantum.bit

    
    %q13 = quantum.custom "T"() %q12#0 : !quantum.bit   // on q2

    %reg2 = quantum.alloc( 2) : !quantum.reg
    %p0 = quantum.extract %reg2[ 0] : !quantum.reg -> !quantum.bit
    %p1 = quantum.extract %reg2[ 1] : !quantum.reg -> !quantum.bit    

    // %c = arith.constant 1.1212 : f64
    // %q_temp = quantum.custom "RZ"(%c) %q13 : !quantum.bit

    %q14 = quantum.custom "T"() %q13 : !quantum.bit
    %q15 = quantum.custom "T"() %q14 : !quantum.bit
    %q16 = quantum.custom "T"() %q15 : !quantum.bit
    // %q17 = quantum.custom "PauliY"() %q16 : !quantum.bit
    // %q18 = quantum.custom "T"() %q17 : !quantum.bit
    // %q19 = quantum.custom "T"() %q18 : !quantum.bit
    // %q20 = quantum.custom "T"() %q19 : !quantum.bit

    %p2 = quantum.custom "T"() %p0 : !quantum.bit
    %p3 = quantum.custom "Hadamard"() %p1 : !quantum.bit
    %p4:2 = quantum.custom "CNOT"() %p3, %p2 : !quantum.bit, !quantum.bit
    %p5 = quantum.custom "T"() %p4#1 : !quantum.bit
    %p6:2 = quantum.custom "CNOT"() %p4#0, %p5 : !quantum.bit, !quantum.bit
    %p7 = quantum.custom "T"() %p6#1 : !quantum.bit

    %q17 = quantum.custom "PauliY"() %q16 : !quantum.bit

    %qb = quantum.alloc_qb : !quantum.bit
    %h = quantum.custom "PauliX"() %qb : !quantum.bit

    %p8:2 = quantum.custom "SWAP"() %p7, %h : !quantum.bit, !quantum.bit
    %p9 = quantum.custom "S"() %p8#1 : !quantum.bit

    // %tens01 = arith.constant dense<[false, true]> : tensor<2xi1>
    // %q18:2 = quantum.set_basis_state(%tens01) %q12#1, %q17 : (tensor<2xi1>, !quantum.bit, !quantum.bit) -> (!quantum.bit, !quantum.bit)

    // func.return %q12#1, %q12#0 : !quantum.bit, !quantum.bit
    func.return %q12#1, %q12#0 : !quantum.bit, !quantum.bit
}

