// func.func @test_if_1(%arg0: i1) attributes {quantum.node} {
//     // Stats:
//     // T: initial-> 6,  final-> 3. difference-> -3

//     %reg = qref.alloc( 2) : !qref.reg<2>
//     %q0 = qref.get %reg[ 0] : !qref.reg<2> -> !qref.bit
//     %q1 = qref.get %reg[ 1] : !qref.reg<2> -> !qref.bit

//     // %2 = stablehlo.convert %arg0 : (tensor<2xi64>) -> tensor<2xcomplex<f64>>
//     // %3 = qref.set_state(%2) %q1 : (tensor<2xcomplex<f64>>, !qref.bit) -> !qref.bit
//     // %q3 = qref.custom "T"() %3 : !qref.bit

//     %tens01 = arith.constant dense<[true]> : tensor<1xi1>
//     qref.set_basis_state(%tens01) %q0 : tensor<1xi1>, !qref.bit

//     qref.custom "T"() %q0 : !qref.bit   // l0
//     qref.custom "PauliX"() %q0 : !qref.bit

//     %2:2 = scf.if %arg0 -> (!qref.bit, !qref.bit) {
//         qref.custom "T"() %q0 : !qref.bit // l1 // will be removed:
//         qref.custom "T"() %q1 : !qref.bit // l2
//         qref.custom "CNOT"() %q0, %q1 : !qref.bit, !qref.bit
//         qref.custom "T"() %q1 : !qref.bit // l3

//         scf.yield %q0, %q1 : !qref.bit, !qref.bit
//     } 
//     else {
//         qref.custom "Hadamard"() %q1 : !qref.bit
//         qref.custom "T"() %q0 : !qref.bit // l4 // will be removed:

//         scf.yield %q0, %q1 : !qref.bit, !qref.bit
//     }

//     qref.custom "T"() %q0 : !qref.bit // l5 // will be removed:

//     qref.dealloc %reg : !qref.reg<2>
//     return
// }

// func.func @test_if_2(%arg0: i1) attributes {quantum.node} {
//     // Stats:
//     // S: initial-> 0,  final-> 1. difference-> 1
//     // T: initial-> 7,  final-> 4. difference-> -3

//     %reg = qref.alloc( 2) : !qref.reg<2>
//     %q0 = qref.get %reg[ 0] : !qref.reg<2> -> !qref.bit
//     %q1 = qref.get %reg[ 1] : !qref.reg<2> -> !qref.bit

//     %tens01 = arith.constant dense<[true]> : tensor<1xi1>
//     qref.set_basis_state(%tens01) %q0 : tensor<1xi1>, !qref.bit

//     qref.custom "T"() %q0 : !qref.bit   // l0
//     qref.custom "PauliX"() %q0 : !qref.bit

//     %2:2 = scf.if %arg0 -> (!qref.bit, !qref.bit) {
//         qref.custom "T"() %q0 : !qref.bit // l1     // will be removed
//         qref.custom "T"() %q1 : !qref.bit // l2
//         qref.custom "CNOT"() %q0, %q1 : !qref.bit, !qref.bit
//         qref.custom "T"() %q1 : !qref.bit // l3
//         scf.yield %q0, %q1 : !qref.bit, !qref.bit
//     } 
//     else {
//         // qref.custom "S"() %2 : !qref.bit
//         qref.custom "T"() %q1 : !qref.bit // l4     // will be removed

//         qref.custom "CNOT"() %q0, %q1 : !qref.bit, !qref.bit
//         qref.custom "CNOT"() %q1, %q0 : !qref.bit, !qref.bit
//         qref.custom "CNOT"() %q0, %q1 : !qref.bit, !qref.bit

//         qref.custom "Hadamard"() %q1 : !qref.bit

//         qref.custom "CNOT"() %q0, %q1 : !qref.bit, !qref.bit
//         qref.custom "CNOT"() %q1, %q0 : !qref.bit, !qref.bit
//         qref.custom "CNOT"() %q0, %q1 : !qref.bit, !qref.bit
        
//         qref.custom "T"() %q1 : !qref.bit // l5     // will be removed

//         scf.yield %q0, %q1 : !qref.bit, !qref.bit
//     }

//     qref.custom "T"() %q0 : !qref.bit // l6

//     qref.dealloc %reg : !qref.reg<2>
//     return
// }

func.func @test_for() attributes {quantum.node} {
    // Stats:
    // S: initial-> 0,  final-> 1. difference-> 1
    // T: initial-> 2,  final-> 0. difference-> -2

    %start = arith.constant 0 : index
    %step = arith.constant 1 : index
    %stop = arith.constant 37 : index

    %reg = qref.alloc( 2) : !qref.reg<2>
    %q0 = qref.get %reg[ 0] : !qref.reg<2> -> !qref.bit
    %q1 = qref.get %reg[ 1] : !qref.reg<2> -> !qref.bit

    qref.custom "CNOT"() %q0, %q1 : !qref.bit, !qref.bit
    // qref.custom "S"() %2 : !qref.bit
    qref.custom "T"() %q1 : !qref.bit   // l0   // will be removed
    qref.custom "CNOT"() %q0, %q1 : !qref.bit, !qref.bit

    scf.for %i = %start to %stop step %step {
        qref.custom "SWAP"() %q0, %q1 : !qref.bit, !qref.bit
        scf.yield
    } 

    qref.custom "CNOT"() %q0, %q1 : !qref.bit, !qref.bit
    qref.custom "T"() %q1 : !qref.bit   // l1   // will be removed
    qref.custom "CNOT"() %q0, %q1 : !qref.bit, !qref.bit

    qref.dealloc %reg : !qref.reg<2>
    return
}


// func.func @ex_normal(%arg0: i1) {
//     %reg = qref.alloc( 2) : !qref.reg<2>
//     %q0 = qref.get %reg[ 0] : !qref.reg<2> -> !qref.bit
//     %q1 = qref.get %reg[ 1] : !qref.reg<2> -> !qref.bit

//     qref.custom "T"() %q1 : !qref.bit // l3
//     qref.custom "CNOT"() %q0, %q1 : !qref.bit, !qref.bit
//     qref.custom "CNOT"() %q1, %q0 : !qref.bit, !qref.bit
//     qref.custom "CNOT"() %q0, %q1 : !qref.bit, !qref.bit
    
//     qref.custom "Hadamard"() %q1 : !qref.bit
    
//     qref.custom "CNOT"() %q0, %q1 : !qref.bit, !qref.bit
//     qref.custom "CNOT"() %q1, %q0 : !qref.bit, !qref.bit
//     qref.custom "CNOT"() %q0, %q1 : !qref.bit, !qref.bit
//     qref.custom "T"() %q1 : !qref.bit // l4

//     return
// }

// func.func @test_nested(%arg0: i1) attributes {quantum.node} {
//     %start = arith.constant 0 : index
//     %step = arith.constant 1 : index
//     %stop = arith.constant 37 : index

//     %reg = qref.alloc( 2) : !qref.reg<2>
//     %q0 = qref.get %reg[ 0] : !qref.reg<2> -> !qref.bit
//     %q1 = qref.get %reg[ 1] : !qref.reg<2> -> !qref.bit

//     %tens01 = arith.constant dense<[true]> : tensor<1xi1>
//     qref.set_basis_state(%tens01) %q0 : tensor<1xi1>, !qref.bit

//     qref.custom "T"() %q0 : !qref.bit   // l0
//     qref.custom "PauliX"() %q0 : !qref.bit

//     scf.for %i = %start to %stop step %step {
//         qref.custom "T"() %q1 : !qref.bit // l1
//         qref.custom "SWAP"() %q0, %q1 : !qref.bit, !qref.bit

//         %2:2 = scf.if %arg0 -> (!qref.bit, !qref.bit) {
//             qref.custom "T"() %q0 : !qref.bit // l2
//             qref.custom "T"() %q1 : !qref.bit // l3
//             qref.custom "CNOT"() %q0, %q1 : !qref.bit, !qref.bit
//             qref.custom "T"() %q1 : !qref.bit // l4
//             qref.custom "CNOT"() %q0, %q1 : !qref.bit, !qref.bit

//             scf.yield %q0, %q1 : !qref.bit, !qref.bit
//         } 
//         else {
//             qref.custom "T"() %q0 : !qref.bit // l5

//             qref.custom "CNOT"() %q0, %q1 : !qref.bit, !qref.bit
//             qref.custom "CNOT"() %q1, %q0 : !qref.bit, !qref.bit
//             qref.custom "CNOT"() %q0, %q1 : !qref.bit, !qref.bit

//             qref.custom "CNOT"() %q0, %q1 : !qref.bit, !qref.bit
//             qref.custom "CNOT"() %q1, %q0 : !qref.bit, !qref.bit
//             qref.custom "CNOT"() %q0, %q1 : !qref.bit, !qref.bit

//             qref.custom "T"() %q0 : !qref.bit // l6
            
//             scf.yield %q0, %q1 : !qref.bit, !qref.bit
//         }

//         qref.custom "SWAP"() %q0, %q1 : !qref.bit, !qref.bit
//         qref.custom "T"() %q1 : !qref.bit // l7

//         scf.yield
//     } 
    
//     qref.custom "T"() %q0 : !qref.bit // l8

//     qref.dealloc %reg : !qref.reg<2>
//     return
// }