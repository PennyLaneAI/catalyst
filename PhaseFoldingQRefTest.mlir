// Stats:
// S: initial-> 0,  final-> 1. difference-> 1
// T: initial-> 2,  final-> 0. difference-> -2
// func.func @test_if_simple(%arg0: i1) attributes {quantum.node} {
//     %reg = qref.alloc( 2) : !qref.reg<2>
//     %q0 = qref.get %reg[ 0] : !qref.reg<2> -> !qref.bit
//     %q1 = qref.get %reg[ 1] : !qref.reg<2> -> !qref.bit

//     qref.custom "T"() %q0 : !qref.bit   // l0

//     %2:2 = scf.if %arg0 -> (!qref.bit, !qref.bit) {
//         qref.custom "CNOT"() %q0, %q1 : !qref.bit, !qref.bit

//         scf.yield %q0, %q1 : !qref.bit, !qref.bit
//     } 
//     else {
        
//         scf.yield %q0, %q1 : !qref.bit, !qref.bit
//     }

//     qref.custom "T"() %q0 : !qref.bit // l5 // will be removed:

//     qref.dealloc %reg : !qref.reg<2>
//     return
// }

// no change
// limitation of HH
// func.func @test_loop_block() attributes {quantum.node} {
    // %start = arith.constant 0 : index
    // %step = arith.constant 1 : index
    // %stop = arith.constant 37 : index

//     %reg = qref.alloc( 2) : !qref.reg<2>
//     %q0 = qref.get %reg[ 0] : !qref.reg<2> -> !qref.bit
//     %q1 = qref.get %reg[ 1] : !qref.reg<2> -> !qref.bit

//     qref.custom "T"() %q0 : !qref.bit   // l0
//     qref.custom "Hadamard"() %q0 : !qref.bit

//     scf.for %i = %start to %stop step %step {
//         qref.custom "T"() %q0 : !qref.bit   // l1
//         scf.yield
//     } 

//     qref.custom "Hadamard"() %q0 : !qref.bit
//     qref.custom "T"() %q0 : !qref.bit   // l2

//     qref.dealloc %reg : !qref.reg<2>
//     return
// }

// no change
// func.func @test_loop_cycle() attributes {quantum.node} {
//     %start = arith.constant 0 : index
//     %step = arith.constant 1 : index
//     %stop = arith.constant 37 : index

//     %reg = qref.alloc( 3) : !qref.reg<3>
//     %q0 = qref.get %reg[ 0] : !qref.reg<3> -> !qref.bit
//     %q1 = qref.get %reg[ 1] : !qref.reg<3> -> !qref.bit
//     %q2 = qref.get %reg[ 2] : !qref.reg<3> -> !qref.bit

//     qref.custom "T"() %q1 : !qref.bit   // l0

//     scf.for %i = %start to %stop step %step {
//         qref.custom "SWAP"() %q0, %q1 : !qref.bit, !qref.bit
//         qref.custom "SWAP"() %q1, %q2 : !qref.bit, !qref.bit
//         scf.yield
//     }

//     qref.custom "T"() %q1 : !qref.bit   // l1

//     qref.dealloc %reg : !qref.reg<3>
//     return
// }

// Stats:
// S: initial-> 0,  final-> 1. difference-> 1
// T: initial-> 2,  final-> 0. difference-> -2
// func.func @test_loop_h() attributes {quantum.node} {
//     %start = arith.constant 0 : index
//     %step = arith.constant 1 : index
//     %stop = arith.constant 37 : index

//     %reg = qref.alloc( 2) : !qref.reg<2>
//     %q0 = qref.get %reg[ 0] : !qref.reg<2> -> !qref.bit
//     %q1 = qref.get %reg[ 1] : !qref.reg<2> -> !qref.bit

//     qref.custom "T"() %q1 : !qref.bit   // l0   // will be removed
    
//     scf.for %i = %start to %stop step %step {
//         qref.custom "Hadamard"() %q0 : !qref.bit
//         scf.yield
//     } 

//     qref.custom "T"() %q1 : !qref.bit   // l1   // will be removed

//     qref.dealloc %reg : !qref.reg<2>
//     return
// }

// correct?!
// Stats:
// T: initial-> 3,  final-> 2. difference-> -1
func.func @test_loop_nested() attributes {quantum.node} {
    %start = arith.constant 0 : index
    %step = arith.constant 1 : index
    %stop = arith.constant 37 : index

    %reg = qref.alloc( 2) : !qref.reg<2>
    %q0 = qref.get %reg[ 0] : !qref.reg<2> -> !qref.bit
    %q1 = qref.get %reg[ 1] : !qref.reg<2> -> !qref.bit

    %cst0 = arith.constant dense<[false]> : tensor<1xi1>
    qref.set_basis_state(%cst0) %q0 : tensor<1xi1>, !qref.bit

    qref.custom "T"() %q1 : !qref.bit   // l0

    scf.for %i = %start to %stop step %step {
        qref.custom "T"() %q0 : !qref.bit   // l1      // will be removed
        
        scf.for %j = %start to %stop step %step {
            qref.custom "PauliX"() %q1 : !qref.bit
            
            scf.yield
        }

        scf.yield
    }

    qref.custom "T"() %q1 : !qref.bit   // l2

    qref.dealloc %reg : !qref.reg<2>
    return
}

// Stats:
// S: initial-> 0,  final-> 1. difference-> 1
// T: initial-> 4,  final-> 1. difference-> -3
// func.func @test_loop_null() attributes {quantum.node} {
//     %start = arith.constant 0 : index
//     %step = arith.constant 1 : index
//     %stop = arith.constant 37 : index

//     %reg = qref.alloc( 2) : !qref.reg<2>
//     %q0 = qref.get %reg[ 0] : !qref.reg<2> -> !qref.bit
//     %q1 = qref.get %reg[ 1] : !qref.reg<2> -> !qref.bit

//     qref.custom "T"() %q1 : !qref.bit   // l0

//     %tens01 = arith.constant dense<[false]> : tensor<1xi1>
//     qref.set_basis_state(%tens01) %q0 : tensor<1xi1>, !qref.bit   

//     scf.for %i = %start to %stop step %step {
//         qref.custom "T"() %q1 : !qref.bit   // l1
//         qref.custom "T"() %q0 : !qref.bit   // l2   // will be removed
        
//         scf.yield
//     }

//     qref.custom "T"() %q1 : !qref.bit   // l1   // will be removed

//     qref.dealloc %reg : !qref.reg<2>
//     return
// }

// Stats:
// S: initial-> 0,  final-> 1. difference-> 1
// T: initial-> 2,  final-> 0. difference-> -2
// func.func @test_loop_simple() attributes {quantum.node} {
//     %start = arith.constant 0 : index
//     %step = arith.constant 1 : index
//     %stop = arith.constant 37 : index

//     %reg = qref.alloc( 2) : !qref.reg<2>
//     %q0 = qref.get %reg[ 0] : !qref.reg<2> -> !qref.bit
//     %q1 = qref.get %reg[ 1] : !qref.reg<2> -> !qref.bit

//     qref.custom "T"() %q0 : !qref.bit   // l0   // will be removed

//     scf.for %i = %start to %stop step %step {
//         qref.custom "CNOT"() %q0, %q1 : !qref.bit, !qref.bit
        
//         scf.yield
//     }

//     qref.custom "T"() %q0 : !qref.bit   // l1   // will be removed

//     qref.dealloc %reg : !qref.reg<2>
//     return
// }

// Stats:
// S: initial-> 0,  final-> 1. difference-> 1
// T: initial-> 2,  final-> 0. difference-> -2
// func.func @test_loop_swap() attributes {quantum.node} {
//     %start = arith.constant 0 : index
//     %step = arith.constant 1 : index
//     %stop = arith.constant 37 : index

//     %reg = qref.alloc( 2) : !qref.reg<2>
//     %q0 = qref.get %reg[ 0] : !qref.reg<2> -> !qref.bit
//     %q1 = qref.get %reg[ 1] : !qref.reg<2> -> !qref.bit

//     qref.custom "CNOT"() %q0, %q1 : !qref.bit, !qref.bit
//     qref.custom "T"() %q1 : !qref.bit   // l0
//     qref.custom "CNOT"() %q0, %q1 : !qref.bit, !qref.bit

//     scf.for %i = %start to %stop step %step {
//         qref.custom "SWAP"() %q0, %q1 : !qref.bit, !qref.bit
        
//         scf.yield
//     }

//     qref.custom "CNOT"() %q0, %q1 : !qref.bit, !qref.bit
//     qref.custom "T"() %q1 : !qref.bit   // l0
//     qref.custom "CNOT"() %q0, %q1 : !qref.bit, !qref.bit

//     qref.dealloc %reg : !qref.reg<2>
//     return
// }

// Stats:
// T: initial-> 2,  final-> 1. difference-> -1
// func.func @test_reset_simple() attributes {quantum.node} {
//     %reg = qref.alloc( 1) : !qref.reg<1>
//     %q0 = qref.get %reg[ 0] : !qref.reg<1> -> !qref.bit

//     qref.custom "T"() %q0 : !qref.bit   // l0
    
//     %tens01 = arith.constant dense<[false]> : tensor<1xi1>
//     qref.set_basis_state(%tens01) %q0 : tensor<1xi1>, !qref.bit

//     qref.custom "T"() %q0 : !qref.bit   // l1   // will be removed

//     qref.dealloc %reg : !qref.reg<1>
//     return
// }

// Stats:
// S: initial-> 0,  final-> 1. difference-> 1
// T: initial-> 6,  final-> 1. difference-> -5
// func.func @test_if_1(%arg0: i1) attributes {quantum.node} {
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
//         qref.custom "T"() %q0 : !qref.bit // l1 // RM
//         qref.custom "T"() %q1 : !qref.bit // l2 // RM
//         qref.custom "CNOT"() %q0, %q1 : !qref.bit, !qref.bit
//         qref.custom "T"() %q1 : !qref.bit // l3 // RM
//         // qref.custom "S"() %2 : !qref.bit

//         scf.yield %q0, %q1 : !qref.bit, !qref.bit
//     } 
//     else {
//         qref.custom "Hadamard"() %q1 : !qref.bit
//         qref.custom "T"() %q0 : !qref.bit // l4 // RM

//         scf.yield %q0, %q1 : !qref.bit, !qref.bit
//     }

//     qref.custom "T"() %q0 : !qref.bit // l5 // RM

//     qref.dealloc %reg : !qref.reg<2>
//     return
// }

// Stats:
// S: initial-> 0,  final-> 2. difference-> 2
// T: initial-> 7,  final-> 2. difference-> -5
// func.func @test_if_2(%arg0: i1) attributes {quantum.node} {
//     %reg = qref.alloc( 2) : !qref.reg<2>
//     %q0 = qref.get %reg[ 0] : !qref.reg<2> -> !qref.bit
//     %q1 = qref.get %reg[ 1] : !qref.reg<2> -> !qref.bit

//     %tens01 = arith.constant dense<[true]> : tensor<1xi1>
//     qref.set_basis_state(%tens01) %q0 : tensor<1xi1>, !qref.bit

//     qref.custom "T"() %q0 : !qref.bit   // l0
//     qref.custom "PauliX"() %q0 : !qref.bit

//     %2:2 = scf.if %arg0 -> (!qref.bit, !qref.bit) {
//         qref.custom "T"() %q0 : !qref.bit // l1     // RM
//         qref.custom "T"() %q1 : !qref.bit // l2     // RM
//         // qref.custom "S"() %2 : !qref.bit // ADD
//         qref.custom "CNOT"() %q0, %q1 : !qref.bit, !qref.bit
//         qref.custom "T"() %q1 : !qref.bit // l3     // RM
//         scf.yield %q0, %q1 : !qref.bit, !qref.bit
//     } 
//     else {
//         // qref.custom "S"() %2 : !qref.bit
//         qref.custom "T"() %q1 : !qref.bit // l4     // RM

//         qref.custom "CNOT"() %q0, %q1 : !qref.bit, !qref.bit
//         qref.custom "CNOT"() %q1, %q0 : !qref.bit, !qref.bit
//         qref.custom "CNOT"() %q0, %q1 : !qref.bit, !qref.bit

//         qref.custom "Hadamard"() %q1 : !qref.bit

//         qref.custom "CNOT"() %q0, %q1 : !qref.bit, !qref.bit
//         qref.custom "CNOT"() %q1, %q0 : !qref.bit, !qref.bit
//         qref.custom "CNOT"() %q0, %q1 : !qref.bit, !qref.bit
        
//         qref.custom "T"() %q1 : !qref.bit // l5     // RM

//         scf.yield %q0, %q1 : !qref.bit, !qref.bit
//     }

//     qref.custom "T"() %q0 : !qref.bit

//     qref.dealloc %reg : !qref.reg<2>
//     return
// }

// Stats:
// S: initial-> 0,  final-> 2. difference-> 2
// T: initial-> 4,  final-> 0. difference-> -4
// func.func @test_for() attributes {quantum.node} {
//     %start = arith.constant 0 : index
//     %step = arith.constant 1 : index
//     %stop = arith.constant 37 : index

//     %reg = qref.alloc( 2) : !qref.reg<2>
//     %q0 = qref.get %reg[ 0] : !qref.reg<2> -> !qref.bit
//     %q1 = qref.get %reg[ 1] : !qref.reg<2> -> !qref.bit

//     qref.custom "CNOT"() %q0, %q1 : !qref.bit, !qref.bit
//     // qref.custom "S"() %2 : !qref.bit
//     qref.custom "T"() %q1 : !qref.bit   // l0   // RM
//     qref.custom "CNOT"() %q0, %q1 : !qref.bit, !qref.bit

//     scf.for %i = %start to %stop step %step {
//         qref.custom "T"() %q0 : !qref.bit   // l1   // RM
//         // qref.custom "S"() %1 : !qref.bit
//         qref.custom "SWAP"() %q0, %q1 : !qref.bit, !qref.bit
//         qref.custom "T"() %q1 : !qref.bit   // l2   // RM
//         scf.yield
//     } 

//     qref.custom "CNOT"() %q0, %q1 : !qref.bit, !qref.bit
//     qref.custom "T"() %q1 : !qref.bit   // l3   // RM
//     qref.custom "CNOT"() %q0, %q1 : !qref.bit, !qref.bit

//     qref.dealloc %reg : !qref.reg<2>
//     return
// }

// Stats:
// S: initial-> 0,  final-> 1. difference-> 1
// T: initial-> 2,  final-> 0. difference-> -2
// func.func @ex_normal(%arg0: i1) attributes {quantum.node} {
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

// Stats:
// S: initial-> 0,  final-> 2. difference-> 2
// T: initial-> 9,  final-> 4. difference-> -5
func.func @test_nested(%arg0: i1) attributes {quantum.node} {
    %start = arith.constant 0 : index
    %step = arith.constant 1 : index
    %stop = arith.constant 37 : index

    %reg = qref.alloc( 2) : !qref.reg<2>
    %q0 = qref.get %reg[ 0] : !qref.reg<2> -> !qref.bit
    %q1 = qref.get %reg[ 1] : !qref.reg<2> -> !qref.bit

    %tens01 = arith.constant dense<[true]> : tensor<1xi1>
    qref.set_basis_state(%tens01) %q0 : tensor<1xi1>, !qref.bit

    qref.custom "T"() %q0 : !qref.bit   // l0
    qref.custom "PauliX"() %q0 : !qref.bit

    scf.for %i = %start to %stop step %step {
        qref.custom "T"() %q1 : !qref.bit // l1
        qref.custom "SWAP"() %q0, %q1 : !qref.bit, !qref.bit

        %2:2 = scf.if %arg0 -> (!qref.bit, !qref.bit) {
            qref.custom "T"() %q0 : !qref.bit // l2
            qref.custom "T"() %q1 : !qref.bit // l3     // doesn't capture this!
            qref.custom "CNOT"() %q0, %q1 : !qref.bit, !qref.bit
            qref.custom "T"() %q1 : !qref.bit // l4
            qref.custom "CNOT"() %q0, %q1 : !qref.bit, !qref.bit

            scf.yield %q0, %q1 : !qref.bit, !qref.bit
        } 
        else {
            qref.custom "T"() %q0 : !qref.bit // l5

            qref.custom "CNOT"() %q0, %q1 : !qref.bit, !qref.bit
            qref.custom "CNOT"() %q1, %q0 : !qref.bit, !qref.bit
            qref.custom "CNOT"() %q0, %q1 : !qref.bit, !qref.bit

            // qref.custom "H"() %q0 : !qref.bit

            qref.custom "CNOT"() %q0, %q1 : !qref.bit, !qref.bit
            qref.custom "CNOT"() %q1, %q0 : !qref.bit, !qref.bit
            qref.custom "CNOT"() %q0, %q1 : !qref.bit, !qref.bit

            qref.custom "T"() %q0 : !qref.bit // l6
            
            scf.yield %q0, %q1 : !qref.bit, !qref.bit
        }

        qref.custom "T"() %q0 : !qref.bit // l7
        qref.custom "SWAP"() %q0, %q1 : !qref.bit, !qref.bit
        // qref.custom "T"() %q1 : !qref.bit // l7

        scf.yield
    } 
    
    qref.custom "T"() %q0 : !qref.bit // l8

    qref.dealloc %reg : !qref.reg<2>
    return
}



















// alloc_qb
// Stats:
// S: initial-> 0,  final-> 2. difference-> 2
// T: initial-> 4,  final-> 0. difference-> -4
// func.func @test_temp_anc() attributes {quantum.node} {
//     %start = arith.constant 0 : index
//     %step = arith.constant 1 : index
//     %stop = arith.constant 37 : index

//     %reg = qref.alloc( 2) : !qref.reg<2>
//     %q0 = qref.get %reg[ 0] : !qref.reg<2> -> !qref.bit
//     %q1 = qref.get %reg[ 1] : !qref.reg<2> -> !qref.bit

//     qref.custom "CNOT"() %q0, %q1 : !qref.bit, !qref.bit
//     // qref.custom "S"() %2 : !qref.bit
//     qref.custom "T"() %q1 : !qref.bit   // l0   // will be removed
//     qref.custom "CNOT"() %q0, %q1 : !qref.bit, !qref.bit

//     scf.for %i = %start to %stop step %step {
//         // qref.custom "S"() %1 : !qref.bit
//         qref.custom "T"() %q0 : !qref.bit   // l1   // will be removed
//         qref.custom "SWAP"() %q0, %q1 : !qref.bit, !qref.bit

//         %anc = qref.alloc_qb : !qref.bit
//         qref.custom "Hadamard"() %anc : !qref.bit
//         %mres = qref.measure %anc : i1

//         qref.custom "T"() %q1 : !qref.bit   // l2   // will be removed

//         scf.yield
//     } 

//     qref.custom "CNOT"() %q0, %q1 : !qref.bit, !qref.bit
//     qref.custom "T"() %q1 : !qref.bit   // l3   // will be removed
//     qref.custom "CNOT"() %q0, %q1 : !qref.bit, !qref.bit

//     qref.dealloc %reg : !qref.reg<2>
//     return
// }

// Stats:
// S: initial-> 0,  final-> 1. difference-> 1
// T: initial-> 2,  final-> 0. difference-> -2
// func.func @test_loop() attributes {quantum.node} {
//     %start = arith.constant 0 : index
//     %step = arith.constant 1 : index
//     %stop = arith.constant 37 : index

//     %reg = qref.alloc( 3) : !qref.reg<3>
//     %q0 = qref.get %reg[ 0] : !qref.reg<3> -> !qref.bit
//     %q1 = qref.get %reg[ 1] : !qref.reg<3> -> !qref.bit
//     %q2 = qref.get %reg[ 2] : !qref.reg<3> -> !qref.bit

//     // qref.custom "S"() %2 : !qref.bit
//     qref.custom "T"() %q0 : !qref.bit   // l0   // will be removed

//     scf.for %i = %start to %stop step %step {
//         qref.custom "CNOT"() %q0, %q2 : !qref.bit, !qref.bit

//         qref.custom "Hadamard"() %q1 : !qref.bit
        
//         qref.custom "CNOT"() %q2, %q1 : !qref.bit, !qref.bit

//         scf.yield
//     } 

//     qref.custom "T"() %q0 : !qref.bit   // l2   // will be removed
    
//     qref.dealloc %reg : !qref.reg<3>
//     return
// }

// Stats:
// T: initial-> 2,  final-> 0. difference-> -2
// module @module_circuit_base {
//   func.func public @circuit_base(%arg0: tensor<f64>) -> tensor<4xf64> attributes {diff_method = "parameter-shift", llvm.linkage = #llvm.linkage<internal>, quantum.node} {
//     %c0_i64 = arith.constant 0 : i64
//     %cst = stablehlo.constant dense<1.400000e+00> : tensor<f64>
//     %0 = stablehlo.compare  GT, %arg0, %cst,  FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
//     quantum.device shots(%c0_i64) ["/Users/sara.babaeekhanehsar/Desktop/Home/Coding/Catalyst/catalyst/.venv/lib/python3.14/site-packages/pennylane_lightning/liblightning_qubit_catalyst.dylib", "LightningSimulator", "{'mcmc': False, 'num_burnin': 0, 'kernel_name': None}"]
    
//     %1 = qref.alloc( 2) : !qref.reg<2>
//     %2 = qref.get %1[ 0] : !qref.reg<2> -> !qref.bit
    
//     qref.custom "T"() %2 : !qref.bit

//     %extracted = tensor.extract %0[] : tensor<i1>

//     scf.if %extracted {
//       %6 = qref.get %1[ 0] : !qref.reg<2> -> !qref.bit
//       qref.custom "PauliX"() %6 : !qref.bit

//       %7 = qref.get %1[ 1] : !qref.reg<2> -> !qref.bit
//       qref.custom "Hadamard"() %7 : !qref.bit
//     } else {
//       %6 = qref.get %1[ 0] : !qref.reg<2> -> !qref.bit
//       qref.custom "PauliY"() %6 : !qref.bit

//       %7 = qref.get %1[ 1] : !qref.reg<2> -> !qref.bit
//       %extracted_0 = tensor.extract %arg0[] : tensor<f64>
//       qref.custom "RZ"(%extracted_0) %7 : !qref.bit
//     }
//     %3 = qref.get %1[ 0] : !qref.reg<2> -> !qref.bit
//     qref.custom "T"() %3 : !qref.bit

//     %4 = qref.compbasis(qreg %1 : !qref.reg<2>) : !quantum.obs
//     %5 = quantum.probs %4 : tensor<4xf64>
//     qref.dealloc %1 : !qref.reg<2>
//     quantum.device_release
//     return %5 : tensor<4xf64>
//   }
// }

// not captured in feynman
// Stats:
// S: initial-> 0,  final-> 1. difference-> 1
// T: initial-> 5,  final-> 1. difference-> -4
// func.func @test_reset_in_block(%arg0: i1) attributes {quantum.node} {
//     %reg = qref.alloc( 2) : !qref.reg<2>
//     %q0 = qref.get %reg[ 0] : !qref.reg<2> -> !qref.bit
//     %q1 = qref.get %reg[ 1] : !qref.reg<2> -> !qref.bit

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
//         scf.yield %q0, %q1 : !qref.bit, !qref.bit
//     }

//     qref.custom "T"() %q0 : !qref.bit // l4 // will be removed:

//     qref.dealloc %reg : !qref.reg<2>
//     return
// }


// module @module_tof_decomp {
//   func.func public @tof_decomp() -> tensor<8xf64> attributes {decompose_gatesets = [["CNOT", "GlobalPhase", "Hadamard", "Identity", "PauliX", "PauliY", "PauliZ", "RZ", "S", "SWAP", "T"]], diff_method = "parameter-shift", llvm.linkage = #llvm.linkage<internal>, quantum.node} {
//     %cst = arith.constant dense<[0, 1, 2]> : tensor<3xi64>
//     %c0_i64 = arith.constant 0 : i64
//     quantum.device shots(%c0_i64) ["/Users/sara.babaeekhanehsar/Desktop/Home/Coding/Catalyst/catalyst/.venv/lib/python3.14/site-packages/pennylane_lightning/liblightning_qubit_catalyst.dylib", "LightningSimulator", "{'mcmc': False, 'num_burnin': 0, 'kernel_name': None}"]
//     %0 = qref.alloc( 3) : !qref.reg<3>
//     %1 = stablehlo.slice %cst [2:3] : (tensor<3xi64>) -> tensor<1xi64>
//     %2 = stablehlo.reshape %1 : (tensor<1xi64>) -> tensor<i64>
//     %extracted = tensor.extract %2[] : tensor<i64>
//     %3 = qref.get %0[%extracted] : !qref.reg<3>, i64 -> !qref.bit
//     qref.custom "Hadamard"() %3 : !qref.bit
//     %4 = stablehlo.slice %cst [1:2] : (tensor<3xi64>) -> tensor<1xi64>
//     %5 = stablehlo.reshape %4 : (tensor<1xi64>) -> tensor<i64>
//     %extracted_0 = tensor.extract %5[] : tensor<i64>
//     %6 = qref.get %0[%extracted_0] : !qref.reg<3>, i64 -> !qref.bit
//     qref.custom "CNOT"() %6, %3 : !qref.bit, !qref.bit
//     qref.custom "T"() %3 adj : !qref.bit
//     %7 = stablehlo.slice %cst [0:1] : (tensor<3xi64>) -> tensor<1xi64>
//     %8 = stablehlo.reshape %7 : (tensor<1xi64>) -> tensor<i64>
//     %extracted_1 = tensor.extract %8[] : tensor<i64>
//     %9 = qref.get %0[%extracted_1] : !qref.reg<3>, i64 -> !qref.bit
//     qref.custom "CNOT"() %9, %3 : !qref.bit, !qref.bit
//     qref.custom "T"() %3 : !qref.bit
//     qref.custom "CNOT"() %6, %3 : !qref.bit, !qref.bit
//     qref.custom "T"() %3 adj : !qref.bit
//     qref.custom "CNOT"() %9, %3 : !qref.bit, !qref.bit
//     qref.custom "T"() %3 : !qref.bit
//     qref.custom "T"() %6 : !qref.bit
//     qref.custom "CNOT"() %9, %6 : !qref.bit, !qref.bit
//     qref.custom "Hadamard"() %3 : !qref.bit
//     qref.custom "T"() %9 : !qref.bit
//     qref.custom "T"() %6 adj : !qref.bit
//     qref.custom "CNOT"() %9, %6 : !qref.bit, !qref.bit
//     %10 = qref.compbasis(qreg %0 : !qref.reg<3>) : !quantum.obs
//     %11 = quantum.probs %10 : tensor<8xf64>
//     qref.dealloc %0 : !qref.reg<3>
//     quantum.device_release
//     return %11 : tensor<8xf64>
//   }
// }