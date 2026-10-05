module @module_qfunc {
  func.func public @qfunc() -> tensor<f64> attributes {diff_method = "parameter-shift", llvm.linkage = #llvm.linkage<internal>, quantum.node} {
    %cst = arith.constant dense<[0, 1, 2]> : tensor<3xi64>
    %c0_i64 = arith.constant 0 : i64
    quantum.device shots(%c0_i64) ["/Users/haider.sajjad/catalyst/frontend/catalyst/utils/../../../runtime/build/lib/librtd_null_qubit.dylib", "NullQubit", "{'track_resources': False}"]
    %0 = quantum.alloc( 3) : !quantum.reg
    %1 = stablehlo.slice %cst [2:3] : (tensor<3xi64>) -> tensor<1xi64>
    %2 = stablehlo.reshape %1 : (tensor<1xi64>) -> tensor<i64>
    %extracted = tensor.extract %2[] : tensor<i64>
    %3 = quantum.extract %0[%extracted] : !quantum.reg -> !quantum.bit
    %out_qubits = quantum.custom "Hadamard"() %3 : !quantum.bit
    %4 = quantum.insert %0[%extracted], %out_qubits : !quantum.reg, !quantum.bit
    %5 = stablehlo.slice %cst [1:2] : (tensor<3xi64>) -> tensor<1xi64>
    %6 = stablehlo.reshape %5 : (tensor<1xi64>) -> tensor<i64>
    %7 = stablehlo.slice %cst [2:3] : (tensor<3xi64>) -> tensor<1xi64>
    %8 = stablehlo.reshape %7 : (tensor<1xi64>) -> tensor<i64>
    %extracted_0 = tensor.extract %6[] : tensor<i64>
    %extracted_1 = tensor.extract %8[] : tensor<i64>
    %9 = quantum.extract %4[%extracted_0] : !quantum.reg -> !quantum.bit
    %10 = quantum.extract %4[%extracted_1] : !quantum.reg -> !quantum.bit
    %out_qubits_2:2 = quantum.custom "CNOT"() %9, %10 : !quantum.bit, !quantum.bit
    %11 = quantum.insert %4[%extracted_0], %out_qubits_2#0 : !quantum.reg, !quantum.bit
    %12 = quantum.insert %11[%extracted_1], %out_qubits_2#1 : !quantum.reg, !quantum.bit
    %13 = stablehlo.slice %cst [2:3] : (tensor<3xi64>) -> tensor<1xi64>
    %14 = stablehlo.reshape %13 : (tensor<1xi64>) -> tensor<i64>
    %extracted_3 = tensor.extract %14[] : tensor<i64>
    %15 = quantum.extract %12[%extracted_3] : !quantum.reg -> !quantum.bit
    %out_qubits_4 = quantum.custom "T"() %15 adj : !quantum.bit
    %16 = quantum.insert %12[%extracted_3], %out_qubits_4 : !quantum.reg, !quantum.bit
    %17 = stablehlo.slice %cst [0:1] : (tensor<3xi64>) -> tensor<1xi64>
    %18 = stablehlo.reshape %17 : (tensor<1xi64>) -> tensor<i64>
    %19 = stablehlo.slice %cst [2:3] : (tensor<3xi64>) -> tensor<1xi64>
    %20 = stablehlo.reshape %19 : (tensor<1xi64>) -> tensor<i64>
    %extracted_5 = tensor.extract %18[] : tensor<i64>
    %extracted_6 = tensor.extract %20[] : tensor<i64>
    %21 = quantum.extract %16[%extracted_5] : !quantum.reg -> !quantum.bit
    %22 = quantum.extract %16[%extracted_6] : !quantum.reg -> !quantum.bit
    %out_qubits_7:2 = quantum.custom "CNOT"() %21, %22 : !quantum.bit, !quantum.bit
    %23 = quantum.insert %16[%extracted_5], %out_qubits_7#0 : !quantum.reg, !quantum.bit
    %24 = quantum.insert %23[%extracted_6], %out_qubits_7#1 : !quantum.reg, !quantum.bit
    %25 = stablehlo.slice %cst [2:3] : (tensor<3xi64>) -> tensor<1xi64>
    %26 = stablehlo.reshape %25 : (tensor<1xi64>) -> tensor<i64>
    %extracted_8 = tensor.extract %26[] : tensor<i64>
    %27 = quantum.extract %24[%extracted_8] : !quantum.reg -> !quantum.bit
    %out_qubits_9 = quantum.custom "T"() %27 : !quantum.bit
    %28 = quantum.insert %24[%extracted_8], %out_qubits_9 : !quantum.reg, !quantum.bit
    %29 = stablehlo.slice %cst [1:2] : (tensor<3xi64>) -> tensor<1xi64>
    %30 = stablehlo.reshape %29 : (tensor<1xi64>) -> tensor<i64>
    %31 = stablehlo.slice %cst [2:3] : (tensor<3xi64>) -> tensor<1xi64>
    %32 = stablehlo.reshape %31 : (tensor<1xi64>) -> tensor<i64>
    %extracted_10 = tensor.extract %30[] : tensor<i64>
    %extracted_11 = tensor.extract %32[] : tensor<i64>
    %33 = quantum.extract %28[%extracted_10] : !quantum.reg -> !quantum.bit
    %34 = quantum.extract %28[%extracted_11] : !quantum.reg -> !quantum.bit
    %out_qubits_12:2 = quantum.custom "CNOT"() %33, %34 : !quantum.bit, !quantum.bit
    %35 = quantum.insert %28[%extracted_10], %out_qubits_12#0 : !quantum.reg, !quantum.bit
    %36 = quantum.insert %35[%extracted_11], %out_qubits_12#1 : !quantum.reg, !quantum.bit
    %37 = stablehlo.slice %cst [2:3] : (tensor<3xi64>) -> tensor<1xi64>
    %38 = stablehlo.reshape %37 : (tensor<1xi64>) -> tensor<i64>
    %extracted_13 = tensor.extract %38[] : tensor<i64>
    %39 = quantum.extract %36[%extracted_13] : !quantum.reg -> !quantum.bit
    %out_qubits_14 = quantum.custom "T"() %39 adj : !quantum.bit
    %40 = quantum.insert %36[%extracted_13], %out_qubits_14 : !quantum.reg, !quantum.bit
    %41 = stablehlo.slice %cst [0:1] : (tensor<3xi64>) -> tensor<1xi64>
    %42 = stablehlo.reshape %41 : (tensor<1xi64>) -> tensor<i64>
    %43 = stablehlo.slice %cst [2:3] : (tensor<3xi64>) -> tensor<1xi64>
    %44 = stablehlo.reshape %43 : (tensor<1xi64>) -> tensor<i64>
    %extracted_15 = tensor.extract %42[] : tensor<i64>
    %extracted_16 = tensor.extract %44[] : tensor<i64>
    %45 = quantum.extract %40[%extracted_15] : !quantum.reg -> !quantum.bit
    %46 = quantum.extract %40[%extracted_16] : !quantum.reg -> !quantum.bit
    %out_qubits_17:2 = quantum.custom "CNOT"() %45, %46 : !quantum.bit, !quantum.bit
    %47 = quantum.insert %40[%extracted_15], %out_qubits_17#0 : !quantum.reg, !quantum.bit
    %48 = quantum.insert %47[%extracted_16], %out_qubits_17#1 : !quantum.reg, !quantum.bit
    %49 = stablehlo.slice %cst [2:3] : (tensor<3xi64>) -> tensor<1xi64>
    %50 = stablehlo.reshape %49 : (tensor<1xi64>) -> tensor<i64>
    %extracted_18 = tensor.extract %50[] : tensor<i64>
    %51 = quantum.extract %48[%extracted_18] : !quantum.reg -> !quantum.bit
    %out_qubits_19 = quantum.custom "T"() %51 : !quantum.bit
    %52 = quantum.insert %48[%extracted_18], %out_qubits_19 : !quantum.reg, !quantum.bit
    %53 = stablehlo.slice %cst [1:2] : (tensor<3xi64>) -> tensor<1xi64>
    %54 = stablehlo.reshape %53 : (tensor<1xi64>) -> tensor<i64>
    %extracted_20 = tensor.extract %54[] : tensor<i64>
    %55 = quantum.extract %52[%extracted_20] : !quantum.reg -> !quantum.bit
    %out_qubits_21 = quantum.custom "T"() %55 : !quantum.bit
    %56 = quantum.insert %52[%extracted_20], %out_qubits_21 : !quantum.reg, !quantum.bit
    %57 = stablehlo.slice %cst [0:1] : (tensor<3xi64>) -> tensor<1xi64>
    %58 = stablehlo.reshape %57 : (tensor<1xi64>) -> tensor<i64>
    %59 = stablehlo.slice %cst [1:2] : (tensor<3xi64>) -> tensor<1xi64>
    %60 = stablehlo.reshape %59 : (tensor<1xi64>) -> tensor<i64>
    %extracted_22 = tensor.extract %58[] : tensor<i64>
    %extracted_23 = tensor.extract %60[] : tensor<i64>
    %61 = quantum.extract %56[%extracted_22] : !quantum.reg -> !quantum.bit
    %62 = quantum.extract %56[%extracted_23] : !quantum.reg -> !quantum.bit
    %out_qubits_24:2 = quantum.custom "CNOT"() %61, %62 : !quantum.bit, !quantum.bit
    %63 = quantum.insert %56[%extracted_22], %out_qubits_24#0 : !quantum.reg, !quantum.bit
    %64 = quantum.insert %63[%extracted_23], %out_qubits_24#1 : !quantum.reg, !quantum.bit
    %65 = stablehlo.slice %cst [2:3] : (tensor<3xi64>) -> tensor<1xi64>
    %66 = stablehlo.reshape %65 : (tensor<1xi64>) -> tensor<i64>
    %extracted_25 = tensor.extract %66[] : tensor<i64>
    %67 = quantum.extract %64[%extracted_25] : !quantum.reg -> !quantum.bit
    %out_qubits_26 = quantum.custom "Hadamard"() %67 : !quantum.bit
    %68 = quantum.insert %64[%extracted_25], %out_qubits_26 : !quantum.reg, !quantum.bit
    %69 = stablehlo.slice %cst [0:1] : (tensor<3xi64>) -> tensor<1xi64>
    %70 = stablehlo.reshape %69 : (tensor<1xi64>) -> tensor<i64>
    %extracted_27 = tensor.extract %70[] : tensor<i64>
    %71 = quantum.extract %68[%extracted_27] : !quantum.reg -> !quantum.bit
    %out_qubits_28 = quantum.custom "T"() %71 : !quantum.bit
    %72 = quantum.insert %68[%extracted_27], %out_qubits_28 : !quantum.reg, !quantum.bit
    %73 = stablehlo.slice %cst [1:2] : (tensor<3xi64>) -> tensor<1xi64>
    %74 = stablehlo.reshape %73 : (tensor<1xi64>) -> tensor<i64>
    %extracted_29 = tensor.extract %74[] : tensor<i64>
    %75 = quantum.extract %72[%extracted_29] : !quantum.reg -> !quantum.bit
    %out_qubits_30 = quantum.custom "T"() %75 adj : !quantum.bit
    %76 = quantum.insert %72[%extracted_29], %out_qubits_30 : !quantum.reg, !quantum.bit
    %77 = stablehlo.slice %cst [0:1] : (tensor<3xi64>) -> tensor<1xi64>
    %78 = stablehlo.reshape %77 : (tensor<1xi64>) -> tensor<i64>
    %79 = stablehlo.slice %cst [1:2] : (tensor<3xi64>) -> tensor<1xi64>
    %80 = stablehlo.reshape %79 : (tensor<1xi64>) -> tensor<i64>
    %extracted_31 = tensor.extract %78[] : tensor<i64>
    %extracted_32 = tensor.extract %80[] : tensor<i64>
    %81 = quantum.extract %76[%extracted_31] : !quantum.reg -> !quantum.bit
    %82 = quantum.extract %76[%extracted_32] : !quantum.reg -> !quantum.bit
    %out_qubits_33:2 = quantum.custom "CNOT"() %81, %82 : !quantum.bit, !quantum.bit
    %83 = quantum.insert %76[%extracted_31], %out_qubits_33#0 : !quantum.reg, !quantum.bit
    %84 = quantum.insert %83[%extracted_32], %out_qubits_33#1 : !quantum.reg, !quantum.bit
    %85 = quantum.extract %84[ 0] : !quantum.reg -> !quantum.bit
    %out_qubits_34 = quantum.custom "Hadamard"() %85 : !quantum.bit
    %out_qubits_35 = quantum.custom "Hadamard"() %out_qubits_34 : !quantum.bit
    %86 = quantum.namedobs %out_qubits_35[ PauliZ] : !quantum.obs
    %87 = quantum.insert %84[ 0], %out_qubits_35 : !quantum.reg, !quantum.bit
    %88 = quantum.expval %86 : f64
    %from_elements = tensor.from_elements %88 : tensor<f64>
    quantum.dealloc %87 : !quantum.reg
    quantum.device_release
    return %from_elements : tensor<f64>
  }
  func.func private @"__builtin__toffoli_Toffoli{}{wires:3}{}"(%arg0: tensor<3xi64>, %arg1: !quantum.reg) -> !quantum.reg attributes {frontend_name = "_toffoli", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"Adjoint(T){}{wires:1}{}" = 3 : i64, "CNOT{}{wires:2}{}" = 6 : i64, "Hadamard{}{wires:1}{}" = 2 : i64, "T{}{wires:1}{}" = 4 : i64}}, target_gate = "Toffoli{}{wires:3}{}"} {
    %0 = stablehlo.slice %arg0 [2:3] : (tensor<3xi64>) -> tensor<1xi64>
    %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
    %extracted = tensor.extract %1[] : tensor<i64>
    %2 = quantum.extract %arg1[%extracted] : !quantum.reg -> !quantum.bit
    %out_qubits = quantum.custom "Hadamard"() %2 : !quantum.bit
    %3 = quantum.insert %arg1[%extracted], %out_qubits : !quantum.reg, !quantum.bit
    %4 = stablehlo.slice %arg0 [1:2] : (tensor<3xi64>) -> tensor<1xi64>
    %5 = stablehlo.reshape %4 : (tensor<1xi64>) -> tensor<i64>
    %6 = stablehlo.slice %arg0 [2:3] : (tensor<3xi64>) -> tensor<1xi64>
    %7 = stablehlo.reshape %6 : (tensor<1xi64>) -> tensor<i64>
    %extracted_0 = tensor.extract %5[] : tensor<i64>
    %extracted_1 = tensor.extract %7[] : tensor<i64>
    %8 = quantum.extract %3[%extracted_0] : !quantum.reg -> !quantum.bit
    %9 = quantum.extract %3[%extracted_1] : !quantum.reg -> !quantum.bit
    %out_qubits_2:2 = quantum.custom "CNOT"() %8, %9 : !quantum.bit, !quantum.bit
    %10 = quantum.insert %3[%extracted_0], %out_qubits_2#0 : !quantum.reg, !quantum.bit
    %11 = quantum.insert %10[%extracted_1], %out_qubits_2#1 : !quantum.reg, !quantum.bit
    %12 = stablehlo.slice %arg0 [2:3] : (tensor<3xi64>) -> tensor<1xi64>
    %13 = stablehlo.reshape %12 : (tensor<1xi64>) -> tensor<i64>
    %extracted_3 = tensor.extract %13[] : tensor<i64>
    %14 = quantum.extract %11[%extracted_3] : !quantum.reg -> !quantum.bit
    %out_qubits_4 = quantum.custom "T"() %14 adj : !quantum.bit
    %15 = quantum.insert %11[%extracted_3], %out_qubits_4 : !quantum.reg, !quantum.bit
    %16 = stablehlo.slice %arg0 [0:1] : (tensor<3xi64>) -> tensor<1xi64>
    %17 = stablehlo.reshape %16 : (tensor<1xi64>) -> tensor<i64>
    %18 = stablehlo.slice %arg0 [2:3] : (tensor<3xi64>) -> tensor<1xi64>
    %19 = stablehlo.reshape %18 : (tensor<1xi64>) -> tensor<i64>
    %extracted_5 = tensor.extract %17[] : tensor<i64>
    %extracted_6 = tensor.extract %19[] : tensor<i64>
    %20 = quantum.extract %15[%extracted_5] : !quantum.reg -> !quantum.bit
    %21 = quantum.extract %15[%extracted_6] : !quantum.reg -> !quantum.bit
    %out_qubits_7:2 = quantum.custom "CNOT"() %20, %21 : !quantum.bit, !quantum.bit
    %22 = quantum.insert %15[%extracted_5], %out_qubits_7#0 : !quantum.reg, !quantum.bit
    %23 = quantum.insert %22[%extracted_6], %out_qubits_7#1 : !quantum.reg, !quantum.bit
    %24 = stablehlo.slice %arg0 [2:3] : (tensor<3xi64>) -> tensor<1xi64>
    %25 = stablehlo.reshape %24 : (tensor<1xi64>) -> tensor<i64>
    %extracted_8 = tensor.extract %25[] : tensor<i64>
    %26 = quantum.extract %23[%extracted_8] : !quantum.reg -> !quantum.bit
    %out_qubits_9 = quantum.custom "T"() %26 : !quantum.bit
    %27 = quantum.insert %23[%extracted_8], %out_qubits_9 : !quantum.reg, !quantum.bit
    %28 = stablehlo.slice %arg0 [1:2] : (tensor<3xi64>) -> tensor<1xi64>
    %29 = stablehlo.reshape %28 : (tensor<1xi64>) -> tensor<i64>
    %30 = stablehlo.slice %arg0 [2:3] : (tensor<3xi64>) -> tensor<1xi64>
    %31 = stablehlo.reshape %30 : (tensor<1xi64>) -> tensor<i64>
    %extracted_10 = tensor.extract %29[] : tensor<i64>
    %extracted_11 = tensor.extract %31[] : tensor<i64>
    %32 = quantum.extract %27[%extracted_10] : !quantum.reg -> !quantum.bit
    %33 = quantum.extract %27[%extracted_11] : !quantum.reg -> !quantum.bit
    %out_qubits_12:2 = quantum.custom "CNOT"() %32, %33 : !quantum.bit, !quantum.bit
    %34 = quantum.insert %27[%extracted_10], %out_qubits_12#0 : !quantum.reg, !quantum.bit
    %35 = quantum.insert %34[%extracted_11], %out_qubits_12#1 : !quantum.reg, !quantum.bit
    %36 = stablehlo.slice %arg0 [2:3] : (tensor<3xi64>) -> tensor<1xi64>
    %37 = stablehlo.reshape %36 : (tensor<1xi64>) -> tensor<i64>
    %extracted_13 = tensor.extract %37[] : tensor<i64>
    %38 = quantum.extract %35[%extracted_13] : !quantum.reg -> !quantum.bit
    %out_qubits_14 = quantum.custom "T"() %38 adj : !quantum.bit
    %39 = quantum.insert %35[%extracted_13], %out_qubits_14 : !quantum.reg, !quantum.bit
    %40 = stablehlo.slice %arg0 [0:1] : (tensor<3xi64>) -> tensor<1xi64>
    %41 = stablehlo.reshape %40 : (tensor<1xi64>) -> tensor<i64>
    %42 = stablehlo.slice %arg0 [2:3] : (tensor<3xi64>) -> tensor<1xi64>
    %43 = stablehlo.reshape %42 : (tensor<1xi64>) -> tensor<i64>
    %extracted_15 = tensor.extract %41[] : tensor<i64>
    %extracted_16 = tensor.extract %43[] : tensor<i64>
    %44 = quantum.extract %39[%extracted_15] : !quantum.reg -> !quantum.bit
    %45 = quantum.extract %39[%extracted_16] : !quantum.reg -> !quantum.bit
    %out_qubits_17:2 = quantum.custom "CNOT"() %44, %45 : !quantum.bit, !quantum.bit
    %46 = quantum.insert %39[%extracted_15], %out_qubits_17#0 : !quantum.reg, !quantum.bit
    %47 = quantum.insert %46[%extracted_16], %out_qubits_17#1 : !quantum.reg, !quantum.bit
    %48 = stablehlo.slice %arg0 [2:3] : (tensor<3xi64>) -> tensor<1xi64>
    %49 = stablehlo.reshape %48 : (tensor<1xi64>) -> tensor<i64>
    %extracted_18 = tensor.extract %49[] : tensor<i64>
    %50 = quantum.extract %47[%extracted_18] : !quantum.reg -> !quantum.bit
    %out_qubits_19 = quantum.custom "T"() %50 : !quantum.bit
    %51 = quantum.insert %47[%extracted_18], %out_qubits_19 : !quantum.reg, !quantum.bit
    %52 = stablehlo.slice %arg0 [1:2] : (tensor<3xi64>) -> tensor<1xi64>
    %53 = stablehlo.reshape %52 : (tensor<1xi64>) -> tensor<i64>
    %extracted_20 = tensor.extract %53[] : tensor<i64>
    %54 = quantum.extract %51[%extracted_20] : !quantum.reg -> !quantum.bit
    %out_qubits_21 = quantum.custom "T"() %54 : !quantum.bit
    %55 = quantum.insert %51[%extracted_20], %out_qubits_21 : !quantum.reg, !quantum.bit
    %56 = stablehlo.slice %arg0 [0:1] : (tensor<3xi64>) -> tensor<1xi64>
    %57 = stablehlo.reshape %56 : (tensor<1xi64>) -> tensor<i64>
    %58 = stablehlo.slice %arg0 [1:2] : (tensor<3xi64>) -> tensor<1xi64>
    %59 = stablehlo.reshape %58 : (tensor<1xi64>) -> tensor<i64>
    %extracted_22 = tensor.extract %57[] : tensor<i64>
    %extracted_23 = tensor.extract %59[] : tensor<i64>
    %60 = quantum.extract %55[%extracted_22] : !quantum.reg -> !quantum.bit
    %61 = quantum.extract %55[%extracted_23] : !quantum.reg -> !quantum.bit
    %out_qubits_24:2 = quantum.custom "CNOT"() %60, %61 : !quantum.bit, !quantum.bit
    %62 = quantum.insert %55[%extracted_22], %out_qubits_24#0 : !quantum.reg, !quantum.bit
    %63 = quantum.insert %62[%extracted_23], %out_qubits_24#1 : !quantum.reg, !quantum.bit
    %64 = stablehlo.slice %arg0 [2:3] : (tensor<3xi64>) -> tensor<1xi64>
    %65 = stablehlo.reshape %64 : (tensor<1xi64>) -> tensor<i64>
    %extracted_25 = tensor.extract %65[] : tensor<i64>
    %66 = quantum.extract %63[%extracted_25] : !quantum.reg -> !quantum.bit
    %out_qubits_26 = quantum.custom "Hadamard"() %66 : !quantum.bit
    %67 = quantum.insert %63[%extracted_25], %out_qubits_26 : !quantum.reg, !quantum.bit
    %68 = stablehlo.slice %arg0 [0:1] : (tensor<3xi64>) -> tensor<1xi64>
    %69 = stablehlo.reshape %68 : (tensor<1xi64>) -> tensor<i64>
    %extracted_27 = tensor.extract %69[] : tensor<i64>
    %70 = quantum.extract %67[%extracted_27] : !quantum.reg -> !quantum.bit
    %out_qubits_28 = quantum.custom "T"() %70 : !quantum.bit
    %71 = quantum.insert %67[%extracted_27], %out_qubits_28 : !quantum.reg, !quantum.bit
    %72 = stablehlo.slice %arg0 [1:2] : (tensor<3xi64>) -> tensor<1xi64>
    %73 = stablehlo.reshape %72 : (tensor<1xi64>) -> tensor<i64>
    %extracted_29 = tensor.extract %73[] : tensor<i64>
    %74 = quantum.extract %71[%extracted_29] : !quantum.reg -> !quantum.bit
    %out_qubits_30 = quantum.custom "T"() %74 adj : !quantum.bit
    %75 = quantum.insert %71[%extracted_29], %out_qubits_30 : !quantum.reg, !quantum.bit
    %76 = stablehlo.slice %arg0 [0:1] : (tensor<3xi64>) -> tensor<1xi64>
    %77 = stablehlo.reshape %76 : (tensor<1xi64>) -> tensor<i64>
    %78 = stablehlo.slice %arg0 [1:2] : (tensor<3xi64>) -> tensor<1xi64>
    %79 = stablehlo.reshape %78 : (tensor<1xi64>) -> tensor<i64>
    %extracted_31 = tensor.extract %77[] : tensor<i64>
    %extracted_32 = tensor.extract %79[] : tensor<i64>
    %80 = quantum.extract %75[%extracted_31] : !quantum.reg -> !quantum.bit
    %81 = quantum.extract %75[%extracted_32] : !quantum.reg -> !quantum.bit
    %out_qubits_33:2 = quantum.custom "CNOT"() %80, %81 : !quantum.bit, !quantum.bit
    %82 = quantum.insert %75[%extracted_31], %out_qubits_33#0 : !quantum.reg, !quantum.bit
    %83 = quantum.insert %82[%extracted_32], %out_qubits_33#1 : !quantum.reg, !quantum.bit
    return %83 : !quantum.reg
  }
  func.func private @"__builtin__toffoli_to_ppr_Toffoli{}{wires:3}{}"(%arg0: tensor<3xi64>, %arg1: !quantum.reg) -> !quantum.reg attributes {frontend_name = "_toffoli_to_ppr", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"GlobalPhase{phi:[f64]}{}{}" = 1 : i64, "PPR{}{wires:1}{angle_denominator = 8 : i64, pauli_word = \22X\22}" = 1 : i64, "PPR{}{wires:1}{angle_denominator = 8 : i64, pauli_word = \22Z\22}" = 2 : i64, "PPR{}{wires:2}{angle_denominator = -8 : si64, pauli_word = \22ZX\22}" = 2 : i64, "PPR{}{wires:2}{angle_denominator = -8 : si64, pauli_word = \22ZZ\22}" = 1 : i64, "PPR{}{wires:3}{angle_denominator = 8 : i64, pauli_word = \22ZZX\22}" = 1 : i64}}, target_gate = "Toffoli{}{wires:3}{}"} {
    %cst = arith.constant -0.39269908169872414 : f64
    %0 = stablehlo.slice %arg0 [0:2] : (tensor<3xi64>) -> tensor<2xi64>
    %1 = stablehlo.slice %0 [0:1] : (tensor<2xi64>) -> tensor<1xi64>
    %2 = stablehlo.reshape %1 : (tensor<1xi64>) -> tensor<i64>
    %3 = stablehlo.slice %0 [1:2] : (tensor<2xi64>) -> tensor<1xi64>
    %4 = stablehlo.reshape %3 : (tensor<1xi64>) -> tensor<i64>
    %extracted = tensor.extract %2[] : tensor<i64>
    %extracted_0 = tensor.extract %4[] : tensor<i64>
    %5 = quantum.extract %arg1[%extracted] : !quantum.reg -> !quantum.bit
    %6 = quantum.extract %arg1[%extracted_0] : !quantum.reg -> !quantum.bit
    %out_qubits:2 = quantum.operator "PPR"() qubits(%5, %6)
      static_data = {angle_denominator = -8 : si64, pauli_word = "ZZ"}
      qubit_map = {wires = [0, 1]}
    %7 = quantum.insert %arg1[%extracted], %out_qubits#0 : !quantum.reg, !quantum.bit
    %8 = quantum.insert %7[%extracted_0], %out_qubits#1 : !quantum.reg, !quantum.bit
    %9 = stablehlo.slice %arg0 [0:1] : (tensor<3xi64>) -> tensor<1xi64>
    %10 = stablehlo.reshape %9 : (tensor<1xi64>) -> tensor<i64>
    %11 = stablehlo.slice %arg0 [2:3] : (tensor<3xi64>) -> tensor<1xi64>
    %12 = stablehlo.reshape %11 : (tensor<1xi64>) -> tensor<i64>
    %extracted_1 = tensor.extract %10[] : tensor<i64>
    %extracted_2 = tensor.extract %12[] : tensor<i64>
    %13 = quantum.extract %8[%extracted_1] : !quantum.reg -> !quantum.bit
    %14 = quantum.extract %8[%extracted_2] : !quantum.reg -> !quantum.bit
    %out_qubits_3:2 = quantum.operator "PPR"() qubits(%13, %14)
      static_data = {angle_denominator = -8 : si64, pauli_word = "ZX"}
      qubit_map = {wires = [0, 1]}
    %15 = quantum.insert %8[%extracted_1], %out_qubits_3#0 : !quantum.reg, !quantum.bit
    %16 = quantum.insert %15[%extracted_2], %out_qubits_3#1 : !quantum.reg, !quantum.bit
    %17 = stablehlo.slice %arg0 [1:3] : (tensor<3xi64>) -> tensor<2xi64>
    %18 = stablehlo.slice %17 [0:1] : (tensor<2xi64>) -> tensor<1xi64>
    %19 = stablehlo.reshape %18 : (tensor<1xi64>) -> tensor<i64>
    %20 = stablehlo.slice %17 [1:2] : (tensor<2xi64>) -> tensor<1xi64>
    %21 = stablehlo.reshape %20 : (tensor<1xi64>) -> tensor<i64>
    %extracted_4 = tensor.extract %19[] : tensor<i64>
    %extracted_5 = tensor.extract %21[] : tensor<i64>
    %22 = quantum.extract %16[%extracted_4] : !quantum.reg -> !quantum.bit
    %23 = quantum.extract %16[%extracted_5] : !quantum.reg -> !quantum.bit
    %out_qubits_6:2 = quantum.operator "PPR"() qubits(%22, %23)
      static_data = {angle_denominator = -8 : si64, pauli_word = "ZX"}
      qubit_map = {wires = [0, 1]}
    %24 = quantum.insert %16[%extracted_4], %out_qubits_6#0 : !quantum.reg, !quantum.bit
    %25 = quantum.insert %24[%extracted_5], %out_qubits_6#1 : !quantum.reg, !quantum.bit
    %26 = stablehlo.slice %arg0 [0:1] : (tensor<3xi64>) -> tensor<1xi64>
    %27 = stablehlo.reshape %26 : (tensor<1xi64>) -> tensor<i64>
    %28 = stablehlo.slice %arg0 [1:2] : (tensor<3xi64>) -> tensor<1xi64>
    %29 = stablehlo.reshape %28 : (tensor<1xi64>) -> tensor<i64>
    %30 = stablehlo.slice %arg0 [2:3] : (tensor<3xi64>) -> tensor<1xi64>
    %31 = stablehlo.reshape %30 : (tensor<1xi64>) -> tensor<i64>
    %extracted_7 = tensor.extract %27[] : tensor<i64>
    %extracted_8 = tensor.extract %29[] : tensor<i64>
    %extracted_9 = tensor.extract %31[] : tensor<i64>
    %32 = quantum.extract %25[%extracted_7] : !quantum.reg -> !quantum.bit
    %33 = quantum.extract %25[%extracted_8] : !quantum.reg -> !quantum.bit
    %34 = quantum.extract %25[%extracted_9] : !quantum.reg -> !quantum.bit
    %out_qubits_10:3 = quantum.operator "PPR"() qubits(%32, %33, %34)
      static_data = {angle_denominator = 8 : i64, pauli_word = "ZZX"}
      qubit_map = {wires = [0, 1, 2]}
    %35 = quantum.insert %25[%extracted_7], %out_qubits_10#0 : !quantum.reg, !quantum.bit
    %36 = quantum.insert %35[%extracted_8], %out_qubits_10#1 : !quantum.reg, !quantum.bit
    %37 = quantum.insert %36[%extracted_9], %out_qubits_10#2 : !quantum.reg, !quantum.bit
    %38 = stablehlo.slice %arg0 [2:3] : (tensor<3xi64>) -> tensor<1xi64>
    %39 = stablehlo.reshape %38 : (tensor<1xi64>) -> tensor<i64>
    %extracted_11 = tensor.extract %39[] : tensor<i64>
    %40 = quantum.extract %37[%extracted_11] : !quantum.reg -> !quantum.bit
    %out_qubits_12 = quantum.operator "PPR"() qubits(%40)
      static_data = {angle_denominator = 8 : i64, pauli_word = "X"}
      qubit_map = {wires = [0]}
    %41 = quantum.insert %37[%extracted_11], %out_qubits_12 : !quantum.reg, !quantum.bit
    %42 = stablehlo.slice %arg0 [1:2] : (tensor<3xi64>) -> tensor<1xi64>
    %43 = stablehlo.reshape %42 : (tensor<1xi64>) -> tensor<i64>
    %extracted_13 = tensor.extract %43[] : tensor<i64>
    %44 = quantum.extract %41[%extracted_13] : !quantum.reg -> !quantum.bit
    %out_qubits_14 = quantum.operator "PPR"() qubits(%44)
      static_data = {angle_denominator = 8 : i64, pauli_word = "Z"}
      qubit_map = {wires = [0]}
    %45 = quantum.insert %41[%extracted_13], %out_qubits_14 : !quantum.reg, !quantum.bit
    %46 = stablehlo.slice %arg0 [0:1] : (tensor<3xi64>) -> tensor<1xi64>
    %47 = stablehlo.reshape %46 : (tensor<1xi64>) -> tensor<i64>
    %extracted_15 = tensor.extract %47[] : tensor<i64>
    %48 = quantum.extract %45[%extracted_15] : !quantum.reg -> !quantum.bit
    %out_qubits_16 = quantum.operator "PPR"() qubits(%48)
      static_data = {angle_denominator = 8 : i64, pauli_word = "Z"}
      qubit_map = {wires = [0]}
    %49 = quantum.insert %45[%extracted_15], %out_qubits_16 : !quantum.reg, !quantum.bit
    quantum.gphase(%cst)
    return %49 : !quantum.reg
  }
  func.func private @"__builtin__hadamard_to_rz_rx_Hadamard{}{wires:1}{}"(%arg0: tensor<1xi64>, %arg1: !quantum.reg) -> !quantum.reg attributes {frontend_name = "_hadamard_to_rz_rx", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"GlobalPhase{phi:[f64]}{}{}" = 1 : i64, "RX{0:[f64]}{wires:1}{}" = 1 : i64, "RZ{0:[f64]}{wires:1}{}" = 2 : i64}}, target_gate = "Hadamard{}{wires:1}{}"} {
    %cst = arith.constant -1.5707963267948966 : f64
    %cst_0 = arith.constant 1.5707963267948966 : f64
    %0 = stablehlo.slice %arg0 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
    %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
    %extracted = tensor.extract %1[] : tensor<i64>
    %2 = quantum.extract %arg1[%extracted] : !quantum.reg -> !quantum.bit
    %out_qubits = quantum.custom "RZ"(%cst_0) %2 : !quantum.bit
    %3 = quantum.insert %arg1[%extracted], %out_qubits : !quantum.reg, !quantum.bit
    %4 = stablehlo.slice %arg0 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
    %5 = stablehlo.reshape %4 : (tensor<1xi64>) -> tensor<i64>
    %extracted_1 = tensor.extract %5[] : tensor<i64>
    %6 = quantum.extract %3[%extracted_1] : !quantum.reg -> !quantum.bit
    %out_qubits_2 = quantum.custom "RX"(%cst_0) %6 : !quantum.bit
    %7 = quantum.insert %3[%extracted_1], %out_qubits_2 : !quantum.reg, !quantum.bit
    %8 = stablehlo.slice %arg0 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
    %9 = stablehlo.reshape %8 : (tensor<1xi64>) -> tensor<i64>
    %extracted_3 = tensor.extract %9[] : tensor<i64>
    %10 = quantum.extract %7[%extracted_3] : !quantum.reg -> !quantum.bit
    %out_qubits_4 = quantum.custom "RZ"(%cst_0) %10 : !quantum.bit
    %11 = quantum.insert %7[%extracted_3], %out_qubits_4 : !quantum.reg, !quantum.bit
    quantum.gphase(%cst)
    return %11 : !quantum.reg
  }
  func.func private @"__builtin__hadamard_to_rz_ry_Hadamard{}{wires:1}{}"(%arg0: tensor<1xi64>, %arg1: !quantum.reg) -> !quantum.reg attributes {frontend_name = "_hadamard_to_rz_ry", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"GlobalPhase{phi:[f64]}{}{}" = 1 : i64, "RY{0:[f64]}{wires:1}{}" = 1 : i64, "RZ{0:[f64]}{wires:1}{}" = 1 : i64}}, target_gate = "Hadamard{}{wires:1}{}"} {
    %cst = arith.constant -1.5707963267948966 : f64
    %cst_0 = arith.constant 1.5707963267948966 : f64
    %cst_1 = arith.constant 3.1415926535897931 : f64
    %0 = stablehlo.slice %arg0 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
    %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
    %extracted = tensor.extract %1[] : tensor<i64>
    %2 = quantum.extract %arg1[%extracted] : !quantum.reg -> !quantum.bit
    %out_qubits = quantum.custom "RZ"(%cst_1) %2 : !quantum.bit
    %3 = quantum.insert %arg1[%extracted], %out_qubits : !quantum.reg, !quantum.bit
    %4 = stablehlo.slice %arg0 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
    %5 = stablehlo.reshape %4 : (tensor<1xi64>) -> tensor<i64>
    %extracted_2 = tensor.extract %5[] : tensor<i64>
    %6 = quantum.extract %3[%extracted_2] : !quantum.reg -> !quantum.bit
    %out_qubits_3 = quantum.custom "RY"(%cst_0) %6 : !quantum.bit
    %7 = quantum.insert %3[%extracted_2], %out_qubits_3 : !quantum.reg, !quantum.bit
    quantum.gphase(%cst)
    return %7 : !quantum.reg
  }
  func.func private @"__builtin__cnot_to_cz_h_CNOT{}{wires:2}{}"(%arg0: tensor<2xi64>, %arg1: !quantum.reg) -> !quantum.reg attributes {frontend_name = "_cnot_to_cz_h", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"CZ{}{wires:2}{}" = 1 : i64, "Hadamard{}{wires:1}{}" = 2 : i64}}, target_gate = "CNOT{}{wires:2}{}"} {
    %0 = stablehlo.slice %arg0 [1:2] : (tensor<2xi64>) -> tensor<1xi64>
    %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
    %extracted = tensor.extract %1[] : tensor<i64>
    %2 = quantum.extract %arg1[%extracted] : !quantum.reg -> !quantum.bit
    %out_qubits = quantum.custom "Hadamard"() %2 : !quantum.bit
    %3 = quantum.insert %arg1[%extracted], %out_qubits : !quantum.reg, !quantum.bit
    %4 = stablehlo.slice %arg0 [0:1] : (tensor<2xi64>) -> tensor<1xi64>
    %5 = stablehlo.reshape %4 : (tensor<1xi64>) -> tensor<i64>
    %6 = stablehlo.slice %arg0 [1:2] : (tensor<2xi64>) -> tensor<1xi64>
    %7 = stablehlo.reshape %6 : (tensor<1xi64>) -> tensor<i64>
    %extracted_0 = tensor.extract %5[] : tensor<i64>
    %extracted_1 = tensor.extract %7[] : tensor<i64>
    %8 = quantum.extract %3[%extracted_0] : !quantum.reg -> !quantum.bit
    %9 = quantum.extract %3[%extracted_1] : !quantum.reg -> !quantum.bit
    %out_qubits_2:2 = quantum.custom "CZ"() %8, %9 : !quantum.bit, !quantum.bit
    %10 = quantum.insert %3[%extracted_0], %out_qubits_2#0 : !quantum.reg, !quantum.bit
    %11 = quantum.insert %10[%extracted_1], %out_qubits_2#1 : !quantum.reg, !quantum.bit
    %12 = stablehlo.slice %arg0 [1:2] : (tensor<2xi64>) -> tensor<1xi64>
    %13 = stablehlo.reshape %12 : (tensor<1xi64>) -> tensor<i64>
    %extracted_3 = tensor.extract %13[] : tensor<i64>
    %14 = quantum.extract %11[%extracted_3] : !quantum.reg -> !quantum.bit
    %out_qubits_4 = quantum.custom "Hadamard"() %14 : !quantum.bit
    %15 = quantum.insert %11[%extracted_3], %out_qubits_4 : !quantum.reg, !quantum.bit
    return %15 : !quantum.reg
  }
  func.func private @"__builtin__cnot_to_ppr_CNOT{}{wires:2}{}"(%arg0: tensor<2xi64>, %arg1: !quantum.reg) -> !quantum.reg attributes {frontend_name = "_cnot_to_ppr", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"GlobalPhase{phi:[f64]}{}{}" = 1 : i64, "PPR{}{wires:1}{angle_denominator = -4 : si64, pauli_word = \22X\22}" = 1 : i64, "PPR{}{wires:1}{angle_denominator = -4 : si64, pauli_word = \22Z\22}" = 1 : i64, "PPR{}{wires:2}{angle_denominator = 4 : i64, pauli_word = \22ZX\22}" = 1 : i64}}, target_gate = "CNOT{}{wires:2}{}"} {
    %cst = arith.constant 0.78539816339744828 : f64
    %0 = stablehlo.slice %arg0 [0:1] : (tensor<2xi64>) -> tensor<1xi64>
    %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
    %extracted = tensor.extract %1[] : tensor<i64>
    %2 = quantum.extract %arg1[%extracted] : !quantum.reg -> !quantum.bit
    %out_qubits = quantum.operator "PPR"() qubits(%2)
      static_data = {angle_denominator = -4 : si64, pauli_word = "Z"}
      qubit_map = {wires = [0]}
    %3 = quantum.insert %arg1[%extracted], %out_qubits : !quantum.reg, !quantum.bit
    %4 = stablehlo.slice %arg0 [1:2] : (tensor<2xi64>) -> tensor<1xi64>
    %5 = stablehlo.reshape %4 : (tensor<1xi64>) -> tensor<i64>
    %extracted_0 = tensor.extract %5[] : tensor<i64>
    %6 = quantum.extract %3[%extracted_0] : !quantum.reg -> !quantum.bit
    %out_qubits_1 = quantum.operator "PPR"() qubits(%6)
      static_data = {angle_denominator = -4 : si64, pauli_word = "X"}
      qubit_map = {wires = [0]}
    %7 = quantum.insert %3[%extracted_0], %out_qubits_1 : !quantum.reg, !quantum.bit
    %8 = stablehlo.slice %arg0 [0:1] : (tensor<2xi64>) -> tensor<1xi64>
    %9 = stablehlo.reshape %8 : (tensor<1xi64>) -> tensor<i64>
    %10 = stablehlo.slice %arg0 [1:2] : (tensor<2xi64>) -> tensor<1xi64>
    %11 = stablehlo.reshape %10 : (tensor<1xi64>) -> tensor<i64>
    %extracted_2 = tensor.extract %9[] : tensor<i64>
    %extracted_3 = tensor.extract %11[] : tensor<i64>
    %12 = quantum.extract %7[%extracted_2] : !quantum.reg -> !quantum.bit
    %13 = quantum.extract %7[%extracted_3] : !quantum.reg -> !quantum.bit
    %out_qubits_4:2 = quantum.operator "PPR"() qubits(%12, %13)
      static_data = {angle_denominator = 4 : i64, pauli_word = "ZX"}
      qubit_map = {wires = [0, 1]}
    %14 = quantum.insert %7[%extracted_2], %out_qubits_4#0 : !quantum.reg, !quantum.bit
    %15 = quantum.insert %14[%extracted_3], %out_qubits_4#1 : !quantum.reg, !quantum.bit
    quantum.gphase(%cst)
    return %15 : !quantum.reg
  }
  func.func private @"__builtin__t_phaseshift_T{}{wires:1}{}"(%arg0: tensor<1xi64>, %arg1: !quantum.reg) -> !quantum.reg attributes {frontend_name = "_t_phaseshift", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"PhaseShift{0:[f64]}{wires:1}{}" = 1 : i64}}, target_gate = "T{}{wires:1}{}"} {
    %cst = arith.constant 0.78539816339744828 : f64
    %0 = stablehlo.slice %arg0 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
    %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
    %extracted = tensor.extract %1[] : tensor<i64>
    %2 = quantum.extract %arg1[%extracted] : !quantum.reg -> !quantum.bit
    %out_qubits = quantum.custom "PhaseShift"(%cst) %2 : !quantum.bit
    %3 = quantum.insert %arg1[%extracted], %out_qubits : !quantum.reg, !quantum.bit
    return %3 : !quantum.reg
  }
  func.func private @"__builtin__t_phaseshift_Adjoint(T){}{wires:1}{}"(%arg0: tensor<1xi64>, %arg1: !quantum.reg) -> !quantum.reg attributes {frontend_name = "_t_phaseshift", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"Adjoint(PhaseShift){0:[f64]}{wires:1}{}" = 1 : i64}}, target_gate = "Adjoint(T){}{wires:1}{}"} {
    %cst = arith.constant 0.78539816339744828 : f64
    %0 = catalyst.list_init : <f64>
    %1 = catalyst.list_init : <i64>
    %2 = catalyst.list_init : <i64>
    %3 = stablehlo.slice %arg0 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
    %4 = stablehlo.reshape %3 : (tensor<1xi64>) -> tensor<i64>
    %extracted = tensor.extract %4[] : tensor<i64>
    catalyst.list_push %extracted, %2 : <i64>
    catalyst.list_push %extracted, %2 : <i64>
    %5 = catalyst.list_pop %2 : <i64>
    %6 = quantum.extract %arg1[%5] : !quantum.reg -> !quantum.bit
    %out_qubits = quantum.custom "PhaseShift"(%cst) %6 adj : !quantum.bit
    %7 = quantum.insert %arg1[%5], %out_qubits : !quantum.reg, !quantum.bit
    %8 = catalyst.list_pop %2 : <i64>
    catalyst.list_dealloc %0 : <f64>
    catalyst.list_dealloc %1 : <i64>
    catalyst.list_dealloc %2 : <i64>
    return %7 : !quantum.reg
  }
  func.func private @"__builtin__ppr_to_paulirot_PPR{}{wires:2}{angle_denominator = -8 : si64, pauli_word = \22ZZ\22}"(%arg0: tensor<2xi64>, %arg1: !quantum.reg) -> !quantum.reg attributes {frontend_name = "_ppr_to_paulirot", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"PauliRot{theta:[f64]}{wires:2}{pauli_word = \22ZZ\22}" = 1 : i64}}, target_gate = "PPR{}{wires:2}{angle_denominator = -8 : si64, pauli_word = \22ZZ\22}"} {
    %cst = arith.constant -0.78539816339744828 : f64
    %0 = stablehlo.slice %arg0 [0:1] : (tensor<2xi64>) -> tensor<1xi64>
    %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
    %2 = stablehlo.slice %arg0 [1:2] : (tensor<2xi64>) -> tensor<1xi64>
    %3 = stablehlo.reshape %2 : (tensor<1xi64>) -> tensor<i64>
    %extracted = tensor.extract %1[] : tensor<i64>
    %extracted_0 = tensor.extract %3[] : tensor<i64>
    %4 = quantum.extract %arg1[%extracted] : !quantum.reg -> !quantum.bit
    %5 = quantum.extract %arg1[%extracted_0] : !quantum.reg -> !quantum.bit
    %out_qubits:2 = quantum.paulirot ["Z", "Z"](%cst) %4, %5 : !quantum.bit, !quantum.bit
    %6 = quantum.insert %arg1[%extracted], %out_qubits#0 : !quantum.reg, !quantum.bit
    %7 = quantum.insert %6[%extracted_0], %out_qubits#1 : !quantum.reg, !quantum.bit
    return %7 : !quantum.reg
  }
  func.func private @"__builtin__ppr_to_paulirot_PPR{}{wires:2}{angle_denominator = -8 : si64, pauli_word = \22ZX\22}"(%arg0: tensor<2xi64>, %arg1: !quantum.reg) -> !quantum.reg attributes {frontend_name = "_ppr_to_paulirot", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"PauliRot{theta:[f64]}{wires:2}{pauli_word = \22ZX\22}" = 1 : i64}}, target_gate = "PPR{}{wires:2}{angle_denominator = -8 : si64, pauli_word = \22ZX\22}"} {
    %cst = arith.constant -0.78539816339744828 : f64
    %0 = stablehlo.slice %arg0 [0:1] : (tensor<2xi64>) -> tensor<1xi64>
    %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
    %2 = stablehlo.slice %arg0 [1:2] : (tensor<2xi64>) -> tensor<1xi64>
    %3 = stablehlo.reshape %2 : (tensor<1xi64>) -> tensor<i64>
    %extracted = tensor.extract %1[] : tensor<i64>
    %extracted_0 = tensor.extract %3[] : tensor<i64>
    %4 = quantum.extract %arg1[%extracted] : !quantum.reg -> !quantum.bit
    %5 = quantum.extract %arg1[%extracted_0] : !quantum.reg -> !quantum.bit
    %out_qubits:2 = quantum.paulirot ["Z", "X"](%cst) %4, %5 : !quantum.bit, !quantum.bit
    %6 = quantum.insert %arg1[%extracted], %out_qubits#0 : !quantum.reg, !quantum.bit
    %7 = quantum.insert %6[%extracted_0], %out_qubits#1 : !quantum.reg, !quantum.bit
    return %7 : !quantum.reg
  }
  func.func private @"__builtin__ppr_to_paulirot_PPR{}{wires:3}{angle_denominator = 8 : i64, pauli_word = \22ZZX\22}"(%arg0: tensor<3xi64>, %arg1: !quantum.reg) -> !quantum.reg attributes {frontend_name = "_ppr_to_paulirot", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"PauliRot{theta:[f64]}{wires:3}{pauli_word = \22ZZX\22}" = 1 : i64}}, target_gate = "PPR{}{wires:3}{angle_denominator = 8 : i64, pauli_word = \22ZZX\22}"} {
    %cst = arith.constant 0.78539816339744828 : f64
    %0 = stablehlo.slice %arg0 [0:1] : (tensor<3xi64>) -> tensor<1xi64>
    %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
    %2 = stablehlo.slice %arg0 [1:2] : (tensor<3xi64>) -> tensor<1xi64>
    %3 = stablehlo.reshape %2 : (tensor<1xi64>) -> tensor<i64>
    %4 = stablehlo.slice %arg0 [2:3] : (tensor<3xi64>) -> tensor<1xi64>
    %5 = stablehlo.reshape %4 : (tensor<1xi64>) -> tensor<i64>
    %extracted = tensor.extract %1[] : tensor<i64>
    %extracted_0 = tensor.extract %3[] : tensor<i64>
    %extracted_1 = tensor.extract %5[] : tensor<i64>
    %6 = quantum.extract %arg1[%extracted] : !quantum.reg -> !quantum.bit
    %7 = quantum.extract %arg1[%extracted_0] : !quantum.reg -> !quantum.bit
    %8 = quantum.extract %arg1[%extracted_1] : !quantum.reg -> !quantum.bit
    %out_qubits:3 = quantum.paulirot ["Z", "Z", "X"](%cst) %6, %7, %8 : !quantum.bit, !quantum.bit, !quantum.bit
    %9 = quantum.insert %arg1[%extracted], %out_qubits#0 : !quantum.reg, !quantum.bit
    %10 = quantum.insert %9[%extracted_0], %out_qubits#1 : !quantum.reg, !quantum.bit
    %11 = quantum.insert %10[%extracted_1], %out_qubits#2 : !quantum.reg, !quantum.bit
    return %11 : !quantum.reg
  }
  func.func private @"__builtin__ppr_to_paulirot_PPR{}{wires:1}{angle_denominator = 8 : i64, pauli_word = \22X\22}"(%arg0: tensor<1xi64>, %arg1: !quantum.reg) -> !quantum.reg attributes {frontend_name = "_ppr_to_paulirot", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"PauliRot{theta:[f64]}{wires:1}{pauli_word = \22X\22}" = 1 : i64}}, target_gate = "PPR{}{wires:1}{angle_denominator = 8 : i64, pauli_word = \22X\22}"} {
    %cst = arith.constant 0.78539816339744828 : f64
    %0 = stablehlo.slice %arg0 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
    %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
    %extracted = tensor.extract %1[] : tensor<i64>
    %2 = quantum.extract %arg1[%extracted] : !quantum.reg -> !quantum.bit
    %out_qubits = quantum.paulirot ["X"](%cst) %2 : !quantum.bit
    %3 = quantum.insert %arg1[%extracted], %out_qubits : !quantum.reg, !quantum.bit
    return %3 : !quantum.reg
  }
  func.func private @"__builtin__ppr_to_paulirot_PPR{}{wires:1}{angle_denominator = 8 : i64, pauli_word = \22Z\22}"(%arg0: tensor<1xi64>, %arg1: !quantum.reg) -> !quantum.reg attributes {frontend_name = "_ppr_to_paulirot", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"PauliRot{theta:[f64]}{wires:1}{pauli_word = \22Z\22}" = 1 : i64}}, target_gate = "PPR{}{wires:1}{angle_denominator = 8 : i64, pauli_word = \22Z\22}"} {
    %cst = arith.constant 0.78539816339744828 : f64
    %0 = stablehlo.slice %arg0 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
    %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
    %extracted = tensor.extract %1[] : tensor<i64>
    %2 = quantum.extract %arg1[%extracted] : !quantum.reg -> !quantum.bit
    %out_qubits = quantum.paulirot ["Z"](%cst) %2 : !quantum.bit
    %3 = quantum.insert %arg1[%extracted], %out_qubits : !quantum.reg, !quantum.bit
    return %3 : !quantum.reg
  }
  func.func private @"__builtin__rz_to_ps_RZ{0:[f64]}{wires:1}{}"(%arg0: tensor<f64>, %arg1: tensor<1xi64>, %arg2: !quantum.reg) -> !quantum.reg attributes {frontend_name = "_rz_to_ps", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"GlobalPhase{phi:[f64]}{}{}" = 1 : i64, "PhaseShift{0:[f64]}{wires:1}{}" = 1 : i64}}, target_gate = "RZ{0:[f64]}{wires:1}{}"} {
    %cst = stablehlo.constant dense<2.000000e+00> : tensor<f64>
    %0 = stablehlo.slice %arg1 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
    %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
    %extracted = tensor.extract %1[] : tensor<i64>
    %extracted_0 = tensor.extract %arg0[] : tensor<f64>
    %2 = quantum.extract %arg2[%extracted] : !quantum.reg -> !quantum.bit
    %out_qubits = quantum.custom "PhaseShift"(%extracted_0) %2 : !quantum.bit
    %3 = quantum.insert %arg2[%extracted], %out_qubits : !quantum.reg, !quantum.bit
    %4 = stablehlo.divide %arg0, %cst : tensor<f64>
    %extracted_1 = tensor.extract %4[] : tensor<f64>
    quantum.gphase(%extracted_1)
    return %3 : !quantum.reg
  }
  func.func private @"__builtin__rz_to_rot_RZ{0:[f64]}{wires:1}{}"(%arg0: tensor<f64>, %arg1: tensor<1xi64>, %arg2: !quantum.reg) -> !quantum.reg attributes {frontend_name = "_rz_to_rot", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"Rot{0:[f64],1:[f64],2:[f64]}{wires:1}{}" = 1 : i64}}, target_gate = "RZ{0:[f64]}{wires:1}{}"} {
    %c = stablehlo.constant dense<0> : tensor<i64>
    %0 = stablehlo.slice %arg1 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
    %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
    %extracted = tensor.extract %1[] : tensor<i64>
    %2 = stablehlo.convert %c : (tensor<i64>) -> tensor<f64>
    %extracted_0 = tensor.extract %2[] : tensor<f64>
    %3 = stablehlo.convert %c : (tensor<i64>) -> tensor<f64>
    %extracted_1 = tensor.extract %3[] : tensor<f64>
    %extracted_2 = tensor.extract %arg0[] : tensor<f64>
    %4 = quantum.extract %arg2[%extracted] : !quantum.reg -> !quantum.bit
    %out_qubits = quantum.custom "Rot"(%extracted_0, %extracted_1, %extracted_2) %4 : !quantum.bit
    %5 = quantum.insert %arg2[%extracted], %out_qubits : !quantum.reg, !quantum.bit
    return %5 : !quantum.reg
  }
  func.func private @"__builtin__rz_to_ry_rx_RZ{0:[f64]}{wires:1}{}"(%arg0: tensor<f64>, %arg1: tensor<1xi64>, %arg2: !quantum.reg) -> !quantum.reg attributes {frontend_name = "_rz_to_ry_rx", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"RX{0:[f64]}{wires:1}{}" = 1 : i64, "RY{0:[f64]}{wires:1}{}" = 2 : i64}}, target_gate = "RZ{0:[f64]}{wires:1}{}"} {
    %cst = arith.constant -1.5707963267948966 : f64
    %cst_0 = arith.constant 1.5707963267948966 : f64
    %0 = stablehlo.slice %arg1 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
    %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
    %extracted = tensor.extract %1[] : tensor<i64>
    %2 = quantum.extract %arg2[%extracted] : !quantum.reg -> !quantum.bit
    %out_qubits = quantum.custom "RY"(%cst_0) %2 : !quantum.bit
    %3 = quantum.insert %arg2[%extracted], %out_qubits : !quantum.reg, !quantum.bit
    %4 = stablehlo.slice %arg1 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
    %5 = stablehlo.reshape %4 : (tensor<1xi64>) -> tensor<i64>
    %extracted_1 = tensor.extract %5[] : tensor<i64>
    %extracted_2 = tensor.extract %arg0[] : tensor<f64>
    %6 = quantum.extract %3[%extracted_1] : !quantum.reg -> !quantum.bit
    %out_qubits_3 = quantum.custom "RX"(%extracted_2) %6 : !quantum.bit
    %7 = quantum.insert %3[%extracted_1], %out_qubits_3 : !quantum.reg, !quantum.bit
    %8 = stablehlo.slice %arg1 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
    %9 = stablehlo.reshape %8 : (tensor<1xi64>) -> tensor<i64>
    %extracted_4 = tensor.extract %9[] : tensor<i64>
    %10 = quantum.extract %7[%extracted_4] : !quantum.reg -> !quantum.bit
    %out_qubits_5 = quantum.custom "RY"(%cst) %10 : !quantum.bit
    %11 = quantum.insert %7[%extracted_4], %out_qubits_5 : !quantum.reg, !quantum.bit
    return %11 : !quantum.reg
  }
  func.func private @"__builtin__rz_to_ppr_RZ{0:[f64]}{wires:1}{}"(%arg0: tensor<f64>, %arg1: tensor<1xi64>, %arg2: !quantum.reg) -> !quantum.reg attributes {frontend_name = "_rz_to_ppr", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"PauliRot{theta:[f64]}{wires:1}{pauli_word = \22Z\22}" = 1 : i64}}, target_gate = "RZ{0:[f64]}{wires:1}{}"} {
    %0 = stablehlo.slice %arg1 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
    %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
    %extracted = tensor.extract %1[] : tensor<i64>
    %extracted_0 = tensor.extract %arg0[] : tensor<f64>
    %2 = quantum.extract %arg2[%extracted] : !quantum.reg -> !quantum.bit
    %out_qubits = quantum.paulirot ["Z"](%extracted_0) %2 : !quantum.bit
    %3 = quantum.insert %arg2[%extracted], %out_qubits : !quantum.reg, !quantum.bit
    return %3 : !quantum.reg
  }
  func.func private @"__builtin__rz_to_rx_cliff_RZ{0:[f64]}{wires:1}{}"(%arg0: tensor<f64>, %arg1: tensor<1xi64>, %arg2: !quantum.reg) -> !quantum.reg attributes {frontend_name = "_rz_to_rx_cliff", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"Hadamard{}{wires:1}{}" = 2 : i64, "RX{0:[f64]}{wires:1}{}" = 1 : i64}}, target_gate = "RZ{0:[f64]}{wires:1}{}"} {
    %0 = stablehlo.slice %arg1 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
    %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
    %2 = stablehlo.slice %arg1 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
    %3 = stablehlo.reshape %2 : (tensor<1xi64>) -> tensor<i64>
    %4 = stablehlo.slice %arg1 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
    %5 = stablehlo.reshape %4 : (tensor<1xi64>) -> tensor<i64>
    %extracted = tensor.extract %1[] : tensor<i64>
    %6 = quantum.extract %arg2[%extracted] : !quantum.reg -> !quantum.bit
    %out_qubits = quantum.custom "Hadamard"() %6 : !quantum.bit
    %7 = quantum.insert %arg2[%extracted], %out_qubits : !quantum.reg, !quantum.bit
    %extracted_0 = tensor.extract %3[] : tensor<i64>
    %extracted_1 = tensor.extract %arg0[] : tensor<f64>
    %8 = quantum.extract %7[%extracted_0] : !quantum.reg -> !quantum.bit
    %out_qubits_2 = quantum.custom "RX"(%extracted_1) %8 : !quantum.bit
    %9 = quantum.insert %7[%extracted_0], %out_qubits_2 : !quantum.reg, !quantum.bit
    %extracted_3 = tensor.extract %5[] : tensor<i64>
    %10 = quantum.extract %9[%extracted_3] : !quantum.reg -> !quantum.bit
    %out_qubits_4 = quantum.custom "Hadamard"() %10 : !quantum.bit
    %11 = quantum.insert %9[%extracted_3], %out_qubits_4 : !quantum.reg, !quantum.bit
    return %11 : !quantum.reg
  }
  func.func private @"__builtin__rz_to_ry_cliff_RZ{0:[f64]}{wires:1}{}"(%arg0: tensor<f64>, %arg1: tensor<1xi64>, %arg2: !quantum.reg) -> !quantum.reg attributes {frontend_name = "_rz_to_ry_cliff", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"Adjoint(S){}{wires:1}{}" = 1 : i64, "Hadamard{}{wires:1}{}" = 2 : i64, "RY{0:[f64]}{wires:1}{}" = 1 : i64, "S{}{wires:1}{}" = 1 : i64}}, target_gate = "RZ{0:[f64]}{wires:1}{}"} {
    %0 = stablehlo.slice %arg1 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
    %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
    %2 = stablehlo.slice %arg1 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
    %3 = stablehlo.reshape %2 : (tensor<1xi64>) -> tensor<i64>
    %extracted = tensor.extract %3[] : tensor<i64>
    %4 = quantum.extract %arg2[%extracted] : !quantum.reg -> !quantum.bit
    %out_qubits = quantum.custom "Hadamard"() %4 : !quantum.bit
    %5 = quantum.insert %arg2[%extracted], %out_qubits : !quantum.reg, !quantum.bit
    %6 = stablehlo.slice %arg1 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
    %7 = stablehlo.reshape %6 : (tensor<1xi64>) -> tensor<i64>
    %extracted_0 = tensor.extract %7[] : tensor<i64>
    %8 = quantum.extract %5[%extracted_0] : !quantum.reg -> !quantum.bit
    %out_qubits_1 = quantum.custom "S"() %8 : !quantum.bit
    %9 = quantum.insert %5[%extracted_0], %out_qubits_1 : !quantum.reg, !quantum.bit
    %extracted_2 = tensor.extract %1[] : tensor<i64>
    %extracted_3 = tensor.extract %arg0[] : tensor<f64>
    %10 = quantum.extract %9[%extracted_2] : !quantum.reg -> !quantum.bit
    %out_qubits_4 = quantum.custom "RY"(%extracted_3) %10 : !quantum.bit
    %11 = quantum.insert %9[%extracted_2], %out_qubits_4 : !quantum.reg, !quantum.bit
    %12 = stablehlo.slice %arg1 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
    %13 = stablehlo.reshape %12 : (tensor<1xi64>) -> tensor<i64>
    %extracted_5 = tensor.extract %13[] : tensor<i64>
    %14 = quantum.extract %11[%extracted_5] : !quantum.reg -> !quantum.bit
    %out_qubits_6 = quantum.custom "S"() %14 adj : !quantum.bit
    %15 = quantum.insert %11[%extracted_5], %out_qubits_6 : !quantum.reg, !quantum.bit
    %16 = stablehlo.slice %arg1 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
    %17 = stablehlo.reshape %16 : (tensor<1xi64>) -> tensor<i64>
    %extracted_7 = tensor.extract %17[] : tensor<i64>
    %18 = quantum.extract %15[%extracted_7] : !quantum.reg -> !quantum.bit
    %out_qubits_8 = quantum.custom "Hadamard"() %18 : !quantum.bit
    %19 = quantum.insert %15[%extracted_7], %out_qubits_8 : !quantum.reg, !quantum.bit
    return %19 : !quantum.reg
  }
  func.func private @"__builtin__rx_to_rot_RX{0:[f64]}{wires:1}{}"(%arg0: tensor<f64>, %arg1: tensor<1xi64>, %arg2: !quantum.reg) -> !quantum.reg attributes {frontend_name = "_rx_to_rot", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"Rot{0:[f64],1:[f64],2:[f64]}{wires:1}{}" = 1 : i64}}, target_gate = "RX{0:[f64]}{wires:1}{}"} {
    %cst = arith.constant 10.995574287564276 : f64
    %cst_0 = arith.constant 1.5707963267948966 : f64
    %0 = stablehlo.slice %arg1 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
    %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
    %extracted = tensor.extract %1[] : tensor<i64>
    %extracted_1 = tensor.extract %arg0[] : tensor<f64>
    %2 = quantum.extract %arg2[%extracted] : !quantum.reg -> !quantum.bit
    %out_qubits = quantum.custom "Rot"(%cst_0, %extracted_1, %cst) %2 : !quantum.bit
    %3 = quantum.insert %arg2[%extracted], %out_qubits : !quantum.reg, !quantum.bit
    return %3 : !quantum.reg
  }
  func.func private @"__builtin__rx_to_rz_ry_RX{0:[f64]}{wires:1}{}"(%arg0: tensor<f64>, %arg1: tensor<1xi64>, %arg2: !quantum.reg) -> !quantum.reg attributes {frontend_name = "_rx_to_rz_ry", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"RY{0:[f64]}{wires:1}{}" = 1 : i64, "RZ{0:[f64]}{wires:1}{}" = 2 : i64}}, target_gate = "RX{0:[f64]}{wires:1}{}"} {
    %cst = arith.constant -1.5707963267948966 : f64
    %cst_0 = arith.constant 1.5707963267948966 : f64
    %0 = stablehlo.slice %arg1 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
    %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
    %extracted = tensor.extract %1[] : tensor<i64>
    %2 = quantum.extract %arg2[%extracted] : !quantum.reg -> !quantum.bit
    %out_qubits = quantum.custom "RZ"(%cst_0) %2 : !quantum.bit
    %3 = quantum.insert %arg2[%extracted], %out_qubits : !quantum.reg, !quantum.bit
    %4 = stablehlo.slice %arg1 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
    %5 = stablehlo.reshape %4 : (tensor<1xi64>) -> tensor<i64>
    %extracted_1 = tensor.extract %5[] : tensor<i64>
    %extracted_2 = tensor.extract %arg0[] : tensor<f64>
    %6 = quantum.extract %3[%extracted_1] : !quantum.reg -> !quantum.bit
    %out_qubits_3 = quantum.custom "RY"(%extracted_2) %6 : !quantum.bit
    %7 = quantum.insert %3[%extracted_1], %out_qubits_3 : !quantum.reg, !quantum.bit
    %8 = stablehlo.slice %arg1 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
    %9 = stablehlo.reshape %8 : (tensor<1xi64>) -> tensor<i64>
    %extracted_4 = tensor.extract %9[] : tensor<i64>
    %10 = quantum.extract %7[%extracted_4] : !quantum.reg -> !quantum.bit
    %out_qubits_5 = quantum.custom "RZ"(%cst) %10 : !quantum.bit
    %11 = quantum.insert %7[%extracted_4], %out_qubits_5 : !quantum.reg, !quantum.bit
    return %11 : !quantum.reg
  }
  func.func private @"__builtin__rx_to_ppr_RX{0:[f64]}{wires:1}{}"(%arg0: tensor<f64>, %arg1: tensor<1xi64>, %arg2: !quantum.reg) -> !quantum.reg attributes {frontend_name = "_rx_to_ppr", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"PauliRot{theta:[f64]}{wires:1}{pauli_word = \22X\22}" = 1 : i64}}, target_gate = "RX{0:[f64]}{wires:1}{}"} {
    %0 = stablehlo.slice %arg1 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
    %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
    %extracted = tensor.extract %1[] : tensor<i64>
    %extracted_0 = tensor.extract %arg0[] : tensor<f64>
    %2 = quantum.extract %arg2[%extracted] : !quantum.reg -> !quantum.bit
    %out_qubits = quantum.paulirot ["X"](%extracted_0) %2 : !quantum.bit
    %3 = quantum.insert %arg2[%extracted], %out_qubits : !quantum.reg, !quantum.bit
    return %3 : !quantum.reg
  }
  func.func private @"__builtin__rx_to_ry_cliff_RX{0:[f64]}{wires:1}{}"(%arg0: tensor<f64>, %arg1: tensor<1xi64>, %arg2: !quantum.reg) -> !quantum.reg attributes {frontend_name = "_rx_to_ry_cliff", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"Adjoint(S){}{wires:1}{}" = 1 : i64, "RY{0:[f64]}{wires:1}{}" = 1 : i64, "S{}{wires:1}{}" = 1 : i64}}, target_gate = "RX{0:[f64]}{wires:1}{}"} {
    %0 = stablehlo.slice %arg1 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
    %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
    %2 = stablehlo.slice %arg1 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
    %3 = stablehlo.reshape %2 : (tensor<1xi64>) -> tensor<i64>
    %extracted = tensor.extract %1[] : tensor<i64>
    %4 = quantum.extract %arg2[%extracted] : !quantum.reg -> !quantum.bit
    %out_qubits = quantum.custom "S"() %4 : !quantum.bit
    %5 = quantum.insert %arg2[%extracted], %out_qubits : !quantum.reg, !quantum.bit
    %extracted_0 = tensor.extract %3[] : tensor<i64>
    %extracted_1 = tensor.extract %arg0[] : tensor<f64>
    %6 = quantum.extract %5[%extracted_0] : !quantum.reg -> !quantum.bit
    %out_qubits_2 = quantum.custom "RY"(%extracted_1) %6 : !quantum.bit
    %7 = quantum.insert %5[%extracted_0], %out_qubits_2 : !quantum.reg, !quantum.bit
    %extracted_3 = tensor.extract %1[] : tensor<i64>
    %8 = quantum.extract %7[%extracted_3] : !quantum.reg -> !quantum.bit
    %out_qubits_4 = quantum.custom "S"() %8 adj : !quantum.bit
    %9 = quantum.insert %7[%extracted_3], %out_qubits_4 : !quantum.reg, !quantum.bit
    return %9 : !quantum.reg
  }
  func.func private @"__builtin__rx_to_rz_cliff_RX{0:[f64]}{wires:1}{}"(%arg0: tensor<f64>, %arg1: tensor<1xi64>, %arg2: !quantum.reg) -> !quantum.reg attributes {frontend_name = "_rx_to_rz_cliff", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"Hadamard{}{wires:1}{}" = 2 : i64, "RZ{0:[f64]}{wires:1}{}" = 1 : i64}}, target_gate = "RX{0:[f64]}{wires:1}{}"} {
    %0 = stablehlo.slice %arg1 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
    %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
    %2 = stablehlo.slice %arg1 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
    %3 = stablehlo.reshape %2 : (tensor<1xi64>) -> tensor<i64>
    %4 = stablehlo.slice %arg1 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
    %5 = stablehlo.reshape %4 : (tensor<1xi64>) -> tensor<i64>
    %extracted = tensor.extract %1[] : tensor<i64>
    %6 = quantum.extract %arg2[%extracted] : !quantum.reg -> !quantum.bit
    %out_qubits = quantum.custom "Hadamard"() %6 : !quantum.bit
    %7 = quantum.insert %arg2[%extracted], %out_qubits : !quantum.reg, !quantum.bit
    %extracted_0 = tensor.extract %3[] : tensor<i64>
    %extracted_1 = tensor.extract %arg0[] : tensor<f64>
    %8 = quantum.extract %7[%extracted_0] : !quantum.reg -> !quantum.bit
    %out_qubits_2 = quantum.custom "RZ"(%extracted_1) %8 : !quantum.bit
    %9 = quantum.insert %7[%extracted_0], %out_qubits_2 : !quantum.reg, !quantum.bit
    %extracted_3 = tensor.extract %5[] : tensor<i64>
    %10 = quantum.extract %9[%extracted_3] : !quantum.reg -> !quantum.bit
    %out_qubits_4 = quantum.custom "Hadamard"() %10 : !quantum.bit
    %11 = quantum.insert %9[%extracted_3], %out_qubits_4 : !quantum.reg, !quantum.bit
    return %11 : !quantum.reg
  }
  func.func private @"__builtin__ry_to_rot_RY{0:[f64]}{wires:1}{}"(%arg0: tensor<f64>, %arg1: tensor<1xi64>, %arg2: !quantum.reg) -> !quantum.reg attributes {frontend_name = "_ry_to_rot", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"Rot{0:[f64],1:[f64],2:[f64]}{wires:1}{}" = 1 : i64}}, target_gate = "RY{0:[f64]}{wires:1}{}"} {
    %c = stablehlo.constant dense<0> : tensor<i64>
    %0 = stablehlo.slice %arg1 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
    %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
    %extracted = tensor.extract %1[] : tensor<i64>
    %2 = stablehlo.convert %c : (tensor<i64>) -> tensor<f64>
    %extracted_0 = tensor.extract %2[] : tensor<f64>
    %extracted_1 = tensor.extract %arg0[] : tensor<f64>
    %3 = stablehlo.convert %c : (tensor<i64>) -> tensor<f64>
    %extracted_2 = tensor.extract %3[] : tensor<f64>
    %4 = quantum.extract %arg2[%extracted] : !quantum.reg -> !quantum.bit
    %out_qubits = quantum.custom "Rot"(%extracted_0, %extracted_1, %extracted_2) %4 : !quantum.bit
    %5 = quantum.insert %arg2[%extracted], %out_qubits : !quantum.reg, !quantum.bit
    return %5 : !quantum.reg
  }
  func.func private @"__builtin__ry_to_rz_rx_RY{0:[f64]}{wires:1}{}"(%arg0: tensor<f64>, %arg1: tensor<1xi64>, %arg2: !quantum.reg) -> !quantum.reg attributes {frontend_name = "_ry_to_rz_rx", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"RX{0:[f64]}{wires:1}{}" = 1 : i64, "RZ{0:[f64]}{wires:1}{}" = 2 : i64}}, target_gate = "RY{0:[f64]}{wires:1}{}"} {
    %cst = arith.constant 1.5707963267948966 : f64
    %cst_0 = arith.constant -1.5707963267948966 : f64
    %0 = stablehlo.slice %arg1 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
    %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
    %extracted = tensor.extract %1[] : tensor<i64>
    %2 = quantum.extract %arg2[%extracted] : !quantum.reg -> !quantum.bit
    %out_qubits = quantum.custom "RZ"(%cst_0) %2 : !quantum.bit
    %3 = quantum.insert %arg2[%extracted], %out_qubits : !quantum.reg, !quantum.bit
    %4 = stablehlo.slice %arg1 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
    %5 = stablehlo.reshape %4 : (tensor<1xi64>) -> tensor<i64>
    %extracted_1 = tensor.extract %5[] : tensor<i64>
    %extracted_2 = tensor.extract %arg0[] : tensor<f64>
    %6 = quantum.extract %3[%extracted_1] : !quantum.reg -> !quantum.bit
    %out_qubits_3 = quantum.custom "RX"(%extracted_2) %6 : !quantum.bit
    %7 = quantum.insert %3[%extracted_1], %out_qubits_3 : !quantum.reg, !quantum.bit
    %8 = stablehlo.slice %arg1 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
    %9 = stablehlo.reshape %8 : (tensor<1xi64>) -> tensor<i64>
    %extracted_4 = tensor.extract %9[] : tensor<i64>
    %10 = quantum.extract %7[%extracted_4] : !quantum.reg -> !quantum.bit
    %out_qubits_5 = quantum.custom "RZ"(%cst) %10 : !quantum.bit
    %11 = quantum.insert %7[%extracted_4], %out_qubits_5 : !quantum.reg, !quantum.bit
    return %11 : !quantum.reg
  }
  func.func private @"__builtin__ry_to_ppr_RY{0:[f64]}{wires:1}{}"(%arg0: tensor<f64>, %arg1: tensor<1xi64>, %arg2: !quantum.reg) -> !quantum.reg attributes {frontend_name = "_ry_to_ppr", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"PauliRot{theta:[f64]}{wires:1}{pauli_word = \22Y\22}" = 1 : i64}}, target_gate = "RY{0:[f64]}{wires:1}{}"} {
    %0 = stablehlo.slice %arg1 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
    %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
    %extracted = tensor.extract %1[] : tensor<i64>
    %extracted_0 = tensor.extract %arg0[] : tensor<f64>
    %2 = quantum.extract %arg2[%extracted] : !quantum.reg -> !quantum.bit
    %out_qubits = quantum.paulirot ["Y"](%extracted_0) %2 : !quantum.bit
    %3 = quantum.insert %arg2[%extracted], %out_qubits : !quantum.reg, !quantum.bit
    return %3 : !quantum.reg
  }
  func.func private @"__builtin__ry_to_rx_cliff_RY{0:[f64]}{wires:1}{}"(%arg0: tensor<f64>, %arg1: tensor<1xi64>, %arg2: !quantum.reg) -> !quantum.reg attributes {frontend_name = "_ry_to_rx_cliff", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"Adjoint(S){}{wires:1}{}" = 1 : i64, "RX{0:[f64]}{wires:1}{}" = 1 : i64, "S{}{wires:1}{}" = 1 : i64}}, target_gate = "RY{0:[f64]}{wires:1}{}"} {
    %0 = stablehlo.slice %arg1 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
    %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
    %2 = stablehlo.slice %arg1 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
    %3 = stablehlo.reshape %2 : (tensor<1xi64>) -> tensor<i64>
    %4 = stablehlo.slice %arg1 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
    %5 = stablehlo.reshape %4 : (tensor<1xi64>) -> tensor<i64>
    %extracted = tensor.extract %1[] : tensor<i64>
    %6 = quantum.extract %arg2[%extracted] : !quantum.reg -> !quantum.bit
    %out_qubits = quantum.custom "S"() %6 adj : !quantum.bit
    %7 = quantum.insert %arg2[%extracted], %out_qubits : !quantum.reg, !quantum.bit
    %extracted_0 = tensor.extract %3[] : tensor<i64>
    %extracted_1 = tensor.extract %arg0[] : tensor<f64>
    %8 = quantum.extract %7[%extracted_0] : !quantum.reg -> !quantum.bit
    %out_qubits_2 = quantum.custom "RX"(%extracted_1) %8 : !quantum.bit
    %9 = quantum.insert %7[%extracted_0], %out_qubits_2 : !quantum.reg, !quantum.bit
    %extracted_3 = tensor.extract %5[] : tensor<i64>
    %10 = quantum.extract %9[%extracted_3] : !quantum.reg -> !quantum.bit
    %out_qubits_4 = quantum.custom "S"() %10 : !quantum.bit
    %11 = quantum.insert %9[%extracted_3], %out_qubits_4 : !quantum.reg, !quantum.bit
    return %11 : !quantum.reg
  }
  func.func private @"__builtin__ry_to_rz_cliff_RY{0:[f64]}{wires:1}{}"(%arg0: tensor<f64>, %arg1: tensor<1xi64>, %arg2: !quantum.reg) -> !quantum.reg attributes {frontend_name = "_ry_to_rz_cliff", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"Adjoint(S){}{wires:1}{}" = 1 : i64, "Hadamard{}{wires:1}{}" = 2 : i64, "RZ{0:[f64]}{wires:1}{}" = 1 : i64, "S{}{wires:1}{}" = 1 : i64}}, target_gate = "RY{0:[f64]}{wires:1}{}"} {
    %0 = stablehlo.slice %arg1 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
    %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
    %2 = stablehlo.slice %arg1 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
    %3 = stablehlo.reshape %2 : (tensor<1xi64>) -> tensor<i64>
    %extracted = tensor.extract %3[] : tensor<i64>
    %4 = quantum.extract %arg2[%extracted] : !quantum.reg -> !quantum.bit
    %out_qubits = quantum.custom "S"() %4 adj : !quantum.bit
    %5 = quantum.insert %arg2[%extracted], %out_qubits : !quantum.reg, !quantum.bit
    %6 = stablehlo.slice %arg1 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
    %7 = stablehlo.reshape %6 : (tensor<1xi64>) -> tensor<i64>
    %extracted_0 = tensor.extract %7[] : tensor<i64>
    %8 = quantum.extract %5[%extracted_0] : !quantum.reg -> !quantum.bit
    %out_qubits_1 = quantum.custom "Hadamard"() %8 : !quantum.bit
    %9 = quantum.insert %5[%extracted_0], %out_qubits_1 : !quantum.reg, !quantum.bit
    %extracted_2 = tensor.extract %1[] : tensor<i64>
    %extracted_3 = tensor.extract %arg0[] : tensor<f64>
    %10 = quantum.extract %9[%extracted_2] : !quantum.reg -> !quantum.bit
    %out_qubits_4 = quantum.custom "RZ"(%extracted_3) %10 : !quantum.bit
    %11 = quantum.insert %9[%extracted_2], %out_qubits_4 : !quantum.reg, !quantum.bit
    %12 = stablehlo.slice %arg1 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
    %13 = stablehlo.reshape %12 : (tensor<1xi64>) -> tensor<i64>
    %extracted_5 = tensor.extract %13[] : tensor<i64>
    %14 = quantum.extract %11[%extracted_5] : !quantum.reg -> !quantum.bit
    %out_qubits_6 = quantum.custom "Hadamard"() %14 : !quantum.bit
    %15 = quantum.insert %11[%extracted_5], %out_qubits_6 : !quantum.reg, !quantum.bit
    %16 = stablehlo.slice %arg1 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
    %17 = stablehlo.reshape %16 : (tensor<1xi64>) -> tensor<i64>
    %extracted_7 = tensor.extract %17[] : tensor<i64>
    %18 = quantum.extract %15[%extracted_7] : !quantum.reg -> !quantum.bit
    %out_qubits_8 = quantum.custom "S"() %18 : !quantum.bit
    %19 = quantum.insert %15[%extracted_7], %out_qubits_8 : !quantum.reg, !quantum.bit
    return %19 : !quantum.reg
  }
  func.func private @"__builtin__cz_to_cps_CZ{}{wires:2}{}"(%arg0: tensor<2xi64>, %arg1: !quantum.reg) -> !quantum.reg attributes {frontend_name = "_cz_to_cps", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"ControlledPhaseShift{0:[f64]}{wires:2}{}" = 1 : i64}}, target_gate = "CZ{}{wires:2}{}"} {
    %cst = arith.constant 3.1415926535897931 : f64
    %0 = stablehlo.slice %arg0 [0:1] : (tensor<2xi64>) -> tensor<1xi64>
    %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
    %2 = stablehlo.slice %arg0 [1:2] : (tensor<2xi64>) -> tensor<1xi64>
    %3 = stablehlo.reshape %2 : (tensor<1xi64>) -> tensor<i64>
    %extracted = tensor.extract %1[] : tensor<i64>
    %extracted_0 = tensor.extract %3[] : tensor<i64>
    %4 = quantum.extract %arg1[%extracted] : !quantum.reg -> !quantum.bit
    %5 = quantum.extract %arg1[%extracted_0] : !quantum.reg -> !quantum.bit
    %out_qubits:2 = quantum.custom "ControlledPhaseShift"(%cst) %4, %5 : !quantum.bit, !quantum.bit
    %6 = quantum.insert %arg1[%extracted], %out_qubits#0 : !quantum.reg, !quantum.bit
    %7 = quantum.insert %6[%extracted_0], %out_qubits#1 : !quantum.reg, !quantum.bit
    return %7 : !quantum.reg
  }
  func.func private @"__builtin__cz_to_cnot_CZ{}{wires:2}{}"(%arg0: tensor<2xi64>, %arg1: !quantum.reg) -> !quantum.reg attributes {frontend_name = "_cz_to_cnot", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"CNOT{}{wires:2}{}" = 1 : i64, "Hadamard{}{wires:1}{}" = 2 : i64}}, target_gate = "CZ{}{wires:2}{}"} {
    %0 = stablehlo.slice %arg0 [1:2] : (tensor<2xi64>) -> tensor<1xi64>
    %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
    %extracted = tensor.extract %1[] : tensor<i64>
    %2 = quantum.extract %arg1[%extracted] : !quantum.reg -> !quantum.bit
    %out_qubits = quantum.custom "Hadamard"() %2 : !quantum.bit
    %3 = quantum.insert %arg1[%extracted], %out_qubits : !quantum.reg, !quantum.bit
    %4 = stablehlo.slice %arg0 [0:1] : (tensor<2xi64>) -> tensor<1xi64>
    %5 = stablehlo.reshape %4 : (tensor<1xi64>) -> tensor<i64>
    %6 = stablehlo.slice %arg0 [1:2] : (tensor<2xi64>) -> tensor<1xi64>
    %7 = stablehlo.reshape %6 : (tensor<1xi64>) -> tensor<i64>
    %extracted_0 = tensor.extract %5[] : tensor<i64>
    %extracted_1 = tensor.extract %7[] : tensor<i64>
    %8 = quantum.extract %3[%extracted_0] : !quantum.reg -> !quantum.bit
    %9 = quantum.extract %3[%extracted_1] : !quantum.reg -> !quantum.bit
    %out_qubits_2:2 = quantum.custom "CNOT"() %8, %9 : !quantum.bit, !quantum.bit
    %10 = quantum.insert %3[%extracted_0], %out_qubits_2#0 : !quantum.reg, !quantum.bit
    %11 = quantum.insert %10[%extracted_1], %out_qubits_2#1 : !quantum.reg, !quantum.bit
    %12 = stablehlo.slice %arg0 [1:2] : (tensor<2xi64>) -> tensor<1xi64>
    %13 = stablehlo.reshape %12 : (tensor<1xi64>) -> tensor<i64>
    %extracted_3 = tensor.extract %13[] : tensor<i64>
    %14 = quantum.extract %11[%extracted_3] : !quantum.reg -> !quantum.bit
    %out_qubits_4 = quantum.custom "Hadamard"() %14 : !quantum.bit
    %15 = quantum.insert %11[%extracted_3], %out_qubits_4 : !quantum.reg, !quantum.bit
    return %15 : !quantum.reg
  }
  func.func private @"__builtin__cz_to_ppr_CZ{}{wires:2}{}"(%arg0: tensor<2xi64>, %arg1: !quantum.reg) -> !quantum.reg attributes {frontend_name = "_cz_to_ppr", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"GlobalPhase{phi:[f64]}{}{}" = 1 : i64, "PPR{}{wires:1}{angle_denominator = -4 : si64, pauli_word = \22Z\22}" = 2 : i64, "PPR{}{wires:2}{angle_denominator = 4 : i64, pauli_word = \22ZZ\22}" = 1 : i64}}, target_gate = "CZ{}{wires:2}{}"} {
    %cst = arith.constant 0.78539816339744828 : f64
    %0 = stablehlo.slice %arg0 [0:1] : (tensor<2xi64>) -> tensor<1xi64>
    %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
    %extracted = tensor.extract %1[] : tensor<i64>
    %2 = quantum.extract %arg1[%extracted] : !quantum.reg -> !quantum.bit
    %out_qubits = quantum.operator "PPR"() qubits(%2)
      static_data = {angle_denominator = -4 : si64, pauli_word = "Z"}
      qubit_map = {wires = [0]}
    %3 = quantum.insert %arg1[%extracted], %out_qubits : !quantum.reg, !quantum.bit
    %4 = stablehlo.slice %arg0 [1:2] : (tensor<2xi64>) -> tensor<1xi64>
    %5 = stablehlo.reshape %4 : (tensor<1xi64>) -> tensor<i64>
    %extracted_0 = tensor.extract %5[] : tensor<i64>
    %6 = quantum.extract %3[%extracted_0] : !quantum.reg -> !quantum.bit
    %out_qubits_1 = quantum.operator "PPR"() qubits(%6)
      static_data = {angle_denominator = -4 : si64, pauli_word = "Z"}
      qubit_map = {wires = [0]}
    %7 = quantum.insert %3[%extracted_0], %out_qubits_1 : !quantum.reg, !quantum.bit
    %8 = stablehlo.slice %arg0 [0:1] : (tensor<2xi64>) -> tensor<1xi64>
    %9 = stablehlo.reshape %8 : (tensor<1xi64>) -> tensor<i64>
    %10 = stablehlo.slice %arg0 [1:2] : (tensor<2xi64>) -> tensor<1xi64>
    %11 = stablehlo.reshape %10 : (tensor<1xi64>) -> tensor<i64>
    %extracted_2 = tensor.extract %9[] : tensor<i64>
    %extracted_3 = tensor.extract %11[] : tensor<i64>
    %12 = quantum.extract %7[%extracted_2] : !quantum.reg -> !quantum.bit
    %13 = quantum.extract %7[%extracted_3] : !quantum.reg -> !quantum.bit
    %out_qubits_4:2 = quantum.operator "PPR"() qubits(%12, %13)
      static_data = {angle_denominator = 4 : i64, pauli_word = "ZZ"}
      qubit_map = {wires = [0, 1]}
    %14 = quantum.insert %7[%extracted_2], %out_qubits_4#0 : !quantum.reg, !quantum.bit
    %15 = quantum.insert %14[%extracted_3], %out_qubits_4#1 : !quantum.reg, !quantum.bit
    quantum.gphase(%cst)
    return %15 : !quantum.reg
  }
  func.func private @"__builtin__ppr_to_paulirot_PPR{}{wires:1}{angle_denominator = -4 : si64, pauli_word = \22Z\22}"(%arg0: tensor<1xi64>, %arg1: !quantum.reg) -> !quantum.reg attributes {frontend_name = "_ppr_to_paulirot", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"PauliRot{theta:[f64]}{wires:1}{pauli_word = \22Z\22}" = 1 : i64}}, target_gate = "PPR{}{wires:1}{angle_denominator = -4 : si64, pauli_word = \22Z\22}"} {
    %cst = arith.constant -1.5707963267948966 : f64
    %0 = stablehlo.slice %arg0 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
    %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
    %extracted = tensor.extract %1[] : tensor<i64>
    %2 = quantum.extract %arg1[%extracted] : !quantum.reg -> !quantum.bit
    %out_qubits = quantum.paulirot ["Z"](%cst) %2 : !quantum.bit
    %3 = quantum.insert %arg1[%extracted], %out_qubits : !quantum.reg, !quantum.bit
    return %3 : !quantum.reg
  }
  func.func private @"__builtin__ppr_to_paulirot_PPR{}{wires:1}{angle_denominator = -4 : si64, pauli_word = \22X\22}"(%arg0: tensor<1xi64>, %arg1: !quantum.reg) -> !quantum.reg attributes {frontend_name = "_ppr_to_paulirot", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"PauliRot{theta:[f64]}{wires:1}{pauli_word = \22X\22}" = 1 : i64}}, target_gate = "PPR{}{wires:1}{angle_denominator = -4 : si64, pauli_word = \22X\22}"} {
    %cst = arith.constant -1.5707963267948966 : f64
    %0 = stablehlo.slice %arg0 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
    %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
    %extracted = tensor.extract %1[] : tensor<i64>
    %2 = quantum.extract %arg1[%extracted] : !quantum.reg -> !quantum.bit
    %out_qubits = quantum.paulirot ["X"](%cst) %2 : !quantum.bit
    %3 = quantum.insert %arg1[%extracted], %out_qubits : !quantum.reg, !quantum.bit
    return %3 : !quantum.reg
  }
  func.func private @"__builtin__ppr_to_paulirot_PPR{}{wires:2}{angle_denominator = 4 : i64, pauli_word = \22ZX\22}"(%arg0: tensor<2xi64>, %arg1: !quantum.reg) -> !quantum.reg attributes {frontend_name = "_ppr_to_paulirot", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"PauliRot{theta:[f64]}{wires:2}{pauli_word = \22ZX\22}" = 1 : i64}}, target_gate = "PPR{}{wires:2}{angle_denominator = 4 : i64, pauli_word = \22ZX\22}"} {
    %cst = arith.constant 1.5707963267948966 : f64
    %0 = stablehlo.slice %arg0 [0:1] : (tensor<2xi64>) -> tensor<1xi64>
    %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
    %2 = stablehlo.slice %arg0 [1:2] : (tensor<2xi64>) -> tensor<1xi64>
    %3 = stablehlo.reshape %2 : (tensor<1xi64>) -> tensor<i64>
    %extracted = tensor.extract %1[] : tensor<i64>
    %extracted_0 = tensor.extract %3[] : tensor<i64>
    %4 = quantum.extract %arg1[%extracted] : !quantum.reg -> !quantum.bit
    %5 = quantum.extract %arg1[%extracted_0] : !quantum.reg -> !quantum.bit
    %out_qubits:2 = quantum.paulirot ["Z", "X"](%cst) %4, %5 : !quantum.bit, !quantum.bit
    %6 = quantum.insert %arg1[%extracted], %out_qubits#0 : !quantum.reg, !quantum.bit
    %7 = quantum.insert %6[%extracted_0], %out_qubits#1 : !quantum.reg, !quantum.bit
    return %7 : !quantum.reg
  }
  func.func private @"__builtin__phaseshift_to_rz_gp_PhaseShift{0:[f64]}{wires:1}{}"(%arg0: tensor<f64>, %arg1: tensor<1xi64>, %arg2: !quantum.reg) -> !quantum.reg attributes {frontend_name = "_phaseshift_to_rz_gp", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"GlobalPhase{phi:[f64]}{}{}" = 1 : i64, "RZ{0:[f64]}{wires:1}{}" = 1 : i64}}, target_gate = "PhaseShift{0:[f64]}{wires:1}{}"} {
    %cst = stablehlo.constant dense<2.000000e+00> : tensor<f64>
    %0 = stablehlo.slice %arg1 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
    %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
    %extracted = tensor.extract %1[] : tensor<i64>
    %extracted_0 = tensor.extract %arg0[] : tensor<f64>
    %2 = quantum.extract %arg2[%extracted] : !quantum.reg -> !quantum.bit
    %out_qubits = quantum.custom "RZ"(%extracted_0) %2 : !quantum.bit
    %3 = quantum.insert %arg2[%extracted], %out_qubits : !quantum.reg, !quantum.bit
    %4 = stablehlo.negate %arg0 : tensor<f64>
    %5 = stablehlo.divide %4, %cst : tensor<f64>
    %extracted_1 = tensor.extract %5[] : tensor<f64>
    quantum.gphase(%extracted_1)
    return %3 : !quantum.reg
  }
  func.func private @"__builtin_adjoint_rotation_Adjoint(PhaseShift){0:[f64]}{wires:1}{}"(%arg0: tensor<f64>, %arg1: tensor<1xi64>, %arg2: !quantum.reg) -> !quantum.reg attributes {frontend_name = "adjoint_rotation", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"PhaseShift{0:[f64]}{wires:1}{}" = 1 : i64}}, target_gate = "Adjoint(PhaseShift){0:[f64]}{wires:1}{}"} {
    %0 = stablehlo.slice %arg1 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
    %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
    %2 = stablehlo.negate %arg0 : tensor<f64>
    %extracted = tensor.extract %1[] : tensor<i64>
    %extracted_0 = tensor.extract %2[] : tensor<f64>
    %3 = quantum.extract %arg2[%extracted] : !quantum.reg -> !quantum.bit
    %out_qubits = quantum.custom "PhaseShift"(%extracted_0) %3 : !quantum.bit
    %4 = quantum.insert %arg2[%extracted], %out_qubits : !quantum.reg, !quantum.bit
    return %4 : !quantum.reg
  }
  func.func private @"__builtin__phaseshift_to_rz_gp_Adjoint(PhaseShift){0:[f64]}{wires:1}{}"(%arg0: tensor<f64>, %arg1: tensor<1xi64>, %arg2: !quantum.reg) -> !quantum.reg attributes {frontend_name = "_phaseshift_to_rz_gp", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"Adjoint(GlobalPhase){phi:[f64]}{}{}" = 1 : i64, "Adjoint(RZ){0:[f64]}{wires:1}{}" = 1 : i64}}, target_gate = "Adjoint(PhaseShift){0:[f64]}{wires:1}{}"} {
    %cst = stablehlo.constant dense<2.000000e+00> : tensor<f64>
    %0 = catalyst.list_init : <f64>
    %1 = catalyst.list_init : <i64>
    %2 = catalyst.list_init : <i64>
    %3 = stablehlo.slice %arg1 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
    %4 = stablehlo.reshape %3 : (tensor<1xi64>) -> tensor<i64>
    %extracted = tensor.extract %4[] : tensor<i64>
    %extracted_0 = tensor.extract %arg0[] : tensor<f64>
    catalyst.list_push %extracted, %2 : <i64>
    catalyst.list_push %extracted, %2 : <i64>
    %5 = stablehlo.negate %arg0 : tensor<f64>
    %6 = stablehlo.divide %5, %cst : tensor<f64>
    %extracted_1 = tensor.extract %6[] : tensor<f64>
    quantum.gphase(%extracted_1) adj
    %7 = catalyst.list_pop %2 : <i64>
    %8 = quantum.extract %arg2[%7] : !quantum.reg -> !quantum.bit
    %out_qubits = quantum.custom "RZ"(%extracted_0) %8 adj : !quantum.bit
    %9 = quantum.insert %arg2[%7], %out_qubits : !quantum.reg, !quantum.bit
    %10 = catalyst.list_pop %2 : <i64>
    catalyst.list_dealloc %0 : <f64>
    catalyst.list_dealloc %1 : <i64>
    catalyst.list_dealloc %2 : <i64>
    return %9 : !quantum.reg
  }
  func.func private @"__builtin__pauli_rot_decomposition_PauliRot{theta:[f64]}{wires:2}{pauli_word = \22ZZ\22}"(%arg0: tensor<f64>, %arg1: tensor<2xi64>, %arg2: !quantum.reg) -> !quantum.reg attributes {frontend_name = "_pauli_rot_decomposition", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"MultiRZ{theta:[f64]}{wires:2}{}" = 1 : i64}}, target_gate = "PauliRot{theta:[f64]}{wires:2}{pauli_word = \22ZZ\22}"} {
    %0 = stablehlo.slice %arg1 [0:1] : (tensor<2xi64>) -> tensor<1xi64>
    %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
    %2 = stablehlo.slice %arg1 [1:2] : (tensor<2xi64>) -> tensor<1xi64>
    %3 = stablehlo.reshape %2 : (tensor<1xi64>) -> tensor<i64>
    %extracted = tensor.extract %1[] : tensor<i64>
    %extracted_0 = tensor.extract %3[] : tensor<i64>
    %extracted_1 = tensor.extract %arg0[] : tensor<f64>
    %4 = quantum.extract %arg2[%extracted] : !quantum.reg -> !quantum.bit
    %5 = quantum.extract %arg2[%extracted_0] : !quantum.reg -> !quantum.bit
    %out_qubits:2 = quantum.multirz(%extracted_1) %4, %5 : !quantum.bit, !quantum.bit
    %6 = quantum.insert %arg2[%extracted], %out_qubits#0 : !quantum.reg, !quantum.bit
    %7 = quantum.insert %6[%extracted_0], %out_qubits#1 : !quantum.reg, !quantum.bit
    return %7 : !quantum.reg
  }
  func.func private @"__builtin__pauli_rot_decomposition_PauliRot{theta:[f64]}{wires:2}{pauli_word = \22ZX\22}"(%arg0: tensor<f64>, %arg1: tensor<2xi64>, %arg2: !quantum.reg) -> !quantum.reg attributes {frontend_name = "_pauli_rot_decomposition", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"Hadamard{}{wires:1}{}" = 2 : i64, "MultiRZ{theta:[f64]}{wires:2}{}" = 1 : i64}}, target_gate = "PauliRot{theta:[f64]}{wires:2}{pauli_word = \22ZX\22}"} {
    %0 = stablehlo.slice %arg1 [0:1] : (tensor<2xi64>) -> tensor<1xi64>
    %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
    %2 = stablehlo.slice %arg1 [1:2] : (tensor<2xi64>) -> tensor<1xi64>
    %3 = stablehlo.reshape %2 : (tensor<1xi64>) -> tensor<i64>
    %extracted = tensor.extract %3[] : tensor<i64>
    %4 = quantum.extract %arg2[%extracted] : !quantum.reg -> !quantum.bit
    %out_qubits = quantum.custom "Hadamard"() %4 : !quantum.bit
    %5 = quantum.insert %arg2[%extracted], %out_qubits : !quantum.reg, !quantum.bit
    %extracted_0 = tensor.extract %1[] : tensor<i64>
    %extracted_1 = tensor.extract %3[] : tensor<i64>
    %extracted_2 = tensor.extract %arg0[] : tensor<f64>
    %6 = quantum.extract %5[%extracted_0] : !quantum.reg -> !quantum.bit
    %7 = quantum.extract %5[%extracted_1] : !quantum.reg -> !quantum.bit
    %out_qubits_3:2 = quantum.multirz(%extracted_2) %6, %7 : !quantum.bit, !quantum.bit
    %8 = quantum.insert %5[%extracted_0], %out_qubits_3#0 : !quantum.reg, !quantum.bit
    %9 = quantum.insert %8[%extracted_1], %out_qubits_3#1 : !quantum.reg, !quantum.bit
    %extracted_4 = tensor.extract %3[] : tensor<i64>
    %10 = quantum.extract %9[%extracted_4] : !quantum.reg -> !quantum.bit
    %out_qubits_5 = quantum.custom "Hadamard"() %10 : !quantum.bit
    %11 = quantum.insert %9[%extracted_4], %out_qubits_5 : !quantum.reg, !quantum.bit
    return %11 : !quantum.reg
  }
  func.func private @"__builtin__pauli_rot_decomposition_PauliRot{theta:[f64]}{wires:3}{pauli_word = \22ZZX\22}"(%arg0: tensor<f64>, %arg1: tensor<3xi64>, %arg2: !quantum.reg) -> !quantum.reg attributes {frontend_name = "_pauli_rot_decomposition", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"Hadamard{}{wires:1}{}" = 2 : i64, "MultiRZ{theta:[f64]}{wires:3}{}" = 1 : i64}}, target_gate = "PauliRot{theta:[f64]}{wires:3}{pauli_word = \22ZZX\22}"} {
    %0 = stablehlo.slice %arg1 [0:1] : (tensor<3xi64>) -> tensor<1xi64>
    %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
    %2 = stablehlo.slice %arg1 [1:2] : (tensor<3xi64>) -> tensor<1xi64>
    %3 = stablehlo.reshape %2 : (tensor<1xi64>) -> tensor<i64>
    %4 = stablehlo.slice %arg1 [2:3] : (tensor<3xi64>) -> tensor<1xi64>
    %5 = stablehlo.reshape %4 : (tensor<1xi64>) -> tensor<i64>
    %extracted = tensor.extract %5[] : tensor<i64>
    %6 = quantum.extract %arg2[%extracted] : !quantum.reg -> !quantum.bit
    %out_qubits = quantum.custom "Hadamard"() %6 : !quantum.bit
    %7 = quantum.insert %arg2[%extracted], %out_qubits : !quantum.reg, !quantum.bit
    %extracted_0 = tensor.extract %1[] : tensor<i64>
    %extracted_1 = tensor.extract %3[] : tensor<i64>
    %extracted_2 = tensor.extract %5[] : tensor<i64>
    %extracted_3 = tensor.extract %arg0[] : tensor<f64>
    %8 = quantum.extract %7[%extracted_0] : !quantum.reg -> !quantum.bit
    %9 = quantum.extract %7[%extracted_1] : !quantum.reg -> !quantum.bit
    %10 = quantum.extract %7[%extracted_2] : !quantum.reg -> !quantum.bit
    %out_qubits_4:3 = quantum.multirz(%extracted_3) %8, %9, %10 : !quantum.bit, !quantum.bit, !quantum.bit
    %11 = quantum.insert %7[%extracted_0], %out_qubits_4#0 : !quantum.reg, !quantum.bit
    %12 = quantum.insert %11[%extracted_1], %out_qubits_4#1 : !quantum.reg, !quantum.bit
    %13 = quantum.insert %12[%extracted_2], %out_qubits_4#2 : !quantum.reg, !quantum.bit
    %extracted_5 = tensor.extract %5[] : tensor<i64>
    %14 = quantum.extract %13[%extracted_5] : !quantum.reg -> !quantum.bit
    %out_qubits_6 = quantum.custom "Hadamard"() %14 : !quantum.bit
    %15 = quantum.insert %13[%extracted_5], %out_qubits_6 : !quantum.reg, !quantum.bit
    return %15 : !quantum.reg
  }
  func.func private @"__builtin__pauli_rot_decomposition_PauliRot{theta:[f64]}{wires:1}{pauli_word = \22X\22}"(%arg0: tensor<f64>, %arg1: tensor<1xi64>, %arg2: !quantum.reg) -> !quantum.reg attributes {frontend_name = "_pauli_rot_decomposition", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"Hadamard{}{wires:1}{}" = 2 : i64, "MultiRZ{theta:[f64]}{wires:1}{}" = 1 : i64}}, target_gate = "PauliRot{theta:[f64]}{wires:1}{pauli_word = \22X\22}"} {
    %0 = stablehlo.slice %arg1 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
    %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
    %extracted = tensor.extract %1[] : tensor<i64>
    %2 = quantum.extract %arg2[%extracted] : !quantum.reg -> !quantum.bit
    %out_qubits = quantum.custom "Hadamard"() %2 : !quantum.bit
    %3 = quantum.insert %arg2[%extracted], %out_qubits : !quantum.reg, !quantum.bit
    %extracted_0 = tensor.extract %1[] : tensor<i64>
    %extracted_1 = tensor.extract %arg0[] : tensor<f64>
    %4 = quantum.extract %3[%extracted_0] : !quantum.reg -> !quantum.bit
    %out_qubits_2 = quantum.multirz(%extracted_1) %4 : !quantum.bit
    %5 = quantum.insert %3[%extracted_0], %out_qubits_2 : !quantum.reg, !quantum.bit
    %extracted_3 = tensor.extract %1[] : tensor<i64>
    %6 = quantum.extract %5[%extracted_3] : !quantum.reg -> !quantum.bit
    %out_qubits_4 = quantum.custom "Hadamard"() %6 : !quantum.bit
    %7 = quantum.insert %5[%extracted_3], %out_qubits_4 : !quantum.reg, !quantum.bit
    return %7 : !quantum.reg
  }
  func.func private @"__builtin__pauli_rot_decomposition_PauliRot{theta:[f64]}{wires:1}{pauli_word = \22Z\22}"(%arg0: tensor<f64>, %arg1: tensor<1xi64>, %arg2: !quantum.reg) -> !quantum.reg attributes {frontend_name = "_pauli_rot_decomposition", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"MultiRZ{theta:[f64]}{wires:1}{}" = 1 : i64}}, target_gate = "PauliRot{theta:[f64]}{wires:1}{pauli_word = \22Z\22}"} {
    %0 = stablehlo.slice %arg1 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
    %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
    %extracted = tensor.extract %1[] : tensor<i64>
    %extracted_0 = tensor.extract %arg0[] : tensor<f64>
    %2 = quantum.extract %arg2[%extracted] : !quantum.reg -> !quantum.bit
    %out_qubits = quantum.multirz(%extracted_0) %2 : !quantum.bit
    %3 = quantum.insert %arg2[%extracted], %out_qubits : !quantum.reg, !quantum.bit
    return %3 : !quantum.reg
  }
  func.func private @"__builtin__rot_to_rz_ry_rz_Rot{0:[f64],1:[f64],2:[f64]}{wires:1}{}"(%arg0: tensor<f64>, %arg1: tensor<f64>, %arg2: tensor<f64>, %arg3: tensor<1xi64>, %arg4: !quantum.reg) -> !quantum.reg attributes {frontend_name = "_rot_to_rz_ry_rz", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"RY{0:[f64]}{wires:1}{}" = 1 : i64, "RZ{0:[f64]}{wires:1}{}" = 2 : i64}}, target_gate = "Rot{0:[f64],1:[f64],2:[f64]}{wires:1}{}"} {
    %0 = stablehlo.slice %arg3 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
    %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
    %extracted = tensor.extract %1[] : tensor<i64>
    %extracted_0 = tensor.extract %arg0[] : tensor<f64>
    %2 = quantum.extract %arg4[%extracted] : !quantum.reg -> !quantum.bit
    %out_qubits = quantum.custom "RZ"(%extracted_0) %2 : !quantum.bit
    %3 = quantum.insert %arg4[%extracted], %out_qubits : !quantum.reg, !quantum.bit
    %4 = stablehlo.slice %arg3 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
    %5 = stablehlo.reshape %4 : (tensor<1xi64>) -> tensor<i64>
    %extracted_1 = tensor.extract %5[] : tensor<i64>
    %extracted_2 = tensor.extract %arg1[] : tensor<f64>
    %6 = quantum.extract %3[%extracted_1] : !quantum.reg -> !quantum.bit
    %out_qubits_3 = quantum.custom "RY"(%extracted_2) %6 : !quantum.bit
    %7 = quantum.insert %3[%extracted_1], %out_qubits_3 : !quantum.reg, !quantum.bit
    %8 = stablehlo.slice %arg3 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
    %9 = stablehlo.reshape %8 : (tensor<1xi64>) -> tensor<i64>
    %extracted_4 = tensor.extract %9[] : tensor<i64>
    %extracted_5 = tensor.extract %arg2[] : tensor<f64>
    %10 = quantum.extract %7[%extracted_4] : !quantum.reg -> !quantum.bit
    %out_qubits_6 = quantum.custom "RZ"(%extracted_5) %10 : !quantum.bit
    %11 = quantum.insert %7[%extracted_4], %out_qubits_6 : !quantum.reg, !quantum.bit
    return %11 : !quantum.reg
  }
  func.func private @"__builtin__s_phaseshift_S{}{wires:1}{}"(%arg0: tensor<1xi64>, %arg1: !quantum.reg) -> !quantum.reg attributes {frontend_name = "_s_phaseshift", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"PhaseShift{0:[f64]}{wires:1}{}" = 1 : i64}}, target_gate = "S{}{wires:1}{}"} {
    %cst = arith.constant 1.5707963267948966 : f64
    %0 = stablehlo.slice %arg0 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
    %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
    %extracted = tensor.extract %1[] : tensor<i64>
    %2 = quantum.extract %arg1[%extracted] : !quantum.reg -> !quantum.bit
    %out_qubits = quantum.custom "PhaseShift"(%cst) %2 : !quantum.bit
    %3 = quantum.insert %arg1[%extracted], %out_qubits : !quantum.reg, !quantum.bit
    return %3 : !quantum.reg
  }
  func.func private @"__builtin__s_phaseshift_Adjoint(S){}{wires:1}{}"(%arg0: tensor<1xi64>, %arg1: !quantum.reg) -> !quantum.reg attributes {frontend_name = "_s_phaseshift", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"Adjoint(PhaseShift){0:[f64]}{wires:1}{}" = 1 : i64}}, target_gate = "Adjoint(S){}{wires:1}{}"} {
    %cst = arith.constant 1.5707963267948966 : f64
    %0 = catalyst.list_init : <f64>
    %1 = catalyst.list_init : <i64>
    %2 = catalyst.list_init : <i64>
    %3 = stablehlo.slice %arg0 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
    %4 = stablehlo.reshape %3 : (tensor<1xi64>) -> tensor<i64>
    %extracted = tensor.extract %4[] : tensor<i64>
    catalyst.list_push %extracted, %2 : <i64>
    catalyst.list_push %extracted, %2 : <i64>
    %5 = catalyst.list_pop %2 : <i64>
    %6 = quantum.extract %arg1[%5] : !quantum.reg -> !quantum.bit
    %out_qubits = quantum.custom "PhaseShift"(%cst) %6 adj : !quantum.bit
    %7 = quantum.insert %arg1[%5], %out_qubits : !quantum.reg, !quantum.bit
    %8 = catalyst.list_pop %2 : <i64>
    catalyst.list_dealloc %0 : <f64>
    catalyst.list_dealloc %1 : <i64>
    catalyst.list_dealloc %2 : <i64>
    return %7 : !quantum.reg
  }
  func.func private @"__builtin__pauli_rot_decomposition_PauliRot{theta:[f64]}{wires:1}{pauli_word = \22Y\22}"(%arg0: tensor<f64>, %arg1: tensor<1xi64>, %arg2: !quantum.reg) -> !quantum.reg attributes {frontend_name = "_pauli_rot_decomposition", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"MultiRZ{theta:[f64]}{wires:1}{}" = 1 : i64, "RX{0:[f64]}{wires:1}{}" = 2 : i64}}, target_gate = "PauliRot{theta:[f64]}{wires:1}{pauli_word = \22Y\22}"} {
    %cst = arith.constant -1.5707963267948966 : f64
    %cst_0 = arith.constant 1.5707963267948966 : f64
    %0 = stablehlo.slice %arg1 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
    %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
    %extracted = tensor.extract %1[] : tensor<i64>
    %2 = quantum.extract %arg2[%extracted] : !quantum.reg -> !quantum.bit
    %out_qubits = quantum.custom "RX"(%cst_0) %2 : !quantum.bit
    %3 = quantum.insert %arg2[%extracted], %out_qubits : !quantum.reg, !quantum.bit
    %extracted_1 = tensor.extract %1[] : tensor<i64>
    %extracted_2 = tensor.extract %arg0[] : tensor<f64>
    %4 = quantum.extract %3[%extracted_1] : !quantum.reg -> !quantum.bit
    %out_qubits_3 = quantum.multirz(%extracted_2) %4 : !quantum.bit
    %5 = quantum.insert %3[%extracted_1], %out_qubits_3 : !quantum.reg, !quantum.bit
    %extracted_4 = tensor.extract %1[] : tensor<i64>
    %6 = quantum.extract %5[%extracted_4] : !quantum.reg -> !quantum.bit
    %out_qubits_5 = quantum.custom "RX"(%cst) %6 : !quantum.bit
    %7 = quantum.insert %5[%extracted_4], %out_qubits_5 : !quantum.reg, !quantum.bit
    return %7 : !quantum.reg
  }
  func.func private @"__builtin__cphase_to_rz_cnot_ControlledPhaseShift{0:[f64]}{wires:2}{}"(%arg0: tensor<f64>, %arg1: tensor<2xi64>, %arg2: !quantum.reg) -> !quantum.reg attributes {frontend_name = "_cphase_to_rz_cnot", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"CNOT{}{wires:2}{}" = 2 : i64, "GlobalPhase{phi:[f64]}{}{}" = 1 : i64, "RZ{0:[f64]}{wires:1}{}" = 3 : i64}}, target_gate = "ControlledPhaseShift{0:[f64]}{wires:2}{}"} {
    %cst = stablehlo.constant dense<4.000000e+00> : tensor<f64>
    %cst_0 = stablehlo.constant dense<2.000000e+00> : tensor<f64>
    %0 = stablehlo.divide %arg0, %cst_0 : tensor<f64>
    %1 = stablehlo.slice %arg1 [0:1] : (tensor<2xi64>) -> tensor<1xi64>
    %2 = stablehlo.reshape %1 : (tensor<1xi64>) -> tensor<i64>
    %extracted = tensor.extract %2[] : tensor<i64>
    %extracted_1 = tensor.extract %0[] : tensor<f64>
    %3 = quantum.extract %arg2[%extracted] : !quantum.reg -> !quantum.bit
    %out_qubits = quantum.custom "RZ"(%extracted_1) %3 : !quantum.bit
    %4 = quantum.insert %arg2[%extracted], %out_qubits : !quantum.reg, !quantum.bit
    %5 = stablehlo.slice %arg1 [0:1] : (tensor<2xi64>) -> tensor<1xi64>
    %6 = stablehlo.reshape %5 : (tensor<1xi64>) -> tensor<i64>
    %7 = stablehlo.slice %arg1 [1:2] : (tensor<2xi64>) -> tensor<1xi64>
    %8 = stablehlo.reshape %7 : (tensor<1xi64>) -> tensor<i64>
    %extracted_2 = tensor.extract %6[] : tensor<i64>
    %extracted_3 = tensor.extract %8[] : tensor<i64>
    %9 = quantum.extract %4[%extracted_2] : !quantum.reg -> !quantum.bit
    %10 = quantum.extract %4[%extracted_3] : !quantum.reg -> !quantum.bit
    %out_qubits_4:2 = quantum.custom "CNOT"() %9, %10 : !quantum.bit, !quantum.bit
    %11 = quantum.insert %4[%extracted_2], %out_qubits_4#0 : !quantum.reg, !quantum.bit
    %12 = quantum.insert %11[%extracted_3], %out_qubits_4#1 : !quantum.reg, !quantum.bit
    %13 = stablehlo.negate %arg0 : tensor<f64>
    %14 = stablehlo.divide %13, %cst_0 : tensor<f64>
    %15 = stablehlo.slice %arg1 [1:2] : (tensor<2xi64>) -> tensor<1xi64>
    %16 = stablehlo.reshape %15 : (tensor<1xi64>) -> tensor<i64>
    %extracted_5 = tensor.extract %16[] : tensor<i64>
    %extracted_6 = tensor.extract %14[] : tensor<f64>
    %17 = quantum.extract %12[%extracted_5] : !quantum.reg -> !quantum.bit
    %out_qubits_7 = quantum.custom "RZ"(%extracted_6) %17 : !quantum.bit
    %18 = quantum.insert %12[%extracted_5], %out_qubits_7 : !quantum.reg, !quantum.bit
    %19 = stablehlo.slice %arg1 [0:1] : (tensor<2xi64>) -> tensor<1xi64>
    %20 = stablehlo.reshape %19 : (tensor<1xi64>) -> tensor<i64>
    %21 = stablehlo.slice %arg1 [1:2] : (tensor<2xi64>) -> tensor<1xi64>
    %22 = stablehlo.reshape %21 : (tensor<1xi64>) -> tensor<i64>
    %extracted_8 = tensor.extract %20[] : tensor<i64>
    %extracted_9 = tensor.extract %22[] : tensor<i64>
    %23 = quantum.extract %18[%extracted_8] : !quantum.reg -> !quantum.bit
    %24 = quantum.extract %18[%extracted_9] : !quantum.reg -> !quantum.bit
    %out_qubits_10:2 = quantum.custom "CNOT"() %23, %24 : !quantum.bit, !quantum.bit
    %25 = quantum.insert %18[%extracted_8], %out_qubits_10#0 : !quantum.reg, !quantum.bit
    %26 = quantum.insert %25[%extracted_9], %out_qubits_10#1 : !quantum.reg, !quantum.bit
    %27 = stablehlo.divide %arg0, %cst_0 : tensor<f64>
    %28 = stablehlo.slice %arg1 [1:2] : (tensor<2xi64>) -> tensor<1xi64>
    %29 = stablehlo.reshape %28 : (tensor<1xi64>) -> tensor<i64>
    %extracted_11 = tensor.extract %29[] : tensor<i64>
    %extracted_12 = tensor.extract %27[] : tensor<f64>
    %30 = quantum.extract %26[%extracted_11] : !quantum.reg -> !quantum.bit
    %out_qubits_13 = quantum.custom "RZ"(%extracted_12) %30 : !quantum.bit
    %31 = quantum.insert %26[%extracted_11], %out_qubits_13 : !quantum.reg, !quantum.bit
    %32 = stablehlo.negate %arg0 : tensor<f64>
    %33 = stablehlo.divide %32, %cst : tensor<f64>
    %extracted_14 = tensor.extract %33[] : tensor<f64>
    quantum.gphase(%extracted_14)
    return %31 : !quantum.reg
  }
  func.func private @"__builtin__cphase_to_ppr_ControlledPhaseShift{0:[f64]}{wires:2}{}"(%arg0: tensor<f64>, %arg1: tensor<2xi64>, %arg2: !quantum.reg) -> !quantum.reg attributes {frontend_name = "_cphase_to_ppr", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"GlobalPhase{phi:[f64]}{}{}" = 1 : i64, "PauliRot{theta:[f64]}{wires:1}{pauli_word = \22Z\22}" = 2 : i64, "PauliRot{theta:[f64]}{wires:2}{pauli_word = \22ZZ\22}" = 1 : i64}}, target_gate = "ControlledPhaseShift{0:[f64]}{wires:2}{}"} {
    %cst = stablehlo.constant dense<4.000000e+00> : tensor<f64>
    %cst_0 = stablehlo.constant dense<2.000000e+00> : tensor<f64>
    %0 = stablehlo.negate %arg0 : tensor<f64>
    %1 = stablehlo.divide %0, %cst_0 : tensor<f64>
    %2 = stablehlo.slice %arg1 [0:1] : (tensor<2xi64>) -> tensor<1xi64>
    %3 = stablehlo.reshape %2 : (tensor<1xi64>) -> tensor<i64>
    %4 = stablehlo.slice %arg1 [1:2] : (tensor<2xi64>) -> tensor<1xi64>
    %5 = stablehlo.reshape %4 : (tensor<1xi64>) -> tensor<i64>
    %extracted = tensor.extract %3[] : tensor<i64>
    %extracted_1 = tensor.extract %5[] : tensor<i64>
    %extracted_2 = tensor.extract %1[] : tensor<f64>
    %6 = quantum.extract %arg2[%extracted] : !quantum.reg -> !quantum.bit
    %7 = quantum.extract %arg2[%extracted_1] : !quantum.reg -> !quantum.bit
    %out_qubits:2 = quantum.paulirot ["Z", "Z"](%extracted_2) %6, %7 : !quantum.bit, !quantum.bit
    %8 = quantum.insert %arg2[%extracted], %out_qubits#0 : !quantum.reg, !quantum.bit
    %9 = quantum.insert %8[%extracted_1], %out_qubits#1 : !quantum.reg, !quantum.bit
    %10 = stablehlo.divide %arg0, %cst_0 : tensor<f64>
    %11 = stablehlo.slice %arg1 [1:2] : (tensor<2xi64>) -> tensor<1xi64>
    %12 = stablehlo.reshape %11 : (tensor<1xi64>) -> tensor<i64>
    %extracted_3 = tensor.extract %12[] : tensor<i64>
    %extracted_4 = tensor.extract %10[] : tensor<f64>
    %13 = quantum.extract %9[%extracted_3] : !quantum.reg -> !quantum.bit
    %out_qubits_5 = quantum.paulirot ["Z"](%extracted_4) %13 : !quantum.bit
    %14 = quantum.insert %9[%extracted_3], %out_qubits_5 : !quantum.reg, !quantum.bit
    %15 = stablehlo.divide %arg0, %cst_0 : tensor<f64>
    %16 = stablehlo.slice %arg1 [0:1] : (tensor<2xi64>) -> tensor<1xi64>
    %17 = stablehlo.reshape %16 : (tensor<1xi64>) -> tensor<i64>
    %extracted_6 = tensor.extract %17[] : tensor<i64>
    %extracted_7 = tensor.extract %15[] : tensor<f64>
    %18 = quantum.extract %14[%extracted_6] : !quantum.reg -> !quantum.bit
    %out_qubits_8 = quantum.paulirot ["Z"](%extracted_7) %18 : !quantum.bit
    %19 = quantum.insert %14[%extracted_6], %out_qubits_8 : !quantum.reg, !quantum.bit
    %20 = stablehlo.negate %arg0 : tensor<f64>
    %21 = stablehlo.divide %20, %cst : tensor<f64>
    %extracted_9 = tensor.extract %21[] : tensor<f64>
    quantum.gphase(%extracted_9)
    return %19 : !quantum.reg
  }
  func.func private @"__builtin__ppr_to_paulirot_PPR{}{wires:2}{angle_denominator = 4 : i64, pauli_word = \22ZZ\22}"(%arg0: tensor<2xi64>, %arg1: !quantum.reg) -> !quantum.reg attributes {frontend_name = "_ppr_to_paulirot", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"PauliRot{theta:[f64]}{wires:2}{pauli_word = \22ZZ\22}" = 1 : i64}}, target_gate = "PPR{}{wires:2}{angle_denominator = 4 : i64, pauli_word = \22ZZ\22}"} {
    %cst = arith.constant 1.5707963267948966 : f64
    %0 = stablehlo.slice %arg0 [0:1] : (tensor<2xi64>) -> tensor<1xi64>
    %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
    %2 = stablehlo.slice %arg0 [1:2] : (tensor<2xi64>) -> tensor<1xi64>
    %3 = stablehlo.reshape %2 : (tensor<1xi64>) -> tensor<i64>
    %extracted = tensor.extract %1[] : tensor<i64>
    %extracted_0 = tensor.extract %3[] : tensor<i64>
    %4 = quantum.extract %arg1[%extracted] : !quantum.reg -> !quantum.bit
    %5 = quantum.extract %arg1[%extracted_0] : !quantum.reg -> !quantum.bit
    %out_qubits:2 = quantum.paulirot ["Z", "Z"](%cst) %4, %5 : !quantum.bit, !quantum.bit
    %6 = quantum.insert %arg1[%extracted], %out_qubits#0 : !quantum.reg, !quantum.bit
    %7 = quantum.insert %6[%extracted_0], %out_qubits#1 : !quantum.reg, !quantum.bit
    return %7 : !quantum.reg
  }
  func.func private @"__builtin_adjoint_rotation_Adjoint(RZ){0:[f64]}{wires:1}{}"(%arg0: tensor<f64>, %arg1: tensor<1xi64>, %arg2: !quantum.reg) -> !quantum.reg attributes {frontend_name = "adjoint_rotation", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"RZ{0:[f64]}{wires:1}{}" = 1 : i64}}, target_gate = "Adjoint(RZ){0:[f64]}{wires:1}{}"} {
    %0 = stablehlo.slice %arg1 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
    %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
    %2 = stablehlo.negate %arg0 : tensor<f64>
    %extracted = tensor.extract %1[] : tensor<i64>
    %extracted_0 = tensor.extract %2[] : tensor<f64>
    %3 = quantum.extract %arg2[%extracted] : !quantum.reg -> !quantum.bit
    %out_qubits = quantum.custom "RZ"(%extracted_0) %3 : !quantum.bit
    %4 = quantum.insert %arg2[%extracted], %out_qubits : !quantum.reg, !quantum.bit
    return %4 : !quantum.reg
  }
  func.func private @"__builtin__rz_to_ps_Adjoint(RZ){0:[f64]}{wires:1}{}"(%arg0: tensor<f64>, %arg1: tensor<1xi64>, %arg2: !quantum.reg) -> !quantum.reg attributes {frontend_name = "_rz_to_ps", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"Adjoint(GlobalPhase){phi:[f64]}{}{}" = 1 : i64, "Adjoint(PhaseShift){0:[f64]}{wires:1}{}" = 1 : i64}}, target_gate = "Adjoint(RZ){0:[f64]}{wires:1}{}"} {
    %cst = stablehlo.constant dense<2.000000e+00> : tensor<f64>
    %0 = catalyst.list_init : <f64>
    %1 = catalyst.list_init : <i64>
    %2 = catalyst.list_init : <i64>
    %3 = stablehlo.slice %arg1 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
    %4 = stablehlo.reshape %3 : (tensor<1xi64>) -> tensor<i64>
    %extracted = tensor.extract %4[] : tensor<i64>
    %extracted_0 = tensor.extract %arg0[] : tensor<f64>
    catalyst.list_push %extracted, %2 : <i64>
    catalyst.list_push %extracted, %2 : <i64>
    %5 = stablehlo.divide %arg0, %cst : tensor<f64>
    %extracted_1 = tensor.extract %5[] : tensor<f64>
    quantum.gphase(%extracted_1) adj
    %6 = catalyst.list_pop %2 : <i64>
    %7 = quantum.extract %arg2[%6] : !quantum.reg -> !quantum.bit
    %out_qubits = quantum.custom "PhaseShift"(%extracted_0) %7 adj : !quantum.bit
    %8 = quantum.insert %arg2[%6], %out_qubits : !quantum.reg, !quantum.bit
    %9 = catalyst.list_pop %2 : <i64>
    catalyst.list_dealloc %0 : <f64>
    catalyst.list_dealloc %1 : <i64>
    catalyst.list_dealloc %2 : <i64>
    return %8 : !quantum.reg
  }
  func.func private @"__builtin__rz_to_rot_Adjoint(RZ){0:[f64]}{wires:1}{}"(%arg0: tensor<f64>, %arg1: tensor<1xi64>, %arg2: !quantum.reg) -> !quantum.reg attributes {frontend_name = "_rz_to_rot", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"Adjoint(Rot){0:[f64],1:[f64],2:[f64]}{wires:1}{}" = 1 : i64}}, target_gate = "Adjoint(RZ){0:[f64]}{wires:1}{}"} {
    %c = stablehlo.constant dense<0> : tensor<i64>
    %0 = catalyst.list_init : <f64>
    %1 = catalyst.list_init : <i64>
    %2 = catalyst.list_init : <i64>
    %3 = stablehlo.slice %arg1 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
    %4 = stablehlo.reshape %3 : (tensor<1xi64>) -> tensor<i64>
    %extracted = tensor.extract %4[] : tensor<i64>
    %5 = stablehlo.convert %c : (tensor<i64>) -> tensor<f64>
    %extracted_0 = tensor.extract %5[] : tensor<f64>
    %6 = stablehlo.convert %c : (tensor<i64>) -> tensor<f64>
    %extracted_1 = tensor.extract %6[] : tensor<f64>
    %extracted_2 = tensor.extract %arg0[] : tensor<f64>
    catalyst.list_push %extracted, %2 : <i64>
    catalyst.list_push %extracted, %2 : <i64>
    %7 = catalyst.list_pop %2 : <i64>
    %8 = quantum.extract %arg2[%7] : !quantum.reg -> !quantum.bit
    %out_qubits = quantum.custom "Rot"(%extracted_0, %extracted_1, %extracted_2) %8 adj : !quantum.bit
    %9 = quantum.insert %arg2[%7], %out_qubits : !quantum.reg, !quantum.bit
    %10 = catalyst.list_pop %2 : <i64>
    catalyst.list_dealloc %0 : <f64>
    catalyst.list_dealloc %1 : <i64>
    catalyst.list_dealloc %2 : <i64>
    return %9 : !quantum.reg
  }
  func.func private @"__builtin__rz_to_ry_rx_Adjoint(RZ){0:[f64]}{wires:1}{}"(%arg0: tensor<f64>, %arg1: tensor<1xi64>, %arg2: !quantum.reg) -> !quantum.reg attributes {frontend_name = "_rz_to_ry_rx", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"Adjoint(RX){0:[f64]}{wires:1}{}" = 1 : i64, "Adjoint(RY){0:[f64]}{wires:1}{}" = 2 : i64}}, target_gate = "Adjoint(RZ){0:[f64]}{wires:1}{}"} {
    %cst = arith.constant -1.5707963267948966 : f64
    %cst_0 = arith.constant 1.5707963267948966 : f64
    %0 = catalyst.list_init : <f64>
    %1 = catalyst.list_init : <i64>
    %2 = catalyst.list_init : <i64>
    %3 = stablehlo.slice %arg1 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
    %4 = stablehlo.reshape %3 : (tensor<1xi64>) -> tensor<i64>
    %extracted = tensor.extract %4[] : tensor<i64>
    catalyst.list_push %extracted, %2 : <i64>
    catalyst.list_push %extracted, %2 : <i64>
    %5 = stablehlo.slice %arg1 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
    %6 = stablehlo.reshape %5 : (tensor<1xi64>) -> tensor<i64>
    %extracted_1 = tensor.extract %6[] : tensor<i64>
    %extracted_2 = tensor.extract %arg0[] : tensor<f64>
    catalyst.list_push %extracted_1, %2 : <i64>
    catalyst.list_push %extracted_1, %2 : <i64>
    %7 = stablehlo.slice %arg1 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
    %8 = stablehlo.reshape %7 : (tensor<1xi64>) -> tensor<i64>
    %extracted_3 = tensor.extract %8[] : tensor<i64>
    catalyst.list_push %extracted_3, %2 : <i64>
    catalyst.list_push %extracted_3, %2 : <i64>
    %9 = catalyst.list_pop %2 : <i64>
    %10 = quantum.extract %arg2[%9] : !quantum.reg -> !quantum.bit
    %out_qubits = quantum.custom "RY"(%cst) %10 adj : !quantum.bit
    %11 = quantum.insert %arg2[%9], %out_qubits : !quantum.reg, !quantum.bit
    %12 = catalyst.list_pop %2 : <i64>
    %13 = catalyst.list_pop %2 : <i64>
    %14 = quantum.extract %11[%13] : !quantum.reg -> !quantum.bit
    %out_qubits_4 = quantum.custom "RX"(%extracted_2) %14 adj : !quantum.bit
    %15 = quantum.insert %11[%13], %out_qubits_4 : !quantum.reg, !quantum.bit
    %16 = catalyst.list_pop %2 : <i64>
    %17 = catalyst.list_pop %2 : <i64>
    %18 = quantum.extract %15[%17] : !quantum.reg -> !quantum.bit
    %out_qubits_5 = quantum.custom "RY"(%cst_0) %18 adj : !quantum.bit
    %19 = quantum.insert %15[%17], %out_qubits_5 : !quantum.reg, !quantum.bit
    %20 = catalyst.list_pop %2 : <i64>
    catalyst.list_dealloc %0 : <f64>
    catalyst.list_dealloc %1 : <i64>
    catalyst.list_dealloc %2 : <i64>
    return %19 : !quantum.reg
  }
  func.func private @"__builtin__rz_to_ppr_Adjoint(RZ){0:[f64]}{wires:1}{}"(%arg0: tensor<f64>, %arg1: tensor<1xi64>, %arg2: !quantum.reg) -> !quantum.reg attributes {frontend_name = "_rz_to_ppr", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"Adjoint(PauliRot){theta:[f64]}{wires:1}{pauli_word = \22Z\22}" = 1 : i64}}, target_gate = "Adjoint(RZ){0:[f64]}{wires:1}{}"} {
    %0 = catalyst.list_init : <f64>
    %1 = catalyst.list_init : <i64>
    %2 = catalyst.list_init : <i64>
    %3 = stablehlo.slice %arg1 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
    %4 = stablehlo.reshape %3 : (tensor<1xi64>) -> tensor<i64>
    %extracted = tensor.extract %4[] : tensor<i64>
    %extracted_0 = tensor.extract %arg0[] : tensor<f64>
    catalyst.list_push %extracted, %2 : <i64>
    catalyst.list_push %extracted, %2 : <i64>
    %5 = catalyst.list_pop %2 : <i64>
    %6 = quantum.extract %arg2[%5] : !quantum.reg -> !quantum.bit
    %out_qubits = quantum.paulirot ["Z"](%extracted_0) %6 adj : !quantum.bit
    %7 = quantum.insert %arg2[%5], %out_qubits : !quantum.reg, !quantum.bit
    %8 = catalyst.list_pop %2 : <i64>
    catalyst.list_dealloc %0 : <f64>
    catalyst.list_dealloc %1 : <i64>
    catalyst.list_dealloc %2 : <i64>
    return %7 : !quantum.reg
  }
  func.func private @"__builtin__rz_to_rx_cliff_Adjoint(RZ){0:[f64]}{wires:1}{}"(%arg0: tensor<f64>, %arg1: tensor<1xi64>, %arg2: !quantum.reg) -> !quantum.reg attributes {frontend_name = "_rz_to_rx_cliff", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"Adjoint(Hadamard){}{wires:1}{}" = 2 : i64, "Adjoint(RX){0:[f64]}{wires:1}{}" = 1 : i64}}, target_gate = "Adjoint(RZ){0:[f64]}{wires:1}{}"} {
    %0 = catalyst.list_init : <f64>
    %1 = catalyst.list_init : <i64>
    %2 = catalyst.list_init : <i64>
    %3 = stablehlo.slice %arg1 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
    %4 = stablehlo.reshape %3 : (tensor<1xi64>) -> tensor<i64>
    %5 = stablehlo.slice %arg1 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
    %6 = stablehlo.reshape %5 : (tensor<1xi64>) -> tensor<i64>
    %7 = stablehlo.slice %arg1 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
    %8 = stablehlo.reshape %7 : (tensor<1xi64>) -> tensor<i64>
    %extracted = tensor.extract %4[] : tensor<i64>
    catalyst.list_push %extracted, %2 : <i64>
    catalyst.list_push %extracted, %2 : <i64>
    %extracted_0 = tensor.extract %6[] : tensor<i64>
    %extracted_1 = tensor.extract %arg0[] : tensor<f64>
    catalyst.list_push %extracted_0, %2 : <i64>
    catalyst.list_push %extracted_0, %2 : <i64>
    %extracted_2 = tensor.extract %8[] : tensor<i64>
    catalyst.list_push %extracted_2, %2 : <i64>
    catalyst.list_push %extracted_2, %2 : <i64>
    %9 = catalyst.list_pop %2 : <i64>
    %10 = quantum.extract %arg2[%9] : !quantum.reg -> !quantum.bit
    %out_qubits = quantum.custom "Hadamard"() %10 adj : !quantum.bit
    %11 = quantum.insert %arg2[%9], %out_qubits : !quantum.reg, !quantum.bit
    %12 = catalyst.list_pop %2 : <i64>
    %13 = catalyst.list_pop %2 : <i64>
    %14 = quantum.extract %11[%13] : !quantum.reg -> !quantum.bit
    %out_qubits_3 = quantum.custom "RX"(%extracted_1) %14 adj : !quantum.bit
    %15 = quantum.insert %11[%13], %out_qubits_3 : !quantum.reg, !quantum.bit
    %16 = catalyst.list_pop %2 : <i64>
    %17 = catalyst.list_pop %2 : <i64>
    %18 = quantum.extract %15[%17] : !quantum.reg -> !quantum.bit
    %out_qubits_4 = quantum.custom "Hadamard"() %18 adj : !quantum.bit
    %19 = quantum.insert %15[%17], %out_qubits_4 : !quantum.reg, !quantum.bit
    %20 = catalyst.list_pop %2 : <i64>
    catalyst.list_dealloc %0 : <f64>
    catalyst.list_dealloc %1 : <i64>
    catalyst.list_dealloc %2 : <i64>
    return %19 : !quantum.reg
  }
  func.func private @"__builtin__rz_to_ry_cliff_Adjoint(RZ){0:[f64]}{wires:1}{}"(%arg0: tensor<f64>, %arg1: tensor<1xi64>, %arg2: !quantum.reg) -> !quantum.reg attributes {frontend_name = "_rz_to_ry_cliff", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"Adjoint(Hadamard){}{wires:1}{}" = 2 : i64, "Adjoint(RY){0:[f64]}{wires:1}{}" = 1 : i64, "Adjoint(S){}{wires:1}{}" = 1 : i64, "S{}{wires:1}{}" = 1 : i64}}, target_gate = "Adjoint(RZ){0:[f64]}{wires:1}{}"} {
    %0 = catalyst.list_init : <f64>
    %1 = catalyst.list_init : <i64>
    %2 = catalyst.list_init : <i64>
    %3 = stablehlo.slice %arg1 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
    %4 = stablehlo.reshape %3 : (tensor<1xi64>) -> tensor<i64>
    %5 = stablehlo.slice %arg1 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
    %6 = stablehlo.reshape %5 : (tensor<1xi64>) -> tensor<i64>
    %extracted = tensor.extract %6[] : tensor<i64>
    catalyst.list_push %extracted, %2 : <i64>
    catalyst.list_push %extracted, %2 : <i64>
    %7 = stablehlo.slice %arg1 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
    %8 = stablehlo.reshape %7 : (tensor<1xi64>) -> tensor<i64>
    %extracted_0 = tensor.extract %8[] : tensor<i64>
    catalyst.list_push %extracted_0, %2 : <i64>
    catalyst.list_push %extracted_0, %2 : <i64>
    %extracted_1 = tensor.extract %4[] : tensor<i64>
    %extracted_2 = tensor.extract %arg0[] : tensor<f64>
    catalyst.list_push %extracted_1, %2 : <i64>
    catalyst.list_push %extracted_1, %2 : <i64>
    %9 = stablehlo.slice %arg1 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
    %10 = stablehlo.reshape %9 : (tensor<1xi64>) -> tensor<i64>
    %extracted_3 = tensor.extract %10[] : tensor<i64>
    catalyst.list_push %extracted_3, %2 : <i64>
    catalyst.list_push %extracted_3, %2 : <i64>
    %11 = stablehlo.slice %arg1 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
    %12 = stablehlo.reshape %11 : (tensor<1xi64>) -> tensor<i64>
    %extracted_4 = tensor.extract %12[] : tensor<i64>
    catalyst.list_push %extracted_4, %2 : <i64>
    catalyst.list_push %extracted_4, %2 : <i64>
    %13 = catalyst.list_pop %2 : <i64>
    %14 = quantum.extract %arg2[%13] : !quantum.reg -> !quantum.bit
    %out_qubits = quantum.custom "Hadamard"() %14 adj : !quantum.bit
    %15 = quantum.insert %arg2[%13], %out_qubits : !quantum.reg, !quantum.bit
    %16 = catalyst.list_pop %2 : <i64>
    %17 = catalyst.list_pop %2 : <i64>
    %18 = quantum.extract %15[%17] : !quantum.reg -> !quantum.bit
    %out_qubits_5 = quantum.custom "S"() %18 : !quantum.bit
    %19 = quantum.insert %15[%17], %out_qubits_5 : !quantum.reg, !quantum.bit
    %20 = catalyst.list_pop %2 : <i64>
    %21 = catalyst.list_pop %2 : <i64>
    %22 = quantum.extract %19[%21] : !quantum.reg -> !quantum.bit
    %out_qubits_6 = quantum.custom "RY"(%extracted_2) %22 adj : !quantum.bit
    %23 = quantum.insert %19[%21], %out_qubits_6 : !quantum.reg, !quantum.bit
    %24 = catalyst.list_pop %2 : <i64>
    %25 = catalyst.list_pop %2 : <i64>
    %26 = quantum.extract %23[%25] : !quantum.reg -> !quantum.bit
    %out_qubits_7 = quantum.custom "S"() %26 adj : !quantum.bit
    %27 = quantum.insert %23[%25], %out_qubits_7 : !quantum.reg, !quantum.bit
    %28 = catalyst.list_pop %2 : <i64>
    %29 = catalyst.list_pop %2 : <i64>
    %30 = quantum.extract %27[%29] : !quantum.reg -> !quantum.bit
    %out_qubits_8 = quantum.custom "Hadamard"() %30 adj : !quantum.bit
    %31 = quantum.insert %27[%29], %out_qubits_8 : !quantum.reg, !quantum.bit
    %32 = catalyst.list_pop %2 : <i64>
    catalyst.list_dealloc %0 : <f64>
    catalyst.list_dealloc %1 : <i64>
    catalyst.list_dealloc %2 : <i64>
    return %31 : !quantum.reg
  }
  func.func private @"__builtin_adjoint_rotation_Adjoint(GlobalPhase){phi:[f64]}{}{}"(%arg0: tensor<f64>, %arg1: tensor<0xi64>) attributes {frontend_name = "adjoint_rotation", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"GlobalPhase{phi:[f64]}{}{}" = 1 : i64}}, target_gate = "Adjoint(GlobalPhase){phi:[f64]}{}{}"} {
    %0 = stablehlo.negate %arg0 : tensor<f64>
    %extracted = tensor.extract %0[] : tensor<f64>
    quantum.gphase(%extracted)
    return
  }
  func.func private @"__builtin__multi_rz_decomposition_MultiRZ{theta:[f64]}{wires:2}{}"(%arg0: tensor<f64>, %arg1: tensor<2xi64>, %arg2: !quantum.reg) -> !quantum.reg attributes {frontend_name = "_multi_rz_decomposition", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"CNOT{}{wires:2}{}" = 2 : i64, "RZ{0:[f64]}{wires:1}{}" = 1 : i64}}, target_gate = "MultiRZ{theta:[f64]}{wires:2}{}"} {
    %cst = arith.constant dense<1> : tensor<i64>
    %c = stablehlo.constant dense<2> : tensor<i64>
    %c_0 = stablehlo.constant dense<-1> : tensor<i64>
    %c_1 = stablehlo.constant dense<0> : tensor<i64>
    %c_2 = stablehlo.constant dense<1> : tensor<i64>
    %cst_3 = arith.constant dense<0> : tensor<i64>
    %0 = stablehlo.multiply %c_0, %cst_3 : tensor<i64>
    %1 = stablehlo.add %c_2, %0 : tensor<i64>
    %2 = stablehlo.compare  LT, %1, %c_1,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
    %3 = stablehlo.convert %1 : tensor<i64>
    %4 = stablehlo.add %3, %c : tensor<i64>
    %5 = stablehlo.select %2, %4, %1 : tensor<i1>, tensor<i64>
    %6 = stablehlo.dynamic_slice %arg1, %5, sizes = [1] : (tensor<2xi64>, tensor<i64>) -> tensor<1xi64>
    %7 = stablehlo.reshape %6 : (tensor<1xi64>) -> tensor<i64>
    %8 = stablehlo.subtract %1, %c_2 : tensor<i64>
    %9 = stablehlo.compare  LT, %8, %c_1,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
    %10 = stablehlo.convert %8 : tensor<i64>
    %11 = stablehlo.add %10, %c : tensor<i64>
    %12 = stablehlo.select %9, %11, %8 : tensor<i1>, tensor<i64>
    %13 = stablehlo.dynamic_slice %arg1, %12, sizes = [1] : (tensor<2xi64>, tensor<i64>) -> tensor<1xi64>
    %14 = stablehlo.reshape %13 : (tensor<1xi64>) -> tensor<i64>
    %extracted = tensor.extract %7[] : tensor<i64>
    %extracted_4 = tensor.extract %14[] : tensor<i64>
    %15 = quantum.extract %arg2[%extracted] : !quantum.reg -> !quantum.bit
    %16 = quantum.extract %arg2[%extracted_4] : !quantum.reg -> !quantum.bit
    %out_qubits:2 = quantum.custom "CNOT"() %15, %16 : !quantum.bit, !quantum.bit
    %17 = quantum.insert %arg2[%extracted], %out_qubits#0 : !quantum.reg, !quantum.bit
    %18 = quantum.insert %17[%extracted_4], %out_qubits#1 : !quantum.reg, !quantum.bit
    %19 = stablehlo.slice %arg1 [0:1] : (tensor<2xi64>) -> tensor<1xi64>
    %20 = stablehlo.reshape %19 : (tensor<1xi64>) -> tensor<i64>
    %extracted_5 = tensor.extract %20[] : tensor<i64>
    %extracted_6 = tensor.extract %arg0[] : tensor<f64>
    %21 = quantum.extract %18[%extracted_5] : !quantum.reg -> !quantum.bit
    %out_qubits_7 = quantum.custom "RZ"(%extracted_6) %21 : !quantum.bit
    %22 = quantum.insert %18[%extracted_5], %out_qubits_7 : !quantum.reg, !quantum.bit
    %23 = stablehlo.compare  LT, %cst, %c_1,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
    %24 = stablehlo.convert %cst : tensor<i64>
    %25 = stablehlo.add %24, %c : tensor<i64>
    %26 = stablehlo.select %23, %25, %cst : tensor<i1>, tensor<i64>
    %27 = stablehlo.dynamic_slice %arg1, %26, sizes = [1] : (tensor<2xi64>, tensor<i64>) -> tensor<1xi64>
    %28 = stablehlo.reshape %27 : (tensor<1xi64>) -> tensor<i64>
    %29 = stablehlo.subtract %cst, %c_2 : tensor<i64>
    %30 = stablehlo.compare  LT, %29, %c_1,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
    %31 = stablehlo.convert %29 : tensor<i64>
    %32 = stablehlo.add %31, %c : tensor<i64>
    %33 = stablehlo.select %30, %32, %29 : tensor<i1>, tensor<i64>
    %34 = stablehlo.dynamic_slice %arg1, %33, sizes = [1] : (tensor<2xi64>, tensor<i64>) -> tensor<1xi64>
    %35 = stablehlo.reshape %34 : (tensor<1xi64>) -> tensor<i64>
    %extracted_8 = tensor.extract %28[] : tensor<i64>
    %extracted_9 = tensor.extract %35[] : tensor<i64>
    %36 = quantum.extract %22[%extracted_8] : !quantum.reg -> !quantum.bit
    %37 = quantum.extract %22[%extracted_9] : !quantum.reg -> !quantum.bit
    %out_qubits_10:2 = quantum.custom "CNOT"() %36, %37 : !quantum.bit, !quantum.bit
    %38 = quantum.insert %22[%extracted_8], %out_qubits_10#0 : !quantum.reg, !quantum.bit
    %39 = quantum.insert %38[%extracted_9], %out_qubits_10#1 : !quantum.reg, !quantum.bit
    return %39 : !quantum.reg
  }
  func.func private @"__builtin__multi_rz_decomposition_MultiRZ{theta:[f64]}{wires:3}{}"(%arg0: tensor<f64>, %arg1: tensor<3xi64>, %arg2: !quantum.reg) -> !quantum.reg attributes {frontend_name = "_multi_rz_decomposition", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"CNOT{}{wires:2}{}" = 4 : i64, "RZ{0:[f64]}{wires:1}{}" = 1 : i64}}, target_gate = "MultiRZ{theta:[f64]}{wires:3}{}"} {
    %c3 = arith.constant 3 : index
    %c = stablehlo.constant dense<3> : tensor<i64>
    %c_0 = stablehlo.constant dense<-1> : tensor<i64>
    %c_1 = stablehlo.constant dense<0> : tensor<i64>
    %c_2 = stablehlo.constant dense<2> : tensor<i64>
    %c_3 = stablehlo.constant dense<1> : tensor<i64>
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %0 = scf.for %arg3 = %c0 to %c2 step %c1 iter_args(%arg4 = %arg2) -> (!quantum.reg) {
      %6 = arith.index_cast %arg3 : index to i64
      %from_elements = tensor.from_elements %6 : tensor<i64>
      %7 = stablehlo.multiply %c_0, %from_elements : tensor<i64>
      %8 = stablehlo.add %c_2, %7 : tensor<i64>
      %9 = stablehlo.compare  LT, %8, %c_1,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
      %10 = stablehlo.convert %8 : tensor<i64>
      %11 = stablehlo.add %10, %c : tensor<i64>
      %12 = stablehlo.select %9, %11, %8 : tensor<i1>, tensor<i64>
      %13 = stablehlo.dynamic_slice %arg1, %12, sizes = [1] : (tensor<3xi64>, tensor<i64>) -> tensor<1xi64>
      %14 = stablehlo.reshape %13 : (tensor<1xi64>) -> tensor<i64>
      %15 = stablehlo.subtract %8, %c_3 : tensor<i64>
      %16 = stablehlo.compare  LT, %15, %c_1,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
      %17 = stablehlo.convert %15 : tensor<i64>
      %18 = stablehlo.add %17, %c : tensor<i64>
      %19 = stablehlo.select %16, %18, %15 : tensor<i1>, tensor<i64>
      %20 = stablehlo.dynamic_slice %arg1, %19, sizes = [1] : (tensor<3xi64>, tensor<i64>) -> tensor<1xi64>
      %21 = stablehlo.reshape %20 : (tensor<1xi64>) -> tensor<i64>
      %extracted_5 = tensor.extract %14[] : tensor<i64>
      %extracted_6 = tensor.extract %21[] : tensor<i64>
      %22 = quantum.extract %arg4[%extracted_5] : !quantum.reg -> !quantum.bit
      %23 = quantum.extract %arg4[%extracted_6] : !quantum.reg -> !quantum.bit
      %out_qubits_7:2 = quantum.custom "CNOT"() %22, %23 : !quantum.bit, !quantum.bit
      %24 = quantum.insert %arg4[%extracted_5], %out_qubits_7#0 : !quantum.reg, !quantum.bit
      %25 = quantum.insert %24[%extracted_6], %out_qubits_7#1 : !quantum.reg, !quantum.bit
      scf.yield %25 : !quantum.reg
    }
    %1 = stablehlo.slice %arg1 [0:1] : (tensor<3xi64>) -> tensor<1xi64>
    %2 = stablehlo.reshape %1 : (tensor<1xi64>) -> tensor<i64>
    %extracted = tensor.extract %2[] : tensor<i64>
    %extracted_4 = tensor.extract %arg0[] : tensor<f64>
    %3 = quantum.extract %0[%extracted] : !quantum.reg -> !quantum.bit
    %out_qubits = quantum.custom "RZ"(%extracted_4) %3 : !quantum.bit
    %4 = quantum.insert %0[%extracted], %out_qubits : !quantum.reg, !quantum.bit
    %5 = scf.for %arg3 = %c1 to %c3 step %c1 iter_args(%arg4 = %4) -> (!quantum.reg) {
      %6 = arith.index_cast %arg3 : index to i64
      %from_elements = tensor.from_elements %6 : tensor<i64>
      %7 = stablehlo.compare  LT, %from_elements, %c_1,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
      %8 = stablehlo.convert %from_elements : tensor<i64>
      %9 = stablehlo.add %8, %c : tensor<i64>
      %10 = stablehlo.select %7, %9, %from_elements : tensor<i1>, tensor<i64>
      %11 = stablehlo.dynamic_slice %arg1, %10, sizes = [1] : (tensor<3xi64>, tensor<i64>) -> tensor<1xi64>
      %12 = stablehlo.reshape %11 : (tensor<1xi64>) -> tensor<i64>
      %13 = stablehlo.subtract %from_elements, %c_3 : tensor<i64>
      %14 = stablehlo.compare  LT, %13, %c_1,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
      %15 = stablehlo.convert %13 : tensor<i64>
      %16 = stablehlo.add %15, %c : tensor<i64>
      %17 = stablehlo.select %14, %16, %13 : tensor<i1>, tensor<i64>
      %18 = stablehlo.dynamic_slice %arg1, %17, sizes = [1] : (tensor<3xi64>, tensor<i64>) -> tensor<1xi64>
      %19 = stablehlo.reshape %18 : (tensor<1xi64>) -> tensor<i64>
      %extracted_5 = tensor.extract %12[] : tensor<i64>
      %extracted_6 = tensor.extract %19[] : tensor<i64>
      %20 = quantum.extract %arg4[%extracted_5] : !quantum.reg -> !quantum.bit
      %21 = quantum.extract %arg4[%extracted_6] : !quantum.reg -> !quantum.bit
      %out_qubits_7:2 = quantum.custom "CNOT"() %20, %21 : !quantum.bit, !quantum.bit
      %22 = quantum.insert %arg4[%extracted_5], %out_qubits_7#0 : !quantum.reg, !quantum.bit
      %23 = quantum.insert %22[%extracted_6], %out_qubits_7#1 : !quantum.reg, !quantum.bit
      scf.yield %23 : !quantum.reg
    }
    return %5 : !quantum.reg
  }
  func.func private @"__builtin__multi_rz_decomposition_MultiRZ{theta:[f64]}{wires:1}{}"(%arg0: tensor<f64>, %arg1: tensor<1xi64>, %arg2: !quantum.reg) -> !quantum.reg attributes {frontend_name = "_multi_rz_decomposition", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"RZ{0:[f64]}{wires:1}{}" = 1 : i64}}, target_gate = "MultiRZ{theta:[f64]}{wires:1}{}"} {
    %0 = stablehlo.slice %arg1 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
    %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
    %extracted = tensor.extract %1[] : tensor<i64>
    %extracted_0 = tensor.extract %arg0[] : tensor<f64>
    %2 = quantum.extract %arg2[%extracted] : !quantum.reg -> !quantum.bit
    %out_qubits = quantum.custom "RZ"(%extracted_0) %2 : !quantum.bit
    %3 = quantum.insert %arg2[%extracted], %out_qubits : !quantum.reg, !quantum.bit
    return %3 : !quantum.reg
  }
  func.func private @"__builtin__adjoint_rot_Adjoint(Rot){0:[f64],1:[f64],2:[f64]}{wires:1}{}"(%arg0: tensor<f64>, %arg1: tensor<f64>, %arg2: tensor<f64>, %arg3: tensor<1xi64>, %arg4: !quantum.reg) -> !quantum.reg attributes {frontend_name = "_adjoint_rot", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"Rot{0:[f64],1:[f64],2:[f64]}{wires:1}{}" = 1 : i64}}, target_gate = "Adjoint(Rot){0:[f64],1:[f64],2:[f64]}{wires:1}{}"} {
    %0 = stablehlo.slice %arg3 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
    %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
    %2 = stablehlo.negate %arg2 : tensor<f64>
    %3 = stablehlo.negate %arg1 : tensor<f64>
    %4 = stablehlo.negate %arg0 : tensor<f64>
    %extracted = tensor.extract %1[] : tensor<i64>
    %extracted_0 = tensor.extract %2[] : tensor<f64>
    %extracted_1 = tensor.extract %3[] : tensor<f64>
    %extracted_2 = tensor.extract %4[] : tensor<f64>
    %5 = quantum.extract %arg4[%extracted] : !quantum.reg -> !quantum.bit
    %out_qubits = quantum.custom "Rot"(%extracted_0, %extracted_1, %extracted_2) %5 : !quantum.bit
    %6 = quantum.insert %arg4[%extracted], %out_qubits : !quantum.reg, !quantum.bit
    return %6 : !quantum.reg
  }
  func.func private @"__builtin__rot_to_rz_ry_rz_Adjoint(Rot){0:[f64],1:[f64],2:[f64]}{wires:1}{}"(%arg0: tensor<f64>, %arg1: tensor<f64>, %arg2: tensor<f64>, %arg3: tensor<1xi64>, %arg4: !quantum.reg) -> !quantum.reg attributes {frontend_name = "_rot_to_rz_ry_rz", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"Adjoint(RY){0:[f64]}{wires:1}{}" = 1 : i64, "Adjoint(RZ){0:[f64]}{wires:1}{}" = 2 : i64}}, target_gate = "Adjoint(Rot){0:[f64],1:[f64],2:[f64]}{wires:1}{}"} {
    %0 = catalyst.list_init : <f64>
    %1 = catalyst.list_init : <i64>
    %2 = catalyst.list_init : <i64>
    %3 = stablehlo.slice %arg3 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
    %4 = stablehlo.reshape %3 : (tensor<1xi64>) -> tensor<i64>
    %extracted = tensor.extract %4[] : tensor<i64>
    %extracted_0 = tensor.extract %arg0[] : tensor<f64>
    catalyst.list_push %extracted, %2 : <i64>
    catalyst.list_push %extracted, %2 : <i64>
    %5 = stablehlo.slice %arg3 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
    %6 = stablehlo.reshape %5 : (tensor<1xi64>) -> tensor<i64>
    %extracted_1 = tensor.extract %6[] : tensor<i64>
    %extracted_2 = tensor.extract %arg1[] : tensor<f64>
    catalyst.list_push %extracted_1, %2 : <i64>
    catalyst.list_push %extracted_1, %2 : <i64>
    %7 = stablehlo.slice %arg3 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
    %8 = stablehlo.reshape %7 : (tensor<1xi64>) -> tensor<i64>
    %extracted_3 = tensor.extract %8[] : tensor<i64>
    %extracted_4 = tensor.extract %arg2[] : tensor<f64>
    catalyst.list_push %extracted_3, %2 : <i64>
    catalyst.list_push %extracted_3, %2 : <i64>
    %9 = catalyst.list_pop %2 : <i64>
    %10 = quantum.extract %arg4[%9] : !quantum.reg -> !quantum.bit
    %out_qubits = quantum.custom "RZ"(%extracted_4) %10 adj : !quantum.bit
    %11 = quantum.insert %arg4[%9], %out_qubits : !quantum.reg, !quantum.bit
    %12 = catalyst.list_pop %2 : <i64>
    %13 = catalyst.list_pop %2 : <i64>
    %14 = quantum.extract %11[%13] : !quantum.reg -> !quantum.bit
    %out_qubits_5 = quantum.custom "RY"(%extracted_2) %14 adj : !quantum.bit
    %15 = quantum.insert %11[%13], %out_qubits_5 : !quantum.reg, !quantum.bit
    %16 = catalyst.list_pop %2 : <i64>
    %17 = catalyst.list_pop %2 : <i64>
    %18 = quantum.extract %15[%17] : !quantum.reg -> !quantum.bit
    %out_qubits_6 = quantum.custom "RZ"(%extracted_0) %18 adj : !quantum.bit
    %19 = quantum.insert %15[%17], %out_qubits_6 : !quantum.reg, !quantum.bit
    %20 = catalyst.list_pop %2 : <i64>
    catalyst.list_dealloc %0 : <f64>
    catalyst.list_dealloc %1 : <i64>
    catalyst.list_dealloc %2 : <i64>
    return %19 : !quantum.reg
  }
  func.func private @"__builtin_adjoint_rotation_Adjoint(RY){0:[f64]}{wires:1}{}"(%arg0: tensor<f64>, %arg1: tensor<1xi64>, %arg2: !quantum.reg) -> !quantum.reg attributes {frontend_name = "adjoint_rotation", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"RY{0:[f64]}{wires:1}{}" = 1 : i64}}, target_gate = "Adjoint(RY){0:[f64]}{wires:1}{}"} {
    %0 = stablehlo.slice %arg1 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
    %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
    %2 = stablehlo.negate %arg0 : tensor<f64>
    %extracted = tensor.extract %1[] : tensor<i64>
    %extracted_0 = tensor.extract %2[] : tensor<f64>
    %3 = quantum.extract %arg2[%extracted] : !quantum.reg -> !quantum.bit
    %out_qubits = quantum.custom "RY"(%extracted_0) %3 : !quantum.bit
    %4 = quantum.insert %arg2[%extracted], %out_qubits : !quantum.reg, !quantum.bit
    return %4 : !quantum.reg
  }
  func.func private @"__builtin__ry_to_rot_Adjoint(RY){0:[f64]}{wires:1}{}"(%arg0: tensor<f64>, %arg1: tensor<1xi64>, %arg2: !quantum.reg) -> !quantum.reg attributes {frontend_name = "_ry_to_rot", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"Adjoint(Rot){0:[f64],1:[f64],2:[f64]}{wires:1}{}" = 1 : i64}}, target_gate = "Adjoint(RY){0:[f64]}{wires:1}{}"} {
    %c = stablehlo.constant dense<0> : tensor<i64>
    %0 = catalyst.list_init : <f64>
    %1 = catalyst.list_init : <i64>
    %2 = catalyst.list_init : <i64>
    %3 = stablehlo.slice %arg1 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
    %4 = stablehlo.reshape %3 : (tensor<1xi64>) -> tensor<i64>
    %extracted = tensor.extract %4[] : tensor<i64>
    %5 = stablehlo.convert %c : (tensor<i64>) -> tensor<f64>
    %extracted_0 = tensor.extract %5[] : tensor<f64>
    %extracted_1 = tensor.extract %arg0[] : tensor<f64>
    %6 = stablehlo.convert %c : (tensor<i64>) -> tensor<f64>
    %extracted_2 = tensor.extract %6[] : tensor<f64>
    catalyst.list_push %extracted, %2 : <i64>
    catalyst.list_push %extracted, %2 : <i64>
    %7 = catalyst.list_pop %2 : <i64>
    %8 = quantum.extract %arg2[%7] : !quantum.reg -> !quantum.bit
    %out_qubits = quantum.custom "Rot"(%extracted_0, %extracted_1, %extracted_2) %8 adj : !quantum.bit
    %9 = quantum.insert %arg2[%7], %out_qubits : !quantum.reg, !quantum.bit
    %10 = catalyst.list_pop %2 : <i64>
    catalyst.list_dealloc %0 : <f64>
    catalyst.list_dealloc %1 : <i64>
    catalyst.list_dealloc %2 : <i64>
    return %9 : !quantum.reg
  }
  func.func private @"__builtin__ry_to_rz_rx_Adjoint(RY){0:[f64]}{wires:1}{}"(%arg0: tensor<f64>, %arg1: tensor<1xi64>, %arg2: !quantum.reg) -> !quantum.reg attributes {frontend_name = "_ry_to_rz_rx", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"Adjoint(RX){0:[f64]}{wires:1}{}" = 1 : i64, "Adjoint(RZ){0:[f64]}{wires:1}{}" = 2 : i64}}, target_gate = "Adjoint(RY){0:[f64]}{wires:1}{}"} {
    %cst = arith.constant 1.5707963267948966 : f64
    %cst_0 = arith.constant -1.5707963267948966 : f64
    %0 = catalyst.list_init : <f64>
    %1 = catalyst.list_init : <i64>
    %2 = catalyst.list_init : <i64>
    %3 = stablehlo.slice %arg1 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
    %4 = stablehlo.reshape %3 : (tensor<1xi64>) -> tensor<i64>
    %extracted = tensor.extract %4[] : tensor<i64>
    catalyst.list_push %extracted, %2 : <i64>
    catalyst.list_push %extracted, %2 : <i64>
    %5 = stablehlo.slice %arg1 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
    %6 = stablehlo.reshape %5 : (tensor<1xi64>) -> tensor<i64>
    %extracted_1 = tensor.extract %6[] : tensor<i64>
    %extracted_2 = tensor.extract %arg0[] : tensor<f64>
    catalyst.list_push %extracted_1, %2 : <i64>
    catalyst.list_push %extracted_1, %2 : <i64>
    %7 = stablehlo.slice %arg1 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
    %8 = stablehlo.reshape %7 : (tensor<1xi64>) -> tensor<i64>
    %extracted_3 = tensor.extract %8[] : tensor<i64>
    catalyst.list_push %extracted_3, %2 : <i64>
    catalyst.list_push %extracted_3, %2 : <i64>
    %9 = catalyst.list_pop %2 : <i64>
    %10 = quantum.extract %arg2[%9] : !quantum.reg -> !quantum.bit
    %out_qubits = quantum.custom "RZ"(%cst) %10 adj : !quantum.bit
    %11 = quantum.insert %arg2[%9], %out_qubits : !quantum.reg, !quantum.bit
    %12 = catalyst.list_pop %2 : <i64>
    %13 = catalyst.list_pop %2 : <i64>
    %14 = quantum.extract %11[%13] : !quantum.reg -> !quantum.bit
    %out_qubits_4 = quantum.custom "RX"(%extracted_2) %14 adj : !quantum.bit
    %15 = quantum.insert %11[%13], %out_qubits_4 : !quantum.reg, !quantum.bit
    %16 = catalyst.list_pop %2 : <i64>
    %17 = catalyst.list_pop %2 : <i64>
    %18 = quantum.extract %15[%17] : !quantum.reg -> !quantum.bit
    %out_qubits_5 = quantum.custom "RZ"(%cst_0) %18 adj : !quantum.bit
    %19 = quantum.insert %15[%17], %out_qubits_5 : !quantum.reg, !quantum.bit
    %20 = catalyst.list_pop %2 : <i64>
    catalyst.list_dealloc %0 : <f64>
    catalyst.list_dealloc %1 : <i64>
    catalyst.list_dealloc %2 : <i64>
    return %19 : !quantum.reg
  }
  func.func private @"__builtin__ry_to_ppr_Adjoint(RY){0:[f64]}{wires:1}{}"(%arg0: tensor<f64>, %arg1: tensor<1xi64>, %arg2: !quantum.reg) -> !quantum.reg attributes {frontend_name = "_ry_to_ppr", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"Adjoint(PauliRot){theta:[f64]}{wires:1}{pauli_word = \22Y\22}" = 1 : i64}}, target_gate = "Adjoint(RY){0:[f64]}{wires:1}{}"} {
    %0 = catalyst.list_init : <f64>
    %1 = catalyst.list_init : <i64>
    %2 = catalyst.list_init : <i64>
    %3 = stablehlo.slice %arg1 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
    %4 = stablehlo.reshape %3 : (tensor<1xi64>) -> tensor<i64>
    %extracted = tensor.extract %4[] : tensor<i64>
    %extracted_0 = tensor.extract %arg0[] : tensor<f64>
    catalyst.list_push %extracted, %2 : <i64>
    catalyst.list_push %extracted, %2 : <i64>
    %5 = catalyst.list_pop %2 : <i64>
    %6 = quantum.extract %arg2[%5] : !quantum.reg -> !quantum.bit
    %out_qubits = quantum.paulirot ["Y"](%extracted_0) %6 adj : !quantum.bit
    %7 = quantum.insert %arg2[%5], %out_qubits : !quantum.reg, !quantum.bit
    %8 = catalyst.list_pop %2 : <i64>
    catalyst.list_dealloc %0 : <f64>
    catalyst.list_dealloc %1 : <i64>
    catalyst.list_dealloc %2 : <i64>
    return %7 : !quantum.reg
  }
  func.func private @"__builtin__ry_to_rx_cliff_Adjoint(RY){0:[f64]}{wires:1}{}"(%arg0: tensor<f64>, %arg1: tensor<1xi64>, %arg2: !quantum.reg) -> !quantum.reg attributes {frontend_name = "_ry_to_rx_cliff", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"Adjoint(RX){0:[f64]}{wires:1}{}" = 1 : i64, "Adjoint(S){}{wires:1}{}" = 1 : i64, "S{}{wires:1}{}" = 1 : i64}}, target_gate = "Adjoint(RY){0:[f64]}{wires:1}{}"} {
    %0 = catalyst.list_init : <f64>
    %1 = catalyst.list_init : <i64>
    %2 = catalyst.list_init : <i64>
    %3 = stablehlo.slice %arg1 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
    %4 = stablehlo.reshape %3 : (tensor<1xi64>) -> tensor<i64>
    %5 = stablehlo.slice %arg1 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
    %6 = stablehlo.reshape %5 : (tensor<1xi64>) -> tensor<i64>
    %7 = stablehlo.slice %arg1 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
    %8 = stablehlo.reshape %7 : (tensor<1xi64>) -> tensor<i64>
    %extracted = tensor.extract %4[] : tensor<i64>
    catalyst.list_push %extracted, %2 : <i64>
    catalyst.list_push %extracted, %2 : <i64>
    %extracted_0 = tensor.extract %6[] : tensor<i64>
    %extracted_1 = tensor.extract %arg0[] : tensor<f64>
    catalyst.list_push %extracted_0, %2 : <i64>
    catalyst.list_push %extracted_0, %2 : <i64>
    %extracted_2 = tensor.extract %8[] : tensor<i64>
    catalyst.list_push %extracted_2, %2 : <i64>
    catalyst.list_push %extracted_2, %2 : <i64>
    %9 = catalyst.list_pop %2 : <i64>
    %10 = quantum.extract %arg2[%9] : !quantum.reg -> !quantum.bit
    %out_qubits = quantum.custom "S"() %10 adj : !quantum.bit
    %11 = quantum.insert %arg2[%9], %out_qubits : !quantum.reg, !quantum.bit
    %12 = catalyst.list_pop %2 : <i64>
    %13 = catalyst.list_pop %2 : <i64>
    %14 = quantum.extract %11[%13] : !quantum.reg -> !quantum.bit
    %out_qubits_3 = quantum.custom "RX"(%extracted_1) %14 adj : !quantum.bit
    %15 = quantum.insert %11[%13], %out_qubits_3 : !quantum.reg, !quantum.bit
    %16 = catalyst.list_pop %2 : <i64>
    %17 = catalyst.list_pop %2 : <i64>
    %18 = quantum.extract %15[%17] : !quantum.reg -> !quantum.bit
    %out_qubits_4 = quantum.custom "S"() %18 : !quantum.bit
    %19 = quantum.insert %15[%17], %out_qubits_4 : !quantum.reg, !quantum.bit
    %20 = catalyst.list_pop %2 : <i64>
    catalyst.list_dealloc %0 : <f64>
    catalyst.list_dealloc %1 : <i64>
    catalyst.list_dealloc %2 : <i64>
    return %19 : !quantum.reg
  }
  func.func private @"__builtin__ry_to_rz_cliff_Adjoint(RY){0:[f64]}{wires:1}{}"(%arg0: tensor<f64>, %arg1: tensor<1xi64>, %arg2: !quantum.reg) -> !quantum.reg attributes {frontend_name = "_ry_to_rz_cliff", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"Adjoint(Hadamard){}{wires:1}{}" = 2 : i64, "Adjoint(RZ){0:[f64]}{wires:1}{}" = 1 : i64, "Adjoint(S){}{wires:1}{}" = 1 : i64, "S{}{wires:1}{}" = 1 : i64}}, target_gate = "Adjoint(RY){0:[f64]}{wires:1}{}"} {
    %0 = catalyst.list_init : <f64>
    %1 = catalyst.list_init : <i64>
    %2 = catalyst.list_init : <i64>
    %3 = stablehlo.slice %arg1 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
    %4 = stablehlo.reshape %3 : (tensor<1xi64>) -> tensor<i64>
    %5 = stablehlo.slice %arg1 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
    %6 = stablehlo.reshape %5 : (tensor<1xi64>) -> tensor<i64>
    %extracted = tensor.extract %6[] : tensor<i64>
    catalyst.list_push %extracted, %2 : <i64>
    catalyst.list_push %extracted, %2 : <i64>
    %7 = stablehlo.slice %arg1 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
    %8 = stablehlo.reshape %7 : (tensor<1xi64>) -> tensor<i64>
    %extracted_0 = tensor.extract %8[] : tensor<i64>
    catalyst.list_push %extracted_0, %2 : <i64>
    catalyst.list_push %extracted_0, %2 : <i64>
    %extracted_1 = tensor.extract %4[] : tensor<i64>
    %extracted_2 = tensor.extract %arg0[] : tensor<f64>
    catalyst.list_push %extracted_1, %2 : <i64>
    catalyst.list_push %extracted_1, %2 : <i64>
    %9 = stablehlo.slice %arg1 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
    %10 = stablehlo.reshape %9 : (tensor<1xi64>) -> tensor<i64>
    %extracted_3 = tensor.extract %10[] : tensor<i64>
    catalyst.list_push %extracted_3, %2 : <i64>
    catalyst.list_push %extracted_3, %2 : <i64>
    %11 = stablehlo.slice %arg1 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
    %12 = stablehlo.reshape %11 : (tensor<1xi64>) -> tensor<i64>
    %extracted_4 = tensor.extract %12[] : tensor<i64>
    catalyst.list_push %extracted_4, %2 : <i64>
    catalyst.list_push %extracted_4, %2 : <i64>
    %13 = catalyst.list_pop %2 : <i64>
    %14 = quantum.extract %arg2[%13] : !quantum.reg -> !quantum.bit
    %out_qubits = quantum.custom "S"() %14 adj : !quantum.bit
    %15 = quantum.insert %arg2[%13], %out_qubits : !quantum.reg, !quantum.bit
    %16 = catalyst.list_pop %2 : <i64>
    %17 = catalyst.list_pop %2 : <i64>
    %18 = quantum.extract %15[%17] : !quantum.reg -> !quantum.bit
    %out_qubits_5 = quantum.custom "Hadamard"() %18 adj : !quantum.bit
    %19 = quantum.insert %15[%17], %out_qubits_5 : !quantum.reg, !quantum.bit
    %20 = catalyst.list_pop %2 : <i64>
    %21 = catalyst.list_pop %2 : <i64>
    %22 = quantum.extract %19[%21] : !quantum.reg -> !quantum.bit
    %out_qubits_6 = quantum.custom "RZ"(%extracted_2) %22 adj : !quantum.bit
    %23 = quantum.insert %19[%21], %out_qubits_6 : !quantum.reg, !quantum.bit
    %24 = catalyst.list_pop %2 : <i64>
    %25 = catalyst.list_pop %2 : <i64>
    %26 = quantum.extract %23[%25] : !quantum.reg -> !quantum.bit
    %out_qubits_7 = quantum.custom "Hadamard"() %26 adj : !quantum.bit
    %27 = quantum.insert %23[%25], %out_qubits_7 : !quantum.reg, !quantum.bit
    %28 = catalyst.list_pop %2 : <i64>
    %29 = catalyst.list_pop %2 : <i64>
    %30 = quantum.extract %27[%29] : !quantum.reg -> !quantum.bit
    %out_qubits_8 = quantum.custom "S"() %30 : !quantum.bit
    %31 = quantum.insert %27[%29], %out_qubits_8 : !quantum.reg, !quantum.bit
    %32 = catalyst.list_pop %2 : <i64>
    catalyst.list_dealloc %0 : <f64>
    catalyst.list_dealloc %1 : <i64>
    catalyst.list_dealloc %2 : <i64>
    return %31 : !quantum.reg
  }
  func.func private @"__builtin_adjoint_rotation_Adjoint(RX){0:[f64]}{wires:1}{}"(%arg0: tensor<f64>, %arg1: tensor<1xi64>, %arg2: !quantum.reg) -> !quantum.reg attributes {frontend_name = "adjoint_rotation", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"RX{0:[f64]}{wires:1}{}" = 1 : i64}}, target_gate = "Adjoint(RX){0:[f64]}{wires:1}{}"} {
    %0 = stablehlo.slice %arg1 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
    %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
    %2 = stablehlo.negate %arg0 : tensor<f64>
    %extracted = tensor.extract %1[] : tensor<i64>
    %extracted_0 = tensor.extract %2[] : tensor<f64>
    %3 = quantum.extract %arg2[%extracted] : !quantum.reg -> !quantum.bit
    %out_qubits = quantum.custom "RX"(%extracted_0) %3 : !quantum.bit
    %4 = quantum.insert %arg2[%extracted], %out_qubits : !quantum.reg, !quantum.bit
    return %4 : !quantum.reg
  }
  func.func private @"__builtin__rx_to_rot_Adjoint(RX){0:[f64]}{wires:1}{}"(%arg0: tensor<f64>, %arg1: tensor<1xi64>, %arg2: !quantum.reg) -> !quantum.reg attributes {frontend_name = "_rx_to_rot", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"Adjoint(Rot){0:[f64],1:[f64],2:[f64]}{wires:1}{}" = 1 : i64}}, target_gate = "Adjoint(RX){0:[f64]}{wires:1}{}"} {
    %cst = arith.constant 10.995574287564276 : f64
    %cst_0 = arith.constant 1.5707963267948966 : f64
    %0 = catalyst.list_init : <f64>
    %1 = catalyst.list_init : <i64>
    %2 = catalyst.list_init : <i64>
    %3 = stablehlo.slice %arg1 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
    %4 = stablehlo.reshape %3 : (tensor<1xi64>) -> tensor<i64>
    %extracted = tensor.extract %4[] : tensor<i64>
    %extracted_1 = tensor.extract %arg0[] : tensor<f64>
    catalyst.list_push %extracted, %2 : <i64>
    catalyst.list_push %extracted, %2 : <i64>
    %5 = catalyst.list_pop %2 : <i64>
    %6 = quantum.extract %arg2[%5] : !quantum.reg -> !quantum.bit
    %out_qubits = quantum.custom "Rot"(%cst_0, %extracted_1, %cst) %6 adj : !quantum.bit
    %7 = quantum.insert %arg2[%5], %out_qubits : !quantum.reg, !quantum.bit
    %8 = catalyst.list_pop %2 : <i64>
    catalyst.list_dealloc %0 : <f64>
    catalyst.list_dealloc %1 : <i64>
    catalyst.list_dealloc %2 : <i64>
    return %7 : !quantum.reg
  }
  func.func private @"__builtin__rx_to_rz_ry_Adjoint(RX){0:[f64]}{wires:1}{}"(%arg0: tensor<f64>, %arg1: tensor<1xi64>, %arg2: !quantum.reg) -> !quantum.reg attributes {frontend_name = "_rx_to_rz_ry", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"Adjoint(RY){0:[f64]}{wires:1}{}" = 1 : i64, "Adjoint(RZ){0:[f64]}{wires:1}{}" = 2 : i64}}, target_gate = "Adjoint(RX){0:[f64]}{wires:1}{}"} {
    %cst = arith.constant -1.5707963267948966 : f64
    %cst_0 = arith.constant 1.5707963267948966 : f64
    %0 = catalyst.list_init : <f64>
    %1 = catalyst.list_init : <i64>
    %2 = catalyst.list_init : <i64>
    %3 = stablehlo.slice %arg1 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
    %4 = stablehlo.reshape %3 : (tensor<1xi64>) -> tensor<i64>
    %extracted = tensor.extract %4[] : tensor<i64>
    catalyst.list_push %extracted, %2 : <i64>
    catalyst.list_push %extracted, %2 : <i64>
    %5 = stablehlo.slice %arg1 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
    %6 = stablehlo.reshape %5 : (tensor<1xi64>) -> tensor<i64>
    %extracted_1 = tensor.extract %6[] : tensor<i64>
    %extracted_2 = tensor.extract %arg0[] : tensor<f64>
    catalyst.list_push %extracted_1, %2 : <i64>
    catalyst.list_push %extracted_1, %2 : <i64>
    %7 = stablehlo.slice %arg1 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
    %8 = stablehlo.reshape %7 : (tensor<1xi64>) -> tensor<i64>
    %extracted_3 = tensor.extract %8[] : tensor<i64>
    catalyst.list_push %extracted_3, %2 : <i64>
    catalyst.list_push %extracted_3, %2 : <i64>
    %9 = catalyst.list_pop %2 : <i64>
    %10 = quantum.extract %arg2[%9] : !quantum.reg -> !quantum.bit
    %out_qubits = quantum.custom "RZ"(%cst) %10 adj : !quantum.bit
    %11 = quantum.insert %arg2[%9], %out_qubits : !quantum.reg, !quantum.bit
    %12 = catalyst.list_pop %2 : <i64>
    %13 = catalyst.list_pop %2 : <i64>
    %14 = quantum.extract %11[%13] : !quantum.reg -> !quantum.bit
    %out_qubits_4 = quantum.custom "RY"(%extracted_2) %14 adj : !quantum.bit
    %15 = quantum.insert %11[%13], %out_qubits_4 : !quantum.reg, !quantum.bit
    %16 = catalyst.list_pop %2 : <i64>
    %17 = catalyst.list_pop %2 : <i64>
    %18 = quantum.extract %15[%17] : !quantum.reg -> !quantum.bit
    %out_qubits_5 = quantum.custom "RZ"(%cst_0) %18 adj : !quantum.bit
    %19 = quantum.insert %15[%17], %out_qubits_5 : !quantum.reg, !quantum.bit
    %20 = catalyst.list_pop %2 : <i64>
    catalyst.list_dealloc %0 : <f64>
    catalyst.list_dealloc %1 : <i64>
    catalyst.list_dealloc %2 : <i64>
    return %19 : !quantum.reg
  }
  func.func private @"__builtin__rx_to_ppr_Adjoint(RX){0:[f64]}{wires:1}{}"(%arg0: tensor<f64>, %arg1: tensor<1xi64>, %arg2: !quantum.reg) -> !quantum.reg attributes {frontend_name = "_rx_to_ppr", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"Adjoint(PauliRot){theta:[f64]}{wires:1}{pauli_word = \22X\22}" = 1 : i64}}, target_gate = "Adjoint(RX){0:[f64]}{wires:1}{}"} {
    %0 = catalyst.list_init : <f64>
    %1 = catalyst.list_init : <i64>
    %2 = catalyst.list_init : <i64>
    %3 = stablehlo.slice %arg1 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
    %4 = stablehlo.reshape %3 : (tensor<1xi64>) -> tensor<i64>
    %extracted = tensor.extract %4[] : tensor<i64>
    %extracted_0 = tensor.extract %arg0[] : tensor<f64>
    catalyst.list_push %extracted, %2 : <i64>
    catalyst.list_push %extracted, %2 : <i64>
    %5 = catalyst.list_pop %2 : <i64>
    %6 = quantum.extract %arg2[%5] : !quantum.reg -> !quantum.bit
    %out_qubits = quantum.paulirot ["X"](%extracted_0) %6 adj : !quantum.bit
    %7 = quantum.insert %arg2[%5], %out_qubits : !quantum.reg, !quantum.bit
    %8 = catalyst.list_pop %2 : <i64>
    catalyst.list_dealloc %0 : <f64>
    catalyst.list_dealloc %1 : <i64>
    catalyst.list_dealloc %2 : <i64>
    return %7 : !quantum.reg
  }
  func.func private @"__builtin__rx_to_ry_cliff_Adjoint(RX){0:[f64]}{wires:1}{}"(%arg0: tensor<f64>, %arg1: tensor<1xi64>, %arg2: !quantum.reg) -> !quantum.reg attributes {frontend_name = "_rx_to_ry_cliff", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"Adjoint(RY){0:[f64]}{wires:1}{}" = 1 : i64, "Adjoint(S){}{wires:1}{}" = 1 : i64, "S{}{wires:1}{}" = 1 : i64}}, target_gate = "Adjoint(RX){0:[f64]}{wires:1}{}"} {
    %0 = catalyst.list_init : <f64>
    %1 = catalyst.list_init : <i64>
    %2 = catalyst.list_init : <i64>
    %3 = stablehlo.slice %arg1 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
    %4 = stablehlo.reshape %3 : (tensor<1xi64>) -> tensor<i64>
    %5 = stablehlo.slice %arg1 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
    %6 = stablehlo.reshape %5 : (tensor<1xi64>) -> tensor<i64>
    %extracted = tensor.extract %4[] : tensor<i64>
    catalyst.list_push %extracted, %2 : <i64>
    catalyst.list_push %extracted, %2 : <i64>
    %extracted_0 = tensor.extract %6[] : tensor<i64>
    %extracted_1 = tensor.extract %arg0[] : tensor<f64>
    catalyst.list_push %extracted_0, %2 : <i64>
    catalyst.list_push %extracted_0, %2 : <i64>
    %extracted_2 = tensor.extract %4[] : tensor<i64>
    catalyst.list_push %extracted_2, %2 : <i64>
    catalyst.list_push %extracted_2, %2 : <i64>
    %7 = catalyst.list_pop %2 : <i64>
    %8 = quantum.extract %arg2[%7] : !quantum.reg -> !quantum.bit
    %out_qubits = quantum.custom "S"() %8 : !quantum.bit
    %9 = quantum.insert %arg2[%7], %out_qubits : !quantum.reg, !quantum.bit
    %10 = catalyst.list_pop %2 : <i64>
    %11 = catalyst.list_pop %2 : <i64>
    %12 = quantum.extract %9[%11] : !quantum.reg -> !quantum.bit
    %out_qubits_3 = quantum.custom "RY"(%extracted_1) %12 adj : !quantum.bit
    %13 = quantum.insert %9[%11], %out_qubits_3 : !quantum.reg, !quantum.bit
    %14 = catalyst.list_pop %2 : <i64>
    %15 = catalyst.list_pop %2 : <i64>
    %16 = quantum.extract %13[%15] : !quantum.reg -> !quantum.bit
    %out_qubits_4 = quantum.custom "S"() %16 adj : !quantum.bit
    %17 = quantum.insert %13[%15], %out_qubits_4 : !quantum.reg, !quantum.bit
    %18 = catalyst.list_pop %2 : <i64>
    catalyst.list_dealloc %0 : <f64>
    catalyst.list_dealloc %1 : <i64>
    catalyst.list_dealloc %2 : <i64>
    return %17 : !quantum.reg
  }
  func.func private @"__builtin__rx_to_rz_cliff_Adjoint(RX){0:[f64]}{wires:1}{}"(%arg0: tensor<f64>, %arg1: tensor<1xi64>, %arg2: !quantum.reg) -> !quantum.reg attributes {frontend_name = "_rx_to_rz_cliff", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"Adjoint(Hadamard){}{wires:1}{}" = 2 : i64, "Adjoint(RZ){0:[f64]}{wires:1}{}" = 1 : i64}}, target_gate = "Adjoint(RX){0:[f64]}{wires:1}{}"} {
    %0 = catalyst.list_init : <f64>
    %1 = catalyst.list_init : <i64>
    %2 = catalyst.list_init : <i64>
    %3 = stablehlo.slice %arg1 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
    %4 = stablehlo.reshape %3 : (tensor<1xi64>) -> tensor<i64>
    %5 = stablehlo.slice %arg1 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
    %6 = stablehlo.reshape %5 : (tensor<1xi64>) -> tensor<i64>
    %7 = stablehlo.slice %arg1 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
    %8 = stablehlo.reshape %7 : (tensor<1xi64>) -> tensor<i64>
    %extracted = tensor.extract %4[] : tensor<i64>
    catalyst.list_push %extracted, %2 : <i64>
    catalyst.list_push %extracted, %2 : <i64>
    %extracted_0 = tensor.extract %6[] : tensor<i64>
    %extracted_1 = tensor.extract %arg0[] : tensor<f64>
    catalyst.list_push %extracted_0, %2 : <i64>
    catalyst.list_push %extracted_0, %2 : <i64>
    %extracted_2 = tensor.extract %8[] : tensor<i64>
    catalyst.list_push %extracted_2, %2 : <i64>
    catalyst.list_push %extracted_2, %2 : <i64>
    %9 = catalyst.list_pop %2 : <i64>
    %10 = quantum.extract %arg2[%9] : !quantum.reg -> !quantum.bit
    %out_qubits = quantum.custom "Hadamard"() %10 adj : !quantum.bit
    %11 = quantum.insert %arg2[%9], %out_qubits : !quantum.reg, !quantum.bit
    %12 = catalyst.list_pop %2 : <i64>
    %13 = catalyst.list_pop %2 : <i64>
    %14 = quantum.extract %11[%13] : !quantum.reg -> !quantum.bit
    %out_qubits_3 = quantum.custom "RZ"(%extracted_1) %14 adj : !quantum.bit
    %15 = quantum.insert %11[%13], %out_qubits_3 : !quantum.reg, !quantum.bit
    %16 = catalyst.list_pop %2 : <i64>
    %17 = catalyst.list_pop %2 : <i64>
    %18 = quantum.extract %15[%17] : !quantum.reg -> !quantum.bit
    %out_qubits_4 = quantum.custom "Hadamard"() %18 adj : !quantum.bit
    %19 = quantum.insert %15[%17], %out_qubits_4 : !quantum.reg, !quantum.bit
    %20 = catalyst.list_pop %2 : <i64>
    catalyst.list_dealloc %0 : <f64>
    catalyst.list_dealloc %1 : <i64>
    catalyst.list_dealloc %2 : <i64>
    return %19 : !quantum.reg
  }
  func.func private @"__builtin_adjoint_rotation_Adjoint(PauliRot){theta:[f64]}{wires:1}{pauli_word = \22Z\22}"(%arg0: tensor<f64>, %arg1: tensor<1xi64>, %arg2: !quantum.reg) -> !quantum.reg attributes {frontend_name = "adjoint_rotation", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"PauliRot{theta:[f64]}{wires:1}{pauli_word = \22Z\22}" = 1 : i64}}, target_gate = "Adjoint(PauliRot){theta:[f64]}{wires:1}{pauli_word = \22Z\22}"} {
    %0 = stablehlo.slice %arg1 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
    %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
    %2 = stablehlo.negate %arg0 : tensor<f64>
    %extracted = tensor.extract %1[] : tensor<i64>
    %extracted_0 = tensor.extract %2[] : tensor<f64>
    %3 = quantum.extract %arg2[%extracted] : !quantum.reg -> !quantum.bit
    %out_qubits = quantum.paulirot ["Z"](%extracted_0) %3 : !quantum.bit
    %4 = quantum.insert %arg2[%extracted], %out_qubits : !quantum.reg, !quantum.bit
    return %4 : !quantum.reg
  }
  func.func private @"__builtin__pauli_rot_decomposition_Adjoint(PauliRot){theta:[f64]}{wires:1}{pauli_word = \22Z\22}"(%arg0: tensor<f64>, %arg1: tensor<1xi64>, %arg2: !quantum.reg) -> !quantum.reg attributes {frontend_name = "_pauli_rot_decomposition", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"Adjoint(MultiRZ){theta:[f64]}{wires:1}{}" = 1 : i64}}, target_gate = "Adjoint(PauliRot){theta:[f64]}{wires:1}{pauli_word = \22Z\22}"} {
    %0 = catalyst.list_init : <f64>
    %1 = catalyst.list_init : <i64>
    %2 = catalyst.list_init : <i64>
    %3 = stablehlo.slice %arg1 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
    %4 = stablehlo.reshape %3 : (tensor<1xi64>) -> tensor<i64>
    %extracted = tensor.extract %4[] : tensor<i64>
    %extracted_0 = tensor.extract %arg0[] : tensor<f64>
    catalyst.list_push %extracted, %2 : <i64>
    catalyst.list_push %extracted, %2 : <i64>
    %5 = catalyst.list_pop %2 : <i64>
    %6 = quantum.extract %arg2[%5] : !quantum.reg -> !quantum.bit
    %out_qubits = quantum.multirz(%extracted_0) %6 adj : !quantum.bit
    %7 = quantum.insert %arg2[%5], %out_qubits : !quantum.reg, !quantum.bit
    %8 = catalyst.list_pop %2 : <i64>
    catalyst.list_dealloc %0 : <f64>
    catalyst.list_dealloc %1 : <i64>
    catalyst.list_dealloc %2 : <i64>
    return %7 : !quantum.reg
  }
  func.func private @"__builtin_decompose_to_base_Adjoint(Hadamard){}{wires:1}{}"(%arg0: tensor<1xi64>, %arg1: !quantum.reg) -> !quantum.reg attributes {frontend_name = "decompose_to_base", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"Hadamard{}{wires:1}{}" = 1 : i64}}, target_gate = "Adjoint(Hadamard){}{wires:1}{}"} {
    %0 = stablehlo.slice %arg0 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
    %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
    %extracted = tensor.extract %1[] : tensor<i64>
    %2 = quantum.extract %arg1[%extracted] : !quantum.reg -> !quantum.bit
    %out_qubits = quantum.custom "Hadamard"() %2 : !quantum.bit
    %3 = quantum.insert %arg1[%extracted], %out_qubits : !quantum.reg, !quantum.bit
    return %3 : !quantum.reg
  }
  func.func private @"__builtin__hadamard_to_rz_rx_Adjoint(Hadamard){}{wires:1}{}"(%arg0: tensor<1xi64>, %arg1: !quantum.reg) -> !quantum.reg attributes {frontend_name = "_hadamard_to_rz_rx", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"Adjoint(GlobalPhase){phi:[f64]}{}{}" = 1 : i64, "Adjoint(RX){0:[f64]}{wires:1}{}" = 1 : i64, "Adjoint(RZ){0:[f64]}{wires:1}{}" = 2 : i64}}, target_gate = "Adjoint(Hadamard){}{wires:1}{}"} {
    %cst = arith.constant -1.5707963267948966 : f64
    %cst_0 = arith.constant 1.5707963267948966 : f64
    %0 = catalyst.list_init : <f64>
    %1 = catalyst.list_init : <i64>
    %2 = catalyst.list_init : <i64>
    %3 = stablehlo.slice %arg0 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
    %4 = stablehlo.reshape %3 : (tensor<1xi64>) -> tensor<i64>
    %extracted = tensor.extract %4[] : tensor<i64>
    catalyst.list_push %extracted, %2 : <i64>
    catalyst.list_push %extracted, %2 : <i64>
    %5 = stablehlo.slice %arg0 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
    %6 = stablehlo.reshape %5 : (tensor<1xi64>) -> tensor<i64>
    %extracted_1 = tensor.extract %6[] : tensor<i64>
    catalyst.list_push %extracted_1, %2 : <i64>
    catalyst.list_push %extracted_1, %2 : <i64>
    %7 = stablehlo.slice %arg0 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
    %8 = stablehlo.reshape %7 : (tensor<1xi64>) -> tensor<i64>
    %extracted_2 = tensor.extract %8[] : tensor<i64>
    catalyst.list_push %extracted_2, %2 : <i64>
    catalyst.list_push %extracted_2, %2 : <i64>
    quantum.gphase(%cst) adj
    %9 = catalyst.list_pop %2 : <i64>
    %10 = quantum.extract %arg1[%9] : !quantum.reg -> !quantum.bit
    %out_qubits = quantum.custom "RZ"(%cst_0) %10 adj : !quantum.bit
    %11 = quantum.insert %arg1[%9], %out_qubits : !quantum.reg, !quantum.bit
    %12 = catalyst.list_pop %2 : <i64>
    %13 = catalyst.list_pop %2 : <i64>
    %14 = quantum.extract %11[%13] : !quantum.reg -> !quantum.bit
    %out_qubits_3 = quantum.custom "RX"(%cst_0) %14 adj : !quantum.bit
    %15 = quantum.insert %11[%13], %out_qubits_3 : !quantum.reg, !quantum.bit
    %16 = catalyst.list_pop %2 : <i64>
    %17 = catalyst.list_pop %2 : <i64>
    %18 = quantum.extract %15[%17] : !quantum.reg -> !quantum.bit
    %out_qubits_4 = quantum.custom "RZ"(%cst_0) %18 adj : !quantum.bit
    %19 = quantum.insert %15[%17], %out_qubits_4 : !quantum.reg, !quantum.bit
    %20 = catalyst.list_pop %2 : <i64>
    catalyst.list_dealloc %0 : <f64>
    catalyst.list_dealloc %1 : <i64>
    catalyst.list_dealloc %2 : <i64>
    return %19 : !quantum.reg
  }
  func.func private @"__builtin__hadamard_to_rz_ry_Adjoint(Hadamard){}{wires:1}{}"(%arg0: tensor<1xi64>, %arg1: !quantum.reg) -> !quantum.reg attributes {frontend_name = "_hadamard_to_rz_ry", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"Adjoint(GlobalPhase){phi:[f64]}{}{}" = 1 : i64, "Adjoint(RY){0:[f64]}{wires:1}{}" = 1 : i64, "Adjoint(RZ){0:[f64]}{wires:1}{}" = 1 : i64}}, target_gate = "Adjoint(Hadamard){}{wires:1}{}"} {
    %cst = arith.constant -1.5707963267948966 : f64
    %cst_0 = arith.constant 1.5707963267948966 : f64
    %cst_1 = arith.constant 3.1415926535897931 : f64
    %0 = catalyst.list_init : <f64>
    %1 = catalyst.list_init : <i64>
    %2 = catalyst.list_init : <i64>
    %3 = stablehlo.slice %arg0 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
    %4 = stablehlo.reshape %3 : (tensor<1xi64>) -> tensor<i64>
    %extracted = tensor.extract %4[] : tensor<i64>
    catalyst.list_push %extracted, %2 : <i64>
    catalyst.list_push %extracted, %2 : <i64>
    %5 = stablehlo.slice %arg0 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
    %6 = stablehlo.reshape %5 : (tensor<1xi64>) -> tensor<i64>
    %extracted_2 = tensor.extract %6[] : tensor<i64>
    catalyst.list_push %extracted_2, %2 : <i64>
    catalyst.list_push %extracted_2, %2 : <i64>
    quantum.gphase(%cst) adj
    %7 = catalyst.list_pop %2 : <i64>
    %8 = quantum.extract %arg1[%7] : !quantum.reg -> !quantum.bit
    %out_qubits = quantum.custom "RY"(%cst_0) %8 adj : !quantum.bit
    %9 = quantum.insert %arg1[%7], %out_qubits : !quantum.reg, !quantum.bit
    %10 = catalyst.list_pop %2 : <i64>
    %11 = catalyst.list_pop %2 : <i64>
    %12 = quantum.extract %9[%11] : !quantum.reg -> !quantum.bit
    %out_qubits_3 = quantum.custom "RZ"(%cst_1) %12 adj : !quantum.bit
    %13 = quantum.insert %9[%11], %out_qubits_3 : !quantum.reg, !quantum.bit
    %14 = catalyst.list_pop %2 : <i64>
    catalyst.list_dealloc %0 : <f64>
    catalyst.list_dealloc %1 : <i64>
    catalyst.list_dealloc %2 : <i64>
    return %13 : !quantum.reg
  }
  func.func private @"__builtin_adjoint_rotation_Adjoint(PauliRot){theta:[f64]}{wires:1}{pauli_word = \22Y\22}"(%arg0: tensor<f64>, %arg1: tensor<1xi64>, %arg2: !quantum.reg) -> !quantum.reg attributes {frontend_name = "adjoint_rotation", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"PauliRot{theta:[f64]}{wires:1}{pauli_word = \22Y\22}" = 1 : i64}}, target_gate = "Adjoint(PauliRot){theta:[f64]}{wires:1}{pauli_word = \22Y\22}"} {
    %0 = stablehlo.slice %arg1 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
    %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
    %2 = stablehlo.negate %arg0 : tensor<f64>
    %extracted = tensor.extract %1[] : tensor<i64>
    %extracted_0 = tensor.extract %2[] : tensor<f64>
    %3 = quantum.extract %arg2[%extracted] : !quantum.reg -> !quantum.bit
    %out_qubits = quantum.paulirot ["Y"](%extracted_0) %3 : !quantum.bit
    %4 = quantum.insert %arg2[%extracted], %out_qubits : !quantum.reg, !quantum.bit
    return %4 : !quantum.reg
  }
  func.func private @"__builtin__pauli_rot_decomposition_Adjoint(PauliRot){theta:[f64]}{wires:1}{pauli_word = \22Y\22}"(%arg0: tensor<f64>, %arg1: tensor<1xi64>, %arg2: !quantum.reg) -> !quantum.reg attributes {frontend_name = "_pauli_rot_decomposition", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"Adjoint(MultiRZ){theta:[f64]}{wires:1}{}" = 1 : i64, "Adjoint(RX){0:[f64]}{wires:1}{}" = 2 : i64}}, target_gate = "Adjoint(PauliRot){theta:[f64]}{wires:1}{pauli_word = \22Y\22}"} {
    %cst = arith.constant -1.5707963267948966 : f64
    %cst_0 = arith.constant 1.5707963267948966 : f64
    %0 = catalyst.list_init : <f64>
    %1 = catalyst.list_init : <i64>
    %2 = catalyst.list_init : <i64>
    %3 = stablehlo.slice %arg1 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
    %4 = stablehlo.reshape %3 : (tensor<1xi64>) -> tensor<i64>
    %extracted = tensor.extract %4[] : tensor<i64>
    catalyst.list_push %extracted, %2 : <i64>
    catalyst.list_push %extracted, %2 : <i64>
    %extracted_1 = tensor.extract %4[] : tensor<i64>
    %extracted_2 = tensor.extract %arg0[] : tensor<f64>
    catalyst.list_push %extracted_1, %2 : <i64>
    catalyst.list_push %extracted_1, %2 : <i64>
    %extracted_3 = tensor.extract %4[] : tensor<i64>
    catalyst.list_push %extracted_3, %2 : <i64>
    catalyst.list_push %extracted_3, %2 : <i64>
    %5 = catalyst.list_pop %2 : <i64>
    %6 = quantum.extract %arg2[%5] : !quantum.reg -> !quantum.bit
    %out_qubits = quantum.custom "RX"(%cst) %6 adj : !quantum.bit
    %7 = quantum.insert %arg2[%5], %out_qubits : !quantum.reg, !quantum.bit
    %8 = catalyst.list_pop %2 : <i64>
    %9 = catalyst.list_pop %2 : <i64>
    %10 = quantum.extract %7[%9] : !quantum.reg -> !quantum.bit
    %out_qubits_4 = quantum.multirz(%extracted_2) %10 adj : !quantum.bit
    %11 = quantum.insert %7[%9], %out_qubits_4 : !quantum.reg, !quantum.bit
    %12 = catalyst.list_pop %2 : <i64>
    %13 = catalyst.list_pop %2 : <i64>
    %14 = quantum.extract %11[%13] : !quantum.reg -> !quantum.bit
    %out_qubits_5 = quantum.custom "RX"(%cst_0) %14 adj : !quantum.bit
    %15 = quantum.insert %11[%13], %out_qubits_5 : !quantum.reg, !quantum.bit
    %16 = catalyst.list_pop %2 : <i64>
    catalyst.list_dealloc %0 : <f64>
    catalyst.list_dealloc %1 : <i64>
    catalyst.list_dealloc %2 : <i64>
    return %15 : !quantum.reg
  }
  func.func private @"__builtin_adjoint_rotation_Adjoint(PauliRot){theta:[f64]}{wires:1}{pauli_word = \22X\22}"(%arg0: tensor<f64>, %arg1: tensor<1xi64>, %arg2: !quantum.reg) -> !quantum.reg attributes {frontend_name = "adjoint_rotation", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"PauliRot{theta:[f64]}{wires:1}{pauli_word = \22X\22}" = 1 : i64}}, target_gate = "Adjoint(PauliRot){theta:[f64]}{wires:1}{pauli_word = \22X\22}"} {
    %0 = stablehlo.slice %arg1 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
    %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
    %2 = stablehlo.negate %arg0 : tensor<f64>
    %extracted = tensor.extract %1[] : tensor<i64>
    %extracted_0 = tensor.extract %2[] : tensor<f64>
    %3 = quantum.extract %arg2[%extracted] : !quantum.reg -> !quantum.bit
    %out_qubits = quantum.paulirot ["X"](%extracted_0) %3 : !quantum.bit
    %4 = quantum.insert %arg2[%extracted], %out_qubits : !quantum.reg, !quantum.bit
    return %4 : !quantum.reg
  }
  func.func private @"__builtin__pauli_rot_decomposition_Adjoint(PauliRot){theta:[f64]}{wires:1}{pauli_word = \22X\22}"(%arg0: tensor<f64>, %arg1: tensor<1xi64>, %arg2: !quantum.reg) -> !quantum.reg attributes {frontend_name = "_pauli_rot_decomposition", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"Adjoint(Hadamard){}{wires:1}{}" = 2 : i64, "Adjoint(MultiRZ){theta:[f64]}{wires:1}{}" = 1 : i64}}, target_gate = "Adjoint(PauliRot){theta:[f64]}{wires:1}{pauli_word = \22X\22}"} {
    %0 = catalyst.list_init : <f64>
    %1 = catalyst.list_init : <i64>
    %2 = catalyst.list_init : <i64>
    %3 = stablehlo.slice %arg1 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
    %4 = stablehlo.reshape %3 : (tensor<1xi64>) -> tensor<i64>
    %extracted = tensor.extract %4[] : tensor<i64>
    catalyst.list_push %extracted, %2 : <i64>
    catalyst.list_push %extracted, %2 : <i64>
    %extracted_0 = tensor.extract %4[] : tensor<i64>
    %extracted_1 = tensor.extract %arg0[] : tensor<f64>
    catalyst.list_push %extracted_0, %2 : <i64>
    catalyst.list_push %extracted_0, %2 : <i64>
    %extracted_2 = tensor.extract %4[] : tensor<i64>
    catalyst.list_push %extracted_2, %2 : <i64>
    catalyst.list_push %extracted_2, %2 : <i64>
    %5 = catalyst.list_pop %2 : <i64>
    %6 = quantum.extract %arg2[%5] : !quantum.reg -> !quantum.bit
    %out_qubits = quantum.custom "Hadamard"() %6 adj : !quantum.bit
    %7 = quantum.insert %arg2[%5], %out_qubits : !quantum.reg, !quantum.bit
    %8 = catalyst.list_pop %2 : <i64>
    %9 = catalyst.list_pop %2 : <i64>
    %10 = quantum.extract %7[%9] : !quantum.reg -> !quantum.bit
    %out_qubits_3 = quantum.multirz(%extracted_1) %10 adj : !quantum.bit
    %11 = quantum.insert %7[%9], %out_qubits_3 : !quantum.reg, !quantum.bit
    %12 = catalyst.list_pop %2 : <i64>
    %13 = catalyst.list_pop %2 : <i64>
    %14 = quantum.extract %11[%13] : !quantum.reg -> !quantum.bit
    %out_qubits_4 = quantum.custom "Hadamard"() %14 adj : !quantum.bit
    %15 = quantum.insert %11[%13], %out_qubits_4 : !quantum.reg, !quantum.bit
    %16 = catalyst.list_pop %2 : <i64>
    catalyst.list_dealloc %0 : <f64>
    catalyst.list_dealloc %1 : <i64>
    catalyst.list_dealloc %2 : <i64>
    return %15 : !quantum.reg
  }
  func.func private @"__builtin_adjoint_rotation_Adjoint(MultiRZ){theta:[f64]}{wires:1}{}"(%arg0: tensor<f64>, %arg1: tensor<1xi64>, %arg2: !quantum.reg) -> !quantum.reg attributes {frontend_name = "adjoint_rotation", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"MultiRZ{theta:[f64]}{wires:1}{}" = 1 : i64}}, target_gate = "Adjoint(MultiRZ){theta:[f64]}{wires:1}{}"} {
    %0 = stablehlo.slice %arg1 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
    %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
    %2 = stablehlo.negate %arg0 : tensor<f64>
    %extracted = tensor.extract %1[] : tensor<i64>
    %extracted_0 = tensor.extract %2[] : tensor<f64>
    %3 = quantum.extract %arg2[%extracted] : !quantum.reg -> !quantum.bit
    %out_qubits = quantum.multirz(%extracted_0) %3 : !quantum.bit
    %4 = quantum.insert %arg2[%extracted], %out_qubits : !quantum.reg, !quantum.bit
    return %4 : !quantum.reg
  }
  func.func private @"__builtin__multi_rz_decomposition_Adjoint(MultiRZ){theta:[f64]}{wires:1}{}"(%arg0: tensor<f64>, %arg1: tensor<1xi64>, %arg2: !quantum.reg) -> !quantum.reg attributes {frontend_name = "_multi_rz_decomposition", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"Adjoint(RZ){0:[f64]}{wires:1}{}" = 1 : i64}}, target_gate = "Adjoint(MultiRZ){theta:[f64]}{wires:1}{}"} {
    %0 = catalyst.list_init : <f64>
    %1 = catalyst.list_init : <i64>
    %2 = catalyst.list_init : <i64>
    %3 = stablehlo.slice %arg1 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
    %4 = stablehlo.reshape %3 : (tensor<1xi64>) -> tensor<i64>
    %extracted = tensor.extract %4[] : tensor<i64>
    %extracted_0 = tensor.extract %arg0[] : tensor<f64>
    catalyst.list_push %extracted, %2 : <i64>
    catalyst.list_push %extracted, %2 : <i64>
    %5 = catalyst.list_pop %2 : <i64>
    %6 = quantum.extract %arg2[%5] : !quantum.reg -> !quantum.bit
    %out_qubits = quantum.custom "RZ"(%extracted_0) %6 adj : !quantum.bit
    %7 = quantum.insert %arg2[%5], %out_qubits : !quantum.reg, !quantum.bit
    %8 = catalyst.list_pop %2 : <i64>
    catalyst.list_dealloc %0 : <f64>
    catalyst.list_dealloc %1 : <i64>
    catalyst.list_dealloc %2 : <i64>
    return %7 : !quantum.reg
  }
}