module @qfunc {
  func.func public @jit_qfunc() -> tensor<f64> attributes {llvm.emit_c_interface} {
    %0 = catalyst.launch_kernel @module_qfunc::@qfunc() : () -> tensor<f64>
    return %0 : tensor<f64>
  }
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
      %extracted_0 = tensor.extract %6[] : tensor<i64>
      %7 = quantum.extract %4[%extracted_0] : !quantum.reg -> !quantum.bit
      %8 = quantum.extract %4[%extracted] : !quantum.reg -> !quantum.bit
      %out_qubits_1:2 = quantum.custom "CNOT"() %7, %8 : !quantum.bit, !quantum.bit
      %9 = quantum.insert %4[%extracted_0], %out_qubits_1#0 : !quantum.reg, !quantum.bit
      %10 = quantum.insert %9[%extracted], %out_qubits_1#1 : !quantum.reg, !quantum.bit
      %11 = quantum.extract %10[%extracted] : !quantum.reg -> !quantum.bit
      %out_qubits_2 = quantum.custom "T"() %11 adj : !quantum.bit
      %12 = quantum.insert %10[%extracted], %out_qubits_2 : !quantum.reg, !quantum.bit
      %13 = stablehlo.slice %cst [0:1] : (tensor<3xi64>) -> tensor<1xi64>
      %14 = stablehlo.reshape %13 : (tensor<1xi64>) -> tensor<i64>
      %extracted_3 = tensor.extract %14[] : tensor<i64>
      %15 = quantum.extract %12[%extracted_3] : !quantum.reg -> !quantum.bit
      %16 = quantum.extract %12[%extracted] : !quantum.reg -> !quantum.bit
      %out_qubits_4:2 = quantum.custom "CNOT"() %15, %16 : !quantum.bit, !quantum.bit
      %17 = quantum.insert %12[%extracted_3], %out_qubits_4#0 : !quantum.reg, !quantum.bit
      %18 = quantum.insert %17[%extracted], %out_qubits_4#1 : !quantum.reg, !quantum.bit
      %19 = quantum.extract %18[%extracted] : !quantum.reg -> !quantum.bit
      %out_qubits_5 = quantum.custom "T"() %19 : !quantum.bit
      %20 = quantum.insert %18[%extracted], %out_qubits_5 : !quantum.reg, !quantum.bit
      %21 = quantum.extract %20[%extracted_0] : !quantum.reg -> !quantum.bit
      %22 = quantum.extract %20[%extracted] : !quantum.reg -> !quantum.bit
      %out_qubits_6:2 = quantum.custom "CNOT"() %21, %22 : !quantum.bit, !quantum.bit
      %23 = quantum.insert %20[%extracted_0], %out_qubits_6#0 : !quantum.reg, !quantum.bit
      %24 = quantum.insert %23[%extracted], %out_qubits_6#1 : !quantum.reg, !quantum.bit
      %25 = quantum.extract %24[%extracted] : !quantum.reg -> !quantum.bit
      %out_qubits_7 = quantum.custom "T"() %25 adj : !quantum.bit
      %26 = quantum.insert %24[%extracted], %out_qubits_7 : !quantum.reg, !quantum.bit
      %27 = quantum.extract %26[%extracted_3] : !quantum.reg -> !quantum.bit
      %28 = quantum.extract %26[%extracted] : !quantum.reg -> !quantum.bit
      %out_qubits_8:2 = quantum.custom "CNOT"() %27, %28 : !quantum.bit, !quantum.bit
      %29 = quantum.insert %26[%extracted_3], %out_qubits_8#0 : !quantum.reg, !quantum.bit
      %30 = quantum.insert %29[%extracted], %out_qubits_8#1 : !quantum.reg, !quantum.bit
      %31 = quantum.extract %30[%extracted] : !quantum.reg -> !quantum.bit
      %out_qubits_9 = quantum.custom "T"() %31 : !quantum.bit
      %32 = quantum.insert %30[%extracted], %out_qubits_9 : !quantum.reg, !quantum.bit
      %33 = quantum.extract %32[%extracted_0] : !quantum.reg -> !quantum.bit
      %out_qubits_10 = quantum.custom "T"() %33 : !quantum.bit
      %34 = quantum.insert %32[%extracted_0], %out_qubits_10 : !quantum.reg, !quantum.bit
      %35 = quantum.extract %34[%extracted_3] : !quantum.reg -> !quantum.bit
      %36 = quantum.extract %34[%extracted_0] : !quantum.reg -> !quantum.bit
      %out_qubits_11:2 = quantum.custom "CNOT"() %35, %36 : !quantum.bit, !quantum.bit
      %37 = quantum.insert %34[%extracted_3], %out_qubits_11#0 : !quantum.reg, !quantum.bit
      %38 = quantum.insert %37[%extracted_0], %out_qubits_11#1 : !quantum.reg, !quantum.bit
      %39 = quantum.extract %38[%extracted] : !quantum.reg -> !quantum.bit
      %out_qubits_12 = quantum.custom "Hadamard"() %39 : !quantum.bit
      %40 = quantum.insert %38[%extracted], %out_qubits_12 : !quantum.reg, !quantum.bit
      %41 = quantum.extract %40[%extracted_3] : !quantum.reg -> !quantum.bit
      %out_qubits_13 = quantum.custom "T"() %41 : !quantum.bit
      %42 = quantum.insert %40[%extracted_3], %out_qubits_13 : !quantum.reg, !quantum.bit
      %43 = quantum.extract %42[%extracted_0] : !quantum.reg -> !quantum.bit
      %out_qubits_14 = quantum.custom "T"() %43 adj : !quantum.bit
      %44 = quantum.insert %42[%extracted_0], %out_qubits_14 : !quantum.reg, !quantum.bit
      %45 = quantum.extract %44[%extracted_3] : !quantum.reg -> !quantum.bit
      %46 = quantum.extract %44[%extracted_0] : !quantum.reg -> !quantum.bit
      %out_qubits_15:2 = quantum.custom "CNOT"() %45, %46 : !quantum.bit, !quantum.bit
      %47 = quantum.insert %44[%extracted_3], %out_qubits_15#0 : !quantum.reg, !quantum.bit
      %48 = quantum.insert %47[%extracted_0], %out_qubits_15#1 : !quantum.reg, !quantum.bit
      %49 = quantum.extract %48[ 0] : !quantum.reg -> !quantum.bit
      %50 = quantum.namedobs %49[ PauliZ] : !quantum.obs
      %51 = quantum.insert %48[ 0], %49 : !quantum.reg, !quantum.bit
      %52 = quantum.expval %50 : f64
      %from_elements = tensor.from_elements %52 : tensor<f64>
      quantum.dealloc %51 : !quantum.reg
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
      %extracted_0 = tensor.extract %5[] : tensor<i64>
      %6 = quantum.extract %3[%extracted_0] : !quantum.reg -> !quantum.bit
      %7 = quantum.extract %3[%extracted] : !quantum.reg -> !quantum.bit
      %out_qubits_1:2 = quantum.custom "CNOT"() %6, %7 : !quantum.bit, !quantum.bit
      %8 = quantum.insert %3[%extracted_0], %out_qubits_1#0 : !quantum.reg, !quantum.bit
      %9 = quantum.insert %8[%extracted], %out_qubits_1#1 : !quantum.reg, !quantum.bit
      %10 = quantum.extract %9[%extracted] : !quantum.reg -> !quantum.bit
      %out_qubits_2 = quantum.custom "T"() %10 adj : !quantum.bit
      %11 = quantum.insert %9[%extracted], %out_qubits_2 : !quantum.reg, !quantum.bit
      %12 = stablehlo.slice %arg0 [0:1] : (tensor<3xi64>) -> tensor<1xi64>
      %13 = stablehlo.reshape %12 : (tensor<1xi64>) -> tensor<i64>
      %extracted_3 = tensor.extract %13[] : tensor<i64>
      %14 = quantum.extract %11[%extracted_3] : !quantum.reg -> !quantum.bit
      %15 = quantum.extract %11[%extracted] : !quantum.reg -> !quantum.bit
      %out_qubits_4:2 = quantum.custom "CNOT"() %14, %15 : !quantum.bit, !quantum.bit
      %16 = quantum.insert %11[%extracted_3], %out_qubits_4#0 : !quantum.reg, !quantum.bit
      %17 = quantum.insert %16[%extracted], %out_qubits_4#1 : !quantum.reg, !quantum.bit
      %18 = quantum.extract %17[%extracted] : !quantum.reg -> !quantum.bit
      %out_qubits_5 = quantum.custom "T"() %18 : !quantum.bit
      %19 = quantum.insert %17[%extracted], %out_qubits_5 : !quantum.reg, !quantum.bit
      %20 = quantum.extract %19[%extracted_0] : !quantum.reg -> !quantum.bit
      %21 = quantum.extract %19[%extracted] : !quantum.reg -> !quantum.bit
      %out_qubits_6:2 = quantum.custom "CNOT"() %20, %21 : !quantum.bit, !quantum.bit
      %22 = quantum.insert %19[%extracted_0], %out_qubits_6#0 : !quantum.reg, !quantum.bit
      %23 = quantum.insert %22[%extracted], %out_qubits_6#1 : !quantum.reg, !quantum.bit
      %24 = quantum.extract %23[%extracted] : !quantum.reg -> !quantum.bit
      %out_qubits_7 = quantum.custom "T"() %24 adj : !quantum.bit
      %25 = quantum.insert %23[%extracted], %out_qubits_7 : !quantum.reg, !quantum.bit
      %26 = quantum.extract %25[%extracted_3] : !quantum.reg -> !quantum.bit
      %27 = quantum.extract %25[%extracted] : !quantum.reg -> !quantum.bit
      %out_qubits_8:2 = quantum.custom "CNOT"() %26, %27 : !quantum.bit, !quantum.bit
      %28 = quantum.insert %25[%extracted_3], %out_qubits_8#0 : !quantum.reg, !quantum.bit
      %29 = quantum.insert %28[%extracted], %out_qubits_8#1 : !quantum.reg, !quantum.bit
      %30 = quantum.extract %29[%extracted] : !quantum.reg -> !quantum.bit
      %out_qubits_9 = quantum.custom "T"() %30 : !quantum.bit
      %31 = quantum.insert %29[%extracted], %out_qubits_9 : !quantum.reg, !quantum.bit
      %32 = quantum.extract %31[%extracted_0] : !quantum.reg -> !quantum.bit
      %out_qubits_10 = quantum.custom "T"() %32 : !quantum.bit
      %33 = quantum.insert %31[%extracted_0], %out_qubits_10 : !quantum.reg, !quantum.bit
      %34 = quantum.extract %33[%extracted_3] : !quantum.reg -> !quantum.bit
      %35 = quantum.extract %33[%extracted_0] : !quantum.reg -> !quantum.bit
      %out_qubits_11:2 = quantum.custom "CNOT"() %34, %35 : !quantum.bit, !quantum.bit
      %36 = quantum.insert %33[%extracted_3], %out_qubits_11#0 : !quantum.reg, !quantum.bit
      %37 = quantum.insert %36[%extracted_0], %out_qubits_11#1 : !quantum.reg, !quantum.bit
      %38 = quantum.extract %37[%extracted] : !quantum.reg -> !quantum.bit
      %out_qubits_12 = quantum.custom "Hadamard"() %38 : !quantum.bit
      %39 = quantum.insert %37[%extracted], %out_qubits_12 : !quantum.reg, !quantum.bit
      %40 = quantum.extract %39[%extracted_3] : !quantum.reg -> !quantum.bit
      %out_qubits_13 = quantum.custom "T"() %40 : !quantum.bit
      %41 = quantum.insert %39[%extracted_3], %out_qubits_13 : !quantum.reg, !quantum.bit
      %42 = quantum.extract %41[%extracted_0] : !quantum.reg -> !quantum.bit
      %out_qubits_14 = quantum.custom "T"() %42 adj : !quantum.bit
      %43 = quantum.insert %41[%extracted_0], %out_qubits_14 : !quantum.reg, !quantum.bit
      %44 = quantum.extract %43[%extracted_3] : !quantum.reg -> !quantum.bit
      %45 = quantum.extract %43[%extracted_0] : !quantum.reg -> !quantum.bit
      %out_qubits_15:2 = quantum.custom "CNOT"() %44, %45 : !quantum.bit, !quantum.bit
      %46 = quantum.insert %43[%extracted_3], %out_qubits_15#0 : !quantum.reg, !quantum.bit
      %47 = quantum.insert %46[%extracted_0], %out_qubits_15#1 : !quantum.reg, !quantum.bit
      return %47 : !quantum.reg
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
      %26 = stablehlo.slice %arg0 [1:2] : (tensor<3xi64>) -> tensor<1xi64>
      %27 = stablehlo.reshape %26 : (tensor<1xi64>) -> tensor<i64>
      %extracted_7 = tensor.extract %27[] : tensor<i64>
      %28 = quantum.extract %25[%extracted_1] : !quantum.reg -> !quantum.bit
      %29 = quantum.extract %25[%extracted_7] : !quantum.reg -> !quantum.bit
      %30 = quantum.extract %25[%extracted_2] : !quantum.reg -> !quantum.bit
      %out_qubits_8:3 = quantum.operator "PPR"() qubits(%28, %29, %30)
        static_data = {angle_denominator = 8 : i64, pauli_word = "ZZX"}
        qubit_map = {wires = [0, 1, 2]}
      %31 = quantum.insert %25[%extracted_1], %out_qubits_8#0 : !quantum.reg, !quantum.bit
      %32 = quantum.insert %31[%extracted_7], %out_qubits_8#1 : !quantum.reg, !quantum.bit
      %33 = quantum.insert %32[%extracted_2], %out_qubits_8#2 : !quantum.reg, !quantum.bit
      %34 = quantum.extract %33[%extracted_2] : !quantum.reg -> !quantum.bit
      %out_qubits_9 = quantum.operator "PPR"() qubits(%34)
        static_data = {angle_denominator = 8 : i64, pauli_word = "X"}
        qubit_map = {wires = [0]}
      %35 = quantum.insert %33[%extracted_2], %out_qubits_9 : !quantum.reg, !quantum.bit
      %36 = quantum.extract %35[%extracted_7] : !quantum.reg -> !quantum.bit
      %out_qubits_10 = quantum.operator "PPR"() qubits(%36)
        static_data = {angle_denominator = 8 : i64, pauli_word = "Z"}
        qubit_map = {wires = [0]}
      %37 = quantum.insert %35[%extracted_7], %out_qubits_10 : !quantum.reg, !quantum.bit
      %38 = quantum.extract %37[%extracted_1] : !quantum.reg -> !quantum.bit
      %out_qubits_11 = quantum.operator "PPR"() qubits(%38)
        static_data = {angle_denominator = 8 : i64, pauli_word = "Z"}
        qubit_map = {wires = [0]}
      %39 = quantum.insert %37[%extracted_1], %out_qubits_11 : !quantum.reg, !quantum.bit
      quantum.gphase(%cst)
      return %39 : !quantum.reg
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
      %4 = quantum.extract %3[%extracted] : !quantum.reg -> !quantum.bit
      %out_qubits_1 = quantum.custom "RX"(%cst_0) %4 : !quantum.bit
      %5 = quantum.insert %3[%extracted], %out_qubits_1 : !quantum.reg, !quantum.bit
      %6 = quantum.extract %5[%extracted] : !quantum.reg -> !quantum.bit
      %out_qubits_2 = quantum.custom "RZ"(%cst_0) %6 : !quantum.bit
      %7 = quantum.insert %5[%extracted], %out_qubits_2 : !quantum.reg, !quantum.bit
      quantum.gphase(%cst)
      return %7 : !quantum.reg
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
      %4 = quantum.extract %3[%extracted] : !quantum.reg -> !quantum.bit
      %out_qubits_2 = quantum.custom "RY"(%cst_0) %4 : !quantum.bit
      %5 = quantum.insert %3[%extracted], %out_qubits_2 : !quantum.reg, !quantum.bit
      quantum.gphase(%cst)
      return %5 : !quantum.reg
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
      %extracted_0 = tensor.extract %5[] : tensor<i64>
      %6 = quantum.extract %3[%extracted_0] : !quantum.reg -> !quantum.bit
      %7 = quantum.extract %3[%extracted] : !quantum.reg -> !quantum.bit
      %out_qubits_1:2 = quantum.custom "CZ"() %6, %7 : !quantum.bit, !quantum.bit
      %8 = quantum.insert %3[%extracted_0], %out_qubits_1#0 : !quantum.reg, !quantum.bit
      %9 = quantum.insert %8[%extracted], %out_qubits_1#1 : !quantum.reg, !quantum.bit
      %10 = quantum.extract %9[%extracted] : !quantum.reg -> !quantum.bit
      %out_qubits_2 = quantum.custom "Hadamard"() %10 : !quantum.bit
      %11 = quantum.insert %9[%extracted], %out_qubits_2 : !quantum.reg, !quantum.bit
      return %11 : !quantum.reg
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
      %8 = quantum.extract %7[%extracted] : !quantum.reg -> !quantum.bit
      %9 = quantum.extract %7[%extracted_0] : !quantum.reg -> !quantum.bit
      %out_qubits_2:2 = quantum.operator "PPR"() qubits(%8, %9)
        static_data = {angle_denominator = 4 : i64, pauli_word = "ZX"}
        qubit_map = {wires = [0, 1]}
      %10 = quantum.insert %7[%extracted], %out_qubits_2#0 : !quantum.reg, !quantum.bit
      %11 = quantum.insert %10[%extracted_0], %out_qubits_2#1 : !quantum.reg, !quantum.bit
      quantum.gphase(%cst)
      return %11 : !quantum.reg
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
      %cst = arith.constant -0.78539816339744828 : f64
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
      %out_qubits = quantum.custom "PhaseShift"(%cst) %6 : !quantum.bit
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
      %extracted_1 = tensor.extract %arg0[] : tensor<f64>
      %3 = quantum.extract %arg2[%extracted] : !quantum.reg -> !quantum.bit
      %out_qubits = quantum.custom "Rot"(%extracted_0, %extracted_0, %extracted_1) %3 : !quantum.bit
      %4 = quantum.insert %arg2[%extracted], %out_qubits : !quantum.reg, !quantum.bit
      return %4 : !quantum.reg
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
      %extracted_1 = tensor.extract %arg0[] : tensor<f64>
      %4 = quantum.extract %3[%extracted] : !quantum.reg -> !quantum.bit
      %out_qubits_2 = quantum.custom "RX"(%extracted_1) %4 : !quantum.bit
      %5 = quantum.insert %3[%extracted], %out_qubits_2 : !quantum.reg, !quantum.bit
      %6 = quantum.extract %5[%extracted] : !quantum.reg -> !quantum.bit
      %out_qubits_3 = quantum.custom "RY"(%cst) %6 : !quantum.bit
      %7 = quantum.insert %5[%extracted], %out_qubits_3 : !quantum.reg, !quantum.bit
      return %7 : !quantum.reg
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
      %extracted = tensor.extract %1[] : tensor<i64>
      %2 = quantum.extract %arg2[%extracted] : !quantum.reg -> !quantum.bit
      %out_qubits = quantum.custom "Hadamard"() %2 : !quantum.bit
      %3 = quantum.insert %arg2[%extracted], %out_qubits : !quantum.reg, !quantum.bit
      %extracted_0 = tensor.extract %arg0[] : tensor<f64>
      %4 = quantum.extract %3[%extracted] : !quantum.reg -> !quantum.bit
      %out_qubits_1 = quantum.custom "RX"(%extracted_0) %4 : !quantum.bit
      %5 = quantum.insert %3[%extracted], %out_qubits_1 : !quantum.reg, !quantum.bit
      %6 = quantum.extract %5[%extracted] : !quantum.reg -> !quantum.bit
      %out_qubits_2 = quantum.custom "Hadamard"() %6 : !quantum.bit
      %7 = quantum.insert %5[%extracted], %out_qubits_2 : !quantum.reg, !quantum.bit
      return %7 : !quantum.reg
    }
    func.func private @"__builtin__rz_to_ry_cliff_RZ{0:[f64]}{wires:1}{}"(%arg0: tensor<f64>, %arg1: tensor<1xi64>, %arg2: !quantum.reg) -> !quantum.reg attributes {frontend_name = "_rz_to_ry_cliff", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"Adjoint(S){}{wires:1}{}" = 1 : i64, "Hadamard{}{wires:1}{}" = 2 : i64, "RY{0:[f64]}{wires:1}{}" = 1 : i64, "S{}{wires:1}{}" = 1 : i64}}, target_gate = "RZ{0:[f64]}{wires:1}{}"} {
      %0 = stablehlo.slice %arg1 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
      %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
      %extracted = tensor.extract %1[] : tensor<i64>
      %2 = quantum.extract %arg2[%extracted] : !quantum.reg -> !quantum.bit
      %out_qubits = quantum.custom "Hadamard"() %2 : !quantum.bit
      %3 = quantum.insert %arg2[%extracted], %out_qubits : !quantum.reg, !quantum.bit
      %4 = quantum.extract %3[%extracted] : !quantum.reg -> !quantum.bit
      %out_qubits_0 = quantum.custom "S"() %4 : !quantum.bit
      %5 = quantum.insert %3[%extracted], %out_qubits_0 : !quantum.reg, !quantum.bit
      %extracted_1 = tensor.extract %arg0[] : tensor<f64>
      %6 = quantum.extract %5[%extracted] : !quantum.reg -> !quantum.bit
      %out_qubits_2 = quantum.custom "RY"(%extracted_1) %6 : !quantum.bit
      %7 = quantum.insert %5[%extracted], %out_qubits_2 : !quantum.reg, !quantum.bit
      %8 = quantum.extract %7[%extracted] : !quantum.reg -> !quantum.bit
      %out_qubits_3 = quantum.custom "S"() %8 adj : !quantum.bit
      %9 = quantum.insert %7[%extracted], %out_qubits_3 : !quantum.reg, !quantum.bit
      %10 = quantum.extract %9[%extracted] : !quantum.reg -> !quantum.bit
      %out_qubits_4 = quantum.custom "Hadamard"() %10 : !quantum.bit
      %11 = quantum.insert %9[%extracted], %out_qubits_4 : !quantum.reg, !quantum.bit
      return %11 : !quantum.reg
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
      %extracted_1 = tensor.extract %arg0[] : tensor<f64>
      %4 = quantum.extract %3[%extracted] : !quantum.reg -> !quantum.bit
      %out_qubits_2 = quantum.custom "RY"(%extracted_1) %4 : !quantum.bit
      %5 = quantum.insert %3[%extracted], %out_qubits_2 : !quantum.reg, !quantum.bit
      %6 = quantum.extract %5[%extracted] : !quantum.reg -> !quantum.bit
      %out_qubits_3 = quantum.custom "RZ"(%cst) %6 : !quantum.bit
      %7 = quantum.insert %5[%extracted], %out_qubits_3 : !quantum.reg, !quantum.bit
      return %7 : !quantum.reg
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
      %extracted = tensor.extract %1[] : tensor<i64>
      %2 = quantum.extract %arg2[%extracted] : !quantum.reg -> !quantum.bit
      %out_qubits = quantum.custom "S"() %2 : !quantum.bit
      %3 = quantum.insert %arg2[%extracted], %out_qubits : !quantum.reg, !quantum.bit
      %extracted_0 = tensor.extract %arg0[] : tensor<f64>
      %4 = quantum.extract %3[%extracted] : !quantum.reg -> !quantum.bit
      %out_qubits_1 = quantum.custom "RY"(%extracted_0) %4 : !quantum.bit
      %5 = quantum.insert %3[%extracted], %out_qubits_1 : !quantum.reg, !quantum.bit
      %6 = quantum.extract %5[%extracted] : !quantum.reg -> !quantum.bit
      %out_qubits_2 = quantum.custom "S"() %6 adj : !quantum.bit
      %7 = quantum.insert %5[%extracted], %out_qubits_2 : !quantum.reg, !quantum.bit
      return %7 : !quantum.reg
    }
    func.func private @"__builtin__rx_to_rz_cliff_RX{0:[f64]}{wires:1}{}"(%arg0: tensor<f64>, %arg1: tensor<1xi64>, %arg2: !quantum.reg) -> !quantum.reg attributes {frontend_name = "_rx_to_rz_cliff", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"Hadamard{}{wires:1}{}" = 2 : i64, "RZ{0:[f64]}{wires:1}{}" = 1 : i64}}, target_gate = "RX{0:[f64]}{wires:1}{}"} {
      %0 = stablehlo.slice %arg1 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
      %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
      %extracted = tensor.extract %1[] : tensor<i64>
      %2 = quantum.extract %arg2[%extracted] : !quantum.reg -> !quantum.bit
      %out_qubits = quantum.custom "Hadamard"() %2 : !quantum.bit
      %3 = quantum.insert %arg2[%extracted], %out_qubits : !quantum.reg, !quantum.bit
      %extracted_0 = tensor.extract %arg0[] : tensor<f64>
      %4 = quantum.extract %3[%extracted] : !quantum.reg -> !quantum.bit
      %out_qubits_1 = quantum.custom "RZ"(%extracted_0) %4 : !quantum.bit
      %5 = quantum.insert %3[%extracted], %out_qubits_1 : !quantum.reg, !quantum.bit
      %6 = quantum.extract %5[%extracted] : !quantum.reg -> !quantum.bit
      %out_qubits_2 = quantum.custom "Hadamard"() %6 : !quantum.bit
      %7 = quantum.insert %5[%extracted], %out_qubits_2 : !quantum.reg, !quantum.bit
      return %7 : !quantum.reg
    }
    func.func private @"__builtin__ry_to_rot_RY{0:[f64]}{wires:1}{}"(%arg0: tensor<f64>, %arg1: tensor<1xi64>, %arg2: !quantum.reg) -> !quantum.reg attributes {frontend_name = "_ry_to_rot", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"Rot{0:[f64],1:[f64],2:[f64]}{wires:1}{}" = 1 : i64}}, target_gate = "RY{0:[f64]}{wires:1}{}"} {
      %c = stablehlo.constant dense<0> : tensor<i64>
      %0 = stablehlo.slice %arg1 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
      %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
      %extracted = tensor.extract %1[] : tensor<i64>
      %2 = stablehlo.convert %c : (tensor<i64>) -> tensor<f64>
      %extracted_0 = tensor.extract %2[] : tensor<f64>
      %extracted_1 = tensor.extract %arg0[] : tensor<f64>
      %3 = quantum.extract %arg2[%extracted] : !quantum.reg -> !quantum.bit
      %out_qubits = quantum.custom "Rot"(%extracted_0, %extracted_1, %extracted_0) %3 : !quantum.bit
      %4 = quantum.insert %arg2[%extracted], %out_qubits : !quantum.reg, !quantum.bit
      return %4 : !quantum.reg
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
      %extracted_1 = tensor.extract %arg0[] : tensor<f64>
      %4 = quantum.extract %3[%extracted] : !quantum.reg -> !quantum.bit
      %out_qubits_2 = quantum.custom "RX"(%extracted_1) %4 : !quantum.bit
      %5 = quantum.insert %3[%extracted], %out_qubits_2 : !quantum.reg, !quantum.bit
      %6 = quantum.extract %5[%extracted] : !quantum.reg -> !quantum.bit
      %out_qubits_3 = quantum.custom "RZ"(%cst) %6 : !quantum.bit
      %7 = quantum.insert %5[%extracted], %out_qubits_3 : !quantum.reg, !quantum.bit
      return %7 : !quantum.reg
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
      %extracted = tensor.extract %1[] : tensor<i64>
      %2 = quantum.extract %arg2[%extracted] : !quantum.reg -> !quantum.bit
      %out_qubits = quantum.custom "S"() %2 adj : !quantum.bit
      %3 = quantum.insert %arg2[%extracted], %out_qubits : !quantum.reg, !quantum.bit
      %extracted_0 = tensor.extract %arg0[] : tensor<f64>
      %4 = quantum.extract %3[%extracted] : !quantum.reg -> !quantum.bit
      %out_qubits_1 = quantum.custom "RX"(%extracted_0) %4 : !quantum.bit
      %5 = quantum.insert %3[%extracted], %out_qubits_1 : !quantum.reg, !quantum.bit
      %6 = quantum.extract %5[%extracted] : !quantum.reg -> !quantum.bit
      %out_qubits_2 = quantum.custom "S"() %6 : !quantum.bit
      %7 = quantum.insert %5[%extracted], %out_qubits_2 : !quantum.reg, !quantum.bit
      return %7 : !quantum.reg
    }
    func.func private @"__builtin__ry_to_rz_cliff_RY{0:[f64]}{wires:1}{}"(%arg0: tensor<f64>, %arg1: tensor<1xi64>, %arg2: !quantum.reg) -> !quantum.reg attributes {frontend_name = "_ry_to_rz_cliff", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"Adjoint(S){}{wires:1}{}" = 1 : i64, "Hadamard{}{wires:1}{}" = 2 : i64, "RZ{0:[f64]}{wires:1}{}" = 1 : i64, "S{}{wires:1}{}" = 1 : i64}}, target_gate = "RY{0:[f64]}{wires:1}{}"} {
      %0 = stablehlo.slice %arg1 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
      %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
      %extracted = tensor.extract %1[] : tensor<i64>
      %2 = quantum.extract %arg2[%extracted] : !quantum.reg -> !quantum.bit
      %out_qubits = quantum.custom "S"() %2 adj : !quantum.bit
      %3 = quantum.insert %arg2[%extracted], %out_qubits : !quantum.reg, !quantum.bit
      %4 = quantum.extract %3[%extracted] : !quantum.reg -> !quantum.bit
      %out_qubits_0 = quantum.custom "Hadamard"() %4 : !quantum.bit
      %5 = quantum.insert %3[%extracted], %out_qubits_0 : !quantum.reg, !quantum.bit
      %extracted_1 = tensor.extract %arg0[] : tensor<f64>
      %6 = quantum.extract %5[%extracted] : !quantum.reg -> !quantum.bit
      %out_qubits_2 = quantum.custom "RZ"(%extracted_1) %6 : !quantum.bit
      %7 = quantum.insert %5[%extracted], %out_qubits_2 : !quantum.reg, !quantum.bit
      %8 = quantum.extract %7[%extracted] : !quantum.reg -> !quantum.bit
      %out_qubits_3 = quantum.custom "Hadamard"() %8 : !quantum.bit
      %9 = quantum.insert %7[%extracted], %out_qubits_3 : !quantum.reg, !quantum.bit
      %10 = quantum.extract %9[%extracted] : !quantum.reg -> !quantum.bit
      %out_qubits_4 = quantum.custom "S"() %10 : !quantum.bit
      %11 = quantum.insert %9[%extracted], %out_qubits_4 : !quantum.reg, !quantum.bit
      return %11 : !quantum.reg
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
      %extracted_0 = tensor.extract %5[] : tensor<i64>
      %6 = quantum.extract %3[%extracted_0] : !quantum.reg -> !quantum.bit
      %7 = quantum.extract %3[%extracted] : !quantum.reg -> !quantum.bit
      %out_qubits_1:2 = quantum.custom "CNOT"() %6, %7 : !quantum.bit, !quantum.bit
      %8 = quantum.insert %3[%extracted_0], %out_qubits_1#0 : !quantum.reg, !quantum.bit
      %9 = quantum.insert %8[%extracted], %out_qubits_1#1 : !quantum.reg, !quantum.bit
      %10 = quantum.extract %9[%extracted] : !quantum.reg -> !quantum.bit
      %out_qubits_2 = quantum.custom "Hadamard"() %10 : !quantum.bit
      %11 = quantum.insert %9[%extracted], %out_qubits_2 : !quantum.reg, !quantum.bit
      return %11 : !quantum.reg
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
      %8 = quantum.extract %7[%extracted] : !quantum.reg -> !quantum.bit
      %9 = quantum.extract %7[%extracted_0] : !quantum.reg -> !quantum.bit
      %out_qubits_2:2 = quantum.operator "PPR"() qubits(%8, %9)
        static_data = {angle_denominator = 4 : i64, pauli_word = "ZZ"}
        qubit_map = {wires = [0, 1]}
      %10 = quantum.insert %7[%extracted], %out_qubits_2#0 : !quantum.reg, !quantum.bit
      %11 = quantum.insert %10[%extracted_0], %out_qubits_2#1 : !quantum.reg, !quantum.bit
      quantum.gphase(%cst)
      return %11 : !quantum.reg
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
      %7 = arith.negf %extracted_1 : f64
      quantum.gphase(%7)
      %8 = catalyst.list_pop %2 : <i64>
      %9 = quantum.extract %arg2[%8] : !quantum.reg -> !quantum.bit
      %10 = arith.negf %extracted_0 : f64
      %out_qubits = quantum.custom "RZ"(%10) %9 : !quantum.bit
      %11 = quantum.insert %arg2[%8], %out_qubits : !quantum.reg, !quantum.bit
      %12 = catalyst.list_pop %2 : <i64>
      catalyst.list_dealloc %0 : <f64>
      catalyst.list_dealloc %1 : <i64>
      catalyst.list_dealloc %2 : <i64>
      return %11 : !quantum.reg
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
      %extracted_1 = tensor.extract %arg0[] : tensor<f64>
      %6 = quantum.extract %5[%extracted_0] : !quantum.reg -> !quantum.bit
      %7 = quantum.extract %5[%extracted] : !quantum.reg -> !quantum.bit
      %out_qubits_2:2 = quantum.multirz(%extracted_1) %6, %7 : !quantum.bit, !quantum.bit
      %8 = quantum.insert %5[%extracted_0], %out_qubits_2#0 : !quantum.reg, !quantum.bit
      %9 = quantum.insert %8[%extracted], %out_qubits_2#1 : !quantum.reg, !quantum.bit
      %10 = quantum.extract %9[%extracted] : !quantum.reg -> !quantum.bit
      %out_qubits_3 = quantum.custom "Hadamard"() %10 : !quantum.bit
      %11 = quantum.insert %9[%extracted], %out_qubits_3 : !quantum.reg, !quantum.bit
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
      %extracted_2 = tensor.extract %arg0[] : tensor<f64>
      %8 = quantum.extract %7[%extracted_0] : !quantum.reg -> !quantum.bit
      %9 = quantum.extract %7[%extracted_1] : !quantum.reg -> !quantum.bit
      %10 = quantum.extract %7[%extracted] : !quantum.reg -> !quantum.bit
      %out_qubits_3:3 = quantum.multirz(%extracted_2) %8, %9, %10 : !quantum.bit, !quantum.bit, !quantum.bit
      %11 = quantum.insert %7[%extracted_0], %out_qubits_3#0 : !quantum.reg, !quantum.bit
      %12 = quantum.insert %11[%extracted_1], %out_qubits_3#1 : !quantum.reg, !quantum.bit
      %13 = quantum.insert %12[%extracted], %out_qubits_3#2 : !quantum.reg, !quantum.bit
      %14 = quantum.extract %13[%extracted] : !quantum.reg -> !quantum.bit
      %out_qubits_4 = quantum.custom "Hadamard"() %14 : !quantum.bit
      %15 = quantum.insert %13[%extracted], %out_qubits_4 : !quantum.reg, !quantum.bit
      return %15 : !quantum.reg
    }
    func.func private @"__builtin__pauli_rot_decomposition_PauliRot{theta:[f64]}{wires:1}{pauli_word = \22X\22}"(%arg0: tensor<f64>, %arg1: tensor<1xi64>, %arg2: !quantum.reg) -> !quantum.reg attributes {frontend_name = "_pauli_rot_decomposition", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"Hadamard{}{wires:1}{}" = 2 : i64, "MultiRZ{theta:[f64]}{wires:1}{}" = 1 : i64}}, target_gate = "PauliRot{theta:[f64]}{wires:1}{pauli_word = \22X\22}"} {
      %0 = stablehlo.slice %arg1 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
      %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
      %extracted = tensor.extract %1[] : tensor<i64>
      %2 = quantum.extract %arg2[%extracted] : !quantum.reg -> !quantum.bit
      %out_qubits = quantum.custom "Hadamard"() %2 : !quantum.bit
      %3 = quantum.insert %arg2[%extracted], %out_qubits : !quantum.reg, !quantum.bit
      %extracted_0 = tensor.extract %arg0[] : tensor<f64>
      %4 = quantum.extract %3[%extracted] : !quantum.reg -> !quantum.bit
      %out_qubits_1 = quantum.multirz(%extracted_0) %4 : !quantum.bit
      %5 = quantum.insert %3[%extracted], %out_qubits_1 : !quantum.reg, !quantum.bit
      %6 = quantum.extract %5[%extracted] : !quantum.reg -> !quantum.bit
      %out_qubits_2 = quantum.custom "Hadamard"() %6 : !quantum.bit
      %7 = quantum.insert %5[%extracted], %out_qubits_2 : !quantum.reg, !quantum.bit
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
      %extracted_1 = tensor.extract %arg1[] : tensor<f64>
      %4 = quantum.extract %3[%extracted] : !quantum.reg -> !quantum.bit
      %out_qubits_2 = quantum.custom "RY"(%extracted_1) %4 : !quantum.bit
      %5 = quantum.insert %3[%extracted], %out_qubits_2 : !quantum.reg, !quantum.bit
      %extracted_3 = tensor.extract %arg2[] : tensor<f64>
      %6 = quantum.extract %5[%extracted] : !quantum.reg -> !quantum.bit
      %out_qubits_4 = quantum.custom "RZ"(%extracted_3) %6 : !quantum.bit
      %7 = quantum.insert %5[%extracted], %out_qubits_4 : !quantum.reg, !quantum.bit
      return %7 : !quantum.reg
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
      %cst = arith.constant -1.5707963267948966 : f64
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
      %out_qubits = quantum.custom "PhaseShift"(%cst) %6 : !quantum.bit
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
      %extracted_1 = tensor.extract %arg0[] : tensor<f64>
      %4 = quantum.extract %3[%extracted] : !quantum.reg -> !quantum.bit
      %out_qubits_2 = quantum.multirz(%extracted_1) %4 : !quantum.bit
      %5 = quantum.insert %3[%extracted], %out_qubits_2 : !quantum.reg, !quantum.bit
      %6 = quantum.extract %5[%extracted] : !quantum.reg -> !quantum.bit
      %out_qubits_3 = quantum.custom "RX"(%cst) %6 : !quantum.bit
      %7 = quantum.insert %5[%extracted], %out_qubits_3 : !quantum.reg, !quantum.bit
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
      %5 = stablehlo.slice %arg1 [1:2] : (tensor<2xi64>) -> tensor<1xi64>
      %6 = stablehlo.reshape %5 : (tensor<1xi64>) -> tensor<i64>
      %extracted_2 = tensor.extract %6[] : tensor<i64>
      %7 = quantum.extract %4[%extracted] : !quantum.reg -> !quantum.bit
      %8 = quantum.extract %4[%extracted_2] : !quantum.reg -> !quantum.bit
      %out_qubits_3:2 = quantum.custom "CNOT"() %7, %8 : !quantum.bit, !quantum.bit
      %9 = quantum.insert %4[%extracted], %out_qubits_3#0 : !quantum.reg, !quantum.bit
      %10 = quantum.insert %9[%extracted_2], %out_qubits_3#1 : !quantum.reg, !quantum.bit
      %11 = stablehlo.negate %arg0 : tensor<f64>
      %12 = stablehlo.divide %11, %cst_0 : tensor<f64>
      %extracted_4 = tensor.extract %12[] : tensor<f64>
      %13 = quantum.extract %10[%extracted_2] : !quantum.reg -> !quantum.bit
      %out_qubits_5 = quantum.custom "RZ"(%extracted_4) %13 : !quantum.bit
      %14 = quantum.insert %10[%extracted_2], %out_qubits_5 : !quantum.reg, !quantum.bit
      %15 = quantum.extract %14[%extracted] : !quantum.reg -> !quantum.bit
      %16 = quantum.extract %14[%extracted_2] : !quantum.reg -> !quantum.bit
      %out_qubits_6:2 = quantum.custom "CNOT"() %15, %16 : !quantum.bit, !quantum.bit
      %17 = quantum.insert %14[%extracted], %out_qubits_6#0 : !quantum.reg, !quantum.bit
      %18 = quantum.insert %17[%extracted_2], %out_qubits_6#1 : !quantum.reg, !quantum.bit
      %19 = quantum.extract %18[%extracted_2] : !quantum.reg -> !quantum.bit
      %out_qubits_7 = quantum.custom "RZ"(%extracted_1) %19 : !quantum.bit
      %20 = quantum.insert %18[%extracted_2], %out_qubits_7 : !quantum.reg, !quantum.bit
      %21 = stablehlo.divide %11, %cst : tensor<f64>
      %extracted_8 = tensor.extract %21[] : tensor<f64>
      quantum.gphase(%extracted_8)
      return %20 : !quantum.reg
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
      %extracted_3 = tensor.extract %10[] : tensor<f64>
      %11 = quantum.extract %9[%extracted_1] : !quantum.reg -> !quantum.bit
      %out_qubits_4 = quantum.paulirot ["Z"](%extracted_3) %11 : !quantum.bit
      %12 = quantum.insert %9[%extracted_1], %out_qubits_4 : !quantum.reg, !quantum.bit
      %13 = quantum.extract %12[%extracted] : !quantum.reg -> !quantum.bit
      %out_qubits_5 = quantum.paulirot ["Z"](%extracted_3) %13 : !quantum.bit
      %14 = quantum.insert %12[%extracted], %out_qubits_5 : !quantum.reg, !quantum.bit
      %15 = stablehlo.divide %0, %cst : tensor<f64>
      %extracted_6 = tensor.extract %15[] : tensor<f64>
      quantum.gphase(%extracted_6)
      return %14 : !quantum.reg
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
      %6 = arith.negf %extracted_1 : f64
      quantum.gphase(%6)
      %7 = catalyst.list_pop %2 : <i64>
      %8 = quantum.extract %arg2[%7] : !quantum.reg -> !quantum.bit
      %9 = arith.negf %extracted_0 : f64
      %out_qubits = quantum.custom "PhaseShift"(%9) %8 : !quantum.bit
      %10 = quantum.insert %arg2[%7], %out_qubits : !quantum.reg, !quantum.bit
      %11 = catalyst.list_pop %2 : <i64>
      catalyst.list_dealloc %0 : <f64>
      catalyst.list_dealloc %1 : <i64>
      catalyst.list_dealloc %2 : <i64>
      return %10 : !quantum.reg
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
      %extracted_1 = tensor.extract %arg0[] : tensor<f64>
      catalyst.list_push %extracted, %2 : <i64>
      catalyst.list_push %extracted, %2 : <i64>
      %6 = catalyst.list_pop %2 : <i64>
      %7 = quantum.extract %arg2[%6] : !quantum.reg -> !quantum.bit
      %out_qubits = quantum.custom "Rot"(%extracted_0, %extracted_0, %extracted_1) %7 adj : !quantum.bit
      %8 = quantum.insert %arg2[%6], %out_qubits : !quantum.reg, !quantum.bit
      %9 = catalyst.list_pop %2 : <i64>
      catalyst.list_dealloc %0 : <f64>
      catalyst.list_dealloc %1 : <i64>
      catalyst.list_dealloc %2 : <i64>
      return %8 : !quantum.reg
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
      %extracted_1 = tensor.extract %arg0[] : tensor<f64>
      catalyst.list_push %extracted, %2 : <i64>
      catalyst.list_push %extracted, %2 : <i64>
      catalyst.list_push %extracted, %2 : <i64>
      catalyst.list_push %extracted, %2 : <i64>
      %5 = catalyst.list_pop %2 : <i64>
      %6 = quantum.extract %arg2[%5] : !quantum.reg -> !quantum.bit
      %out_qubits = quantum.custom "RY"(%cst_0) %6 : !quantum.bit
      %7 = quantum.insert %arg2[%5], %out_qubits : !quantum.reg, !quantum.bit
      %8 = catalyst.list_pop %2 : <i64>
      %9 = catalyst.list_pop %2 : <i64>
      %10 = quantum.extract %7[%9] : !quantum.reg -> !quantum.bit
      %11 = arith.negf %extracted_1 : f64
      %out_qubits_2 = quantum.custom "RX"(%11) %10 : !quantum.bit
      %12 = quantum.insert %7[%9], %out_qubits_2 : !quantum.reg, !quantum.bit
      %13 = catalyst.list_pop %2 : <i64>
      %14 = catalyst.list_pop %2 : <i64>
      %15 = quantum.extract %12[%14] : !quantum.reg -> !quantum.bit
      %out_qubits_3 = quantum.custom "RY"(%cst) %15 : !quantum.bit
      %16 = quantum.insert %12[%14], %out_qubits_3 : !quantum.reg, !quantum.bit
      %17 = catalyst.list_pop %2 : <i64>
      catalyst.list_dealloc %0 : <f64>
      catalyst.list_dealloc %1 : <i64>
      catalyst.list_dealloc %2 : <i64>
      return %16 : !quantum.reg
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
      %7 = arith.negf %extracted_0 : f64
      %out_qubits = quantum.paulirot ["Z"](%7) %6 : !quantum.bit
      %8 = quantum.insert %arg2[%5], %out_qubits : !quantum.reg, !quantum.bit
      %9 = catalyst.list_pop %2 : <i64>
      catalyst.list_dealloc %0 : <f64>
      catalyst.list_dealloc %1 : <i64>
      catalyst.list_dealloc %2 : <i64>
      return %8 : !quantum.reg
    }
    func.func private @"__builtin__rz_to_rx_cliff_Adjoint(RZ){0:[f64]}{wires:1}{}"(%arg0: tensor<f64>, %arg1: tensor<1xi64>, %arg2: !quantum.reg) -> !quantum.reg attributes {frontend_name = "_rz_to_rx_cliff", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"Adjoint(Hadamard){}{wires:1}{}" = 2 : i64, "Adjoint(RX){0:[f64]}{wires:1}{}" = 1 : i64}}, target_gate = "Adjoint(RZ){0:[f64]}{wires:1}{}"} {
      %0 = catalyst.list_init : <f64>
      %1 = catalyst.list_init : <i64>
      %2 = catalyst.list_init : <i64>
      %3 = stablehlo.slice %arg1 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
      %4 = stablehlo.reshape %3 : (tensor<1xi64>) -> tensor<i64>
      %extracted = tensor.extract %4[] : tensor<i64>
      catalyst.list_push %extracted, %2 : <i64>
      catalyst.list_push %extracted, %2 : <i64>
      %extracted_0 = tensor.extract %arg0[] : tensor<f64>
      catalyst.list_push %extracted, %2 : <i64>
      catalyst.list_push %extracted, %2 : <i64>
      catalyst.list_push %extracted, %2 : <i64>
      catalyst.list_push %extracted, %2 : <i64>
      %5 = catalyst.list_pop %2 : <i64>
      %6 = quantum.extract %arg2[%5] : !quantum.reg -> !quantum.bit
      %out_qubits = quantum.custom "Hadamard"() %6 : !quantum.bit
      %7 = quantum.insert %arg2[%5], %out_qubits : !quantum.reg, !quantum.bit
      %8 = catalyst.list_pop %2 : <i64>
      %9 = catalyst.list_pop %2 : <i64>
      %10 = quantum.extract %7[%9] : !quantum.reg -> !quantum.bit
      %11 = arith.negf %extracted_0 : f64
      %out_qubits_1 = quantum.custom "RX"(%11) %10 : !quantum.bit
      %12 = quantum.insert %7[%9], %out_qubits_1 : !quantum.reg, !quantum.bit
      %13 = catalyst.list_pop %2 : <i64>
      %14 = catalyst.list_pop %2 : <i64>
      %15 = quantum.extract %12[%14] : !quantum.reg -> !quantum.bit
      %out_qubits_2 = quantum.custom "Hadamard"() %15 : !quantum.bit
      %16 = quantum.insert %12[%14], %out_qubits_2 : !quantum.reg, !quantum.bit
      %17 = catalyst.list_pop %2 : <i64>
      catalyst.list_dealloc %0 : <f64>
      catalyst.list_dealloc %1 : <i64>
      catalyst.list_dealloc %2 : <i64>
      return %16 : !quantum.reg
    }
    func.func private @"__builtin__rz_to_ry_cliff_Adjoint(RZ){0:[f64]}{wires:1}{}"(%arg0: tensor<f64>, %arg1: tensor<1xi64>, %arg2: !quantum.reg) -> !quantum.reg attributes {frontend_name = "_rz_to_ry_cliff", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"Adjoint(Hadamard){}{wires:1}{}" = 2 : i64, "Adjoint(RY){0:[f64]}{wires:1}{}" = 1 : i64, "Adjoint(S){}{wires:1}{}" = 1 : i64, "S{}{wires:1}{}" = 1 : i64}}, target_gate = "Adjoint(RZ){0:[f64]}{wires:1}{}"} {
      %0 = catalyst.list_init : <f64>
      %1 = catalyst.list_init : <i64>
      %2 = catalyst.list_init : <i64>
      %3 = stablehlo.slice %arg1 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
      %4 = stablehlo.reshape %3 : (tensor<1xi64>) -> tensor<i64>
      %extracted = tensor.extract %4[] : tensor<i64>
      catalyst.list_push %extracted, %2 : <i64>
      catalyst.list_push %extracted, %2 : <i64>
      catalyst.list_push %extracted, %2 : <i64>
      catalyst.list_push %extracted, %2 : <i64>
      %extracted_0 = tensor.extract %arg0[] : tensor<f64>
      catalyst.list_push %extracted, %2 : <i64>
      catalyst.list_push %extracted, %2 : <i64>
      catalyst.list_push %extracted, %2 : <i64>
      catalyst.list_push %extracted, %2 : <i64>
      catalyst.list_push %extracted, %2 : <i64>
      catalyst.list_push %extracted, %2 : <i64>
      %5 = catalyst.list_pop %2 : <i64>
      %6 = quantum.extract %arg2[%5] : !quantum.reg -> !quantum.bit
      %out_qubits = quantum.custom "Hadamard"() %6 : !quantum.bit
      %7 = quantum.insert %arg2[%5], %out_qubits : !quantum.reg, !quantum.bit
      %8 = catalyst.list_pop %2 : <i64>
      %9 = catalyst.list_pop %2 : <i64>
      %10 = quantum.extract %7[%9] : !quantum.reg -> !quantum.bit
      %out_qubits_1 = quantum.custom "S"() %10 : !quantum.bit
      %11 = quantum.insert %7[%9], %out_qubits_1 : !quantum.reg, !quantum.bit
      %12 = catalyst.list_pop %2 : <i64>
      %13 = catalyst.list_pop %2 : <i64>
      %14 = quantum.extract %11[%13] : !quantum.reg -> !quantum.bit
      %15 = arith.negf %extracted_0 : f64
      %out_qubits_2 = quantum.custom "RY"(%15) %14 : !quantum.bit
      %16 = quantum.insert %11[%13], %out_qubits_2 : !quantum.reg, !quantum.bit
      %17 = catalyst.list_pop %2 : <i64>
      %18 = catalyst.list_pop %2 : <i64>
      %19 = quantum.extract %16[%18] : !quantum.reg -> !quantum.bit
      %out_qubits_3 = quantum.custom "S"() %19 adj : !quantum.bit
      %20 = quantum.insert %16[%18], %out_qubits_3 : !quantum.reg, !quantum.bit
      %21 = catalyst.list_pop %2 : <i64>
      %22 = catalyst.list_pop %2 : <i64>
      %23 = quantum.extract %20[%22] : !quantum.reg -> !quantum.bit
      %out_qubits_4 = quantum.custom "Hadamard"() %23 : !quantum.bit
      %24 = quantum.insert %20[%22], %out_qubits_4 : !quantum.reg, !quantum.bit
      %25 = catalyst.list_pop %2 : <i64>
      catalyst.list_dealloc %0 : <f64>
      catalyst.list_dealloc %1 : <i64>
      catalyst.list_dealloc %2 : <i64>
      return %24 : !quantum.reg
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
      %extracted_1 = tensor.extract %arg1[] : tensor<f64>
      catalyst.list_push %extracted, %2 : <i64>
      catalyst.list_push %extracted, %2 : <i64>
      %extracted_2 = tensor.extract %arg2[] : tensor<f64>
      catalyst.list_push %extracted, %2 : <i64>
      catalyst.list_push %extracted, %2 : <i64>
      %5 = catalyst.list_pop %2 : <i64>
      %6 = quantum.extract %arg4[%5] : !quantum.reg -> !quantum.bit
      %7 = arith.negf %extracted_2 : f64
      %out_qubits = quantum.custom "RZ"(%7) %6 : !quantum.bit
      %8 = quantum.insert %arg4[%5], %out_qubits : !quantum.reg, !quantum.bit
      %9 = catalyst.list_pop %2 : <i64>
      %10 = catalyst.list_pop %2 : <i64>
      %11 = quantum.extract %8[%10] : !quantum.reg -> !quantum.bit
      %12 = arith.negf %extracted_1 : f64
      %out_qubits_3 = quantum.custom "RY"(%12) %11 : !quantum.bit
      %13 = quantum.insert %8[%10], %out_qubits_3 : !quantum.reg, !quantum.bit
      %14 = catalyst.list_pop %2 : <i64>
      %15 = catalyst.list_pop %2 : <i64>
      %16 = quantum.extract %13[%15] : !quantum.reg -> !quantum.bit
      %17 = arith.negf %extracted_0 : f64
      %out_qubits_4 = quantum.custom "RZ"(%17) %16 : !quantum.bit
      %18 = quantum.insert %13[%15], %out_qubits_4 : !quantum.reg, !quantum.bit
      %19 = catalyst.list_pop %2 : <i64>
      catalyst.list_dealloc %0 : <f64>
      catalyst.list_dealloc %1 : <i64>
      catalyst.list_dealloc %2 : <i64>
      return %18 : !quantum.reg
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
      catalyst.list_push %extracted, %2 : <i64>
      catalyst.list_push %extracted, %2 : <i64>
      %6 = catalyst.list_pop %2 : <i64>
      %7 = quantum.extract %arg2[%6] : !quantum.reg -> !quantum.bit
      %out_qubits = quantum.custom "Rot"(%extracted_0, %extracted_1, %extracted_0) %7 adj : !quantum.bit
      %8 = quantum.insert %arg2[%6], %out_qubits : !quantum.reg, !quantum.bit
      %9 = catalyst.list_pop %2 : <i64>
      catalyst.list_dealloc %0 : <f64>
      catalyst.list_dealloc %1 : <i64>
      catalyst.list_dealloc %2 : <i64>
      return %8 : !quantum.reg
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
      %extracted_1 = tensor.extract %arg0[] : tensor<f64>
      catalyst.list_push %extracted, %2 : <i64>
      catalyst.list_push %extracted, %2 : <i64>
      catalyst.list_push %extracted, %2 : <i64>
      catalyst.list_push %extracted, %2 : <i64>
      %5 = catalyst.list_pop %2 : <i64>
      %6 = quantum.extract %arg2[%5] : !quantum.reg -> !quantum.bit
      %out_qubits = quantum.custom "RZ"(%cst_0) %6 : !quantum.bit
      %7 = quantum.insert %arg2[%5], %out_qubits : !quantum.reg, !quantum.bit
      %8 = catalyst.list_pop %2 : <i64>
      %9 = catalyst.list_pop %2 : <i64>
      %10 = quantum.extract %7[%9] : !quantum.reg -> !quantum.bit
      %11 = arith.negf %extracted_1 : f64
      %out_qubits_2 = quantum.custom "RX"(%11) %10 : !quantum.bit
      %12 = quantum.insert %7[%9], %out_qubits_2 : !quantum.reg, !quantum.bit
      %13 = catalyst.list_pop %2 : <i64>
      %14 = catalyst.list_pop %2 : <i64>
      %15 = quantum.extract %12[%14] : !quantum.reg -> !quantum.bit
      %out_qubits_3 = quantum.custom "RZ"(%cst) %15 : !quantum.bit
      %16 = quantum.insert %12[%14], %out_qubits_3 : !quantum.reg, !quantum.bit
      %17 = catalyst.list_pop %2 : <i64>
      catalyst.list_dealloc %0 : <f64>
      catalyst.list_dealloc %1 : <i64>
      catalyst.list_dealloc %2 : <i64>
      return %16 : !quantum.reg
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
      %7 = arith.negf %extracted_0 : f64
      %out_qubits = quantum.paulirot ["Y"](%7) %6 : !quantum.bit
      %8 = quantum.insert %arg2[%5], %out_qubits : !quantum.reg, !quantum.bit
      %9 = catalyst.list_pop %2 : <i64>
      catalyst.list_dealloc %0 : <f64>
      catalyst.list_dealloc %1 : <i64>
      catalyst.list_dealloc %2 : <i64>
      return %8 : !quantum.reg
    }
    func.func private @"__builtin__ry_to_rx_cliff_Adjoint(RY){0:[f64]}{wires:1}{}"(%arg0: tensor<f64>, %arg1: tensor<1xi64>, %arg2: !quantum.reg) -> !quantum.reg attributes {frontend_name = "_ry_to_rx_cliff", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"Adjoint(RX){0:[f64]}{wires:1}{}" = 1 : i64, "Adjoint(S){}{wires:1}{}" = 1 : i64, "S{}{wires:1}{}" = 1 : i64}}, target_gate = "Adjoint(RY){0:[f64]}{wires:1}{}"} {
      %0 = catalyst.list_init : <f64>
      %1 = catalyst.list_init : <i64>
      %2 = catalyst.list_init : <i64>
      %3 = stablehlo.slice %arg1 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
      %4 = stablehlo.reshape %3 : (tensor<1xi64>) -> tensor<i64>
      %extracted = tensor.extract %4[] : tensor<i64>
      catalyst.list_push %extracted, %2 : <i64>
      catalyst.list_push %extracted, %2 : <i64>
      %extracted_0 = tensor.extract %arg0[] : tensor<f64>
      catalyst.list_push %extracted, %2 : <i64>
      catalyst.list_push %extracted, %2 : <i64>
      catalyst.list_push %extracted, %2 : <i64>
      catalyst.list_push %extracted, %2 : <i64>
      %5 = catalyst.list_pop %2 : <i64>
      %6 = quantum.extract %arg2[%5] : !quantum.reg -> !quantum.bit
      %out_qubits = quantum.custom "S"() %6 adj : !quantum.bit
      %7 = quantum.insert %arg2[%5], %out_qubits : !quantum.reg, !quantum.bit
      %8 = catalyst.list_pop %2 : <i64>
      %9 = catalyst.list_pop %2 : <i64>
      %10 = quantum.extract %7[%9] : !quantum.reg -> !quantum.bit
      %11 = arith.negf %extracted_0 : f64
      %out_qubits_1 = quantum.custom "RX"(%11) %10 : !quantum.bit
      %12 = quantum.insert %7[%9], %out_qubits_1 : !quantum.reg, !quantum.bit
      %13 = catalyst.list_pop %2 : <i64>
      %14 = catalyst.list_pop %2 : <i64>
      %15 = quantum.extract %12[%14] : !quantum.reg -> !quantum.bit
      %out_qubits_2 = quantum.custom "S"() %15 : !quantum.bit
      %16 = quantum.insert %12[%14], %out_qubits_2 : !quantum.reg, !quantum.bit
      %17 = catalyst.list_pop %2 : <i64>
      catalyst.list_dealloc %0 : <f64>
      catalyst.list_dealloc %1 : <i64>
      catalyst.list_dealloc %2 : <i64>
      return %16 : !quantum.reg
    }
    func.func private @"__builtin__ry_to_rz_cliff_Adjoint(RY){0:[f64]}{wires:1}{}"(%arg0: tensor<f64>, %arg1: tensor<1xi64>, %arg2: !quantum.reg) -> !quantum.reg attributes {frontend_name = "_ry_to_rz_cliff", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"Adjoint(Hadamard){}{wires:1}{}" = 2 : i64, "Adjoint(RZ){0:[f64]}{wires:1}{}" = 1 : i64, "Adjoint(S){}{wires:1}{}" = 1 : i64, "S{}{wires:1}{}" = 1 : i64}}, target_gate = "Adjoint(RY){0:[f64]}{wires:1}{}"} {
      %0 = catalyst.list_init : <f64>
      %1 = catalyst.list_init : <i64>
      %2 = catalyst.list_init : <i64>
      %3 = stablehlo.slice %arg1 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
      %4 = stablehlo.reshape %3 : (tensor<1xi64>) -> tensor<i64>
      %extracted = tensor.extract %4[] : tensor<i64>
      catalyst.list_push %extracted, %2 : <i64>
      catalyst.list_push %extracted, %2 : <i64>
      catalyst.list_push %extracted, %2 : <i64>
      catalyst.list_push %extracted, %2 : <i64>
      %extracted_0 = tensor.extract %arg0[] : tensor<f64>
      catalyst.list_push %extracted, %2 : <i64>
      catalyst.list_push %extracted, %2 : <i64>
      catalyst.list_push %extracted, %2 : <i64>
      catalyst.list_push %extracted, %2 : <i64>
      catalyst.list_push %extracted, %2 : <i64>
      catalyst.list_push %extracted, %2 : <i64>
      %5 = catalyst.list_pop %2 : <i64>
      %6 = quantum.extract %arg2[%5] : !quantum.reg -> !quantum.bit
      %out_qubits = quantum.custom "S"() %6 adj : !quantum.bit
      %7 = quantum.insert %arg2[%5], %out_qubits : !quantum.reg, !quantum.bit
      %8 = catalyst.list_pop %2 : <i64>
      %9 = catalyst.list_pop %2 : <i64>
      %10 = quantum.extract %7[%9] : !quantum.reg -> !quantum.bit
      %out_qubits_1 = quantum.custom "Hadamard"() %10 : !quantum.bit
      %11 = quantum.insert %7[%9], %out_qubits_1 : !quantum.reg, !quantum.bit
      %12 = catalyst.list_pop %2 : <i64>
      %13 = catalyst.list_pop %2 : <i64>
      %14 = quantum.extract %11[%13] : !quantum.reg -> !quantum.bit
      %15 = arith.negf %extracted_0 : f64
      %out_qubits_2 = quantum.custom "RZ"(%15) %14 : !quantum.bit
      %16 = quantum.insert %11[%13], %out_qubits_2 : !quantum.reg, !quantum.bit
      %17 = catalyst.list_pop %2 : <i64>
      %18 = catalyst.list_pop %2 : <i64>
      %19 = quantum.extract %16[%18] : !quantum.reg -> !quantum.bit
      %out_qubits_3 = quantum.custom "Hadamard"() %19 : !quantum.bit
      %20 = quantum.insert %16[%18], %out_qubits_3 : !quantum.reg, !quantum.bit
      %21 = catalyst.list_pop %2 : <i64>
      %22 = catalyst.list_pop %2 : <i64>
      %23 = quantum.extract %20[%22] : !quantum.reg -> !quantum.bit
      %out_qubits_4 = quantum.custom "S"() %23 : !quantum.bit
      %24 = quantum.insert %20[%22], %out_qubits_4 : !quantum.reg, !quantum.bit
      %25 = catalyst.list_pop %2 : <i64>
      catalyst.list_dealloc %0 : <f64>
      catalyst.list_dealloc %1 : <i64>
      catalyst.list_dealloc %2 : <i64>
      return %24 : !quantum.reg
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
      %extracted_1 = tensor.extract %arg0[] : tensor<f64>
      catalyst.list_push %extracted, %2 : <i64>
      catalyst.list_push %extracted, %2 : <i64>
      catalyst.list_push %extracted, %2 : <i64>
      catalyst.list_push %extracted, %2 : <i64>
      %5 = catalyst.list_pop %2 : <i64>
      %6 = quantum.extract %arg2[%5] : !quantum.reg -> !quantum.bit
      %out_qubits = quantum.custom "RZ"(%cst_0) %6 : !quantum.bit
      %7 = quantum.insert %arg2[%5], %out_qubits : !quantum.reg, !quantum.bit
      %8 = catalyst.list_pop %2 : <i64>
      %9 = catalyst.list_pop %2 : <i64>
      %10 = quantum.extract %7[%9] : !quantum.reg -> !quantum.bit
      %11 = arith.negf %extracted_1 : f64
      %out_qubits_2 = quantum.custom "RY"(%11) %10 : !quantum.bit
      %12 = quantum.insert %7[%9], %out_qubits_2 : !quantum.reg, !quantum.bit
      %13 = catalyst.list_pop %2 : <i64>
      %14 = catalyst.list_pop %2 : <i64>
      %15 = quantum.extract %12[%14] : !quantum.reg -> !quantum.bit
      %out_qubits_3 = quantum.custom "RZ"(%cst) %15 : !quantum.bit
      %16 = quantum.insert %12[%14], %out_qubits_3 : !quantum.reg, !quantum.bit
      %17 = catalyst.list_pop %2 : <i64>
      catalyst.list_dealloc %0 : <f64>
      catalyst.list_dealloc %1 : <i64>
      catalyst.list_dealloc %2 : <i64>
      return %16 : !quantum.reg
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
      %7 = arith.negf %extracted_0 : f64
      %out_qubits = quantum.paulirot ["X"](%7) %6 : !quantum.bit
      %8 = quantum.insert %arg2[%5], %out_qubits : !quantum.reg, !quantum.bit
      %9 = catalyst.list_pop %2 : <i64>
      catalyst.list_dealloc %0 : <f64>
      catalyst.list_dealloc %1 : <i64>
      catalyst.list_dealloc %2 : <i64>
      return %8 : !quantum.reg
    }
    func.func private @"__builtin__rx_to_ry_cliff_Adjoint(RX){0:[f64]}{wires:1}{}"(%arg0: tensor<f64>, %arg1: tensor<1xi64>, %arg2: !quantum.reg) -> !quantum.reg attributes {frontend_name = "_rx_to_ry_cliff", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"Adjoint(RY){0:[f64]}{wires:1}{}" = 1 : i64, "Adjoint(S){}{wires:1}{}" = 1 : i64, "S{}{wires:1}{}" = 1 : i64}}, target_gate = "Adjoint(RX){0:[f64]}{wires:1}{}"} {
      %0 = catalyst.list_init : <f64>
      %1 = catalyst.list_init : <i64>
      %2 = catalyst.list_init : <i64>
      %3 = stablehlo.slice %arg1 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
      %4 = stablehlo.reshape %3 : (tensor<1xi64>) -> tensor<i64>
      %extracted = tensor.extract %4[] : tensor<i64>
      catalyst.list_push %extracted, %2 : <i64>
      catalyst.list_push %extracted, %2 : <i64>
      %extracted_0 = tensor.extract %arg0[] : tensor<f64>
      catalyst.list_push %extracted, %2 : <i64>
      catalyst.list_push %extracted, %2 : <i64>
      catalyst.list_push %extracted, %2 : <i64>
      catalyst.list_push %extracted, %2 : <i64>
      %5 = catalyst.list_pop %2 : <i64>
      %6 = quantum.extract %arg2[%5] : !quantum.reg -> !quantum.bit
      %out_qubits = quantum.custom "S"() %6 : !quantum.bit
      %7 = quantum.insert %arg2[%5], %out_qubits : !quantum.reg, !quantum.bit
      %8 = catalyst.list_pop %2 : <i64>
      %9 = catalyst.list_pop %2 : <i64>
      %10 = quantum.extract %7[%9] : !quantum.reg -> !quantum.bit
      %11 = arith.negf %extracted_0 : f64
      %out_qubits_1 = quantum.custom "RY"(%11) %10 : !quantum.bit
      %12 = quantum.insert %7[%9], %out_qubits_1 : !quantum.reg, !quantum.bit
      %13 = catalyst.list_pop %2 : <i64>
      %14 = catalyst.list_pop %2 : <i64>
      %15 = quantum.extract %12[%14] : !quantum.reg -> !quantum.bit
      %out_qubits_2 = quantum.custom "S"() %15 adj : !quantum.bit
      %16 = quantum.insert %12[%14], %out_qubits_2 : !quantum.reg, !quantum.bit
      %17 = catalyst.list_pop %2 : <i64>
      catalyst.list_dealloc %0 : <f64>
      catalyst.list_dealloc %1 : <i64>
      catalyst.list_dealloc %2 : <i64>
      return %16 : !quantum.reg
    }
    func.func private @"__builtin__rx_to_rz_cliff_Adjoint(RX){0:[f64]}{wires:1}{}"(%arg0: tensor<f64>, %arg1: tensor<1xi64>, %arg2: !quantum.reg) -> !quantum.reg attributes {frontend_name = "_rx_to_rz_cliff", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"Adjoint(Hadamard){}{wires:1}{}" = 2 : i64, "Adjoint(RZ){0:[f64]}{wires:1}{}" = 1 : i64}}, target_gate = "Adjoint(RX){0:[f64]}{wires:1}{}"} {
      %0 = catalyst.list_init : <f64>
      %1 = catalyst.list_init : <i64>
      %2 = catalyst.list_init : <i64>
      %3 = stablehlo.slice %arg1 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
      %4 = stablehlo.reshape %3 : (tensor<1xi64>) -> tensor<i64>
      %extracted = tensor.extract %4[] : tensor<i64>
      catalyst.list_push %extracted, %2 : <i64>
      catalyst.list_push %extracted, %2 : <i64>
      %extracted_0 = tensor.extract %arg0[] : tensor<f64>
      catalyst.list_push %extracted, %2 : <i64>
      catalyst.list_push %extracted, %2 : <i64>
      catalyst.list_push %extracted, %2 : <i64>
      catalyst.list_push %extracted, %2 : <i64>
      %5 = catalyst.list_pop %2 : <i64>
      %6 = quantum.extract %arg2[%5] : !quantum.reg -> !quantum.bit
      %out_qubits = quantum.custom "Hadamard"() %6 : !quantum.bit
      %7 = quantum.insert %arg2[%5], %out_qubits : !quantum.reg, !quantum.bit
      %8 = catalyst.list_pop %2 : <i64>
      %9 = catalyst.list_pop %2 : <i64>
      %10 = quantum.extract %7[%9] : !quantum.reg -> !quantum.bit
      %11 = arith.negf %extracted_0 : f64
      %out_qubits_1 = quantum.custom "RZ"(%11) %10 : !quantum.bit
      %12 = quantum.insert %7[%9], %out_qubits_1 : !quantum.reg, !quantum.bit
      %13 = catalyst.list_pop %2 : <i64>
      %14 = catalyst.list_pop %2 : <i64>
      %15 = quantum.extract %12[%14] : !quantum.reg -> !quantum.bit
      %out_qubits_2 = quantum.custom "Hadamard"() %15 : !quantum.bit
      %16 = quantum.insert %12[%14], %out_qubits_2 : !quantum.reg, !quantum.bit
      %17 = catalyst.list_pop %2 : <i64>
      catalyst.list_dealloc %0 : <f64>
      catalyst.list_dealloc %1 : <i64>
      catalyst.list_dealloc %2 : <i64>
      return %16 : !quantum.reg
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
      %7 = arith.negf %extracted_0 : f64
      %out_qubits = quantum.multirz(%7) %6 : !quantum.bit
      %8 = quantum.insert %arg2[%5], %out_qubits : !quantum.reg, !quantum.bit
      %9 = catalyst.list_pop %2 : <i64>
      catalyst.list_dealloc %0 : <f64>
      catalyst.list_dealloc %1 : <i64>
      catalyst.list_dealloc %2 : <i64>
      return %8 : !quantum.reg
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
      catalyst.list_push %extracted, %2 : <i64>
      catalyst.list_push %extracted, %2 : <i64>
      catalyst.list_push %extracted, %2 : <i64>
      catalyst.list_push %extracted, %2 : <i64>
      quantum.gphase(%cst_0)
      %5 = catalyst.list_pop %2 : <i64>
      %6 = quantum.extract %arg1[%5] : !quantum.reg -> !quantum.bit
      %out_qubits = quantum.custom "RZ"(%cst) %6 : !quantum.bit
      %7 = quantum.insert %arg1[%5], %out_qubits : !quantum.reg, !quantum.bit
      %8 = catalyst.list_pop %2 : <i64>
      %9 = catalyst.list_pop %2 : <i64>
      %10 = quantum.extract %7[%9] : !quantum.reg -> !quantum.bit
      %out_qubits_1 = quantum.custom "RX"(%cst) %10 : !quantum.bit
      %11 = quantum.insert %7[%9], %out_qubits_1 : !quantum.reg, !quantum.bit
      %12 = catalyst.list_pop %2 : <i64>
      %13 = catalyst.list_pop %2 : <i64>
      %14 = quantum.extract %11[%13] : !quantum.reg -> !quantum.bit
      %out_qubits_2 = quantum.custom "RZ"(%cst) %14 : !quantum.bit
      %15 = quantum.insert %11[%13], %out_qubits_2 : !quantum.reg, !quantum.bit
      %16 = catalyst.list_pop %2 : <i64>
      catalyst.list_dealloc %0 : <f64>
      catalyst.list_dealloc %1 : <i64>
      catalyst.list_dealloc %2 : <i64>
      return %15 : !quantum.reg
    }
    func.func private @"__builtin__hadamard_to_rz_ry_Adjoint(Hadamard){}{wires:1}{}"(%arg0: tensor<1xi64>, %arg1: !quantum.reg) -> !quantum.reg attributes {frontend_name = "_hadamard_to_rz_ry", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"Adjoint(GlobalPhase){phi:[f64]}{}{}" = 1 : i64, "Adjoint(RY){0:[f64]}{wires:1}{}" = 1 : i64, "Adjoint(RZ){0:[f64]}{wires:1}{}" = 1 : i64}}, target_gate = "Adjoint(Hadamard){}{wires:1}{}"} {
      %cst = arith.constant -3.1415926535897931 : f64
      %cst_0 = arith.constant -1.5707963267948966 : f64
      %cst_1 = arith.constant 1.5707963267948966 : f64
      %0 = catalyst.list_init : <f64>
      %1 = catalyst.list_init : <i64>
      %2 = catalyst.list_init : <i64>
      %3 = stablehlo.slice %arg0 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
      %4 = stablehlo.reshape %3 : (tensor<1xi64>) -> tensor<i64>
      %extracted = tensor.extract %4[] : tensor<i64>
      catalyst.list_push %extracted, %2 : <i64>
      catalyst.list_push %extracted, %2 : <i64>
      catalyst.list_push %extracted, %2 : <i64>
      catalyst.list_push %extracted, %2 : <i64>
      quantum.gphase(%cst_1)
      %5 = catalyst.list_pop %2 : <i64>
      %6 = quantum.extract %arg1[%5] : !quantum.reg -> !quantum.bit
      %out_qubits = quantum.custom "RY"(%cst_0) %6 : !quantum.bit
      %7 = quantum.insert %arg1[%5], %out_qubits : !quantum.reg, !quantum.bit
      %8 = catalyst.list_pop %2 : <i64>
      %9 = catalyst.list_pop %2 : <i64>
      %10 = quantum.extract %7[%9] : !quantum.reg -> !quantum.bit
      %out_qubits_2 = quantum.custom "RZ"(%cst) %10 : !quantum.bit
      %11 = quantum.insert %7[%9], %out_qubits_2 : !quantum.reg, !quantum.bit
      %12 = catalyst.list_pop %2 : <i64>
      catalyst.list_dealloc %0 : <f64>
      catalyst.list_dealloc %1 : <i64>
      catalyst.list_dealloc %2 : <i64>
      return %11 : !quantum.reg
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
      %extracted_1 = tensor.extract %arg0[] : tensor<f64>
      catalyst.list_push %extracted, %2 : <i64>
      catalyst.list_push %extracted, %2 : <i64>
      catalyst.list_push %extracted, %2 : <i64>
      catalyst.list_push %extracted, %2 : <i64>
      %5 = catalyst.list_pop %2 : <i64>
      %6 = quantum.extract %arg2[%5] : !quantum.reg -> !quantum.bit
      %out_qubits = quantum.custom "RX"(%cst_0) %6 : !quantum.bit
      %7 = quantum.insert %arg2[%5], %out_qubits : !quantum.reg, !quantum.bit
      %8 = catalyst.list_pop %2 : <i64>
      %9 = catalyst.list_pop %2 : <i64>
      %10 = quantum.extract %7[%9] : !quantum.reg -> !quantum.bit
      %11 = arith.negf %extracted_1 : f64
      %out_qubits_2 = quantum.multirz(%11) %10 : !quantum.bit
      %12 = quantum.insert %7[%9], %out_qubits_2 : !quantum.reg, !quantum.bit
      %13 = catalyst.list_pop %2 : <i64>
      %14 = catalyst.list_pop %2 : <i64>
      %15 = quantum.extract %12[%14] : !quantum.reg -> !quantum.bit
      %out_qubits_3 = quantum.custom "RX"(%cst) %15 : !quantum.bit
      %16 = quantum.insert %12[%14], %out_qubits_3 : !quantum.reg, !quantum.bit
      %17 = catalyst.list_pop %2 : <i64>
      catalyst.list_dealloc %0 : <f64>
      catalyst.list_dealloc %1 : <i64>
      catalyst.list_dealloc %2 : <i64>
      return %16 : !quantum.reg
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
      %extracted_0 = tensor.extract %arg0[] : tensor<f64>
      catalyst.list_push %extracted, %2 : <i64>
      catalyst.list_push %extracted, %2 : <i64>
      catalyst.list_push %extracted, %2 : <i64>
      catalyst.list_push %extracted, %2 : <i64>
      %5 = catalyst.list_pop %2 : <i64>
      %6 = quantum.extract %arg2[%5] : !quantum.reg -> !quantum.bit
      %out_qubits = quantum.custom "Hadamard"() %6 : !quantum.bit
      %7 = quantum.insert %arg2[%5], %out_qubits : !quantum.reg, !quantum.bit
      %8 = catalyst.list_pop %2 : <i64>
      %9 = catalyst.list_pop %2 : <i64>
      %10 = quantum.extract %7[%9] : !quantum.reg -> !quantum.bit
      %11 = arith.negf %extracted_0 : f64
      %out_qubits_1 = quantum.multirz(%11) %10 : !quantum.bit
      %12 = quantum.insert %7[%9], %out_qubits_1 : !quantum.reg, !quantum.bit
      %13 = catalyst.list_pop %2 : <i64>
      %14 = catalyst.list_pop %2 : <i64>
      %15 = quantum.extract %12[%14] : !quantum.reg -> !quantum.bit
      %out_qubits_2 = quantum.custom "Hadamard"() %15 : !quantum.bit
      %16 = quantum.insert %12[%14], %out_qubits_2 : !quantum.reg, !quantum.bit
      %17 = catalyst.list_pop %2 : <i64>
      catalyst.list_dealloc %0 : <f64>
      catalyst.list_dealloc %1 : <i64>
      catalyst.list_dealloc %2 : <i64>
      return %16 : !quantum.reg
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
      %7 = arith.negf %extracted_0 : f64
      %out_qubits = quantum.custom "RZ"(%7) %6 : !quantum.bit
      %8 = quantum.insert %arg2[%5], %out_qubits : !quantum.reg, !quantum.bit
      %9 = catalyst.list_pop %2 : <i64>
      catalyst.list_dealloc %0 : <f64>
      catalyst.list_dealloc %1 : <i64>
      catalyst.list_dealloc %2 : <i64>
      return %8 : !quantum.reg
    }
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