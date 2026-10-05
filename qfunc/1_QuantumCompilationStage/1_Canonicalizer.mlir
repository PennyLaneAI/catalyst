module @qfunc {
  func.func public @jit_qfunc() -> tensor<f64> attributes {llvm.emit_c_interface} {
    %0 = catalyst.launch_kernel @module_qfunc::@qfunc() : () -> tensor<f64>
    return %0 : tensor<f64>
  }
  module @module_qfunc {
    module attributes {transform.with_named_sequence} {
      transform.named_sequence @__transform_main(%arg0: !transform.op<"builtin.module">) {
        %0 = transform.apply_registered_pass "graph-decomposition" with options = {"bytecode-rules" = "/Users/haider.sajjad/catalyst/frontend/catalyst/utils/../resources/decomposition_rules_0fc9b120185e48ed2255e88b2839956bb9847e78.mlirbc", "gate-set" = {"Adjoint(CNOT)" = 1.000000e+00 : f64, "Adjoint(CY)" = 1.000000e+00 : f64, "Adjoint(CZ)" = 1.000000e+00 : f64, "Adjoint(Hadamard)" = 1.000000e+00 : f64, "Adjoint(ISWAP)" = 1.000000e+00 : f64, "Adjoint(PauliX)" = 1.000000e+00 : f64, "Adjoint(PauliY)" = 1.000000e+00 : f64, "Adjoint(PauliZ)" = 1.000000e+00 : f64, "Adjoint(S)" = 1.000000e+00 : f64, "Adjoint(SWAP)" = 1.000000e+00 : f64, "Adjoint(SX)" = 1.000000e+00 : f64, "Adjoint(T)" = 1.000000e+00 : f64, CNOT = 1.000000e+00 : f64, CY = 1.000000e+00 : f64, CZ = 1.000000e+00 : f64, GlobalPhase = 1.000000e+00 : f64, Hadamard = 1.000000e+00 : f64, ISWAP = 1.000000e+00 : f64, Identity = 1.000000e+00 : f64, MidMeasureMP = 1.000000e+00 : f64, PauliX = 1.000000e+00 : f64, PauliY = 1.000000e+00 : f64, PauliZ = 1.000000e+00 : f64, S = 1.000000e+00 : f64, SWAP = 1.000000e+00 : f64, SX = 1.000000e+00 : f64, T = 1.000000e+00 : f64}, "libQPD-path" = "/Users/haider.sajjad/catalyst/frontend/catalyst/utils/../../../mlir/build/lib", "libpython-path" = "/opt/homebrew/opt/python@3.12/Frameworks/Python.framework/Versions/3.12/Python"} to %arg0 : (!transform.op<"builtin.module">) -> !transform.op<"builtin.module">
        %1 = transform.apply_registered_pass "cancel-inverses" to %0 : (!transform.op<"builtin.module">) -> !transform.op<"builtin.module">
        %2 = transform.apply_registered_pass "merge-rotations" to %1 : (!transform.op<"builtin.module">) -> !transform.op<"builtin.module">
        transform.yield 
      }
      transform.named_sequence @__transform_device(%arg0: !transform.op<"builtin.module">) {
        %0 = transform.apply_registered_pass "empty" with options = {"key" = "verify_operations"} to %arg0 : (!transform.op<"builtin.module">) -> !transform.op<"builtin.module">
        %1 = transform.apply_registered_pass "empty" with options = {"key" = "validate_measurements"} to %0 : (!transform.op<"builtin.module">) -> !transform.op<"builtin.module">
        %2 = transform.apply_registered_pass "empty" with options = {"key" = "verify_no_state_variance_returns"} to %1 : (!transform.op<"builtin.module">) -> !transform.op<"builtin.module">
        transform.yield 
      }
    }
    func.func public @qfunc() -> tensor<f64> attributes {diff_method = "parameter-shift", llvm.linkage = #llvm.linkage<internal>, quantum.node} {
      %c0_i64 = arith.constant 0 : i64
      quantum.device shots(%c0_i64) ["/Users/haider.sajjad/catalyst/frontend/catalyst/utils/../../../runtime/build/lib/librtd_null_qubit.dylib", "NullQubit", "{'track_resources': False}"]
      %0 = qref.alloc( 3) : !qref.reg<3>
      %1 = qref.get %0[ 0] : !qref.reg<3> -> !qref.bit
      %2 = qref.get %0[ 1] : !qref.reg<3> -> !qref.bit
      %3 = qref.get %0[ 2] : !qref.reg<3> -> !qref.bit
      qref.custom "Toffoli"() %1, %2, %3 : !qref.bit, !qref.bit, !qref.bit
      %4 = qref.get %0[ 0] : !qref.reg<3> -> !qref.bit
      qref.custom "Hadamard"() %4 : !qref.bit
      %5 = qref.get %0[ 0] : !qref.reg<3> -> !qref.bit
      qref.custom "Hadamard"() %5 : !qref.bit
      %6 = qref.get %0[ 0] : !qref.reg<3> -> !qref.bit
      %7 = qref.namedobs %6[ PauliZ] : !quantum.obs
      %8 = quantum.expval %7 : f64
      %from_elements = tensor.from_elements %8 : tensor<f64>
      qref.dealloc %0 : !qref.reg<3>
      quantum.device_release
      return %from_elements : tensor<f64>
    }
    func.func private @"__builtin__toffoli_Toffoli{}{wires:3}{}"(%arg0: !qref.reg<?>, %arg1: tensor<3xi64>) attributes {frontend_name = "_toffoli", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"Adjoint(T){}{wires:1}{}" = 3 : i64, "CNOT{}{wires:2}{}" = 6 : i64, "Hadamard{}{wires:1}{}" = 2 : i64, "T{}{wires:1}{}" = 4 : i64}}, target_gate = "Toffoli{}{wires:3}{}"} {
      %0 = stablehlo.slice %arg1 [2:3] : (tensor<3xi64>) -> tensor<1xi64>
      %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
      %extracted = tensor.extract %1[] : tensor<i64>
      %2 = qref.get %arg0[%extracted] : !qref.reg<?>, i64 -> !qref.bit
      qref.custom "Hadamard"() %2 : !qref.bit
      %3 = stablehlo.slice %arg1 [1:2] : (tensor<3xi64>) -> tensor<1xi64>
      %4 = stablehlo.reshape %3 : (tensor<1xi64>) -> tensor<i64>
      %5 = stablehlo.slice %arg1 [2:3] : (tensor<3xi64>) -> tensor<1xi64>
      %6 = stablehlo.reshape %5 : (tensor<1xi64>) -> tensor<i64>
      %extracted_0 = tensor.extract %4[] : tensor<i64>
      %7 = qref.get %arg0[%extracted_0] : !qref.reg<?>, i64 -> !qref.bit
      %extracted_1 = tensor.extract %6[] : tensor<i64>
      %8 = qref.get %arg0[%extracted_1] : !qref.reg<?>, i64 -> !qref.bit
      qref.custom "CNOT"() %7, %8 : !qref.bit, !qref.bit
      %9 = stablehlo.slice %arg1 [2:3] : (tensor<3xi64>) -> tensor<1xi64>
      %10 = stablehlo.reshape %9 : (tensor<1xi64>) -> tensor<i64>
      %extracted_2 = tensor.extract %10[] : tensor<i64>
      %11 = qref.get %arg0[%extracted_2] : !qref.reg<?>, i64 -> !qref.bit
      qref.custom "T"() %11 adj : !qref.bit
      %12 = stablehlo.slice %arg1 [0:1] : (tensor<3xi64>) -> tensor<1xi64>
      %13 = stablehlo.reshape %12 : (tensor<1xi64>) -> tensor<i64>
      %14 = stablehlo.slice %arg1 [2:3] : (tensor<3xi64>) -> tensor<1xi64>
      %15 = stablehlo.reshape %14 : (tensor<1xi64>) -> tensor<i64>
      %extracted_3 = tensor.extract %13[] : tensor<i64>
      %16 = qref.get %arg0[%extracted_3] : !qref.reg<?>, i64 -> !qref.bit
      %extracted_4 = tensor.extract %15[] : tensor<i64>
      %17 = qref.get %arg0[%extracted_4] : !qref.reg<?>, i64 -> !qref.bit
      qref.custom "CNOT"() %16, %17 : !qref.bit, !qref.bit
      %18 = stablehlo.slice %arg1 [2:3] : (tensor<3xi64>) -> tensor<1xi64>
      %19 = stablehlo.reshape %18 : (tensor<1xi64>) -> tensor<i64>
      %extracted_5 = tensor.extract %19[] : tensor<i64>
      %20 = qref.get %arg0[%extracted_5] : !qref.reg<?>, i64 -> !qref.bit
      qref.custom "T"() %20 : !qref.bit
      %21 = stablehlo.slice %arg1 [1:2] : (tensor<3xi64>) -> tensor<1xi64>
      %22 = stablehlo.reshape %21 : (tensor<1xi64>) -> tensor<i64>
      %23 = stablehlo.slice %arg1 [2:3] : (tensor<3xi64>) -> tensor<1xi64>
      %24 = stablehlo.reshape %23 : (tensor<1xi64>) -> tensor<i64>
      %extracted_6 = tensor.extract %22[] : tensor<i64>
      %25 = qref.get %arg0[%extracted_6] : !qref.reg<?>, i64 -> !qref.bit
      %extracted_7 = tensor.extract %24[] : tensor<i64>
      %26 = qref.get %arg0[%extracted_7] : !qref.reg<?>, i64 -> !qref.bit
      qref.custom "CNOT"() %25, %26 : !qref.bit, !qref.bit
      %27 = stablehlo.slice %arg1 [2:3] : (tensor<3xi64>) -> tensor<1xi64>
      %28 = stablehlo.reshape %27 : (tensor<1xi64>) -> tensor<i64>
      %extracted_8 = tensor.extract %28[] : tensor<i64>
      %29 = qref.get %arg0[%extracted_8] : !qref.reg<?>, i64 -> !qref.bit
      qref.custom "T"() %29 adj : !qref.bit
      %30 = stablehlo.slice %arg1 [0:1] : (tensor<3xi64>) -> tensor<1xi64>
      %31 = stablehlo.reshape %30 : (tensor<1xi64>) -> tensor<i64>
      %32 = stablehlo.slice %arg1 [2:3] : (tensor<3xi64>) -> tensor<1xi64>
      %33 = stablehlo.reshape %32 : (tensor<1xi64>) -> tensor<i64>
      %extracted_9 = tensor.extract %31[] : tensor<i64>
      %34 = qref.get %arg0[%extracted_9] : !qref.reg<?>, i64 -> !qref.bit
      %extracted_10 = tensor.extract %33[] : tensor<i64>
      %35 = qref.get %arg0[%extracted_10] : !qref.reg<?>, i64 -> !qref.bit
      qref.custom "CNOT"() %34, %35 : !qref.bit, !qref.bit
      %36 = stablehlo.slice %arg1 [2:3] : (tensor<3xi64>) -> tensor<1xi64>
      %37 = stablehlo.reshape %36 : (tensor<1xi64>) -> tensor<i64>
      %extracted_11 = tensor.extract %37[] : tensor<i64>
      %38 = qref.get %arg0[%extracted_11] : !qref.reg<?>, i64 -> !qref.bit
      qref.custom "T"() %38 : !qref.bit
      %39 = stablehlo.slice %arg1 [1:2] : (tensor<3xi64>) -> tensor<1xi64>
      %40 = stablehlo.reshape %39 : (tensor<1xi64>) -> tensor<i64>
      %extracted_12 = tensor.extract %40[] : tensor<i64>
      %41 = qref.get %arg0[%extracted_12] : !qref.reg<?>, i64 -> !qref.bit
      qref.custom "T"() %41 : !qref.bit
      %42 = stablehlo.slice %arg1 [0:1] : (tensor<3xi64>) -> tensor<1xi64>
      %43 = stablehlo.reshape %42 : (tensor<1xi64>) -> tensor<i64>
      %44 = stablehlo.slice %arg1 [1:2] : (tensor<3xi64>) -> tensor<1xi64>
      %45 = stablehlo.reshape %44 : (tensor<1xi64>) -> tensor<i64>
      %extracted_13 = tensor.extract %43[] : tensor<i64>
      %46 = qref.get %arg0[%extracted_13] : !qref.reg<?>, i64 -> !qref.bit
      %extracted_14 = tensor.extract %45[] : tensor<i64>
      %47 = qref.get %arg0[%extracted_14] : !qref.reg<?>, i64 -> !qref.bit
      qref.custom "CNOT"() %46, %47 : !qref.bit, !qref.bit
      %48 = stablehlo.slice %arg1 [2:3] : (tensor<3xi64>) -> tensor<1xi64>
      %49 = stablehlo.reshape %48 : (tensor<1xi64>) -> tensor<i64>
      %extracted_15 = tensor.extract %49[] : tensor<i64>
      %50 = qref.get %arg0[%extracted_15] : !qref.reg<?>, i64 -> !qref.bit
      qref.custom "Hadamard"() %50 : !qref.bit
      %51 = stablehlo.slice %arg1 [0:1] : (tensor<3xi64>) -> tensor<1xi64>
      %52 = stablehlo.reshape %51 : (tensor<1xi64>) -> tensor<i64>
      %extracted_16 = tensor.extract %52[] : tensor<i64>
      %53 = qref.get %arg0[%extracted_16] : !qref.reg<?>, i64 -> !qref.bit
      qref.custom "T"() %53 : !qref.bit
      %54 = stablehlo.slice %arg1 [1:2] : (tensor<3xi64>) -> tensor<1xi64>
      %55 = stablehlo.reshape %54 : (tensor<1xi64>) -> tensor<i64>
      %extracted_17 = tensor.extract %55[] : tensor<i64>
      %56 = qref.get %arg0[%extracted_17] : !qref.reg<?>, i64 -> !qref.bit
      qref.custom "T"() %56 adj : !qref.bit
      %57 = stablehlo.slice %arg1 [0:1] : (tensor<3xi64>) -> tensor<1xi64>
      %58 = stablehlo.reshape %57 : (tensor<1xi64>) -> tensor<i64>
      %59 = stablehlo.slice %arg1 [1:2] : (tensor<3xi64>) -> tensor<1xi64>
      %60 = stablehlo.reshape %59 : (tensor<1xi64>) -> tensor<i64>
      %extracted_18 = tensor.extract %58[] : tensor<i64>
      %61 = qref.get %arg0[%extracted_18] : !qref.reg<?>, i64 -> !qref.bit
      %extracted_19 = tensor.extract %60[] : tensor<i64>
      %62 = qref.get %arg0[%extracted_19] : !qref.reg<?>, i64 -> !qref.bit
      qref.custom "CNOT"() %61, %62 : !qref.bit, !qref.bit
      return
    }
    func.func private @"__builtin__toffoli_to_ppr_Toffoli{}{wires:3}{}"(%arg0: !qref.reg<?>, %arg1: tensor<3xi64>) attributes {frontend_name = "_toffoli_to_ppr", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"GlobalPhase{phi:[f64]}{}{}" = 1 : i64, "PPR{}{wires:1}{angle_denominator = 8 : i64, pauli_word = \22X\22}" = 1 : i64, "PPR{}{wires:1}{angle_denominator = 8 : i64, pauli_word = \22Z\22}" = 2 : i64, "PPR{}{wires:2}{angle_denominator = -8 : si64, pauli_word = \22ZX\22}" = 2 : i64, "PPR{}{wires:2}{angle_denominator = -8 : si64, pauli_word = \22ZZ\22}" = 1 : i64, "PPR{}{wires:3}{angle_denominator = 8 : i64, pauli_word = \22ZZX\22}" = 1 : i64}}, target_gate = "Toffoli{}{wires:3}{}"} {
      %cst = arith.constant -0.39269908169872414 : f64
      %0 = stablehlo.slice %arg1 [0:2] : (tensor<3xi64>) -> tensor<2xi64>
      %1 = stablehlo.slice %0 [0:1] : (tensor<2xi64>) -> tensor<1xi64>
      %2 = stablehlo.reshape %1 : (tensor<1xi64>) -> tensor<i64>
      %3 = stablehlo.slice %0 [1:2] : (tensor<2xi64>) -> tensor<1xi64>
      %4 = stablehlo.reshape %3 : (tensor<1xi64>) -> tensor<i64>
      %extracted = tensor.extract %2[] : tensor<i64>
      %5 = qref.get %arg0[%extracted] : !qref.reg<?>, i64 -> !qref.bit
      %extracted_0 = tensor.extract %4[] : tensor<i64>
      %6 = qref.get %arg0[%extracted_0] : !qref.reg<?>, i64 -> !qref.bit
      qref.operator "PPR"() qubits(%5, %6)
        static_data = {angle_denominator = -8 : si64, pauli_word = "ZZ"}
        qubit_map = {wires = [0, 1]}
      %7 = stablehlo.slice %arg1 [0:1] : (tensor<3xi64>) -> tensor<1xi64>
      %8 = stablehlo.reshape %7 : (tensor<1xi64>) -> tensor<i64>
      %9 = stablehlo.slice %arg1 [2:3] : (tensor<3xi64>) -> tensor<1xi64>
      %10 = stablehlo.reshape %9 : (tensor<1xi64>) -> tensor<i64>
      %extracted_1 = tensor.extract %8[] : tensor<i64>
      %11 = qref.get %arg0[%extracted_1] : !qref.reg<?>, i64 -> !qref.bit
      %extracted_2 = tensor.extract %10[] : tensor<i64>
      %12 = qref.get %arg0[%extracted_2] : !qref.reg<?>, i64 -> !qref.bit
      qref.operator "PPR"() qubits(%11, %12)
        static_data = {angle_denominator = -8 : si64, pauli_word = "ZX"}
        qubit_map = {wires = [0, 1]}
      %13 = stablehlo.slice %arg1 [1:3] : (tensor<3xi64>) -> tensor<2xi64>
      %14 = stablehlo.slice %13 [0:1] : (tensor<2xi64>) -> tensor<1xi64>
      %15 = stablehlo.reshape %14 : (tensor<1xi64>) -> tensor<i64>
      %16 = stablehlo.slice %13 [1:2] : (tensor<2xi64>) -> tensor<1xi64>
      %17 = stablehlo.reshape %16 : (tensor<1xi64>) -> tensor<i64>
      %extracted_3 = tensor.extract %15[] : tensor<i64>
      %18 = qref.get %arg0[%extracted_3] : !qref.reg<?>, i64 -> !qref.bit
      %extracted_4 = tensor.extract %17[] : tensor<i64>
      %19 = qref.get %arg0[%extracted_4] : !qref.reg<?>, i64 -> !qref.bit
      qref.operator "PPR"() qubits(%18, %19)
        static_data = {angle_denominator = -8 : si64, pauli_word = "ZX"}
        qubit_map = {wires = [0, 1]}
      %20 = stablehlo.slice %arg1 [0:1] : (tensor<3xi64>) -> tensor<1xi64>
      %21 = stablehlo.reshape %20 : (tensor<1xi64>) -> tensor<i64>
      %22 = stablehlo.slice %arg1 [1:2] : (tensor<3xi64>) -> tensor<1xi64>
      %23 = stablehlo.reshape %22 : (tensor<1xi64>) -> tensor<i64>
      %24 = stablehlo.slice %arg1 [2:3] : (tensor<3xi64>) -> tensor<1xi64>
      %25 = stablehlo.reshape %24 : (tensor<1xi64>) -> tensor<i64>
      %extracted_5 = tensor.extract %21[] : tensor<i64>
      %26 = qref.get %arg0[%extracted_5] : !qref.reg<?>, i64 -> !qref.bit
      %extracted_6 = tensor.extract %23[] : tensor<i64>
      %27 = qref.get %arg0[%extracted_6] : !qref.reg<?>, i64 -> !qref.bit
      %extracted_7 = tensor.extract %25[] : tensor<i64>
      %28 = qref.get %arg0[%extracted_7] : !qref.reg<?>, i64 -> !qref.bit
      qref.operator "PPR"() qubits(%26, %27, %28)
        static_data = {angle_denominator = 8 : i64, pauli_word = "ZZX"}
        qubit_map = {wires = [0, 1, 2]}
      %29 = stablehlo.slice %arg1 [2:3] : (tensor<3xi64>) -> tensor<1xi64>
      %30 = stablehlo.reshape %29 : (tensor<1xi64>) -> tensor<i64>
      %extracted_8 = tensor.extract %30[] : tensor<i64>
      %31 = qref.get %arg0[%extracted_8] : !qref.reg<?>, i64 -> !qref.bit
      qref.operator "PPR"() qubits(%31)
        static_data = {angle_denominator = 8 : i64, pauli_word = "X"}
        qubit_map = {wires = [0]}
      %32 = stablehlo.slice %arg1 [1:2] : (tensor<3xi64>) -> tensor<1xi64>
      %33 = stablehlo.reshape %32 : (tensor<1xi64>) -> tensor<i64>
      %extracted_9 = tensor.extract %33[] : tensor<i64>
      %34 = qref.get %arg0[%extracted_9] : !qref.reg<?>, i64 -> !qref.bit
      qref.operator "PPR"() qubits(%34)
        static_data = {angle_denominator = 8 : i64, pauli_word = "Z"}
        qubit_map = {wires = [0]}
      %35 = stablehlo.slice %arg1 [0:1] : (tensor<3xi64>) -> tensor<1xi64>
      %36 = stablehlo.reshape %35 : (tensor<1xi64>) -> tensor<i64>
      %extracted_10 = tensor.extract %36[] : tensor<i64>
      %37 = qref.get %arg0[%extracted_10] : !qref.reg<?>, i64 -> !qref.bit
      qref.operator "PPR"() qubits(%37)
        static_data = {angle_denominator = 8 : i64, pauli_word = "Z"}
        qubit_map = {wires = [0]}
      qref.gphase(%cst)
      return
    }
    func.func private @"__builtin__hadamard_to_rz_rx_Hadamard{}{wires:1}{}"(%arg0: !qref.reg<?>, %arg1: tensor<1xi64>) attributes {frontend_name = "_hadamard_to_rz_rx", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"GlobalPhase{phi:[f64]}{}{}" = 1 : i64, "RX{0:[f64]}{wires:1}{}" = 1 : i64, "RZ{0:[f64]}{wires:1}{}" = 2 : i64}}, target_gate = "Hadamard{}{wires:1}{}"} {
      %cst = arith.constant -1.5707963267948966 : f64
      %cst_0 = arith.constant 1.5707963267948966 : f64
      %0 = stablehlo.slice %arg1 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
      %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
      %extracted = tensor.extract %1[] : tensor<i64>
      %2 = qref.get %arg0[%extracted] : !qref.reg<?>, i64 -> !qref.bit
      qref.custom "RZ"(%cst_0) %2 : !qref.bit
      %3 = stablehlo.slice %arg1 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
      %4 = stablehlo.reshape %3 : (tensor<1xi64>) -> tensor<i64>
      %extracted_1 = tensor.extract %4[] : tensor<i64>
      %5 = qref.get %arg0[%extracted_1] : !qref.reg<?>, i64 -> !qref.bit
      qref.custom "RX"(%cst_0) %5 : !qref.bit
      %6 = stablehlo.slice %arg1 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
      %7 = stablehlo.reshape %6 : (tensor<1xi64>) -> tensor<i64>
      %extracted_2 = tensor.extract %7[] : tensor<i64>
      %8 = qref.get %arg0[%extracted_2] : !qref.reg<?>, i64 -> !qref.bit
      qref.custom "RZ"(%cst_0) %8 : !qref.bit
      qref.gphase(%cst)
      return
    }
    func.func private @"__builtin__hadamard_to_rz_ry_Hadamard{}{wires:1}{}"(%arg0: !qref.reg<?>, %arg1: tensor<1xi64>) attributes {frontend_name = "_hadamard_to_rz_ry", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"GlobalPhase{phi:[f64]}{}{}" = 1 : i64, "RY{0:[f64]}{wires:1}{}" = 1 : i64, "RZ{0:[f64]}{wires:1}{}" = 1 : i64}}, target_gate = "Hadamard{}{wires:1}{}"} {
      %cst = arith.constant -1.5707963267948966 : f64
      %cst_0 = arith.constant 1.5707963267948966 : f64
      %cst_1 = arith.constant 3.1415926535897931 : f64
      %0 = stablehlo.slice %arg1 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
      %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
      %extracted = tensor.extract %1[] : tensor<i64>
      %2 = qref.get %arg0[%extracted] : !qref.reg<?>, i64 -> !qref.bit
      qref.custom "RZ"(%cst_1) %2 : !qref.bit
      %3 = stablehlo.slice %arg1 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
      %4 = stablehlo.reshape %3 : (tensor<1xi64>) -> tensor<i64>
      %extracted_2 = tensor.extract %4[] : tensor<i64>
      %5 = qref.get %arg0[%extracted_2] : !qref.reg<?>, i64 -> !qref.bit
      qref.custom "RY"(%cst_0) %5 : !qref.bit
      qref.gphase(%cst)
      return
    }
    func.func private @"__builtin__cnot_to_cz_h_CNOT{}{wires:2}{}"(%arg0: !qref.reg<?>, %arg1: tensor<2xi64>) attributes {frontend_name = "_cnot_to_cz_h", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"CZ{}{wires:2}{}" = 1 : i64, "Hadamard{}{wires:1}{}" = 2 : i64}}, target_gate = "CNOT{}{wires:2}{}"} {
      %0 = stablehlo.slice %arg1 [1:2] : (tensor<2xi64>) -> tensor<1xi64>
      %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
      %extracted = tensor.extract %1[] : tensor<i64>
      %2 = qref.get %arg0[%extracted] : !qref.reg<?>, i64 -> !qref.bit
      qref.custom "Hadamard"() %2 : !qref.bit
      %3 = stablehlo.slice %arg1 [0:1] : (tensor<2xi64>) -> tensor<1xi64>
      %4 = stablehlo.reshape %3 : (tensor<1xi64>) -> tensor<i64>
      %5 = stablehlo.slice %arg1 [1:2] : (tensor<2xi64>) -> tensor<1xi64>
      %6 = stablehlo.reshape %5 : (tensor<1xi64>) -> tensor<i64>
      %extracted_0 = tensor.extract %4[] : tensor<i64>
      %7 = qref.get %arg0[%extracted_0] : !qref.reg<?>, i64 -> !qref.bit
      %extracted_1 = tensor.extract %6[] : tensor<i64>
      %8 = qref.get %arg0[%extracted_1] : !qref.reg<?>, i64 -> !qref.bit
      qref.custom "CZ"() %7, %8 : !qref.bit, !qref.bit
      %9 = stablehlo.slice %arg1 [1:2] : (tensor<2xi64>) -> tensor<1xi64>
      %10 = stablehlo.reshape %9 : (tensor<1xi64>) -> tensor<i64>
      %extracted_2 = tensor.extract %10[] : tensor<i64>
      %11 = qref.get %arg0[%extracted_2] : !qref.reg<?>, i64 -> !qref.bit
      qref.custom "Hadamard"() %11 : !qref.bit
      return
    }
    func.func private @"__builtin__cnot_to_ppr_CNOT{}{wires:2}{}"(%arg0: !qref.reg<?>, %arg1: tensor<2xi64>) attributes {frontend_name = "_cnot_to_ppr", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"GlobalPhase{phi:[f64]}{}{}" = 1 : i64, "PPR{}{wires:1}{angle_denominator = -4 : si64, pauli_word = \22X\22}" = 1 : i64, "PPR{}{wires:1}{angle_denominator = -4 : si64, pauli_word = \22Z\22}" = 1 : i64, "PPR{}{wires:2}{angle_denominator = 4 : i64, pauli_word = \22ZX\22}" = 1 : i64}}, target_gate = "CNOT{}{wires:2}{}"} {
      %cst = arith.constant 0.78539816339744828 : f64
      %0 = stablehlo.slice %arg1 [0:1] : (tensor<2xi64>) -> tensor<1xi64>
      %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
      %extracted = tensor.extract %1[] : tensor<i64>
      %2 = qref.get %arg0[%extracted] : !qref.reg<?>, i64 -> !qref.bit
      qref.operator "PPR"() qubits(%2)
        static_data = {angle_denominator = -4 : si64, pauli_word = "Z"}
        qubit_map = {wires = [0]}
      %3 = stablehlo.slice %arg1 [1:2] : (tensor<2xi64>) -> tensor<1xi64>
      %4 = stablehlo.reshape %3 : (tensor<1xi64>) -> tensor<i64>
      %extracted_0 = tensor.extract %4[] : tensor<i64>
      %5 = qref.get %arg0[%extracted_0] : !qref.reg<?>, i64 -> !qref.bit
      qref.operator "PPR"() qubits(%5)
        static_data = {angle_denominator = -4 : si64, pauli_word = "X"}
        qubit_map = {wires = [0]}
      %6 = stablehlo.slice %arg1 [0:1] : (tensor<2xi64>) -> tensor<1xi64>
      %7 = stablehlo.reshape %6 : (tensor<1xi64>) -> tensor<i64>
      %8 = stablehlo.slice %arg1 [1:2] : (tensor<2xi64>) -> tensor<1xi64>
      %9 = stablehlo.reshape %8 : (tensor<1xi64>) -> tensor<i64>
      %extracted_1 = tensor.extract %7[] : tensor<i64>
      %10 = qref.get %arg0[%extracted_1] : !qref.reg<?>, i64 -> !qref.bit
      %extracted_2 = tensor.extract %9[] : tensor<i64>
      %11 = qref.get %arg0[%extracted_2] : !qref.reg<?>, i64 -> !qref.bit
      qref.operator "PPR"() qubits(%10, %11)
        static_data = {angle_denominator = 4 : i64, pauli_word = "ZX"}
        qubit_map = {wires = [0, 1]}
      qref.gphase(%cst)
      return
    }
    func.func private @"__builtin__t_phaseshift_T{}{wires:1}{}"(%arg0: !qref.reg<?>, %arg1: tensor<1xi64>) attributes {frontend_name = "_t_phaseshift", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"PhaseShift{0:[f64]}{wires:1}{}" = 1 : i64}}, target_gate = "T{}{wires:1}{}"} {
      %cst = arith.constant 0.78539816339744828 : f64
      %0 = stablehlo.slice %arg1 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
      %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
      %extracted = tensor.extract %1[] : tensor<i64>
      %2 = qref.get %arg0[%extracted] : !qref.reg<?>, i64 -> !qref.bit
      qref.custom "PhaseShift"(%cst) %2 : !qref.bit
      return
    }
    func.func private @"__builtin__t_phaseshift_Adjoint(T){}{wires:1}{}"(%arg0: !qref.reg<?>, %arg1: tensor<1xi64>) attributes {frontend_name = "_t_phaseshift", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"Adjoint(PhaseShift){0:[f64]}{wires:1}{}" = 1 : i64}}, target_gate = "Adjoint(T){}{wires:1}{}"} {
      %cst = arith.constant 0.78539816339744828 : f64
      qref.adjoint {
        %0 = stablehlo.slice %arg1 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
        %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
        %extracted = tensor.extract %1[] : tensor<i64>
        %2 = qref.get %arg0[%extracted] : !qref.reg<?>, i64 -> !qref.bit
        qref.custom "PhaseShift"(%cst) %2 : !qref.bit
      }
      return
    }
    func.func private @"__builtin__ppr_to_paulirot_PPR{}{wires:2}{angle_denominator = -8 : si64, pauli_word = \22ZZ\22}"(%arg0: !qref.reg<?>, %arg1: tensor<2xi64>) attributes {frontend_name = "_ppr_to_paulirot", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"PauliRot{theta:[f64]}{wires:2}{pauli_word = \22ZZ\22}" = 1 : i64}}, target_gate = "PPR{}{wires:2}{angle_denominator = -8 : si64, pauli_word = \22ZZ\22}"} {
      %cst = arith.constant -0.78539816339744828 : f64
      %0 = stablehlo.slice %arg1 [0:1] : (tensor<2xi64>) -> tensor<1xi64>
      %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
      %2 = stablehlo.slice %arg1 [1:2] : (tensor<2xi64>) -> tensor<1xi64>
      %3 = stablehlo.reshape %2 : (tensor<1xi64>) -> tensor<i64>
      %extracted = tensor.extract %1[] : tensor<i64>
      %4 = qref.get %arg0[%extracted] : !qref.reg<?>, i64 -> !qref.bit
      %extracted_0 = tensor.extract %3[] : tensor<i64>
      %5 = qref.get %arg0[%extracted_0] : !qref.reg<?>, i64 -> !qref.bit
      qref.paulirot ["Z", "Z"](%cst) %4, %5 : !qref.bit, !qref.bit
      return
    }
    func.func private @"__builtin__ppr_to_paulirot_PPR{}{wires:2}{angle_denominator = -8 : si64, pauli_word = \22ZX\22}"(%arg0: !qref.reg<?>, %arg1: tensor<2xi64>) attributes {frontend_name = "_ppr_to_paulirot", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"PauliRot{theta:[f64]}{wires:2}{pauli_word = \22ZX\22}" = 1 : i64}}, target_gate = "PPR{}{wires:2}{angle_denominator = -8 : si64, pauli_word = \22ZX\22}"} {
      %cst = arith.constant -0.78539816339744828 : f64
      %0 = stablehlo.slice %arg1 [0:1] : (tensor<2xi64>) -> tensor<1xi64>
      %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
      %2 = stablehlo.slice %arg1 [1:2] : (tensor<2xi64>) -> tensor<1xi64>
      %3 = stablehlo.reshape %2 : (tensor<1xi64>) -> tensor<i64>
      %extracted = tensor.extract %1[] : tensor<i64>
      %4 = qref.get %arg0[%extracted] : !qref.reg<?>, i64 -> !qref.bit
      %extracted_0 = tensor.extract %3[] : tensor<i64>
      %5 = qref.get %arg0[%extracted_0] : !qref.reg<?>, i64 -> !qref.bit
      qref.paulirot ["Z", "X"](%cst) %4, %5 : !qref.bit, !qref.bit
      return
    }
    func.func private @"__builtin__ppr_to_paulirot_PPR{}{wires:3}{angle_denominator = 8 : i64, pauli_word = \22ZZX\22}"(%arg0: !qref.reg<?>, %arg1: tensor<3xi64>) attributes {frontend_name = "_ppr_to_paulirot", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"PauliRot{theta:[f64]}{wires:3}{pauli_word = \22ZZX\22}" = 1 : i64}}, target_gate = "PPR{}{wires:3}{angle_denominator = 8 : i64, pauli_word = \22ZZX\22}"} {
      %cst = arith.constant 0.78539816339744828 : f64
      %0 = stablehlo.slice %arg1 [0:1] : (tensor<3xi64>) -> tensor<1xi64>
      %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
      %2 = stablehlo.slice %arg1 [1:2] : (tensor<3xi64>) -> tensor<1xi64>
      %3 = stablehlo.reshape %2 : (tensor<1xi64>) -> tensor<i64>
      %4 = stablehlo.slice %arg1 [2:3] : (tensor<3xi64>) -> tensor<1xi64>
      %5 = stablehlo.reshape %4 : (tensor<1xi64>) -> tensor<i64>
      %extracted = tensor.extract %1[] : tensor<i64>
      %6 = qref.get %arg0[%extracted] : !qref.reg<?>, i64 -> !qref.bit
      %extracted_0 = tensor.extract %3[] : tensor<i64>
      %7 = qref.get %arg0[%extracted_0] : !qref.reg<?>, i64 -> !qref.bit
      %extracted_1 = tensor.extract %5[] : tensor<i64>
      %8 = qref.get %arg0[%extracted_1] : !qref.reg<?>, i64 -> !qref.bit
      qref.paulirot ["Z", "Z", "X"](%cst) %6, %7, %8 : !qref.bit, !qref.bit, !qref.bit
      return
    }
    func.func private @"__builtin__ppr_to_paulirot_PPR{}{wires:1}{angle_denominator = 8 : i64, pauli_word = \22X\22}"(%arg0: !qref.reg<?>, %arg1: tensor<1xi64>) attributes {frontend_name = "_ppr_to_paulirot", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"PauliRot{theta:[f64]}{wires:1}{pauli_word = \22X\22}" = 1 : i64}}, target_gate = "PPR{}{wires:1}{angle_denominator = 8 : i64, pauli_word = \22X\22}"} {
      %cst = arith.constant 0.78539816339744828 : f64
      %0 = stablehlo.slice %arg1 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
      %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
      %extracted = tensor.extract %1[] : tensor<i64>
      %2 = qref.get %arg0[%extracted] : !qref.reg<?>, i64 -> !qref.bit
      qref.paulirot ["X"](%cst) %2 : !qref.bit
      return
    }
    func.func private @"__builtin__ppr_to_paulirot_PPR{}{wires:1}{angle_denominator = 8 : i64, pauli_word = \22Z\22}"(%arg0: !qref.reg<?>, %arg1: tensor<1xi64>) attributes {frontend_name = "_ppr_to_paulirot", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"PauliRot{theta:[f64]}{wires:1}{pauli_word = \22Z\22}" = 1 : i64}}, target_gate = "PPR{}{wires:1}{angle_denominator = 8 : i64, pauli_word = \22Z\22}"} {
      %cst = arith.constant 0.78539816339744828 : f64
      %0 = stablehlo.slice %arg1 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
      %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
      %extracted = tensor.extract %1[] : tensor<i64>
      %2 = qref.get %arg0[%extracted] : !qref.reg<?>, i64 -> !qref.bit
      qref.paulirot ["Z"](%cst) %2 : !qref.bit
      return
    }
    func.func private @"__builtin__rz_to_ps_RZ{0:[f64]}{wires:1}{}"(%arg0: !qref.reg<?>, %arg1: tensor<f64>, %arg2: tensor<1xi64>) attributes {frontend_name = "_rz_to_ps", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"GlobalPhase{phi:[f64]}{}{}" = 1 : i64, "PhaseShift{0:[f64]}{wires:1}{}" = 1 : i64}}, target_gate = "RZ{0:[f64]}{wires:1}{}"} {
      %cst = stablehlo.constant dense<2.000000e+00> : tensor<f64>
      %0 = stablehlo.slice %arg2 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
      %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
      %extracted = tensor.extract %1[] : tensor<i64>
      %2 = qref.get %arg0[%extracted] : !qref.reg<?>, i64 -> !qref.bit
      %extracted_0 = tensor.extract %arg1[] : tensor<f64>
      qref.custom "PhaseShift"(%extracted_0) %2 : !qref.bit
      %3 = stablehlo.divide %arg1, %cst : tensor<f64>
      %extracted_1 = tensor.extract %3[] : tensor<f64>
      qref.gphase(%extracted_1)
      return
    }
    func.func private @"__builtin__rz_to_rot_RZ{0:[f64]}{wires:1}{}"(%arg0: !qref.reg<?>, %arg1: tensor<f64>, %arg2: tensor<1xi64>) attributes {frontend_name = "_rz_to_rot", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"Rot{0:[f64],1:[f64],2:[f64]}{wires:1}{}" = 1 : i64}}, target_gate = "RZ{0:[f64]}{wires:1}{}"} {
      %c = stablehlo.constant dense<0> : tensor<i64>
      %0 = stablehlo.slice %arg2 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
      %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
      %extracted = tensor.extract %1[] : tensor<i64>
      %2 = qref.get %arg0[%extracted] : !qref.reg<?>, i64 -> !qref.bit
      %3 = stablehlo.convert %c : (tensor<i64>) -> tensor<f64>
      %extracted_0 = tensor.extract %3[] : tensor<f64>
      %4 = stablehlo.convert %c : (tensor<i64>) -> tensor<f64>
      %extracted_1 = tensor.extract %4[] : tensor<f64>
      %extracted_2 = tensor.extract %arg1[] : tensor<f64>
      qref.custom "Rot"(%extracted_0, %extracted_1, %extracted_2) %2 : !qref.bit
      return
    }
    func.func private @"__builtin__rz_to_ry_rx_RZ{0:[f64]}{wires:1}{}"(%arg0: !qref.reg<?>, %arg1: tensor<f64>, %arg2: tensor<1xi64>) attributes {frontend_name = "_rz_to_ry_rx", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"RX{0:[f64]}{wires:1}{}" = 1 : i64, "RY{0:[f64]}{wires:1}{}" = 2 : i64}}, target_gate = "RZ{0:[f64]}{wires:1}{}"} {
      %cst = arith.constant -1.5707963267948966 : f64
      %cst_0 = arith.constant 1.5707963267948966 : f64
      %0 = stablehlo.slice %arg2 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
      %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
      %extracted = tensor.extract %1[] : tensor<i64>
      %2 = qref.get %arg0[%extracted] : !qref.reg<?>, i64 -> !qref.bit
      qref.custom "RY"(%cst_0) %2 : !qref.bit
      %3 = stablehlo.slice %arg2 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
      %4 = stablehlo.reshape %3 : (tensor<1xi64>) -> tensor<i64>
      %extracted_1 = tensor.extract %4[] : tensor<i64>
      %5 = qref.get %arg0[%extracted_1] : !qref.reg<?>, i64 -> !qref.bit
      %extracted_2 = tensor.extract %arg1[] : tensor<f64>
      qref.custom "RX"(%extracted_2) %5 : !qref.bit
      %6 = stablehlo.slice %arg2 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
      %7 = stablehlo.reshape %6 : (tensor<1xi64>) -> tensor<i64>
      %extracted_3 = tensor.extract %7[] : tensor<i64>
      %8 = qref.get %arg0[%extracted_3] : !qref.reg<?>, i64 -> !qref.bit
      qref.custom "RY"(%cst) %8 : !qref.bit
      return
    }
    func.func private @"__builtin__rz_to_ppr_RZ{0:[f64]}{wires:1}{}"(%arg0: !qref.reg<?>, %arg1: tensor<f64>, %arg2: tensor<1xi64>) attributes {frontend_name = "_rz_to_ppr", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"PauliRot{theta:[f64]}{wires:1}{pauli_word = \22Z\22}" = 1 : i64}}, target_gate = "RZ{0:[f64]}{wires:1}{}"} {
      %0 = stablehlo.slice %arg2 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
      %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
      %extracted = tensor.extract %1[] : tensor<i64>
      %2 = qref.get %arg0[%extracted] : !qref.reg<?>, i64 -> !qref.bit
      %extracted_0 = tensor.extract %arg1[] : tensor<f64>
      qref.paulirot ["Z"](%extracted_0) %2 : !qref.bit
      return
    }
    func.func private @"__builtin__rz_to_rx_cliff_RZ{0:[f64]}{wires:1}{}"(%arg0: !qref.reg<?>, %arg1: tensor<f64>, %arg2: tensor<1xi64>) attributes {frontend_name = "_rz_to_rx_cliff", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"Hadamard{}{wires:1}{}" = 2 : i64, "RX{0:[f64]}{wires:1}{}" = 1 : i64}}, target_gate = "RZ{0:[f64]}{wires:1}{}"} {
      %0 = stablehlo.slice %arg2 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
      %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
      %2 = stablehlo.slice %arg2 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
      %3 = stablehlo.reshape %2 : (tensor<1xi64>) -> tensor<i64>
      %4 = stablehlo.slice %arg2 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
      %5 = stablehlo.reshape %4 : (tensor<1xi64>) -> tensor<i64>
      %extracted = tensor.extract %1[] : tensor<i64>
      %6 = qref.get %arg0[%extracted] : !qref.reg<?>, i64 -> !qref.bit
      qref.custom "Hadamard"() %6 : !qref.bit
      %extracted_0 = tensor.extract %3[] : tensor<i64>
      %7 = qref.get %arg0[%extracted_0] : !qref.reg<?>, i64 -> !qref.bit
      %extracted_1 = tensor.extract %arg1[] : tensor<f64>
      qref.custom "RX"(%extracted_1) %7 : !qref.bit
      %extracted_2 = tensor.extract %5[] : tensor<i64>
      %8 = qref.get %arg0[%extracted_2] : !qref.reg<?>, i64 -> !qref.bit
      qref.custom "Hadamard"() %8 : !qref.bit
      return
    }
    func.func private @"__builtin__rz_to_ry_cliff_RZ{0:[f64]}{wires:1}{}"(%arg0: !qref.reg<?>, %arg1: tensor<f64>, %arg2: tensor<1xi64>) attributes {frontend_name = "_rz_to_ry_cliff", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"Adjoint(S){}{wires:1}{}" = 1 : i64, "Hadamard{}{wires:1}{}" = 2 : i64, "RY{0:[f64]}{wires:1}{}" = 1 : i64, "S{}{wires:1}{}" = 1 : i64}}, target_gate = "RZ{0:[f64]}{wires:1}{}"} {
      %0 = stablehlo.slice %arg2 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
      %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
      %2 = stablehlo.slice %arg2 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
      %3 = stablehlo.reshape %2 : (tensor<1xi64>) -> tensor<i64>
      %extracted = tensor.extract %3[] : tensor<i64>
      %4 = qref.get %arg0[%extracted] : !qref.reg<?>, i64 -> !qref.bit
      qref.custom "Hadamard"() %4 : !qref.bit
      %5 = stablehlo.slice %arg2 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
      %6 = stablehlo.reshape %5 : (tensor<1xi64>) -> tensor<i64>
      %extracted_0 = tensor.extract %6[] : tensor<i64>
      %7 = qref.get %arg0[%extracted_0] : !qref.reg<?>, i64 -> !qref.bit
      qref.custom "S"() %7 : !qref.bit
      %extracted_1 = tensor.extract %1[] : tensor<i64>
      %8 = qref.get %arg0[%extracted_1] : !qref.reg<?>, i64 -> !qref.bit
      %extracted_2 = tensor.extract %arg1[] : tensor<f64>
      qref.custom "RY"(%extracted_2) %8 : !qref.bit
      %9 = stablehlo.slice %arg2 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
      %10 = stablehlo.reshape %9 : (tensor<1xi64>) -> tensor<i64>
      %extracted_3 = tensor.extract %10[] : tensor<i64>
      %11 = qref.get %arg0[%extracted_3] : !qref.reg<?>, i64 -> !qref.bit
      qref.custom "S"() %11 adj : !qref.bit
      %12 = stablehlo.slice %arg2 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
      %13 = stablehlo.reshape %12 : (tensor<1xi64>) -> tensor<i64>
      %extracted_4 = tensor.extract %13[] : tensor<i64>
      %14 = qref.get %arg0[%extracted_4] : !qref.reg<?>, i64 -> !qref.bit
      qref.custom "Hadamard"() %14 : !qref.bit
      return
    }
    func.func private @"__builtin__rx_to_rot_RX{0:[f64]}{wires:1}{}"(%arg0: !qref.reg<?>, %arg1: tensor<f64>, %arg2: tensor<1xi64>) attributes {frontend_name = "_rx_to_rot", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"Rot{0:[f64],1:[f64],2:[f64]}{wires:1}{}" = 1 : i64}}, target_gate = "RX{0:[f64]}{wires:1}{}"} {
      %cst = arith.constant 10.995574287564276 : f64
      %cst_0 = arith.constant 1.5707963267948966 : f64
      %0 = stablehlo.slice %arg2 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
      %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
      %extracted = tensor.extract %1[] : tensor<i64>
      %2 = qref.get %arg0[%extracted] : !qref.reg<?>, i64 -> !qref.bit
      %extracted_1 = tensor.extract %arg1[] : tensor<f64>
      qref.custom "Rot"(%cst_0, %extracted_1, %cst) %2 : !qref.bit
      return
    }
    func.func private @"__builtin__rx_to_rz_ry_RX{0:[f64]}{wires:1}{}"(%arg0: !qref.reg<?>, %arg1: tensor<f64>, %arg2: tensor<1xi64>) attributes {frontend_name = "_rx_to_rz_ry", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"RY{0:[f64]}{wires:1}{}" = 1 : i64, "RZ{0:[f64]}{wires:1}{}" = 2 : i64}}, target_gate = "RX{0:[f64]}{wires:1}{}"} {
      %cst = arith.constant -1.5707963267948966 : f64
      %cst_0 = arith.constant 1.5707963267948966 : f64
      %0 = stablehlo.slice %arg2 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
      %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
      %extracted = tensor.extract %1[] : tensor<i64>
      %2 = qref.get %arg0[%extracted] : !qref.reg<?>, i64 -> !qref.bit
      qref.custom "RZ"(%cst_0) %2 : !qref.bit
      %3 = stablehlo.slice %arg2 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
      %4 = stablehlo.reshape %3 : (tensor<1xi64>) -> tensor<i64>
      %extracted_1 = tensor.extract %4[] : tensor<i64>
      %5 = qref.get %arg0[%extracted_1] : !qref.reg<?>, i64 -> !qref.bit
      %extracted_2 = tensor.extract %arg1[] : tensor<f64>
      qref.custom "RY"(%extracted_2) %5 : !qref.bit
      %6 = stablehlo.slice %arg2 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
      %7 = stablehlo.reshape %6 : (tensor<1xi64>) -> tensor<i64>
      %extracted_3 = tensor.extract %7[] : tensor<i64>
      %8 = qref.get %arg0[%extracted_3] : !qref.reg<?>, i64 -> !qref.bit
      qref.custom "RZ"(%cst) %8 : !qref.bit
      return
    }
    func.func private @"__builtin__rx_to_ppr_RX{0:[f64]}{wires:1}{}"(%arg0: !qref.reg<?>, %arg1: tensor<f64>, %arg2: tensor<1xi64>) attributes {frontend_name = "_rx_to_ppr", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"PauliRot{theta:[f64]}{wires:1}{pauli_word = \22X\22}" = 1 : i64}}, target_gate = "RX{0:[f64]}{wires:1}{}"} {
      %0 = stablehlo.slice %arg2 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
      %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
      %extracted = tensor.extract %1[] : tensor<i64>
      %2 = qref.get %arg0[%extracted] : !qref.reg<?>, i64 -> !qref.bit
      %extracted_0 = tensor.extract %arg1[] : tensor<f64>
      qref.paulirot ["X"](%extracted_0) %2 : !qref.bit
      return
    }
    func.func private @"__builtin__rx_to_ry_cliff_RX{0:[f64]}{wires:1}{}"(%arg0: !qref.reg<?>, %arg1: tensor<f64>, %arg2: tensor<1xi64>) attributes {frontend_name = "_rx_to_ry_cliff", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"Adjoint(S){}{wires:1}{}" = 1 : i64, "RY{0:[f64]}{wires:1}{}" = 1 : i64, "S{}{wires:1}{}" = 1 : i64}}, target_gate = "RX{0:[f64]}{wires:1}{}"} {
      %0 = stablehlo.slice %arg2 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
      %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
      %2 = stablehlo.slice %arg2 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
      %3 = stablehlo.reshape %2 : (tensor<1xi64>) -> tensor<i64>
      %extracted = tensor.extract %1[] : tensor<i64>
      %4 = qref.get %arg0[%extracted] : !qref.reg<?>, i64 -> !qref.bit
      qref.custom "S"() %4 : !qref.bit
      %extracted_0 = tensor.extract %3[] : tensor<i64>
      %5 = qref.get %arg0[%extracted_0] : !qref.reg<?>, i64 -> !qref.bit
      %extracted_1 = tensor.extract %arg1[] : tensor<f64>
      qref.custom "RY"(%extracted_1) %5 : !qref.bit
      %extracted_2 = tensor.extract %1[] : tensor<i64>
      %6 = qref.get %arg0[%extracted_2] : !qref.reg<?>, i64 -> !qref.bit
      qref.custom "S"() %6 adj : !qref.bit
      return
    }
    func.func private @"__builtin__rx_to_rz_cliff_RX{0:[f64]}{wires:1}{}"(%arg0: !qref.reg<?>, %arg1: tensor<f64>, %arg2: tensor<1xi64>) attributes {frontend_name = "_rx_to_rz_cliff", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"Hadamard{}{wires:1}{}" = 2 : i64, "RZ{0:[f64]}{wires:1}{}" = 1 : i64}}, target_gate = "RX{0:[f64]}{wires:1}{}"} {
      %0 = stablehlo.slice %arg2 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
      %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
      %2 = stablehlo.slice %arg2 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
      %3 = stablehlo.reshape %2 : (tensor<1xi64>) -> tensor<i64>
      %4 = stablehlo.slice %arg2 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
      %5 = stablehlo.reshape %4 : (tensor<1xi64>) -> tensor<i64>
      %extracted = tensor.extract %1[] : tensor<i64>
      %6 = qref.get %arg0[%extracted] : !qref.reg<?>, i64 -> !qref.bit
      qref.custom "Hadamard"() %6 : !qref.bit
      %extracted_0 = tensor.extract %3[] : tensor<i64>
      %7 = qref.get %arg0[%extracted_0] : !qref.reg<?>, i64 -> !qref.bit
      %extracted_1 = tensor.extract %arg1[] : tensor<f64>
      qref.custom "RZ"(%extracted_1) %7 : !qref.bit
      %extracted_2 = tensor.extract %5[] : tensor<i64>
      %8 = qref.get %arg0[%extracted_2] : !qref.reg<?>, i64 -> !qref.bit
      qref.custom "Hadamard"() %8 : !qref.bit
      return
    }
    func.func private @"__builtin__ry_to_rot_RY{0:[f64]}{wires:1}{}"(%arg0: !qref.reg<?>, %arg1: tensor<f64>, %arg2: tensor<1xi64>) attributes {frontend_name = "_ry_to_rot", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"Rot{0:[f64],1:[f64],2:[f64]}{wires:1}{}" = 1 : i64}}, target_gate = "RY{0:[f64]}{wires:1}{}"} {
      %c = stablehlo.constant dense<0> : tensor<i64>
      %0 = stablehlo.slice %arg2 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
      %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
      %extracted = tensor.extract %1[] : tensor<i64>
      %2 = qref.get %arg0[%extracted] : !qref.reg<?>, i64 -> !qref.bit
      %3 = stablehlo.convert %c : (tensor<i64>) -> tensor<f64>
      %extracted_0 = tensor.extract %3[] : tensor<f64>
      %extracted_1 = tensor.extract %arg1[] : tensor<f64>
      %4 = stablehlo.convert %c : (tensor<i64>) -> tensor<f64>
      %extracted_2 = tensor.extract %4[] : tensor<f64>
      qref.custom "Rot"(%extracted_0, %extracted_1, %extracted_2) %2 : !qref.bit
      return
    }
    func.func private @"__builtin__ry_to_rz_rx_RY{0:[f64]}{wires:1}{}"(%arg0: !qref.reg<?>, %arg1: tensor<f64>, %arg2: tensor<1xi64>) attributes {frontend_name = "_ry_to_rz_rx", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"RX{0:[f64]}{wires:1}{}" = 1 : i64, "RZ{0:[f64]}{wires:1}{}" = 2 : i64}}, target_gate = "RY{0:[f64]}{wires:1}{}"} {
      %cst = arith.constant 1.5707963267948966 : f64
      %cst_0 = arith.constant -1.5707963267948966 : f64
      %0 = stablehlo.slice %arg2 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
      %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
      %extracted = tensor.extract %1[] : tensor<i64>
      %2 = qref.get %arg0[%extracted] : !qref.reg<?>, i64 -> !qref.bit
      qref.custom "RZ"(%cst_0) %2 : !qref.bit
      %3 = stablehlo.slice %arg2 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
      %4 = stablehlo.reshape %3 : (tensor<1xi64>) -> tensor<i64>
      %extracted_1 = tensor.extract %4[] : tensor<i64>
      %5 = qref.get %arg0[%extracted_1] : !qref.reg<?>, i64 -> !qref.bit
      %extracted_2 = tensor.extract %arg1[] : tensor<f64>
      qref.custom "RX"(%extracted_2) %5 : !qref.bit
      %6 = stablehlo.slice %arg2 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
      %7 = stablehlo.reshape %6 : (tensor<1xi64>) -> tensor<i64>
      %extracted_3 = tensor.extract %7[] : tensor<i64>
      %8 = qref.get %arg0[%extracted_3] : !qref.reg<?>, i64 -> !qref.bit
      qref.custom "RZ"(%cst) %8 : !qref.bit
      return
    }
    func.func private @"__builtin__ry_to_ppr_RY{0:[f64]}{wires:1}{}"(%arg0: !qref.reg<?>, %arg1: tensor<f64>, %arg2: tensor<1xi64>) attributes {frontend_name = "_ry_to_ppr", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"PauliRot{theta:[f64]}{wires:1}{pauli_word = \22Y\22}" = 1 : i64}}, target_gate = "RY{0:[f64]}{wires:1}{}"} {
      %0 = stablehlo.slice %arg2 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
      %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
      %extracted = tensor.extract %1[] : tensor<i64>
      %2 = qref.get %arg0[%extracted] : !qref.reg<?>, i64 -> !qref.bit
      %extracted_0 = tensor.extract %arg1[] : tensor<f64>
      qref.paulirot ["Y"](%extracted_0) %2 : !qref.bit
      return
    }
    func.func private @"__builtin__ry_to_rx_cliff_RY{0:[f64]}{wires:1}{}"(%arg0: !qref.reg<?>, %arg1: tensor<f64>, %arg2: tensor<1xi64>) attributes {frontend_name = "_ry_to_rx_cliff", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"Adjoint(S){}{wires:1}{}" = 1 : i64, "RX{0:[f64]}{wires:1}{}" = 1 : i64, "S{}{wires:1}{}" = 1 : i64}}, target_gate = "RY{0:[f64]}{wires:1}{}"} {
      %0 = stablehlo.slice %arg2 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
      %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
      %2 = stablehlo.slice %arg2 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
      %3 = stablehlo.reshape %2 : (tensor<1xi64>) -> tensor<i64>
      %4 = stablehlo.slice %arg2 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
      %5 = stablehlo.reshape %4 : (tensor<1xi64>) -> tensor<i64>
      %extracted = tensor.extract %1[] : tensor<i64>
      %6 = qref.get %arg0[%extracted] : !qref.reg<?>, i64 -> !qref.bit
      qref.custom "S"() %6 adj : !qref.bit
      %extracted_0 = tensor.extract %3[] : tensor<i64>
      %7 = qref.get %arg0[%extracted_0] : !qref.reg<?>, i64 -> !qref.bit
      %extracted_1 = tensor.extract %arg1[] : tensor<f64>
      qref.custom "RX"(%extracted_1) %7 : !qref.bit
      %extracted_2 = tensor.extract %5[] : tensor<i64>
      %8 = qref.get %arg0[%extracted_2] : !qref.reg<?>, i64 -> !qref.bit
      qref.custom "S"() %8 : !qref.bit
      return
    }
    func.func private @"__builtin__ry_to_rz_cliff_RY{0:[f64]}{wires:1}{}"(%arg0: !qref.reg<?>, %arg1: tensor<f64>, %arg2: tensor<1xi64>) attributes {frontend_name = "_ry_to_rz_cliff", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"Adjoint(S){}{wires:1}{}" = 1 : i64, "Hadamard{}{wires:1}{}" = 2 : i64, "RZ{0:[f64]}{wires:1}{}" = 1 : i64, "S{}{wires:1}{}" = 1 : i64}}, target_gate = "RY{0:[f64]}{wires:1}{}"} {
      %0 = stablehlo.slice %arg2 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
      %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
      %2 = stablehlo.slice %arg2 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
      %3 = stablehlo.reshape %2 : (tensor<1xi64>) -> tensor<i64>
      %extracted = tensor.extract %3[] : tensor<i64>
      %4 = qref.get %arg0[%extracted] : !qref.reg<?>, i64 -> !qref.bit
      qref.custom "S"() %4 adj : !qref.bit
      %5 = stablehlo.slice %arg2 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
      %6 = stablehlo.reshape %5 : (tensor<1xi64>) -> tensor<i64>
      %extracted_0 = tensor.extract %6[] : tensor<i64>
      %7 = qref.get %arg0[%extracted_0] : !qref.reg<?>, i64 -> !qref.bit
      qref.custom "Hadamard"() %7 : !qref.bit
      %extracted_1 = tensor.extract %1[] : tensor<i64>
      %8 = qref.get %arg0[%extracted_1] : !qref.reg<?>, i64 -> !qref.bit
      %extracted_2 = tensor.extract %arg1[] : tensor<f64>
      qref.custom "RZ"(%extracted_2) %8 : !qref.bit
      %9 = stablehlo.slice %arg2 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
      %10 = stablehlo.reshape %9 : (tensor<1xi64>) -> tensor<i64>
      %extracted_3 = tensor.extract %10[] : tensor<i64>
      %11 = qref.get %arg0[%extracted_3] : !qref.reg<?>, i64 -> !qref.bit
      qref.custom "Hadamard"() %11 : !qref.bit
      %12 = stablehlo.slice %arg2 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
      %13 = stablehlo.reshape %12 : (tensor<1xi64>) -> tensor<i64>
      %extracted_4 = tensor.extract %13[] : tensor<i64>
      %14 = qref.get %arg0[%extracted_4] : !qref.reg<?>, i64 -> !qref.bit
      qref.custom "S"() %14 : !qref.bit
      return
    }
    func.func private @"__builtin__cz_to_cps_CZ{}{wires:2}{}"(%arg0: !qref.reg<?>, %arg1: tensor<2xi64>) attributes {frontend_name = "_cz_to_cps", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"ControlledPhaseShift{0:[f64]}{wires:2}{}" = 1 : i64}}, target_gate = "CZ{}{wires:2}{}"} {
      %cst = arith.constant 3.1415926535897931 : f64
      %0 = stablehlo.slice %arg1 [0:1] : (tensor<2xi64>) -> tensor<1xi64>
      %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
      %2 = stablehlo.slice %arg1 [1:2] : (tensor<2xi64>) -> tensor<1xi64>
      %3 = stablehlo.reshape %2 : (tensor<1xi64>) -> tensor<i64>
      %extracted = tensor.extract %1[] : tensor<i64>
      %4 = qref.get %arg0[%extracted] : !qref.reg<?>, i64 -> !qref.bit
      %extracted_0 = tensor.extract %3[] : tensor<i64>
      %5 = qref.get %arg0[%extracted_0] : !qref.reg<?>, i64 -> !qref.bit
      qref.custom "ControlledPhaseShift"(%cst) %4, %5 : !qref.bit, !qref.bit
      return
    }
    func.func private @"__builtin__cz_to_cnot_CZ{}{wires:2}{}"(%arg0: !qref.reg<?>, %arg1: tensor<2xi64>) attributes {frontend_name = "_cz_to_cnot", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"CNOT{}{wires:2}{}" = 1 : i64, "Hadamard{}{wires:1}{}" = 2 : i64}}, target_gate = "CZ{}{wires:2}{}"} {
      %0 = stablehlo.slice %arg1 [1:2] : (tensor<2xi64>) -> tensor<1xi64>
      %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
      %extracted = tensor.extract %1[] : tensor<i64>
      %2 = qref.get %arg0[%extracted] : !qref.reg<?>, i64 -> !qref.bit
      qref.custom "Hadamard"() %2 : !qref.bit
      %3 = stablehlo.slice %arg1 [0:1] : (tensor<2xi64>) -> tensor<1xi64>
      %4 = stablehlo.reshape %3 : (tensor<1xi64>) -> tensor<i64>
      %5 = stablehlo.slice %arg1 [1:2] : (tensor<2xi64>) -> tensor<1xi64>
      %6 = stablehlo.reshape %5 : (tensor<1xi64>) -> tensor<i64>
      %extracted_0 = tensor.extract %4[] : tensor<i64>
      %7 = qref.get %arg0[%extracted_0] : !qref.reg<?>, i64 -> !qref.bit
      %extracted_1 = tensor.extract %6[] : tensor<i64>
      %8 = qref.get %arg0[%extracted_1] : !qref.reg<?>, i64 -> !qref.bit
      qref.custom "CNOT"() %7, %8 : !qref.bit, !qref.bit
      %9 = stablehlo.slice %arg1 [1:2] : (tensor<2xi64>) -> tensor<1xi64>
      %10 = stablehlo.reshape %9 : (tensor<1xi64>) -> tensor<i64>
      %extracted_2 = tensor.extract %10[] : tensor<i64>
      %11 = qref.get %arg0[%extracted_2] : !qref.reg<?>, i64 -> !qref.bit
      qref.custom "Hadamard"() %11 : !qref.bit
      return
    }
    func.func private @"__builtin__cz_to_ppr_CZ{}{wires:2}{}"(%arg0: !qref.reg<?>, %arg1: tensor<2xi64>) attributes {frontend_name = "_cz_to_ppr", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"GlobalPhase{phi:[f64]}{}{}" = 1 : i64, "PPR{}{wires:1}{angle_denominator = -4 : si64, pauli_word = \22Z\22}" = 2 : i64, "PPR{}{wires:2}{angle_denominator = 4 : i64, pauli_word = \22ZZ\22}" = 1 : i64}}, target_gate = "CZ{}{wires:2}{}"} {
      %cst = arith.constant 0.78539816339744828 : f64
      %0 = stablehlo.slice %arg1 [0:1] : (tensor<2xi64>) -> tensor<1xi64>
      %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
      %extracted = tensor.extract %1[] : tensor<i64>
      %2 = qref.get %arg0[%extracted] : !qref.reg<?>, i64 -> !qref.bit
      qref.operator "PPR"() qubits(%2)
        static_data = {angle_denominator = -4 : si64, pauli_word = "Z"}
        qubit_map = {wires = [0]}
      %3 = stablehlo.slice %arg1 [1:2] : (tensor<2xi64>) -> tensor<1xi64>
      %4 = stablehlo.reshape %3 : (tensor<1xi64>) -> tensor<i64>
      %extracted_0 = tensor.extract %4[] : tensor<i64>
      %5 = qref.get %arg0[%extracted_0] : !qref.reg<?>, i64 -> !qref.bit
      qref.operator "PPR"() qubits(%5)
        static_data = {angle_denominator = -4 : si64, pauli_word = "Z"}
        qubit_map = {wires = [0]}
      %6 = stablehlo.slice %arg1 [0:1] : (tensor<2xi64>) -> tensor<1xi64>
      %7 = stablehlo.reshape %6 : (tensor<1xi64>) -> tensor<i64>
      %8 = stablehlo.slice %arg1 [1:2] : (tensor<2xi64>) -> tensor<1xi64>
      %9 = stablehlo.reshape %8 : (tensor<1xi64>) -> tensor<i64>
      %extracted_1 = tensor.extract %7[] : tensor<i64>
      %10 = qref.get %arg0[%extracted_1] : !qref.reg<?>, i64 -> !qref.bit
      %extracted_2 = tensor.extract %9[] : tensor<i64>
      %11 = qref.get %arg0[%extracted_2] : !qref.reg<?>, i64 -> !qref.bit
      qref.operator "PPR"() qubits(%10, %11)
        static_data = {angle_denominator = 4 : i64, pauli_word = "ZZ"}
        qubit_map = {wires = [0, 1]}
      qref.gphase(%cst)
      return
    }
    func.func private @"__builtin__ppr_to_paulirot_PPR{}{wires:1}{angle_denominator = -4 : si64, pauli_word = \22Z\22}"(%arg0: !qref.reg<?>, %arg1: tensor<1xi64>) attributes {frontend_name = "_ppr_to_paulirot", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"PauliRot{theta:[f64]}{wires:1}{pauli_word = \22Z\22}" = 1 : i64}}, target_gate = "PPR{}{wires:1}{angle_denominator = -4 : si64, pauli_word = \22Z\22}"} {
      %cst = arith.constant -1.5707963267948966 : f64
      %0 = stablehlo.slice %arg1 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
      %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
      %extracted = tensor.extract %1[] : tensor<i64>
      %2 = qref.get %arg0[%extracted] : !qref.reg<?>, i64 -> !qref.bit
      qref.paulirot ["Z"](%cst) %2 : !qref.bit
      return
    }
    func.func private @"__builtin__ppr_to_paulirot_PPR{}{wires:1}{angle_denominator = -4 : si64, pauli_word = \22X\22}"(%arg0: !qref.reg<?>, %arg1: tensor<1xi64>) attributes {frontend_name = "_ppr_to_paulirot", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"PauliRot{theta:[f64]}{wires:1}{pauli_word = \22X\22}" = 1 : i64}}, target_gate = "PPR{}{wires:1}{angle_denominator = -4 : si64, pauli_word = \22X\22}"} {
      %cst = arith.constant -1.5707963267948966 : f64
      %0 = stablehlo.slice %arg1 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
      %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
      %extracted = tensor.extract %1[] : tensor<i64>
      %2 = qref.get %arg0[%extracted] : !qref.reg<?>, i64 -> !qref.bit
      qref.paulirot ["X"](%cst) %2 : !qref.bit
      return
    }
    func.func private @"__builtin__ppr_to_paulirot_PPR{}{wires:2}{angle_denominator = 4 : i64, pauli_word = \22ZX\22}"(%arg0: !qref.reg<?>, %arg1: tensor<2xi64>) attributes {frontend_name = "_ppr_to_paulirot", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"PauliRot{theta:[f64]}{wires:2}{pauli_word = \22ZX\22}" = 1 : i64}}, target_gate = "PPR{}{wires:2}{angle_denominator = 4 : i64, pauli_word = \22ZX\22}"} {
      %cst = arith.constant 1.5707963267948966 : f64
      %0 = stablehlo.slice %arg1 [0:1] : (tensor<2xi64>) -> tensor<1xi64>
      %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
      %2 = stablehlo.slice %arg1 [1:2] : (tensor<2xi64>) -> tensor<1xi64>
      %3 = stablehlo.reshape %2 : (tensor<1xi64>) -> tensor<i64>
      %extracted = tensor.extract %1[] : tensor<i64>
      %4 = qref.get %arg0[%extracted] : !qref.reg<?>, i64 -> !qref.bit
      %extracted_0 = tensor.extract %3[] : tensor<i64>
      %5 = qref.get %arg0[%extracted_0] : !qref.reg<?>, i64 -> !qref.bit
      qref.paulirot ["Z", "X"](%cst) %4, %5 : !qref.bit, !qref.bit
      return
    }
    func.func private @"__builtin__phaseshift_to_rz_gp_PhaseShift{0:[f64]}{wires:1}{}"(%arg0: !qref.reg<?>, %arg1: tensor<f64>, %arg2: tensor<1xi64>) attributes {frontend_name = "_phaseshift_to_rz_gp", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"GlobalPhase{phi:[f64]}{}{}" = 1 : i64, "RZ{0:[f64]}{wires:1}{}" = 1 : i64}}, target_gate = "PhaseShift{0:[f64]}{wires:1}{}"} {
      %cst = stablehlo.constant dense<2.000000e+00> : tensor<f64>
      %0 = stablehlo.slice %arg2 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
      %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
      %extracted = tensor.extract %1[] : tensor<i64>
      %2 = qref.get %arg0[%extracted] : !qref.reg<?>, i64 -> !qref.bit
      %extracted_0 = tensor.extract %arg1[] : tensor<f64>
      qref.custom "RZ"(%extracted_0) %2 : !qref.bit
      %3 = stablehlo.negate %arg1 : tensor<f64>
      %4 = stablehlo.divide %3, %cst : tensor<f64>
      %extracted_1 = tensor.extract %4[] : tensor<f64>
      qref.gphase(%extracted_1)
      return
    }
    func.func private @"__builtin_adjoint_rotation_Adjoint(PhaseShift){0:[f64]}{wires:1}{}"(%arg0: !qref.reg<?>, %arg1: tensor<f64>, %arg2: tensor<1xi64>) attributes {frontend_name = "adjoint_rotation", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"PhaseShift{0:[f64]}{wires:1}{}" = 1 : i64}}, target_gate = "Adjoint(PhaseShift){0:[f64]}{wires:1}{}"} {
      %0 = stablehlo.slice %arg2 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
      %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
      %2 = stablehlo.negate %arg1 : tensor<f64>
      %extracted = tensor.extract %1[] : tensor<i64>
      %3 = qref.get %arg0[%extracted] : !qref.reg<?>, i64 -> !qref.bit
      %extracted_0 = tensor.extract %2[] : tensor<f64>
      qref.custom "PhaseShift"(%extracted_0) %3 : !qref.bit
      return
    }
    func.func private @"__builtin__phaseshift_to_rz_gp_Adjoint(PhaseShift){0:[f64]}{wires:1}{}"(%arg0: !qref.reg<?>, %arg1: tensor<f64>, %arg2: tensor<1xi64>) attributes {frontend_name = "_phaseshift_to_rz_gp", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"Adjoint(GlobalPhase){phi:[f64]}{}{}" = 1 : i64, "Adjoint(RZ){0:[f64]}{wires:1}{}" = 1 : i64}}, target_gate = "Adjoint(PhaseShift){0:[f64]}{wires:1}{}"} {
      %cst = stablehlo.constant dense<2.000000e+00> : tensor<f64>
      qref.adjoint {
        %0 = stablehlo.slice %arg2 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
        %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
        %extracted = tensor.extract %1[] : tensor<i64>
        %2 = qref.get %arg0[%extracted] : !qref.reg<?>, i64 -> !qref.bit
        %extracted_0 = tensor.extract %arg1[] : tensor<f64>
        qref.custom "RZ"(%extracted_0) %2 : !qref.bit
        %3 = stablehlo.negate %arg1 : tensor<f64>
        %4 = stablehlo.divide %3, %cst : tensor<f64>
        %extracted_1 = tensor.extract %4[] : tensor<f64>
        qref.gphase(%extracted_1)
      }
      return
    }
    func.func private @"__builtin__pauli_rot_decomposition_PauliRot{theta:[f64]}{wires:2}{pauli_word = \22ZZ\22}"(%arg0: !qref.reg<?>, %arg1: tensor<f64>, %arg2: tensor<2xi64>) attributes {frontend_name = "_pauli_rot_decomposition", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"MultiRZ{theta:[f64]}{wires:2}{}" = 1 : i64}}, target_gate = "PauliRot{theta:[f64]}{wires:2}{pauli_word = \22ZZ\22}"} {
      %0 = stablehlo.slice %arg2 [0:1] : (tensor<2xi64>) -> tensor<1xi64>
      %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
      %2 = stablehlo.slice %arg2 [1:2] : (tensor<2xi64>) -> tensor<1xi64>
      %3 = stablehlo.reshape %2 : (tensor<1xi64>) -> tensor<i64>
      %extracted = tensor.extract %1[] : tensor<i64>
      %4 = qref.get %arg0[%extracted] : !qref.reg<?>, i64 -> !qref.bit
      %extracted_0 = tensor.extract %3[] : tensor<i64>
      %5 = qref.get %arg0[%extracted_0] : !qref.reg<?>, i64 -> !qref.bit
      %extracted_1 = tensor.extract %arg1[] : tensor<f64>
      qref.multirz(%extracted_1) %4, %5 : !qref.bit, !qref.bit
      return
    }
    func.func private @"__builtin__pauli_rot_decomposition_PauliRot{theta:[f64]}{wires:2}{pauli_word = \22ZX\22}"(%arg0: !qref.reg<?>, %arg1: tensor<f64>, %arg2: tensor<2xi64>) attributes {frontend_name = "_pauli_rot_decomposition", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"Hadamard{}{wires:1}{}" = 2 : i64, "MultiRZ{theta:[f64]}{wires:2}{}" = 1 : i64}}, target_gate = "PauliRot{theta:[f64]}{wires:2}{pauli_word = \22ZX\22}"} {
      %0 = stablehlo.slice %arg2 [0:1] : (tensor<2xi64>) -> tensor<1xi64>
      %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
      %2 = stablehlo.slice %arg2 [1:2] : (tensor<2xi64>) -> tensor<1xi64>
      %3 = stablehlo.reshape %2 : (tensor<1xi64>) -> tensor<i64>
      %extracted = tensor.extract %3[] : tensor<i64>
      %4 = qref.get %arg0[%extracted] : !qref.reg<?>, i64 -> !qref.bit
      qref.custom "Hadamard"() %4 : !qref.bit
      %extracted_0 = tensor.extract %1[] : tensor<i64>
      %5 = qref.get %arg0[%extracted_0] : !qref.reg<?>, i64 -> !qref.bit
      %extracted_1 = tensor.extract %3[] : tensor<i64>
      %6 = qref.get %arg0[%extracted_1] : !qref.reg<?>, i64 -> !qref.bit
      %extracted_2 = tensor.extract %arg1[] : tensor<f64>
      qref.multirz(%extracted_2) %5, %6 : !qref.bit, !qref.bit
      %extracted_3 = tensor.extract %3[] : tensor<i64>
      %7 = qref.get %arg0[%extracted_3] : !qref.reg<?>, i64 -> !qref.bit
      qref.custom "Hadamard"() %7 : !qref.bit
      return
    }
    func.func private @"__builtin__pauli_rot_decomposition_PauliRot{theta:[f64]}{wires:3}{pauli_word = \22ZZX\22}"(%arg0: !qref.reg<?>, %arg1: tensor<f64>, %arg2: tensor<3xi64>) attributes {frontend_name = "_pauli_rot_decomposition", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"Hadamard{}{wires:1}{}" = 2 : i64, "MultiRZ{theta:[f64]}{wires:3}{}" = 1 : i64}}, target_gate = "PauliRot{theta:[f64]}{wires:3}{pauli_word = \22ZZX\22}"} {
      %0 = stablehlo.slice %arg2 [0:1] : (tensor<3xi64>) -> tensor<1xi64>
      %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
      %2 = stablehlo.slice %arg2 [1:2] : (tensor<3xi64>) -> tensor<1xi64>
      %3 = stablehlo.reshape %2 : (tensor<1xi64>) -> tensor<i64>
      %4 = stablehlo.slice %arg2 [2:3] : (tensor<3xi64>) -> tensor<1xi64>
      %5 = stablehlo.reshape %4 : (tensor<1xi64>) -> tensor<i64>
      %extracted = tensor.extract %5[] : tensor<i64>
      %6 = qref.get %arg0[%extracted] : !qref.reg<?>, i64 -> !qref.bit
      qref.custom "Hadamard"() %6 : !qref.bit
      %extracted_0 = tensor.extract %1[] : tensor<i64>
      %7 = qref.get %arg0[%extracted_0] : !qref.reg<?>, i64 -> !qref.bit
      %extracted_1 = tensor.extract %3[] : tensor<i64>
      %8 = qref.get %arg0[%extracted_1] : !qref.reg<?>, i64 -> !qref.bit
      %extracted_2 = tensor.extract %5[] : tensor<i64>
      %9 = qref.get %arg0[%extracted_2] : !qref.reg<?>, i64 -> !qref.bit
      %extracted_3 = tensor.extract %arg1[] : tensor<f64>
      qref.multirz(%extracted_3) %7, %8, %9 : !qref.bit, !qref.bit, !qref.bit
      %extracted_4 = tensor.extract %5[] : tensor<i64>
      %10 = qref.get %arg0[%extracted_4] : !qref.reg<?>, i64 -> !qref.bit
      qref.custom "Hadamard"() %10 : !qref.bit
      return
    }
    func.func private @"__builtin__pauli_rot_decomposition_PauliRot{theta:[f64]}{wires:1}{pauli_word = \22X\22}"(%arg0: !qref.reg<?>, %arg1: tensor<f64>, %arg2: tensor<1xi64>) attributes {frontend_name = "_pauli_rot_decomposition", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"Hadamard{}{wires:1}{}" = 2 : i64, "MultiRZ{theta:[f64]}{wires:1}{}" = 1 : i64}}, target_gate = "PauliRot{theta:[f64]}{wires:1}{pauli_word = \22X\22}"} {
      %0 = stablehlo.slice %arg2 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
      %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
      %extracted = tensor.extract %1[] : tensor<i64>
      %2 = qref.get %arg0[%extracted] : !qref.reg<?>, i64 -> !qref.bit
      qref.custom "Hadamard"() %2 : !qref.bit
      %extracted_0 = tensor.extract %1[] : tensor<i64>
      %3 = qref.get %arg0[%extracted_0] : !qref.reg<?>, i64 -> !qref.bit
      %extracted_1 = tensor.extract %arg1[] : tensor<f64>
      qref.multirz(%extracted_1) %3 : !qref.bit
      %extracted_2 = tensor.extract %1[] : tensor<i64>
      %4 = qref.get %arg0[%extracted_2] : !qref.reg<?>, i64 -> !qref.bit
      qref.custom "Hadamard"() %4 : !qref.bit
      return
    }
    func.func private @"__builtin__pauli_rot_decomposition_PauliRot{theta:[f64]}{wires:1}{pauli_word = \22Z\22}"(%arg0: !qref.reg<?>, %arg1: tensor<f64>, %arg2: tensor<1xi64>) attributes {frontend_name = "_pauli_rot_decomposition", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"MultiRZ{theta:[f64]}{wires:1}{}" = 1 : i64}}, target_gate = "PauliRot{theta:[f64]}{wires:1}{pauli_word = \22Z\22}"} {
      %0 = stablehlo.slice %arg2 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
      %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
      %extracted = tensor.extract %1[] : tensor<i64>
      %2 = qref.get %arg0[%extracted] : !qref.reg<?>, i64 -> !qref.bit
      %extracted_0 = tensor.extract %arg1[] : tensor<f64>
      qref.multirz(%extracted_0) %2 : !qref.bit
      return
    }
    func.func private @"__builtin__rot_to_rz_ry_rz_Rot{0:[f64],1:[f64],2:[f64]}{wires:1}{}"(%arg0: !qref.reg<?>, %arg1: tensor<f64>, %arg2: tensor<f64>, %arg3: tensor<f64>, %arg4: tensor<1xi64>) attributes {frontend_name = "_rot_to_rz_ry_rz", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"RY{0:[f64]}{wires:1}{}" = 1 : i64, "RZ{0:[f64]}{wires:1}{}" = 2 : i64}}, target_gate = "Rot{0:[f64],1:[f64],2:[f64]}{wires:1}{}"} {
      %0 = stablehlo.slice %arg4 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
      %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
      %extracted = tensor.extract %1[] : tensor<i64>
      %2 = qref.get %arg0[%extracted] : !qref.reg<?>, i64 -> !qref.bit
      %extracted_0 = tensor.extract %arg1[] : tensor<f64>
      qref.custom "RZ"(%extracted_0) %2 : !qref.bit
      %3 = stablehlo.slice %arg4 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
      %4 = stablehlo.reshape %3 : (tensor<1xi64>) -> tensor<i64>
      %extracted_1 = tensor.extract %4[] : tensor<i64>
      %5 = qref.get %arg0[%extracted_1] : !qref.reg<?>, i64 -> !qref.bit
      %extracted_2 = tensor.extract %arg2[] : tensor<f64>
      qref.custom "RY"(%extracted_2) %5 : !qref.bit
      %6 = stablehlo.slice %arg4 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
      %7 = stablehlo.reshape %6 : (tensor<1xi64>) -> tensor<i64>
      %extracted_3 = tensor.extract %7[] : tensor<i64>
      %8 = qref.get %arg0[%extracted_3] : !qref.reg<?>, i64 -> !qref.bit
      %extracted_4 = tensor.extract %arg3[] : tensor<f64>
      qref.custom "RZ"(%extracted_4) %8 : !qref.bit
      return
    }
    func.func private @"__builtin__s_phaseshift_S{}{wires:1}{}"(%arg0: !qref.reg<?>, %arg1: tensor<1xi64>) attributes {frontend_name = "_s_phaseshift", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"PhaseShift{0:[f64]}{wires:1}{}" = 1 : i64}}, target_gate = "S{}{wires:1}{}"} {
      %cst = arith.constant 1.5707963267948966 : f64
      %0 = stablehlo.slice %arg1 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
      %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
      %extracted = tensor.extract %1[] : tensor<i64>
      %2 = qref.get %arg0[%extracted] : !qref.reg<?>, i64 -> !qref.bit
      qref.custom "PhaseShift"(%cst) %2 : !qref.bit
      return
    }
    func.func private @"__builtin__s_phaseshift_Adjoint(S){}{wires:1}{}"(%arg0: !qref.reg<?>, %arg1: tensor<1xi64>) attributes {frontend_name = "_s_phaseshift", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"Adjoint(PhaseShift){0:[f64]}{wires:1}{}" = 1 : i64}}, target_gate = "Adjoint(S){}{wires:1}{}"} {
      %cst = arith.constant 1.5707963267948966 : f64
      qref.adjoint {
        %0 = stablehlo.slice %arg1 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
        %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
        %extracted = tensor.extract %1[] : tensor<i64>
        %2 = qref.get %arg0[%extracted] : !qref.reg<?>, i64 -> !qref.bit
        qref.custom "PhaseShift"(%cst) %2 : !qref.bit
      }
      return
    }
    func.func private @"__builtin__pauli_rot_decomposition_PauliRot{theta:[f64]}{wires:1}{pauli_word = \22Y\22}"(%arg0: !qref.reg<?>, %arg1: tensor<f64>, %arg2: tensor<1xi64>) attributes {frontend_name = "_pauli_rot_decomposition", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"MultiRZ{theta:[f64]}{wires:1}{}" = 1 : i64, "RX{0:[f64]}{wires:1}{}" = 2 : i64}}, target_gate = "PauliRot{theta:[f64]}{wires:1}{pauli_word = \22Y\22}"} {
      %cst = arith.constant -1.5707963267948966 : f64
      %cst_0 = arith.constant 1.5707963267948966 : f64
      %0 = stablehlo.slice %arg2 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
      %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
      %extracted = tensor.extract %1[] : tensor<i64>
      %2 = qref.get %arg0[%extracted] : !qref.reg<?>, i64 -> !qref.bit
      qref.custom "RX"(%cst_0) %2 : !qref.bit
      %extracted_1 = tensor.extract %1[] : tensor<i64>
      %3 = qref.get %arg0[%extracted_1] : !qref.reg<?>, i64 -> !qref.bit
      %extracted_2 = tensor.extract %arg1[] : tensor<f64>
      qref.multirz(%extracted_2) %3 : !qref.bit
      %extracted_3 = tensor.extract %1[] : tensor<i64>
      %4 = qref.get %arg0[%extracted_3] : !qref.reg<?>, i64 -> !qref.bit
      qref.custom "RX"(%cst) %4 : !qref.bit
      return
    }
    func.func private @"__builtin__cphase_to_rz_cnot_ControlledPhaseShift{0:[f64]}{wires:2}{}"(%arg0: !qref.reg<?>, %arg1: tensor<f64>, %arg2: tensor<2xi64>) attributes {frontend_name = "_cphase_to_rz_cnot", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"CNOT{}{wires:2}{}" = 2 : i64, "GlobalPhase{phi:[f64]}{}{}" = 1 : i64, "RZ{0:[f64]}{wires:1}{}" = 3 : i64}}, target_gate = "ControlledPhaseShift{0:[f64]}{wires:2}{}"} {
      %cst = stablehlo.constant dense<4.000000e+00> : tensor<f64>
      %cst_0 = stablehlo.constant dense<2.000000e+00> : tensor<f64>
      %0 = stablehlo.divide %arg1, %cst_0 : tensor<f64>
      %1 = stablehlo.slice %arg2 [0:1] : (tensor<2xi64>) -> tensor<1xi64>
      %2 = stablehlo.reshape %1 : (tensor<1xi64>) -> tensor<i64>
      %extracted = tensor.extract %2[] : tensor<i64>
      %3 = qref.get %arg0[%extracted] : !qref.reg<?>, i64 -> !qref.bit
      %extracted_1 = tensor.extract %0[] : tensor<f64>
      qref.custom "RZ"(%extracted_1) %3 : !qref.bit
      %4 = stablehlo.slice %arg2 [0:1] : (tensor<2xi64>) -> tensor<1xi64>
      %5 = stablehlo.reshape %4 : (tensor<1xi64>) -> tensor<i64>
      %6 = stablehlo.slice %arg2 [1:2] : (tensor<2xi64>) -> tensor<1xi64>
      %7 = stablehlo.reshape %6 : (tensor<1xi64>) -> tensor<i64>
      %extracted_2 = tensor.extract %5[] : tensor<i64>
      %8 = qref.get %arg0[%extracted_2] : !qref.reg<?>, i64 -> !qref.bit
      %extracted_3 = tensor.extract %7[] : tensor<i64>
      %9 = qref.get %arg0[%extracted_3] : !qref.reg<?>, i64 -> !qref.bit
      qref.custom "CNOT"() %8, %9 : !qref.bit, !qref.bit
      %10 = stablehlo.negate %arg1 : tensor<f64>
      %11 = stablehlo.divide %10, %cst_0 : tensor<f64>
      %12 = stablehlo.slice %arg2 [1:2] : (tensor<2xi64>) -> tensor<1xi64>
      %13 = stablehlo.reshape %12 : (tensor<1xi64>) -> tensor<i64>
      %extracted_4 = tensor.extract %13[] : tensor<i64>
      %14 = qref.get %arg0[%extracted_4] : !qref.reg<?>, i64 -> !qref.bit
      %extracted_5 = tensor.extract %11[] : tensor<f64>
      qref.custom "RZ"(%extracted_5) %14 : !qref.bit
      %15 = stablehlo.slice %arg2 [0:1] : (tensor<2xi64>) -> tensor<1xi64>
      %16 = stablehlo.reshape %15 : (tensor<1xi64>) -> tensor<i64>
      %17 = stablehlo.slice %arg2 [1:2] : (tensor<2xi64>) -> tensor<1xi64>
      %18 = stablehlo.reshape %17 : (tensor<1xi64>) -> tensor<i64>
      %extracted_6 = tensor.extract %16[] : tensor<i64>
      %19 = qref.get %arg0[%extracted_6] : !qref.reg<?>, i64 -> !qref.bit
      %extracted_7 = tensor.extract %18[] : tensor<i64>
      %20 = qref.get %arg0[%extracted_7] : !qref.reg<?>, i64 -> !qref.bit
      qref.custom "CNOT"() %19, %20 : !qref.bit, !qref.bit
      %21 = stablehlo.divide %arg1, %cst_0 : tensor<f64>
      %22 = stablehlo.slice %arg2 [1:2] : (tensor<2xi64>) -> tensor<1xi64>
      %23 = stablehlo.reshape %22 : (tensor<1xi64>) -> tensor<i64>
      %extracted_8 = tensor.extract %23[] : tensor<i64>
      %24 = qref.get %arg0[%extracted_8] : !qref.reg<?>, i64 -> !qref.bit
      %extracted_9 = tensor.extract %21[] : tensor<f64>
      qref.custom "RZ"(%extracted_9) %24 : !qref.bit
      %25 = stablehlo.negate %arg1 : tensor<f64>
      %26 = stablehlo.divide %25, %cst : tensor<f64>
      %extracted_10 = tensor.extract %26[] : tensor<f64>
      qref.gphase(%extracted_10)
      return
    }
    func.func private @"__builtin__cphase_to_ppr_ControlledPhaseShift{0:[f64]}{wires:2}{}"(%arg0: !qref.reg<?>, %arg1: tensor<f64>, %arg2: tensor<2xi64>) attributes {frontend_name = "_cphase_to_ppr", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"GlobalPhase{phi:[f64]}{}{}" = 1 : i64, "PauliRot{theta:[f64]}{wires:1}{pauli_word = \22Z\22}" = 2 : i64, "PauliRot{theta:[f64]}{wires:2}{pauli_word = \22ZZ\22}" = 1 : i64}}, target_gate = "ControlledPhaseShift{0:[f64]}{wires:2}{}"} {
      %cst = stablehlo.constant dense<4.000000e+00> : tensor<f64>
      %cst_0 = stablehlo.constant dense<2.000000e+00> : tensor<f64>
      %0 = stablehlo.negate %arg1 : tensor<f64>
      %1 = stablehlo.divide %0, %cst_0 : tensor<f64>
      %2 = stablehlo.slice %arg2 [0:1] : (tensor<2xi64>) -> tensor<1xi64>
      %3 = stablehlo.reshape %2 : (tensor<1xi64>) -> tensor<i64>
      %4 = stablehlo.slice %arg2 [1:2] : (tensor<2xi64>) -> tensor<1xi64>
      %5 = stablehlo.reshape %4 : (tensor<1xi64>) -> tensor<i64>
      %extracted = tensor.extract %3[] : tensor<i64>
      %6 = qref.get %arg0[%extracted] : !qref.reg<?>, i64 -> !qref.bit
      %extracted_1 = tensor.extract %5[] : tensor<i64>
      %7 = qref.get %arg0[%extracted_1] : !qref.reg<?>, i64 -> !qref.bit
      %extracted_2 = tensor.extract %1[] : tensor<f64>
      qref.paulirot ["Z", "Z"](%extracted_2) %6, %7 : !qref.bit, !qref.bit
      %8 = stablehlo.divide %arg1, %cst_0 : tensor<f64>
      %9 = stablehlo.slice %arg2 [1:2] : (tensor<2xi64>) -> tensor<1xi64>
      %10 = stablehlo.reshape %9 : (tensor<1xi64>) -> tensor<i64>
      %extracted_3 = tensor.extract %10[] : tensor<i64>
      %11 = qref.get %arg0[%extracted_3] : !qref.reg<?>, i64 -> !qref.bit
      %extracted_4 = tensor.extract %8[] : tensor<f64>
      qref.paulirot ["Z"](%extracted_4) %11 : !qref.bit
      %12 = stablehlo.divide %arg1, %cst_0 : tensor<f64>
      %13 = stablehlo.slice %arg2 [0:1] : (tensor<2xi64>) -> tensor<1xi64>
      %14 = stablehlo.reshape %13 : (tensor<1xi64>) -> tensor<i64>
      %extracted_5 = tensor.extract %14[] : tensor<i64>
      %15 = qref.get %arg0[%extracted_5] : !qref.reg<?>, i64 -> !qref.bit
      %extracted_6 = tensor.extract %12[] : tensor<f64>
      qref.paulirot ["Z"](%extracted_6) %15 : !qref.bit
      %16 = stablehlo.negate %arg1 : tensor<f64>
      %17 = stablehlo.divide %16, %cst : tensor<f64>
      %extracted_7 = tensor.extract %17[] : tensor<f64>
      qref.gphase(%extracted_7)
      return
    }
    func.func private @"__builtin__ppr_to_paulirot_PPR{}{wires:2}{angle_denominator = 4 : i64, pauli_word = \22ZZ\22}"(%arg0: !qref.reg<?>, %arg1: tensor<2xi64>) attributes {frontend_name = "_ppr_to_paulirot", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"PauliRot{theta:[f64]}{wires:2}{pauli_word = \22ZZ\22}" = 1 : i64}}, target_gate = "PPR{}{wires:2}{angle_denominator = 4 : i64, pauli_word = \22ZZ\22}"} {
      %cst = arith.constant 1.5707963267948966 : f64
      %0 = stablehlo.slice %arg1 [0:1] : (tensor<2xi64>) -> tensor<1xi64>
      %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
      %2 = stablehlo.slice %arg1 [1:2] : (tensor<2xi64>) -> tensor<1xi64>
      %3 = stablehlo.reshape %2 : (tensor<1xi64>) -> tensor<i64>
      %extracted = tensor.extract %1[] : tensor<i64>
      %4 = qref.get %arg0[%extracted] : !qref.reg<?>, i64 -> !qref.bit
      %extracted_0 = tensor.extract %3[] : tensor<i64>
      %5 = qref.get %arg0[%extracted_0] : !qref.reg<?>, i64 -> !qref.bit
      qref.paulirot ["Z", "Z"](%cst) %4, %5 : !qref.bit, !qref.bit
      return
    }
    func.func private @"__builtin_adjoint_rotation_Adjoint(RZ){0:[f64]}{wires:1}{}"(%arg0: !qref.reg<?>, %arg1: tensor<f64>, %arg2: tensor<1xi64>) attributes {frontend_name = "adjoint_rotation", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"RZ{0:[f64]}{wires:1}{}" = 1 : i64}}, target_gate = "Adjoint(RZ){0:[f64]}{wires:1}{}"} {
      %0 = stablehlo.slice %arg2 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
      %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
      %2 = stablehlo.negate %arg1 : tensor<f64>
      %extracted = tensor.extract %1[] : tensor<i64>
      %3 = qref.get %arg0[%extracted] : !qref.reg<?>, i64 -> !qref.bit
      %extracted_0 = tensor.extract %2[] : tensor<f64>
      qref.custom "RZ"(%extracted_0) %3 : !qref.bit
      return
    }
    func.func private @"__builtin__rz_to_ps_Adjoint(RZ){0:[f64]}{wires:1}{}"(%arg0: !qref.reg<?>, %arg1: tensor<f64>, %arg2: tensor<1xi64>) attributes {frontend_name = "_rz_to_ps", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"Adjoint(GlobalPhase){phi:[f64]}{}{}" = 1 : i64, "Adjoint(PhaseShift){0:[f64]}{wires:1}{}" = 1 : i64}}, target_gate = "Adjoint(RZ){0:[f64]}{wires:1}{}"} {
      %cst = stablehlo.constant dense<2.000000e+00> : tensor<f64>
      qref.adjoint {
        %0 = stablehlo.slice %arg2 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
        %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
        %extracted = tensor.extract %1[] : tensor<i64>
        %2 = qref.get %arg0[%extracted] : !qref.reg<?>, i64 -> !qref.bit
        %extracted_0 = tensor.extract %arg1[] : tensor<f64>
        qref.custom "PhaseShift"(%extracted_0) %2 : !qref.bit
        %3 = stablehlo.divide %arg1, %cst : tensor<f64>
        %extracted_1 = tensor.extract %3[] : tensor<f64>
        qref.gphase(%extracted_1)
      }
      return
    }
    func.func private @"__builtin__rz_to_rot_Adjoint(RZ){0:[f64]}{wires:1}{}"(%arg0: !qref.reg<?>, %arg1: tensor<f64>, %arg2: tensor<1xi64>) attributes {frontend_name = "_rz_to_rot", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"Adjoint(Rot){0:[f64],1:[f64],2:[f64]}{wires:1}{}" = 1 : i64}}, target_gate = "Adjoint(RZ){0:[f64]}{wires:1}{}"} {
      %c = stablehlo.constant dense<0> : tensor<i64>
      qref.adjoint {
        %0 = stablehlo.slice %arg2 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
        %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
        %extracted = tensor.extract %1[] : tensor<i64>
        %2 = qref.get %arg0[%extracted] : !qref.reg<?>, i64 -> !qref.bit
        %3 = stablehlo.convert %c : (tensor<i64>) -> tensor<f64>
        %extracted_0 = tensor.extract %3[] : tensor<f64>
        %4 = stablehlo.convert %c : (tensor<i64>) -> tensor<f64>
        %extracted_1 = tensor.extract %4[] : tensor<f64>
        %extracted_2 = tensor.extract %arg1[] : tensor<f64>
        qref.custom "Rot"(%extracted_0, %extracted_1, %extracted_2) %2 : !qref.bit
      }
      return
    }
    func.func private @"__builtin__rz_to_ry_rx_Adjoint(RZ){0:[f64]}{wires:1}{}"(%arg0: !qref.reg<?>, %arg1: tensor<f64>, %arg2: tensor<1xi64>) attributes {frontend_name = "_rz_to_ry_rx", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"Adjoint(RX){0:[f64]}{wires:1}{}" = 1 : i64, "Adjoint(RY){0:[f64]}{wires:1}{}" = 2 : i64}}, target_gate = "Adjoint(RZ){0:[f64]}{wires:1}{}"} {
      %cst = arith.constant -1.5707963267948966 : f64
      %cst_0 = arith.constant 1.5707963267948966 : f64
      qref.adjoint {
        %0 = stablehlo.slice %arg2 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
        %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
        %extracted = tensor.extract %1[] : tensor<i64>
        %2 = qref.get %arg0[%extracted] : !qref.reg<?>, i64 -> !qref.bit
        qref.custom "RY"(%cst_0) %2 : !qref.bit
        %3 = stablehlo.slice %arg2 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
        %4 = stablehlo.reshape %3 : (tensor<1xi64>) -> tensor<i64>
        %extracted_1 = tensor.extract %4[] : tensor<i64>
        %5 = qref.get %arg0[%extracted_1] : !qref.reg<?>, i64 -> !qref.bit
        %extracted_2 = tensor.extract %arg1[] : tensor<f64>
        qref.custom "RX"(%extracted_2) %5 : !qref.bit
        %6 = stablehlo.slice %arg2 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
        %7 = stablehlo.reshape %6 : (tensor<1xi64>) -> tensor<i64>
        %extracted_3 = tensor.extract %7[] : tensor<i64>
        %8 = qref.get %arg0[%extracted_3] : !qref.reg<?>, i64 -> !qref.bit
        qref.custom "RY"(%cst) %8 : !qref.bit
      }
      return
    }
    func.func private @"__builtin__rz_to_ppr_Adjoint(RZ){0:[f64]}{wires:1}{}"(%arg0: !qref.reg<?>, %arg1: tensor<f64>, %arg2: tensor<1xi64>) attributes {frontend_name = "_rz_to_ppr", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"Adjoint(PauliRot){theta:[f64]}{wires:1}{pauli_word = \22Z\22}" = 1 : i64}}, target_gate = "Adjoint(RZ){0:[f64]}{wires:1}{}"} {
      qref.adjoint {
        %0 = stablehlo.slice %arg2 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
        %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
        %extracted = tensor.extract %1[] : tensor<i64>
        %2 = qref.get %arg0[%extracted] : !qref.reg<?>, i64 -> !qref.bit
        %extracted_0 = tensor.extract %arg1[] : tensor<f64>
        qref.paulirot ["Z"](%extracted_0) %2 : !qref.bit
      }
      return
    }
    func.func private @"__builtin__rz_to_rx_cliff_Adjoint(RZ){0:[f64]}{wires:1}{}"(%arg0: !qref.reg<?>, %arg1: tensor<f64>, %arg2: tensor<1xi64>) attributes {frontend_name = "_rz_to_rx_cliff", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"Adjoint(Hadamard){}{wires:1}{}" = 2 : i64, "Adjoint(RX){0:[f64]}{wires:1}{}" = 1 : i64}}, target_gate = "Adjoint(RZ){0:[f64]}{wires:1}{}"} {
      qref.adjoint {
        %0 = stablehlo.slice %arg2 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
        %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
        %2 = stablehlo.slice %arg2 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
        %3 = stablehlo.reshape %2 : (tensor<1xi64>) -> tensor<i64>
        %4 = stablehlo.slice %arg2 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
        %5 = stablehlo.reshape %4 : (tensor<1xi64>) -> tensor<i64>
        %extracted = tensor.extract %1[] : tensor<i64>
        %6 = qref.get %arg0[%extracted] : !qref.reg<?>, i64 -> !qref.bit
        qref.custom "Hadamard"() %6 : !qref.bit
        %extracted_0 = tensor.extract %3[] : tensor<i64>
        %7 = qref.get %arg0[%extracted_0] : !qref.reg<?>, i64 -> !qref.bit
        %extracted_1 = tensor.extract %arg1[] : tensor<f64>
        qref.custom "RX"(%extracted_1) %7 : !qref.bit
        %extracted_2 = tensor.extract %5[] : tensor<i64>
        %8 = qref.get %arg0[%extracted_2] : !qref.reg<?>, i64 -> !qref.bit
        qref.custom "Hadamard"() %8 : !qref.bit
      }
      return
    }
    func.func private @"__builtin__rz_to_ry_cliff_Adjoint(RZ){0:[f64]}{wires:1}{}"(%arg0: !qref.reg<?>, %arg1: tensor<f64>, %arg2: tensor<1xi64>) attributes {frontend_name = "_rz_to_ry_cliff", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"Adjoint(Hadamard){}{wires:1}{}" = 2 : i64, "Adjoint(RY){0:[f64]}{wires:1}{}" = 1 : i64, "Adjoint(S){}{wires:1}{}" = 1 : i64, "S{}{wires:1}{}" = 1 : i64}}, target_gate = "Adjoint(RZ){0:[f64]}{wires:1}{}"} {
      qref.adjoint {
        %0 = stablehlo.slice %arg2 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
        %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
        %2 = stablehlo.slice %arg2 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
        %3 = stablehlo.reshape %2 : (tensor<1xi64>) -> tensor<i64>
        %extracted = tensor.extract %3[] : tensor<i64>
        %4 = qref.get %arg0[%extracted] : !qref.reg<?>, i64 -> !qref.bit
        qref.custom "Hadamard"() %4 : !qref.bit
        %5 = stablehlo.slice %arg2 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
        %6 = stablehlo.reshape %5 : (tensor<1xi64>) -> tensor<i64>
        %extracted_0 = tensor.extract %6[] : tensor<i64>
        %7 = qref.get %arg0[%extracted_0] : !qref.reg<?>, i64 -> !qref.bit
        qref.custom "S"() %7 : !qref.bit
        %extracted_1 = tensor.extract %1[] : tensor<i64>
        %8 = qref.get %arg0[%extracted_1] : !qref.reg<?>, i64 -> !qref.bit
        %extracted_2 = tensor.extract %arg1[] : tensor<f64>
        qref.custom "RY"(%extracted_2) %8 : !qref.bit
        %9 = stablehlo.slice %arg2 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
        %10 = stablehlo.reshape %9 : (tensor<1xi64>) -> tensor<i64>
        %extracted_3 = tensor.extract %10[] : tensor<i64>
        %11 = qref.get %arg0[%extracted_3] : !qref.reg<?>, i64 -> !qref.bit
        qref.custom "S"() %11 adj : !qref.bit
        %12 = stablehlo.slice %arg2 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
        %13 = stablehlo.reshape %12 : (tensor<1xi64>) -> tensor<i64>
        %extracted_4 = tensor.extract %13[] : tensor<i64>
        %14 = qref.get %arg0[%extracted_4] : !qref.reg<?>, i64 -> !qref.bit
        qref.custom "Hadamard"() %14 : !qref.bit
      }
      return
    }
    func.func private @"__builtin_adjoint_rotation_Adjoint(GlobalPhase){phi:[f64]}{}{}"(%arg0: !qref.reg<?>, %arg1: tensor<f64>, %arg2: tensor<0xi64>) attributes {frontend_name = "adjoint_rotation", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"GlobalPhase{phi:[f64]}{}{}" = 1 : i64}}, target_gate = "Adjoint(GlobalPhase){phi:[f64]}{}{}"} {
      %0 = stablehlo.negate %arg1 : tensor<f64>
      %extracted = tensor.extract %0[] : tensor<f64>
      qref.gphase(%extracted)
      return
    }
    func.func private @"__builtin__multi_rz_decomposition_MultiRZ{theta:[f64]}{wires:2}{}"(%arg0: !qref.reg<?>, %arg1: tensor<f64>, %arg2: tensor<2xi64>) attributes {frontend_name = "_multi_rz_decomposition", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"CNOT{}{wires:2}{}" = 2 : i64, "RZ{0:[f64]}{wires:1}{}" = 1 : i64}}, target_gate = "MultiRZ{theta:[f64]}{wires:2}{}"} {
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
      %6 = stablehlo.dynamic_slice %arg2, %5, sizes = [1] : (tensor<2xi64>, tensor<i64>) -> tensor<1xi64>
      %7 = stablehlo.reshape %6 : (tensor<1xi64>) -> tensor<i64>
      %8 = stablehlo.subtract %1, %c_2 : tensor<i64>
      %9 = stablehlo.compare  LT, %8, %c_1,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
      %10 = stablehlo.convert %8 : tensor<i64>
      %11 = stablehlo.add %10, %c : tensor<i64>
      %12 = stablehlo.select %9, %11, %8 : tensor<i1>, tensor<i64>
      %13 = stablehlo.dynamic_slice %arg2, %12, sizes = [1] : (tensor<2xi64>, tensor<i64>) -> tensor<1xi64>
      %14 = stablehlo.reshape %13 : (tensor<1xi64>) -> tensor<i64>
      %extracted = tensor.extract %7[] : tensor<i64>
      %15 = qref.get %arg0[%extracted] : !qref.reg<?>, i64 -> !qref.bit
      %extracted_4 = tensor.extract %14[] : tensor<i64>
      %16 = qref.get %arg0[%extracted_4] : !qref.reg<?>, i64 -> !qref.bit
      qref.custom "CNOT"() %15, %16 : !qref.bit, !qref.bit
      %17 = stablehlo.slice %arg2 [0:1] : (tensor<2xi64>) -> tensor<1xi64>
      %18 = stablehlo.reshape %17 : (tensor<1xi64>) -> tensor<i64>
      %extracted_5 = tensor.extract %18[] : tensor<i64>
      %19 = qref.get %arg0[%extracted_5] : !qref.reg<?>, i64 -> !qref.bit
      %extracted_6 = tensor.extract %arg1[] : tensor<f64>
      qref.custom "RZ"(%extracted_6) %19 : !qref.bit
      %20 = stablehlo.compare  LT, %cst, %c_1,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
      %21 = stablehlo.convert %cst : tensor<i64>
      %22 = stablehlo.add %21, %c : tensor<i64>
      %23 = stablehlo.select %20, %22, %cst : tensor<i1>, tensor<i64>
      %24 = stablehlo.dynamic_slice %arg2, %23, sizes = [1] : (tensor<2xi64>, tensor<i64>) -> tensor<1xi64>
      %25 = stablehlo.reshape %24 : (tensor<1xi64>) -> tensor<i64>
      %26 = stablehlo.subtract %cst, %c_2 : tensor<i64>
      %27 = stablehlo.compare  LT, %26, %c_1,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
      %28 = stablehlo.convert %26 : tensor<i64>
      %29 = stablehlo.add %28, %c : tensor<i64>
      %30 = stablehlo.select %27, %29, %26 : tensor<i1>, tensor<i64>
      %31 = stablehlo.dynamic_slice %arg2, %30, sizes = [1] : (tensor<2xi64>, tensor<i64>) -> tensor<1xi64>
      %32 = stablehlo.reshape %31 : (tensor<1xi64>) -> tensor<i64>
      %extracted_7 = tensor.extract %25[] : tensor<i64>
      %33 = qref.get %arg0[%extracted_7] : !qref.reg<?>, i64 -> !qref.bit
      %extracted_8 = tensor.extract %32[] : tensor<i64>
      %34 = qref.get %arg0[%extracted_8] : !qref.reg<?>, i64 -> !qref.bit
      qref.custom "CNOT"() %33, %34 : !qref.bit, !qref.bit
      return
    }
    func.func private @"__builtin__multi_rz_decomposition_MultiRZ{theta:[f64]}{wires:3}{}"(%arg0: !qref.reg<?>, %arg1: tensor<f64>, %arg2: tensor<3xi64>) attributes {frontend_name = "_multi_rz_decomposition", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"CNOT{}{wires:2}{}" = 4 : i64, "RZ{0:[f64]}{wires:1}{}" = 1 : i64}}, target_gate = "MultiRZ{theta:[f64]}{wires:3}{}"} {
      %c3 = arith.constant 3 : index
      %c = stablehlo.constant dense<3> : tensor<i64>
      %c_0 = stablehlo.constant dense<-1> : tensor<i64>
      %c_1 = stablehlo.constant dense<0> : tensor<i64>
      %c_2 = stablehlo.constant dense<2> : tensor<i64>
      %c_3 = stablehlo.constant dense<1> : tensor<i64>
      %c0 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      scf.for %arg3 = %c0 to %c2 step %c1 {
        %3 = arith.index_cast %arg3 : index to i64
        %from_elements = tensor.from_elements %3 : tensor<i64>
        %4 = stablehlo.multiply %c_0, %from_elements : tensor<i64>
        %5 = stablehlo.add %c_2, %4 : tensor<i64>
        %6 = stablehlo.compare  LT, %5, %c_1,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
        %7 = stablehlo.convert %5 : tensor<i64>
        %8 = stablehlo.add %7, %c : tensor<i64>
        %9 = stablehlo.select %6, %8, %5 : tensor<i1>, tensor<i64>
        %10 = stablehlo.dynamic_slice %arg2, %9, sizes = [1] : (tensor<3xi64>, tensor<i64>) -> tensor<1xi64>
        %11 = stablehlo.reshape %10 : (tensor<1xi64>) -> tensor<i64>
        %12 = stablehlo.subtract %5, %c_3 : tensor<i64>
        %13 = stablehlo.compare  LT, %12, %c_1,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
        %14 = stablehlo.convert %12 : tensor<i64>
        %15 = stablehlo.add %14, %c : tensor<i64>
        %16 = stablehlo.select %13, %15, %12 : tensor<i1>, tensor<i64>
        %17 = stablehlo.dynamic_slice %arg2, %16, sizes = [1] : (tensor<3xi64>, tensor<i64>) -> tensor<1xi64>
        %18 = stablehlo.reshape %17 : (tensor<1xi64>) -> tensor<i64>
        %extracted_5 = tensor.extract %11[] : tensor<i64>
        %19 = qref.get %arg0[%extracted_5] : !qref.reg<?>, i64 -> !qref.bit
        %extracted_6 = tensor.extract %18[] : tensor<i64>
        %20 = qref.get %arg0[%extracted_6] : !qref.reg<?>, i64 -> !qref.bit
        qref.custom "CNOT"() %19, %20 : !qref.bit, !qref.bit
      }
      %0 = stablehlo.slice %arg2 [0:1] : (tensor<3xi64>) -> tensor<1xi64>
      %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
      %extracted = tensor.extract %1[] : tensor<i64>
      %2 = qref.get %arg0[%extracted] : !qref.reg<?>, i64 -> !qref.bit
      %extracted_4 = tensor.extract %arg1[] : tensor<f64>
      qref.custom "RZ"(%extracted_4) %2 : !qref.bit
      scf.for %arg3 = %c1 to %c3 step %c1 {
        %3 = arith.index_cast %arg3 : index to i64
        %from_elements = tensor.from_elements %3 : tensor<i64>
        %4 = stablehlo.compare  LT, %from_elements, %c_1,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
        %5 = stablehlo.convert %from_elements : tensor<i64>
        %6 = stablehlo.add %5, %c : tensor<i64>
        %7 = stablehlo.select %4, %6, %from_elements : tensor<i1>, tensor<i64>
        %8 = stablehlo.dynamic_slice %arg2, %7, sizes = [1] : (tensor<3xi64>, tensor<i64>) -> tensor<1xi64>
        %9 = stablehlo.reshape %8 : (tensor<1xi64>) -> tensor<i64>
        %10 = stablehlo.subtract %from_elements, %c_3 : tensor<i64>
        %11 = stablehlo.compare  LT, %10, %c_1,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
        %12 = stablehlo.convert %10 : tensor<i64>
        %13 = stablehlo.add %12, %c : tensor<i64>
        %14 = stablehlo.select %11, %13, %10 : tensor<i1>, tensor<i64>
        %15 = stablehlo.dynamic_slice %arg2, %14, sizes = [1] : (tensor<3xi64>, tensor<i64>) -> tensor<1xi64>
        %16 = stablehlo.reshape %15 : (tensor<1xi64>) -> tensor<i64>
        %extracted_5 = tensor.extract %9[] : tensor<i64>
        %17 = qref.get %arg0[%extracted_5] : !qref.reg<?>, i64 -> !qref.bit
        %extracted_6 = tensor.extract %16[] : tensor<i64>
        %18 = qref.get %arg0[%extracted_6] : !qref.reg<?>, i64 -> !qref.bit
        qref.custom "CNOT"() %17, %18 : !qref.bit, !qref.bit
      }
      return
    }
    func.func private @"__builtin__multi_rz_decomposition_MultiRZ{theta:[f64]}{wires:1}{}"(%arg0: !qref.reg<?>, %arg1: tensor<f64>, %arg2: tensor<1xi64>) attributes {frontend_name = "_multi_rz_decomposition", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"RZ{0:[f64]}{wires:1}{}" = 1 : i64}}, target_gate = "MultiRZ{theta:[f64]}{wires:1}{}"} {
      %0 = stablehlo.slice %arg2 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
      %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
      %extracted = tensor.extract %1[] : tensor<i64>
      %2 = qref.get %arg0[%extracted] : !qref.reg<?>, i64 -> !qref.bit
      %extracted_0 = tensor.extract %arg1[] : tensor<f64>
      qref.custom "RZ"(%extracted_0) %2 : !qref.bit
      return
    }
    func.func private @"__builtin__adjoint_rot_Adjoint(Rot){0:[f64],1:[f64],2:[f64]}{wires:1}{}"(%arg0: !qref.reg<?>, %arg1: tensor<f64>, %arg2: tensor<f64>, %arg3: tensor<f64>, %arg4: tensor<1xi64>) attributes {frontend_name = "_adjoint_rot", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"Rot{0:[f64],1:[f64],2:[f64]}{wires:1}{}" = 1 : i64}}, target_gate = "Adjoint(Rot){0:[f64],1:[f64],2:[f64]}{wires:1}{}"} {
      %0 = stablehlo.slice %arg4 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
      %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
      %2 = stablehlo.negate %arg3 : tensor<f64>
      %3 = stablehlo.negate %arg2 : tensor<f64>
      %4 = stablehlo.negate %arg1 : tensor<f64>
      %extracted = tensor.extract %1[] : tensor<i64>
      %5 = qref.get %arg0[%extracted] : !qref.reg<?>, i64 -> !qref.bit
      %extracted_0 = tensor.extract %2[] : tensor<f64>
      %extracted_1 = tensor.extract %3[] : tensor<f64>
      %extracted_2 = tensor.extract %4[] : tensor<f64>
      qref.custom "Rot"(%extracted_0, %extracted_1, %extracted_2) %5 : !qref.bit
      return
    }
    func.func private @"__builtin__rot_to_rz_ry_rz_Adjoint(Rot){0:[f64],1:[f64],2:[f64]}{wires:1}{}"(%arg0: !qref.reg<?>, %arg1: tensor<f64>, %arg2: tensor<f64>, %arg3: tensor<f64>, %arg4: tensor<1xi64>) attributes {frontend_name = "_rot_to_rz_ry_rz", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"Adjoint(RY){0:[f64]}{wires:1}{}" = 1 : i64, "Adjoint(RZ){0:[f64]}{wires:1}{}" = 2 : i64}}, target_gate = "Adjoint(Rot){0:[f64],1:[f64],2:[f64]}{wires:1}{}"} {
      qref.adjoint {
        %0 = stablehlo.slice %arg4 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
        %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
        %extracted = tensor.extract %1[] : tensor<i64>
        %2 = qref.get %arg0[%extracted] : !qref.reg<?>, i64 -> !qref.bit
        %extracted_0 = tensor.extract %arg1[] : tensor<f64>
        qref.custom "RZ"(%extracted_0) %2 : !qref.bit
        %3 = stablehlo.slice %arg4 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
        %4 = stablehlo.reshape %3 : (tensor<1xi64>) -> tensor<i64>
        %extracted_1 = tensor.extract %4[] : tensor<i64>
        %5 = qref.get %arg0[%extracted_1] : !qref.reg<?>, i64 -> !qref.bit
        %extracted_2 = tensor.extract %arg2[] : tensor<f64>
        qref.custom "RY"(%extracted_2) %5 : !qref.bit
        %6 = stablehlo.slice %arg4 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
        %7 = stablehlo.reshape %6 : (tensor<1xi64>) -> tensor<i64>
        %extracted_3 = tensor.extract %7[] : tensor<i64>
        %8 = qref.get %arg0[%extracted_3] : !qref.reg<?>, i64 -> !qref.bit
        %extracted_4 = tensor.extract %arg3[] : tensor<f64>
        qref.custom "RZ"(%extracted_4) %8 : !qref.bit
      }
      return
    }
    func.func private @"__builtin_adjoint_rotation_Adjoint(RY){0:[f64]}{wires:1}{}"(%arg0: !qref.reg<?>, %arg1: tensor<f64>, %arg2: tensor<1xi64>) attributes {frontend_name = "adjoint_rotation", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"RY{0:[f64]}{wires:1}{}" = 1 : i64}}, target_gate = "Adjoint(RY){0:[f64]}{wires:1}{}"} {
      %0 = stablehlo.slice %arg2 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
      %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
      %2 = stablehlo.negate %arg1 : tensor<f64>
      %extracted = tensor.extract %1[] : tensor<i64>
      %3 = qref.get %arg0[%extracted] : !qref.reg<?>, i64 -> !qref.bit
      %extracted_0 = tensor.extract %2[] : tensor<f64>
      qref.custom "RY"(%extracted_0) %3 : !qref.bit
      return
    }
    func.func private @"__builtin__ry_to_rot_Adjoint(RY){0:[f64]}{wires:1}{}"(%arg0: !qref.reg<?>, %arg1: tensor<f64>, %arg2: tensor<1xi64>) attributes {frontend_name = "_ry_to_rot", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"Adjoint(Rot){0:[f64],1:[f64],2:[f64]}{wires:1}{}" = 1 : i64}}, target_gate = "Adjoint(RY){0:[f64]}{wires:1}{}"} {
      %c = stablehlo.constant dense<0> : tensor<i64>
      qref.adjoint {
        %0 = stablehlo.slice %arg2 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
        %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
        %extracted = tensor.extract %1[] : tensor<i64>
        %2 = qref.get %arg0[%extracted] : !qref.reg<?>, i64 -> !qref.bit
        %3 = stablehlo.convert %c : (tensor<i64>) -> tensor<f64>
        %extracted_0 = tensor.extract %3[] : tensor<f64>
        %extracted_1 = tensor.extract %arg1[] : tensor<f64>
        %4 = stablehlo.convert %c : (tensor<i64>) -> tensor<f64>
        %extracted_2 = tensor.extract %4[] : tensor<f64>
        qref.custom "Rot"(%extracted_0, %extracted_1, %extracted_2) %2 : !qref.bit
      }
      return
    }
    func.func private @"__builtin__ry_to_rz_rx_Adjoint(RY){0:[f64]}{wires:1}{}"(%arg0: !qref.reg<?>, %arg1: tensor<f64>, %arg2: tensor<1xi64>) attributes {frontend_name = "_ry_to_rz_rx", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"Adjoint(RX){0:[f64]}{wires:1}{}" = 1 : i64, "Adjoint(RZ){0:[f64]}{wires:1}{}" = 2 : i64}}, target_gate = "Adjoint(RY){0:[f64]}{wires:1}{}"} {
      %cst = arith.constant 1.5707963267948966 : f64
      %cst_0 = arith.constant -1.5707963267948966 : f64
      qref.adjoint {
        %0 = stablehlo.slice %arg2 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
        %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
        %extracted = tensor.extract %1[] : tensor<i64>
        %2 = qref.get %arg0[%extracted] : !qref.reg<?>, i64 -> !qref.bit
        qref.custom "RZ"(%cst_0) %2 : !qref.bit
        %3 = stablehlo.slice %arg2 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
        %4 = stablehlo.reshape %3 : (tensor<1xi64>) -> tensor<i64>
        %extracted_1 = tensor.extract %4[] : tensor<i64>
        %5 = qref.get %arg0[%extracted_1] : !qref.reg<?>, i64 -> !qref.bit
        %extracted_2 = tensor.extract %arg1[] : tensor<f64>
        qref.custom "RX"(%extracted_2) %5 : !qref.bit
        %6 = stablehlo.slice %arg2 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
        %7 = stablehlo.reshape %6 : (tensor<1xi64>) -> tensor<i64>
        %extracted_3 = tensor.extract %7[] : tensor<i64>
        %8 = qref.get %arg0[%extracted_3] : !qref.reg<?>, i64 -> !qref.bit
        qref.custom "RZ"(%cst) %8 : !qref.bit
      }
      return
    }
    func.func private @"__builtin__ry_to_ppr_Adjoint(RY){0:[f64]}{wires:1}{}"(%arg0: !qref.reg<?>, %arg1: tensor<f64>, %arg2: tensor<1xi64>) attributes {frontend_name = "_ry_to_ppr", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"Adjoint(PauliRot){theta:[f64]}{wires:1}{pauli_word = \22Y\22}" = 1 : i64}}, target_gate = "Adjoint(RY){0:[f64]}{wires:1}{}"} {
      qref.adjoint {
        %0 = stablehlo.slice %arg2 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
        %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
        %extracted = tensor.extract %1[] : tensor<i64>
        %2 = qref.get %arg0[%extracted] : !qref.reg<?>, i64 -> !qref.bit
        %extracted_0 = tensor.extract %arg1[] : tensor<f64>
        qref.paulirot ["Y"](%extracted_0) %2 : !qref.bit
      }
      return
    }
    func.func private @"__builtin__ry_to_rx_cliff_Adjoint(RY){0:[f64]}{wires:1}{}"(%arg0: !qref.reg<?>, %arg1: tensor<f64>, %arg2: tensor<1xi64>) attributes {frontend_name = "_ry_to_rx_cliff", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"Adjoint(RX){0:[f64]}{wires:1}{}" = 1 : i64, "Adjoint(S){}{wires:1}{}" = 1 : i64, "S{}{wires:1}{}" = 1 : i64}}, target_gate = "Adjoint(RY){0:[f64]}{wires:1}{}"} {
      qref.adjoint {
        %0 = stablehlo.slice %arg2 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
        %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
        %2 = stablehlo.slice %arg2 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
        %3 = stablehlo.reshape %2 : (tensor<1xi64>) -> tensor<i64>
        %4 = stablehlo.slice %arg2 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
        %5 = stablehlo.reshape %4 : (tensor<1xi64>) -> tensor<i64>
        %extracted = tensor.extract %1[] : tensor<i64>
        %6 = qref.get %arg0[%extracted] : !qref.reg<?>, i64 -> !qref.bit
        qref.custom "S"() %6 adj : !qref.bit
        %extracted_0 = tensor.extract %3[] : tensor<i64>
        %7 = qref.get %arg0[%extracted_0] : !qref.reg<?>, i64 -> !qref.bit
        %extracted_1 = tensor.extract %arg1[] : tensor<f64>
        qref.custom "RX"(%extracted_1) %7 : !qref.bit
        %extracted_2 = tensor.extract %5[] : tensor<i64>
        %8 = qref.get %arg0[%extracted_2] : !qref.reg<?>, i64 -> !qref.bit
        qref.custom "S"() %8 : !qref.bit
      }
      return
    }
    func.func private @"__builtin__ry_to_rz_cliff_Adjoint(RY){0:[f64]}{wires:1}{}"(%arg0: !qref.reg<?>, %arg1: tensor<f64>, %arg2: tensor<1xi64>) attributes {frontend_name = "_ry_to_rz_cliff", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"Adjoint(Hadamard){}{wires:1}{}" = 2 : i64, "Adjoint(RZ){0:[f64]}{wires:1}{}" = 1 : i64, "Adjoint(S){}{wires:1}{}" = 1 : i64, "S{}{wires:1}{}" = 1 : i64}}, target_gate = "Adjoint(RY){0:[f64]}{wires:1}{}"} {
      qref.adjoint {
        %0 = stablehlo.slice %arg2 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
        %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
        %2 = stablehlo.slice %arg2 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
        %3 = stablehlo.reshape %2 : (tensor<1xi64>) -> tensor<i64>
        %extracted = tensor.extract %3[] : tensor<i64>
        %4 = qref.get %arg0[%extracted] : !qref.reg<?>, i64 -> !qref.bit
        qref.custom "S"() %4 adj : !qref.bit
        %5 = stablehlo.slice %arg2 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
        %6 = stablehlo.reshape %5 : (tensor<1xi64>) -> tensor<i64>
        %extracted_0 = tensor.extract %6[] : tensor<i64>
        %7 = qref.get %arg0[%extracted_0] : !qref.reg<?>, i64 -> !qref.bit
        qref.custom "Hadamard"() %7 : !qref.bit
        %extracted_1 = tensor.extract %1[] : tensor<i64>
        %8 = qref.get %arg0[%extracted_1] : !qref.reg<?>, i64 -> !qref.bit
        %extracted_2 = tensor.extract %arg1[] : tensor<f64>
        qref.custom "RZ"(%extracted_2) %8 : !qref.bit
        %9 = stablehlo.slice %arg2 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
        %10 = stablehlo.reshape %9 : (tensor<1xi64>) -> tensor<i64>
        %extracted_3 = tensor.extract %10[] : tensor<i64>
        %11 = qref.get %arg0[%extracted_3] : !qref.reg<?>, i64 -> !qref.bit
        qref.custom "Hadamard"() %11 : !qref.bit
        %12 = stablehlo.slice %arg2 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
        %13 = stablehlo.reshape %12 : (tensor<1xi64>) -> tensor<i64>
        %extracted_4 = tensor.extract %13[] : tensor<i64>
        %14 = qref.get %arg0[%extracted_4] : !qref.reg<?>, i64 -> !qref.bit
        qref.custom "S"() %14 : !qref.bit
      }
      return
    }
    func.func private @"__builtin_adjoint_rotation_Adjoint(RX){0:[f64]}{wires:1}{}"(%arg0: !qref.reg<?>, %arg1: tensor<f64>, %arg2: tensor<1xi64>) attributes {frontend_name = "adjoint_rotation", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"RX{0:[f64]}{wires:1}{}" = 1 : i64}}, target_gate = "Adjoint(RX){0:[f64]}{wires:1}{}"} {
      %0 = stablehlo.slice %arg2 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
      %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
      %2 = stablehlo.negate %arg1 : tensor<f64>
      %extracted = tensor.extract %1[] : tensor<i64>
      %3 = qref.get %arg0[%extracted] : !qref.reg<?>, i64 -> !qref.bit
      %extracted_0 = tensor.extract %2[] : tensor<f64>
      qref.custom "RX"(%extracted_0) %3 : !qref.bit
      return
    }
    func.func private @"__builtin__rx_to_rot_Adjoint(RX){0:[f64]}{wires:1}{}"(%arg0: !qref.reg<?>, %arg1: tensor<f64>, %arg2: tensor<1xi64>) attributes {frontend_name = "_rx_to_rot", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"Adjoint(Rot){0:[f64],1:[f64],2:[f64]}{wires:1}{}" = 1 : i64}}, target_gate = "Adjoint(RX){0:[f64]}{wires:1}{}"} {
      %cst = arith.constant 10.995574287564276 : f64
      %cst_0 = arith.constant 1.5707963267948966 : f64
      qref.adjoint {
        %0 = stablehlo.slice %arg2 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
        %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
        %extracted = tensor.extract %1[] : tensor<i64>
        %2 = qref.get %arg0[%extracted] : !qref.reg<?>, i64 -> !qref.bit
        %extracted_1 = tensor.extract %arg1[] : tensor<f64>
        qref.custom "Rot"(%cst_0, %extracted_1, %cst) %2 : !qref.bit
      }
      return
    }
    func.func private @"__builtin__rx_to_rz_ry_Adjoint(RX){0:[f64]}{wires:1}{}"(%arg0: !qref.reg<?>, %arg1: tensor<f64>, %arg2: tensor<1xi64>) attributes {frontend_name = "_rx_to_rz_ry", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"Adjoint(RY){0:[f64]}{wires:1}{}" = 1 : i64, "Adjoint(RZ){0:[f64]}{wires:1}{}" = 2 : i64}}, target_gate = "Adjoint(RX){0:[f64]}{wires:1}{}"} {
      %cst = arith.constant -1.5707963267948966 : f64
      %cst_0 = arith.constant 1.5707963267948966 : f64
      qref.adjoint {
        %0 = stablehlo.slice %arg2 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
        %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
        %extracted = tensor.extract %1[] : tensor<i64>
        %2 = qref.get %arg0[%extracted] : !qref.reg<?>, i64 -> !qref.bit
        qref.custom "RZ"(%cst_0) %2 : !qref.bit
        %3 = stablehlo.slice %arg2 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
        %4 = stablehlo.reshape %3 : (tensor<1xi64>) -> tensor<i64>
        %extracted_1 = tensor.extract %4[] : tensor<i64>
        %5 = qref.get %arg0[%extracted_1] : !qref.reg<?>, i64 -> !qref.bit
        %extracted_2 = tensor.extract %arg1[] : tensor<f64>
        qref.custom "RY"(%extracted_2) %5 : !qref.bit
        %6 = stablehlo.slice %arg2 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
        %7 = stablehlo.reshape %6 : (tensor<1xi64>) -> tensor<i64>
        %extracted_3 = tensor.extract %7[] : tensor<i64>
        %8 = qref.get %arg0[%extracted_3] : !qref.reg<?>, i64 -> !qref.bit
        qref.custom "RZ"(%cst) %8 : !qref.bit
      }
      return
    }
    func.func private @"__builtin__rx_to_ppr_Adjoint(RX){0:[f64]}{wires:1}{}"(%arg0: !qref.reg<?>, %arg1: tensor<f64>, %arg2: tensor<1xi64>) attributes {frontend_name = "_rx_to_ppr", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"Adjoint(PauliRot){theta:[f64]}{wires:1}{pauli_word = \22X\22}" = 1 : i64}}, target_gate = "Adjoint(RX){0:[f64]}{wires:1}{}"} {
      qref.adjoint {
        %0 = stablehlo.slice %arg2 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
        %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
        %extracted = tensor.extract %1[] : tensor<i64>
        %2 = qref.get %arg0[%extracted] : !qref.reg<?>, i64 -> !qref.bit
        %extracted_0 = tensor.extract %arg1[] : tensor<f64>
        qref.paulirot ["X"](%extracted_0) %2 : !qref.bit
      }
      return
    }
    func.func private @"__builtin__rx_to_ry_cliff_Adjoint(RX){0:[f64]}{wires:1}{}"(%arg0: !qref.reg<?>, %arg1: tensor<f64>, %arg2: tensor<1xi64>) attributes {frontend_name = "_rx_to_ry_cliff", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"Adjoint(RY){0:[f64]}{wires:1}{}" = 1 : i64, "Adjoint(S){}{wires:1}{}" = 1 : i64, "S{}{wires:1}{}" = 1 : i64}}, target_gate = "Adjoint(RX){0:[f64]}{wires:1}{}"} {
      qref.adjoint {
        %0 = stablehlo.slice %arg2 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
        %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
        %2 = stablehlo.slice %arg2 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
        %3 = stablehlo.reshape %2 : (tensor<1xi64>) -> tensor<i64>
        %extracted = tensor.extract %1[] : tensor<i64>
        %4 = qref.get %arg0[%extracted] : !qref.reg<?>, i64 -> !qref.bit
        qref.custom "S"() %4 : !qref.bit
        %extracted_0 = tensor.extract %3[] : tensor<i64>
        %5 = qref.get %arg0[%extracted_0] : !qref.reg<?>, i64 -> !qref.bit
        %extracted_1 = tensor.extract %arg1[] : tensor<f64>
        qref.custom "RY"(%extracted_1) %5 : !qref.bit
        %extracted_2 = tensor.extract %1[] : tensor<i64>
        %6 = qref.get %arg0[%extracted_2] : !qref.reg<?>, i64 -> !qref.bit
        qref.custom "S"() %6 adj : !qref.bit
      }
      return
    }
    func.func private @"__builtin__rx_to_rz_cliff_Adjoint(RX){0:[f64]}{wires:1}{}"(%arg0: !qref.reg<?>, %arg1: tensor<f64>, %arg2: tensor<1xi64>) attributes {frontend_name = "_rx_to_rz_cliff", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"Adjoint(Hadamard){}{wires:1}{}" = 2 : i64, "Adjoint(RZ){0:[f64]}{wires:1}{}" = 1 : i64}}, target_gate = "Adjoint(RX){0:[f64]}{wires:1}{}"} {
      qref.adjoint {
        %0 = stablehlo.slice %arg2 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
        %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
        %2 = stablehlo.slice %arg2 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
        %3 = stablehlo.reshape %2 : (tensor<1xi64>) -> tensor<i64>
        %4 = stablehlo.slice %arg2 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
        %5 = stablehlo.reshape %4 : (tensor<1xi64>) -> tensor<i64>
        %extracted = tensor.extract %1[] : tensor<i64>
        %6 = qref.get %arg0[%extracted] : !qref.reg<?>, i64 -> !qref.bit
        qref.custom "Hadamard"() %6 : !qref.bit
        %extracted_0 = tensor.extract %3[] : tensor<i64>
        %7 = qref.get %arg0[%extracted_0] : !qref.reg<?>, i64 -> !qref.bit
        %extracted_1 = tensor.extract %arg1[] : tensor<f64>
        qref.custom "RZ"(%extracted_1) %7 : !qref.bit
        %extracted_2 = tensor.extract %5[] : tensor<i64>
        %8 = qref.get %arg0[%extracted_2] : !qref.reg<?>, i64 -> !qref.bit
        qref.custom "Hadamard"() %8 : !qref.bit
      }
      return
    }
    func.func private @"__builtin_adjoint_rotation_Adjoint(PauliRot){theta:[f64]}{wires:1}{pauli_word = \22Z\22}"(%arg0: !qref.reg<?>, %arg1: tensor<f64>, %arg2: tensor<1xi64>) attributes {frontend_name = "adjoint_rotation", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"PauliRot{theta:[f64]}{wires:1}{pauli_word = \22Z\22}" = 1 : i64}}, target_gate = "Adjoint(PauliRot){theta:[f64]}{wires:1}{pauli_word = \22Z\22}"} {
      %0 = stablehlo.slice %arg2 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
      %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
      %2 = stablehlo.negate %arg1 : tensor<f64>
      %extracted = tensor.extract %1[] : tensor<i64>
      %3 = qref.get %arg0[%extracted] : !qref.reg<?>, i64 -> !qref.bit
      %extracted_0 = tensor.extract %2[] : tensor<f64>
      qref.paulirot ["Z"](%extracted_0) %3 : !qref.bit
      return
    }
    func.func private @"__builtin__pauli_rot_decomposition_Adjoint(PauliRot){theta:[f64]}{wires:1}{pauli_word = \22Z\22}"(%arg0: !qref.reg<?>, %arg1: tensor<f64>, %arg2: tensor<1xi64>) attributes {frontend_name = "_pauli_rot_decomposition", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"Adjoint(MultiRZ){theta:[f64]}{wires:1}{}" = 1 : i64}}, target_gate = "Adjoint(PauliRot){theta:[f64]}{wires:1}{pauli_word = \22Z\22}"} {
      qref.adjoint {
        %0 = stablehlo.slice %arg2 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
        %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
        %extracted = tensor.extract %1[] : tensor<i64>
        %2 = qref.get %arg0[%extracted] : !qref.reg<?>, i64 -> !qref.bit
        %extracted_0 = tensor.extract %arg1[] : tensor<f64>
        qref.multirz(%extracted_0) %2 : !qref.bit
      }
      return
    }
    func.func private @"__builtin_decompose_to_base_Adjoint(Hadamard){}{wires:1}{}"(%arg0: !qref.reg<?>, %arg1: tensor<1xi64>) attributes {frontend_name = "decompose_to_base", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"Hadamard{}{wires:1}{}" = 1 : i64}}, target_gate = "Adjoint(Hadamard){}{wires:1}{}"} {
      %0 = stablehlo.slice %arg1 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
      %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
      %extracted = tensor.extract %1[] : tensor<i64>
      %2 = qref.get %arg0[%extracted] : !qref.reg<?>, i64 -> !qref.bit
      qref.custom "Hadamard"() %2 : !qref.bit
      return
    }
    func.func private @"__builtin__hadamard_to_rz_rx_Adjoint(Hadamard){}{wires:1}{}"(%arg0: !qref.reg<?>, %arg1: tensor<1xi64>) attributes {frontend_name = "_hadamard_to_rz_rx", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"Adjoint(GlobalPhase){phi:[f64]}{}{}" = 1 : i64, "Adjoint(RX){0:[f64]}{wires:1}{}" = 1 : i64, "Adjoint(RZ){0:[f64]}{wires:1}{}" = 2 : i64}}, target_gate = "Adjoint(Hadamard){}{wires:1}{}"} {
      %cst = arith.constant -1.5707963267948966 : f64
      %cst_0 = arith.constant 1.5707963267948966 : f64
      qref.adjoint {
        %0 = stablehlo.slice %arg1 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
        %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
        %extracted = tensor.extract %1[] : tensor<i64>
        %2 = qref.get %arg0[%extracted] : !qref.reg<?>, i64 -> !qref.bit
        qref.custom "RZ"(%cst_0) %2 : !qref.bit
        %3 = stablehlo.slice %arg1 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
        %4 = stablehlo.reshape %3 : (tensor<1xi64>) -> tensor<i64>
        %extracted_1 = tensor.extract %4[] : tensor<i64>
        %5 = qref.get %arg0[%extracted_1] : !qref.reg<?>, i64 -> !qref.bit
        qref.custom "RX"(%cst_0) %5 : !qref.bit
        %6 = stablehlo.slice %arg1 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
        %7 = stablehlo.reshape %6 : (tensor<1xi64>) -> tensor<i64>
        %extracted_2 = tensor.extract %7[] : tensor<i64>
        %8 = qref.get %arg0[%extracted_2] : !qref.reg<?>, i64 -> !qref.bit
        qref.custom "RZ"(%cst_0) %8 : !qref.bit
        qref.gphase(%cst)
      }
      return
    }
    func.func private @"__builtin__hadamard_to_rz_ry_Adjoint(Hadamard){}{wires:1}{}"(%arg0: !qref.reg<?>, %arg1: tensor<1xi64>) attributes {frontend_name = "_hadamard_to_rz_ry", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"Adjoint(GlobalPhase){phi:[f64]}{}{}" = 1 : i64, "Adjoint(RY){0:[f64]}{wires:1}{}" = 1 : i64, "Adjoint(RZ){0:[f64]}{wires:1}{}" = 1 : i64}}, target_gate = "Adjoint(Hadamard){}{wires:1}{}"} {
      %cst = arith.constant -1.5707963267948966 : f64
      %cst_0 = arith.constant 1.5707963267948966 : f64
      %cst_1 = arith.constant 3.1415926535897931 : f64
      qref.adjoint {
        %0 = stablehlo.slice %arg1 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
        %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
        %extracted = tensor.extract %1[] : tensor<i64>
        %2 = qref.get %arg0[%extracted] : !qref.reg<?>, i64 -> !qref.bit
        qref.custom "RZ"(%cst_1) %2 : !qref.bit
        %3 = stablehlo.slice %arg1 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
        %4 = stablehlo.reshape %3 : (tensor<1xi64>) -> tensor<i64>
        %extracted_2 = tensor.extract %4[] : tensor<i64>
        %5 = qref.get %arg0[%extracted_2] : !qref.reg<?>, i64 -> !qref.bit
        qref.custom "RY"(%cst_0) %5 : !qref.bit
        qref.gphase(%cst)
      }
      return
    }
    func.func private @"__builtin_adjoint_rotation_Adjoint(PauliRot){theta:[f64]}{wires:1}{pauli_word = \22Y\22}"(%arg0: !qref.reg<?>, %arg1: tensor<f64>, %arg2: tensor<1xi64>) attributes {frontend_name = "adjoint_rotation", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"PauliRot{theta:[f64]}{wires:1}{pauli_word = \22Y\22}" = 1 : i64}}, target_gate = "Adjoint(PauliRot){theta:[f64]}{wires:1}{pauli_word = \22Y\22}"} {
      %0 = stablehlo.slice %arg2 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
      %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
      %2 = stablehlo.negate %arg1 : tensor<f64>
      %extracted = tensor.extract %1[] : tensor<i64>
      %3 = qref.get %arg0[%extracted] : !qref.reg<?>, i64 -> !qref.bit
      %extracted_0 = tensor.extract %2[] : tensor<f64>
      qref.paulirot ["Y"](%extracted_0) %3 : !qref.bit
      return
    }
    func.func private @"__builtin__pauli_rot_decomposition_Adjoint(PauliRot){theta:[f64]}{wires:1}{pauli_word = \22Y\22}"(%arg0: !qref.reg<?>, %arg1: tensor<f64>, %arg2: tensor<1xi64>) attributes {frontend_name = "_pauli_rot_decomposition", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"Adjoint(MultiRZ){theta:[f64]}{wires:1}{}" = 1 : i64, "Adjoint(RX){0:[f64]}{wires:1}{}" = 2 : i64}}, target_gate = "Adjoint(PauliRot){theta:[f64]}{wires:1}{pauli_word = \22Y\22}"} {
      %cst = arith.constant -1.5707963267948966 : f64
      %cst_0 = arith.constant 1.5707963267948966 : f64
      qref.adjoint {
        %0 = stablehlo.slice %arg2 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
        %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
        %extracted = tensor.extract %1[] : tensor<i64>
        %2 = qref.get %arg0[%extracted] : !qref.reg<?>, i64 -> !qref.bit
        qref.custom "RX"(%cst_0) %2 : !qref.bit
        %extracted_1 = tensor.extract %1[] : tensor<i64>
        %3 = qref.get %arg0[%extracted_1] : !qref.reg<?>, i64 -> !qref.bit
        %extracted_2 = tensor.extract %arg1[] : tensor<f64>
        qref.multirz(%extracted_2) %3 : !qref.bit
        %extracted_3 = tensor.extract %1[] : tensor<i64>
        %4 = qref.get %arg0[%extracted_3] : !qref.reg<?>, i64 -> !qref.bit
        qref.custom "RX"(%cst) %4 : !qref.bit
      }
      return
    }
    func.func private @"__builtin_adjoint_rotation_Adjoint(PauliRot){theta:[f64]}{wires:1}{pauli_word = \22X\22}"(%arg0: !qref.reg<?>, %arg1: tensor<f64>, %arg2: tensor<1xi64>) attributes {frontend_name = "adjoint_rotation", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"PauliRot{theta:[f64]}{wires:1}{pauli_word = \22X\22}" = 1 : i64}}, target_gate = "Adjoint(PauliRot){theta:[f64]}{wires:1}{pauli_word = \22X\22}"} {
      %0 = stablehlo.slice %arg2 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
      %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
      %2 = stablehlo.negate %arg1 : tensor<f64>
      %extracted = tensor.extract %1[] : tensor<i64>
      %3 = qref.get %arg0[%extracted] : !qref.reg<?>, i64 -> !qref.bit
      %extracted_0 = tensor.extract %2[] : tensor<f64>
      qref.paulirot ["X"](%extracted_0) %3 : !qref.bit
      return
    }
    func.func private @"__builtin__pauli_rot_decomposition_Adjoint(PauliRot){theta:[f64]}{wires:1}{pauli_word = \22X\22}"(%arg0: !qref.reg<?>, %arg1: tensor<f64>, %arg2: tensor<1xi64>) attributes {frontend_name = "_pauli_rot_decomposition", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"Adjoint(Hadamard){}{wires:1}{}" = 2 : i64, "Adjoint(MultiRZ){theta:[f64]}{wires:1}{}" = 1 : i64}}, target_gate = "Adjoint(PauliRot){theta:[f64]}{wires:1}{pauli_word = \22X\22}"} {
      qref.adjoint {
        %0 = stablehlo.slice %arg2 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
        %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
        %extracted = tensor.extract %1[] : tensor<i64>
        %2 = qref.get %arg0[%extracted] : !qref.reg<?>, i64 -> !qref.bit
        qref.custom "Hadamard"() %2 : !qref.bit
        %extracted_0 = tensor.extract %1[] : tensor<i64>
        %3 = qref.get %arg0[%extracted_0] : !qref.reg<?>, i64 -> !qref.bit
        %extracted_1 = tensor.extract %arg1[] : tensor<f64>
        qref.multirz(%extracted_1) %3 : !qref.bit
        %extracted_2 = tensor.extract %1[] : tensor<i64>
        %4 = qref.get %arg0[%extracted_2] : !qref.reg<?>, i64 -> !qref.bit
        qref.custom "Hadamard"() %4 : !qref.bit
      }
      return
    }
    func.func private @"__builtin_adjoint_rotation_Adjoint(MultiRZ){theta:[f64]}{wires:1}{}"(%arg0: !qref.reg<?>, %arg1: tensor<f64>, %arg2: tensor<1xi64>) attributes {frontend_name = "adjoint_rotation", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"MultiRZ{theta:[f64]}{wires:1}{}" = 1 : i64}}, target_gate = "Adjoint(MultiRZ){theta:[f64]}{wires:1}{}"} {
      %0 = stablehlo.slice %arg2 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
      %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
      %2 = stablehlo.negate %arg1 : tensor<f64>
      %extracted = tensor.extract %1[] : tensor<i64>
      %3 = qref.get %arg0[%extracted] : !qref.reg<?>, i64 -> !qref.bit
      %extracted_0 = tensor.extract %2[] : tensor<f64>
      qref.multirz(%extracted_0) %3 : !qref.bit
      return
    }
    func.func private @"__builtin__multi_rz_decomposition_Adjoint(MultiRZ){theta:[f64]}{wires:1}{}"(%arg0: !qref.reg<?>, %arg1: tensor<f64>, %arg2: tensor<1xi64>) attributes {frontend_name = "_multi_rz_decomposition", llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"Adjoint(RZ){0:[f64]}{wires:1}{}" = 1 : i64}}, target_gate = "Adjoint(MultiRZ){theta:[f64]}{wires:1}{}"} {
      qref.adjoint {
        %0 = stablehlo.slice %arg2 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
        %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
        %extracted = tensor.extract %1[] : tensor<i64>
        %2 = qref.get %arg0[%extracted] : !qref.reg<?>, i64 -> !qref.bit
        %extracted_0 = tensor.extract %arg1[] : tensor<f64>
        qref.custom "RZ"(%extracted_0) %2 : !qref.bit
      }
      return
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