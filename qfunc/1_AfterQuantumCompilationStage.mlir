module @qfunc {
  func.func public @jit_qfunc() -> tensor<f64> attributes {llvm.emit_c_interface} {
    %0 = call @qfunc_0() : () -> tensor<f64>
    return %0 : tensor<f64>
  }
  func.func public @qfunc_0() -> tensor<f64> attributes {diff_method = "parameter-shift", llvm.linkage = #llvm.linkage<internal>, qnode} {
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
  func.func @setup() {
    quantum.init
    return
  }
  func.func @teardown() {
    quantum.finalize
    return
  }
}