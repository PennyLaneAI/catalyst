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
    %4 = stablehlo.slice %cst [1:2] : (tensor<3xi64>) -> tensor<1xi64>
    %5 = stablehlo.reshape %4 : (tensor<1xi64>) -> tensor<i64>
    %extracted_0 = tensor.extract %5[] : tensor<i64>
    %6 = quantum.extract %0[%extracted_0] : !quantum.reg -> !quantum.bit
    %out_qubits_1:2 = quantum.custom "CNOT"() %6, %out_qubits : !quantum.bit, !quantum.bit
    %7 = quantum.insert %0[%extracted_0], %out_qubits_1#0 : !quantum.reg, !quantum.bit
    %out_qubits_2 = quantum.custom "T"() %out_qubits_1#1 adj : !quantum.bit
    %8 = stablehlo.slice %cst [0:1] : (tensor<3xi64>) -> tensor<1xi64>
    %9 = stablehlo.reshape %8 : (tensor<1xi64>) -> tensor<i64>
    %extracted_3 = tensor.extract %9[] : tensor<i64>
    %10 = quantum.extract %7[%extracted_3] : !quantum.reg -> !quantum.bit
    %out_qubits_4:2 = quantum.custom "CNOT"() %10, %out_qubits_2 : !quantum.bit, !quantum.bit
    %11 = quantum.insert %7[%extracted_3], %out_qubits_4#0 : !quantum.reg, !quantum.bit
    %out_qubits_5 = quantum.custom "T"() %out_qubits_4#1 : !quantum.bit
    %12 = quantum.extract %11[%extracted_0] : !quantum.reg -> !quantum.bit
    %out_qubits_6:2 = quantum.custom "CNOT"() %12, %out_qubits_5 : !quantum.bit, !quantum.bit
    %13 = quantum.insert %11[%extracted_0], %out_qubits_6#0 : !quantum.reg, !quantum.bit
    %out_qubits_7 = quantum.custom "T"() %out_qubits_6#1 adj : !quantum.bit
    %14 = quantum.extract %13[%extracted_3] : !quantum.reg -> !quantum.bit
    %out_qubits_8:2 = quantum.custom "CNOT"() %14, %out_qubits_7 : !quantum.bit, !quantum.bit
    %15 = quantum.insert %13[%extracted_3], %out_qubits_8#0 : !quantum.reg, !quantum.bit
    %out_qubits_9 = quantum.custom "T"() %out_qubits_8#1 : !quantum.bit
    %16 = quantum.insert %15[%extracted], %out_qubits_9 : !quantum.reg, !quantum.bit
    %17 = quantum.extract %16[%extracted_0] : !quantum.reg -> !quantum.bit
    %out_qubits_10 = quantum.custom "T"() %17 : !quantum.bit
    %18 = quantum.extract %16[%extracted_3] : !quantum.reg -> !quantum.bit
    %out_qubits_11:2 = quantum.custom "CNOT"() %18, %out_qubits_10 : !quantum.bit, !quantum.bit
    %19 = quantum.insert %16[%extracted_3], %out_qubits_11#0 : !quantum.reg, !quantum.bit
    %20 = quantum.insert %19[%extracted_0], %out_qubits_11#1 : !quantum.reg, !quantum.bit
    %21 = quantum.extract %20[%extracted] : !quantum.reg -> !quantum.bit
    %out_qubits_12 = quantum.custom "Hadamard"() %21 : !quantum.bit
    %22 = quantum.insert %20[%extracted], %out_qubits_12 : !quantum.reg, !quantum.bit
    %23 = quantum.extract %22[%extracted_3] : !quantum.reg -> !quantum.bit
    %out_qubits_13 = quantum.custom "T"() %23 : !quantum.bit
    %24 = quantum.extract %22[%extracted_0] : !quantum.reg -> !quantum.bit
    %out_qubits_14 = quantum.custom "T"() %24 adj : !quantum.bit
    %out_qubits_15:2 = quantum.custom "CNOT"() %out_qubits_13, %out_qubits_14 : !quantum.bit, !quantum.bit
    %25 = quantum.insert %22[%extracted_3], %out_qubits_15#0 : !quantum.reg, !quantum.bit
    %26 = quantum.insert %25[%extracted_0], %out_qubits_15#1 : !quantum.reg, !quantum.bit
    %27 = quantum.extract %26[ 0] : !quantum.reg -> !quantum.bit
    %28 = quantum.namedobs %27[ PauliZ] : !quantum.obs
    %29 = quantum.insert %26[ 0], %27 : !quantum.reg, !quantum.bit
    %30 = quantum.expval %28 : f64
    %from_elements = tensor.from_elements %30 : tensor<f64>
    quantum.dealloc %29 : !quantum.reg
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