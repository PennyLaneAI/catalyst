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
    %extracted_slice = tensor.extract_slice %cst[2] [1] [1] : tensor<3xi64> to tensor<1xi64>
    %collapsed = tensor.collapse_shape %extracted_slice [] : tensor<1xi64> into tensor<i64>
    %extracted = tensor.extract %collapsed[] : tensor<i64>
    %1 = quantum.extract %0[%extracted] : !quantum.reg -> !quantum.bit
    %out_qubits = quantum.custom "Hadamard"() %1 : !quantum.bit
    %extracted_slice_0 = tensor.extract_slice %cst[1] [1] [1] : tensor<3xi64> to tensor<1xi64>
    %collapsed_1 = tensor.collapse_shape %extracted_slice_0 [] : tensor<1xi64> into tensor<i64>
    %extracted_2 = tensor.extract %collapsed_1[] : tensor<i64>
    %2 = quantum.extract %0[%extracted_2] : !quantum.reg -> !quantum.bit
    %out_qubits_3:2 = quantum.custom "CNOT"() %2, %out_qubits : !quantum.bit, !quantum.bit
    %3 = quantum.insert %0[%extracted_2], %out_qubits_3#0 : !quantum.reg, !quantum.bit
    %out_qubits_4 = quantum.custom "T"() %out_qubits_3#1 adj : !quantum.bit
    %extracted_slice_5 = tensor.extract_slice %cst[0] [1] [1] : tensor<3xi64> to tensor<1xi64>
    %collapsed_6 = tensor.collapse_shape %extracted_slice_5 [] : tensor<1xi64> into tensor<i64>
    %extracted_7 = tensor.extract %collapsed_6[] : tensor<i64>
    %4 = quantum.extract %3[%extracted_7] : !quantum.reg -> !quantum.bit
    %out_qubits_8:2 = quantum.custom "CNOT"() %4, %out_qubits_4 : !quantum.bit, !quantum.bit
    %5 = quantum.insert %3[%extracted_7], %out_qubits_8#0 : !quantum.reg, !quantum.bit
    %out_qubits_9 = quantum.custom "T"() %out_qubits_8#1 : !quantum.bit
    %6 = quantum.extract %5[%extracted_2] : !quantum.reg -> !quantum.bit
    %out_qubits_10:2 = quantum.custom "CNOT"() %6, %out_qubits_9 : !quantum.bit, !quantum.bit
    %7 = quantum.insert %5[%extracted_2], %out_qubits_10#0 : !quantum.reg, !quantum.bit
    %out_qubits_11 = quantum.custom "T"() %out_qubits_10#1 adj : !quantum.bit
    %8 = quantum.extract %7[%extracted_7] : !quantum.reg -> !quantum.bit
    %out_qubits_12:2 = quantum.custom "CNOT"() %8, %out_qubits_11 : !quantum.bit, !quantum.bit
    %9 = quantum.insert %7[%extracted_7], %out_qubits_12#0 : !quantum.reg, !quantum.bit
    %out_qubits_13 = quantum.custom "T"() %out_qubits_12#1 : !quantum.bit
    %10 = quantum.insert %9[%extracted], %out_qubits_13 : !quantum.reg, !quantum.bit
    %11 = quantum.extract %10[%extracted_2] : !quantum.reg -> !quantum.bit
    %out_qubits_14 = quantum.custom "T"() %11 : !quantum.bit
    %12 = quantum.extract %10[%extracted_7] : !quantum.reg -> !quantum.bit
    %out_qubits_15:2 = quantum.custom "CNOT"() %12, %out_qubits_14 : !quantum.bit, !quantum.bit
    %13 = quantum.insert %10[%extracted_7], %out_qubits_15#0 : !quantum.reg, !quantum.bit
    %14 = quantum.insert %13[%extracted_2], %out_qubits_15#1 : !quantum.reg, !quantum.bit
    %15 = quantum.extract %14[%extracted] : !quantum.reg -> !quantum.bit
    %out_qubits_16 = quantum.custom "Hadamard"() %15 : !quantum.bit
    %16 = quantum.insert %14[%extracted], %out_qubits_16 : !quantum.reg, !quantum.bit
    %17 = quantum.extract %16[%extracted_7] : !quantum.reg -> !quantum.bit
    %out_qubits_17 = quantum.custom "T"() %17 : !quantum.bit
    %18 = quantum.extract %16[%extracted_2] : !quantum.reg -> !quantum.bit
    %out_qubits_18 = quantum.custom "T"() %18 adj : !quantum.bit
    %out_qubits_19:2 = quantum.custom "CNOT"() %out_qubits_17, %out_qubits_18 : !quantum.bit, !quantum.bit
    %19 = quantum.insert %16[%extracted_7], %out_qubits_19#0 : !quantum.reg, !quantum.bit
    %20 = quantum.insert %19[%extracted_2], %out_qubits_19#1 : !quantum.reg, !quantum.bit
    %21 = quantum.extract %20[ 0] : !quantum.reg -> !quantum.bit
    %22 = quantum.namedobs %21[ PauliZ] : !quantum.obs
    %23 = quantum.insert %20[ 0], %21 : !quantum.reg, !quantum.bit
    %24 = quantum.expval %22 : f64
    %from_elements = tensor.from_elements %24 : tensor<f64>
    quantum.dealloc %23 : !quantum.reg
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