module @circuit {
  func.func public @jit_circuit(%arg0: tensor<f64>) -> tensor<f64> attributes {llvm.emit_c_interface} {
    %0 = call @circuit_0(%arg0) : (tensor<f64>) -> tensor<f64>
    return %0 : tensor<f64>
  }
  func.func public @circuit_0(%arg0: tensor<f64>) -> tensor<f64> attributes {diff_method = "parameter-shift", llvm.linkage = #llvm.linkage<internal>, qnode} {
    %c0_i64 = arith.constant 0 : i64
    quantum.device shots(%c0_i64) ["/home/ubuntu/catalyst/frontend/catalyst/utils/../../../runtime/build/lib/librtd_openqasm.so", "OpenQasmDevice", "{'device_type': 'braket.local.qubit', 'backend': 'braket_sv'}"]
    %0 = quantum.alloc( 2) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %out_qubits = quantum.custom "Hadamard"() %1 : !quantum.bit
    %extracted = tensor.extract %arg0[] : tensor<f64>
    %out_qubits_0 = quantum.custom "RX"(%extracted) %out_qubits : !quantum.bit
    %2 = quantum.namedobs %out_qubits_0[ PauliZ] : !quantum.obs
    %3 = quantum.expval %2 : f64
    %from_elements = tensor.from_elements %3 : tensor<f64>
    %4 = quantum.insert %0[ 0], %out_qubits_0 : !quantum.reg, !quantum.bit
    quantum.dealloc %4 : !quantum.reg
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