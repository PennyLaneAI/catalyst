module @test_cancel_inverses_keep_original_workflow1 {
  func.func public @jit_test_cancel_inverses_keep_original_workflow1() -> tensor<f64> attributes {llvm.emit_c_interface} {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %0 = call @f_0(%cst) : (tensor<f64>) -> tensor<f64>
    return %0 : tensor<f64>
  }
  func.func public @f_0(%arg0: tensor<f64>) -> tensor<f64> attributes {diff_method = "adjoint", llvm.linkage = #llvm.linkage<internal>, qnode} {
    %c0_i64 = arith.constant 0 : i64
    quantum.device shots(%c0_i64) ["/Users/christina/Prog/catalyst_env/lib/python3.12/site-packages/pennylane_lightning/liblightning_qubit_catalyst.dylib", "LightningSimulator", "{'mcmc': False, 'num_burnin': 0, 'kernel_name': None}"]
    %0 = quantum.alloc( 1) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %extracted = tensor.extract %arg0[] : tensor<f64>
    %out_qubits = quantum.custom "RX"(%extracted) %1 : !quantum.bit
    %2 = quantum.namedobs %out_qubits[ PauliZ] : !quantum.obs
    %3 = quantum.expval %2 : f64
    %from_elements = tensor.from_elements %3 : tensor<f64>
    %4 = quantum.insert %0[ 0], %out_qubits : !quantum.reg, !quantum.bit
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