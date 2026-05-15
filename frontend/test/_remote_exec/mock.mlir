module @circuit attributes {catalyst.runtime_artifacts = ["/Users/mehrdad.malek/catalyst/libxor_ref.dylib"]} {
  func.func public @jit_circuit(%arg0: tensor<f64>, %arg1: tensor<2x2xf64>) -> (tensor<f64>, tensor<4xf64>, tensor<1xi32>) attributes {llvm.emit_c_interface} {
    %0:2 = catalyst.launch_kernel @module_circuit::@circuit(%arg0, %arg1) : (tensor<f64>, tensor<2x2xf64>) -> (tensor<f64>, tensor<4xf64>)
    %c = stablehlo.constant dense<[1, 2, 3, 4, 5, 6]> : tensor<6xi64>
    %4 = catalyst.custom_call fn("xor_reduce") (%c) : (tensor<6xi64>) -> tensor<1xi32>
    return %0#0, %0#1, %4 : tensor<f64>, tensor<4xf64>, tensor<1xi32>
  }
module @module_circuit attributes {catalyst.target = {address = "127.0.0.1:1234", backend = "my-backend", triple = "aarch64-unknown-linux-gnu"}} {
      module attributes {transform.with_named_sequence} {
      transform.named_sequence @__transform_main(%arg0: !transform.op<"builtin.module">) {
        transform.yield 
      }
    }
    func.func public @circuit(%arg0: tensor<f64>, %arg1: tensor<2x2xf64>) -> (tensor<f64>, tensor<4xf64>) attributes {diff_method = "parameter-shift", llvm.linkage = #llvm.linkage<internal>, quantum.node} {
      %c0_i64 = arith.constant 0 : i64
      %0 = stablehlo.slice %arg1 [0:1, 0:1] : (tensor<2x2xf64>) -> tensor<1x1xf64>
      %1 = stablehlo.reshape %0 : (tensor<1x1xf64>) -> tensor<f64>
      quantum.device shots(%c0_i64) ["/Users/mehrdad.malek/catalyst/.venv/python3.12/lib/python3.12/site-packages/pennylane_lightning/liblightning_qubit_catalyst.dylib", "LightningSimulator", "{'mcmc': False, 'num_burnin': 0, 'kernel_name': None}"]
      %2 = quantum.alloc( 2) : !quantum.reg
      %3 = quantum.extract %2[ 0] : !quantum.reg -> !quantum.bit
      %extracted = tensor.extract %arg0[] : tensor<f64>
      %out_qubits = quantum.custom "RX"(%extracted) %3 : !quantum.bit
      %extracted_0 = tensor.extract %1[] : tensor<f64>
      %out_qubits_1 = quantum.custom "RY"(%extracted_0) %out_qubits : !quantum.bit
      %4 = stablehlo.slice %arg1 [1:2, 1:2] : (tensor<2x2xf64>) -> tensor<1x1xf64>
      %5 = stablehlo.reshape %4 : (tensor<1x1xf64>) -> tensor<f64>
      %6 = quantum.extract %2[ 1] : !quantum.reg -> !quantum.bit
      %extracted_2 = tensor.extract %5[] : tensor<f64>
      %out_qubits_3 = quantum.custom "RZ"(%extracted_2) %6 : !quantum.bit
      %7 = quantum.namedobs %out_qubits_1[ PauliZ] : !quantum.obs
      %8 = quantum.expval %7 : f64
      %from_elements = tensor.from_elements %8 : tensor<f64>
      %9 = quantum.compbasis qubits %out_qubits_1, %out_qubits_3 : !quantum.obs
      %10 = quantum.probs %9 : tensor<4xf64>
      %11 = quantum.insert %2[ 0], %out_qubits_1 : !quantum.reg, !quantum.bit
      %12 = quantum.insert %11[ 1], %out_qubits_3 : !quantum.reg, !quantum.bit
      quantum.dealloc %12 : !quantum.reg
      quantum.device_release
      return %from_elements, %10 : tensor<f64>, tensor<4xf64>
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
