module @circuit {
  func.func public @jit_circuit() -> tensor<f64> attributes {llvm.emit_c_interface} {
    %0 = call @circuit_0() : () -> tensor<f64>
    return %0 : tensor<f64>
  }
  func.func public @circuit_0() -> tensor<f64> attributes {diff_method = "parameter-shift", llvm.linkage = #llvm.linkage<internal>, qnode} {
    %cst = arith.constant 0.52359877559829882 : f64
    %c0_i64 = arith.constant 0 : i64
    quantum.device shots(%c0_i64) ["/Users/jeffrey.kam/Code/catalyst/frontend/catalyst/utils/../../../runtime/build/lib/librtd_null_qubit.dylib", "NullQubit", "{'track_resources': False}"]
    %0 = quantum.alloc( 3) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %2 = qec.ppr.arbitrary ["Z"](%cst) %1 : !quantum.bit
    %3 = quantum.namedobs %2[ PauliX] : !quantum.obs
    %4 = quantum.expval %3 : f64
    %from_elements = tensor.from_elements %4 : tensor<f64>
    %5 = quantum.insert %0[ 0], %2 : !quantum.reg, !quantum.bit
    quantum.dealloc %5 : !quantum.reg
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