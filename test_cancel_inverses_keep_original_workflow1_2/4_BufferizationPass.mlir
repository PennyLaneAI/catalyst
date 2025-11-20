module @test_cancel_inverses_keep_original_workflow1 {
  memref.global "private" constant @__constant_xf64 : memref<f64> = dense<1.000000e+00> {alignment = 64 : i64}
  func.func public @jit_test_cancel_inverses_keep_original_workflow1() -> memref<f64> attributes {llvm.copy_memref, llvm.emit_c_interface} {
    %0 = llvm.mlir.constant(3735928559 : index) : i64
    %1 = memref.get_global @__constant_xf64 : memref<f64>
    %2 = call @f_0(%1) : (memref<f64>) -> memref<f64>
    %3 = builtin.unrealized_conversion_cast %2 : memref<f64> to !llvm.struct<(ptr, ptr, i64)>
    %4 = llvm.extractvalue %3[0] : !llvm.struct<(ptr, ptr, i64)> 
    %5 = llvm.ptrtoint %4 : !llvm.ptr to i64
    %6 = llvm.icmp "eq" %0, %5 : i64
    %7 = scf.if %6 -> (memref<f64>) {
      %alloc = memref.alloc() : memref<f64>
      memref.copy %2, %alloc : memref<f64> to memref<f64>
      scf.yield %alloc : memref<f64>
    } else {
      scf.yield %2 : memref<f64>
    }
    return %7 : memref<f64>
  }
  func.func public @f_0(%arg0: memref<f64>) -> memref<f64> attributes {diff_method = "adjoint", llvm.linkage = #llvm.linkage<internal>, qnode} {
    %c0_i64 = arith.constant 0 : i64
    quantum.device shots(%c0_i64) ["/Users/christina/Prog/catalyst_env/lib/python3.12/site-packages/pennylane_lightning/liblightning_qubit_catalyst.dylib", "LightningSimulator", "{'mcmc': False, 'num_burnin': 0, 'kernel_name': None}"]
    %0 = quantum.alloc( 1) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %2 = memref.load %arg0[] : memref<f64>
    %out_qubits = quantum.custom "RX"(%2) %1 : !quantum.bit
    %3 = quantum.namedobs %out_qubits[ PauliZ] : !quantum.obs
    %4 = quantum.expval %3 : f64
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<f64>
    memref.store %4, %alloc[] : memref<f64>
    %5 = quantum.insert %0[ 0], %out_qubits : !quantum.reg, !quantum.bit
    quantum.dealloc %5 : !quantum.reg
    quantum.device_release
    return %alloc : memref<f64>
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