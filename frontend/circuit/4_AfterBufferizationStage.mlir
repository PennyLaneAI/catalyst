module @circuit {
  func.func public @jit_circuit(%arg0: memref<f64>) -> memref<f64> attributes {llvm.copy_memref, llvm.emit_c_interface} {
    %0 = llvm.mlir.constant(3735928559 : index) : i64
    %1 = call @circuit_0(%arg0) : (memref<f64>) -> memref<f64>
    %2 = builtin.unrealized_conversion_cast %1 : memref<f64> to !llvm.struct<(ptr, ptr, i64)>
    %3 = llvm.extractvalue %2[0] : !llvm.struct<(ptr, ptr, i64)> 
    %4 = llvm.ptrtoint %3 : !llvm.ptr to i64
    %5 = llvm.icmp "eq" %0, %4 : i64
    %6 = scf.if %5 -> (memref<f64>) {
      %alloc = memref.alloc() : memref<f64>
      memref.copy %1, %alloc : memref<f64> to memref<f64>
      scf.yield %alloc : memref<f64>
    } else {
      scf.yield %1 : memref<f64>
    }
    return %6 : memref<f64>
  }
  func.func public @circuit_0(%arg0: memref<f64>) -> memref<f64> attributes {diff_method = "parameter-shift", llvm.linkage = #llvm.linkage<internal>, qnode} {
    %c0_i64 = arith.constant 0 : i64
    quantum.device shots(%c0_i64) ["/home/ubuntu/catalyst/frontend/catalyst/utils/../../../runtime/build/lib/librtd_openqasm.so", "OpenQasmDevice", "{'device_type': 'braket.local.qubit', 'backend': 'braket_sv'}"]
    %0 = quantum.alloc( 2) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %out_qubits = quantum.custom "Hadamard"() %1 : !quantum.bit
    %2 = memref.load %arg0[] : memref<f64>
    %out_qubits_0 = quantum.custom "RX"(%2) %out_qubits : !quantum.bit
    %3 = quantum.namedobs %out_qubits_0[ PauliZ] : !quantum.obs
    %4 = quantum.expval %3 : f64
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<f64>
    memref.store %4, %alloc[] : memref<f64>
    %5 = quantum.insert %0[ 0], %out_qubits_0 : !quantum.reg, !quantum.bit
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