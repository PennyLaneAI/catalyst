module @qfunc {
  memref.global "private" constant @__constant_3xi64 : memref<3xi64> = dense<[0, 1, 2]> {alignment = 64 : i64}
  func.func public @jit_qfunc() -> memref<f64> attributes {llvm.copy_memref, llvm.emit_c_interface} {
    %0 = llvm.mlir.constant(3735928559 : index) : i64
    %1 = call @qfunc_0() : () -> memref<f64>
    %2 = builtin.unrealized_conversion_cast %1 : memref<f64> to !llvm.struct<(ptr, ptr, i64)>
    %3 = llvm.extractvalue %2[0] : !llvm.struct<(ptr, ptr, i64)> 
    %4 = llvm.ptrtoint %3 : !llvm.ptr to i64
    %5 = llvm.icmp "eq" %0, %4 : i64
    cf.cond_br %5, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    %alloc = memref.alloc() : memref<f64>
    memref.copy %1, %alloc : memref<f64> to memref<f64>
    cf.br ^bb3(%alloc : memref<f64>)
  ^bb2:  // pred: ^bb0
    cf.br ^bb3(%1 : memref<f64>)
  ^bb3(%6: memref<f64>):  // 2 preds: ^bb1, ^bb2
    cf.br ^bb4
  ^bb4:  // pred: ^bb3
    return %6 : memref<f64>
  }
  func.func public @qfunc_0() -> memref<f64> attributes {diff_method = "parameter-shift", llvm.linkage = #llvm.linkage<internal>, qnode} {
    %0 = llvm.mlir.constant(0 : i64) : i64
    %1 = memref.get_global @__constant_3xi64 : memref<3xi64>
    quantum.device shots(%0) ["/Users/haider.sajjad/catalyst/frontend/catalyst/utils/../../../runtime/build/lib/librtd_null_qubit.dylib", "NullQubit", "{'track_resources': False}"]
    %2 = quantum.alloc( 3) : !quantum.reg
    %reinterpret_cast = memref.reinterpret_cast %1 to offset: [2], sizes: [], strides: [] : memref<3xi64> to memref<i64, strided<[], offset: 2>>
    %3 = memref.load %reinterpret_cast[] : memref<i64, strided<[], offset: 2>>
    %4 = quantum.extract %2[%3] : !quantum.reg -> !quantum.bit
    %out_qubits = quantum.custom "Hadamard"() %4 : !quantum.bit
    %reinterpret_cast_0 = memref.reinterpret_cast %1 to offset: [1], sizes: [], strides: [] : memref<3xi64> to memref<i64, strided<[], offset: 1>>
    %5 = memref.load %reinterpret_cast_0[] : memref<i64, strided<[], offset: 1>>
    %6 = quantum.extract %2[%5] : !quantum.reg -> !quantum.bit
    %out_qubits_1:2 = quantum.custom "CNOT"() %6, %out_qubits : !quantum.bit, !quantum.bit
    %7 = quantum.insert %2[%5], %out_qubits_1#0 : !quantum.reg, !quantum.bit
    %out_qubits_2 = quantum.custom "T"() %out_qubits_1#1 adj : !quantum.bit
    %reinterpret_cast_3 = memref.reinterpret_cast %1 to offset: [0], sizes: [], strides: [] : memref<3xi64> to memref<i64>
    %8 = memref.load %reinterpret_cast_3[] : memref<i64>
    %9 = quantum.extract %7[%8] : !quantum.reg -> !quantum.bit
    %out_qubits_4:2 = quantum.custom "CNOT"() %9, %out_qubits_2 : !quantum.bit, !quantum.bit
    %10 = quantum.insert %7[%8], %out_qubits_4#0 : !quantum.reg, !quantum.bit
    %out_qubits_5 = quantum.custom "T"() %out_qubits_4#1 : !quantum.bit
    %11 = quantum.extract %10[%5] : !quantum.reg -> !quantum.bit
    %out_qubits_6:2 = quantum.custom "CNOT"() %11, %out_qubits_5 : !quantum.bit, !quantum.bit
    %12 = quantum.insert %10[%5], %out_qubits_6#0 : !quantum.reg, !quantum.bit
    %out_qubits_7 = quantum.custom "T"() %out_qubits_6#1 adj : !quantum.bit
    %13 = quantum.extract %12[%8] : !quantum.reg -> !quantum.bit
    %out_qubits_8:2 = quantum.custom "CNOT"() %13, %out_qubits_7 : !quantum.bit, !quantum.bit
    %14 = quantum.insert %12[%8], %out_qubits_8#0 : !quantum.reg, !quantum.bit
    %out_qubits_9 = quantum.custom "T"() %out_qubits_8#1 : !quantum.bit
    %15 = quantum.insert %14[%3], %out_qubits_9 : !quantum.reg, !quantum.bit
    %16 = quantum.extract %15[%5] : !quantum.reg -> !quantum.bit
    %out_qubits_10 = quantum.custom "T"() %16 : !quantum.bit
    %17 = quantum.extract %15[%8] : !quantum.reg -> !quantum.bit
    %out_qubits_11:2 = quantum.custom "CNOT"() %17, %out_qubits_10 : !quantum.bit, !quantum.bit
    %18 = quantum.insert %15[%8], %out_qubits_11#0 : !quantum.reg, !quantum.bit
    %19 = quantum.insert %18[%5], %out_qubits_11#1 : !quantum.reg, !quantum.bit
    %20 = quantum.extract %19[%3] : !quantum.reg -> !quantum.bit
    %out_qubits_12 = quantum.custom "Hadamard"() %20 : !quantum.bit
    %21 = quantum.insert %19[%3], %out_qubits_12 : !quantum.reg, !quantum.bit
    %22 = quantum.extract %21[%8] : !quantum.reg -> !quantum.bit
    %out_qubits_13 = quantum.custom "T"() %22 : !quantum.bit
    %23 = quantum.extract %21[%5] : !quantum.reg -> !quantum.bit
    %out_qubits_14 = quantum.custom "T"() %23 adj : !quantum.bit
    %out_qubits_15:2 = quantum.custom "CNOT"() %out_qubits_13, %out_qubits_14 : !quantum.bit, !quantum.bit
    %24 = quantum.insert %21[%8], %out_qubits_15#0 : !quantum.reg, !quantum.bit
    %25 = quantum.insert %24[%5], %out_qubits_15#1 : !quantum.reg, !quantum.bit
    %26 = quantum.extract %25[ 0] : !quantum.reg -> !quantum.bit
    %27 = quantum.namedobs %26[ PauliZ] : !quantum.obs
    %28 = quantum.insert %25[ 0], %26 : !quantum.reg, !quantum.bit
    %29 = quantum.expval %27 : f64
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<f64>
    memref.store %29, %alloc[] : memref<f64>
    quantum.dealloc %28 : !quantum.reg
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