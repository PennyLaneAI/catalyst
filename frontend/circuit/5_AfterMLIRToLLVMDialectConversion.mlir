module @circuit {
  llvm.func @__catalyst__rt__finalize()
  llvm.func @__catalyst__rt__initialize(!llvm.ptr)
  llvm.func @__catalyst__rt__device_release()
  llvm.func @__catalyst__rt__qubit_release_array(!llvm.ptr)
  llvm.func @__catalyst__qis__Expval(i64) -> f64
  llvm.func @__catalyst__qis__NamedObs(i64, !llvm.ptr) -> i64
  llvm.func @__catalyst__qis__RX(f64, !llvm.ptr, !llvm.ptr)
  llvm.func @__catalyst__qis__Hadamard(!llvm.ptr, !llvm.ptr)
  llvm.func @__catalyst__rt__array_get_element_ptr_1d(!llvm.ptr, i64) -> !llvm.ptr
  llvm.func @__catalyst__rt__qubit_allocate_array(i64) -> !llvm.ptr
  llvm.mlir.global internal constant @"{'device_type': 'braket.local.qubit', 'backend': 'braket_sv'}"("{'device_type': 'braket.local.qubit', 'backend': 'braket_sv'}\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @OpenQasmDevice("OpenQasmDevice\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @"/home/ubuntu/catalyst/frontend/catalyst/utils/../../../runtime/build/lib/librtd_openqasm.so"("/home/ubuntu/catalyst/frontend/catalyst/utils/../../../runtime/build/lib/librtd_openqasm.so\00") {addr_space = 0 : i32}
  llvm.func @__catalyst__rt__device_init(!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i1)
  llvm.func @_mlir_memref_to_llvm_alloc(i64) -> !llvm.ptr
  llvm.func @jit_circuit(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64) -> !llvm.struct<(ptr, ptr, i64)> attributes {llvm.copy_memref, llvm.emit_c_interface} {
    %0 = llvm.mlir.constant(0 : index) : i64
    %1 = llvm.mlir.zero : !llvm.ptr
    %2 = llvm.mlir.constant(1 : index) : i64
    %3 = llvm.mlir.constant(3735928559 : index) : i64
    %4 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64)>
    %5 = llvm.call @circuit_0(%arg0, %arg1, %arg2) : (!llvm.ptr, !llvm.ptr, i64) -> !llvm.struct<(ptr, ptr, i64)>
    %6 = llvm.extractvalue %5[0] : !llvm.struct<(ptr, ptr, i64)> 
    %7 = llvm.ptrtoint %6 : !llvm.ptr to i64
    %8 = llvm.icmp "eq" %3, %7 : i64
    llvm.cond_br %8, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    %9 = llvm.getelementptr %1[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %10 = llvm.ptrtoint %9 : !llvm.ptr to i64
    %11 = llvm.call @_mlir_memref_to_llvm_alloc(%10) : (i64) -> !llvm.ptr
    %12 = llvm.insertvalue %11, %4[0] : !llvm.struct<(ptr, ptr, i64)> 
    %13 = llvm.insertvalue %11, %12[1] : !llvm.struct<(ptr, ptr, i64)> 
    %14 = llvm.insertvalue %0, %13[2] : !llvm.struct<(ptr, ptr, i64)> 
    %15 = llvm.getelementptr %1[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %16 = llvm.ptrtoint %15 : !llvm.ptr to i64
    %17 = llvm.mul %16, %2 : i64
    %18 = llvm.extractvalue %5[1] : !llvm.struct<(ptr, ptr, i64)> 
    %19 = llvm.extractvalue %5[2] : !llvm.struct<(ptr, ptr, i64)> 
    %20 = llvm.getelementptr inbounds %18[%19] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    "llvm.intr.memcpy"(%11, %20, %17) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.br ^bb3(%14 : !llvm.struct<(ptr, ptr, i64)>)
  ^bb2:  // pred: ^bb0
    llvm.br ^bb3(%5 : !llvm.struct<(ptr, ptr, i64)>)
  ^bb3(%21: !llvm.struct<(ptr, ptr, i64)>):  // 2 preds: ^bb1, ^bb2
    llvm.br ^bb4
  ^bb4:  // pred: ^bb3
    llvm.return %21 : !llvm.struct<(ptr, ptr, i64)>
  }
  llvm.func @_catalyst_pyface_jit_circuit(%arg0: !llvm.ptr, %arg1: !llvm.ptr) {
    %0 = llvm.load %arg1 : !llvm.ptr -> !llvm.struct<(ptr, ptr)>
    %1 = llvm.extractvalue %0[0] : !llvm.struct<(ptr, ptr)> 
    llvm.call @_catalyst_ciface_jit_circuit(%arg0, %1) : (!llvm.ptr, !llvm.ptr) -> ()
    llvm.return
  }
  llvm.func @_catalyst_ciface_jit_circuit(%arg0: !llvm.ptr, %arg1: !llvm.ptr) attributes {llvm.copy_memref, llvm.emit_c_interface} {
    %0 = llvm.load %arg1 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64)>
    %1 = llvm.extractvalue %0[0] : !llvm.struct<(ptr, ptr, i64)> 
    %2 = llvm.extractvalue %0[1] : !llvm.struct<(ptr, ptr, i64)> 
    %3 = llvm.extractvalue %0[2] : !llvm.struct<(ptr, ptr, i64)> 
    %4 = llvm.call @jit_circuit(%1, %2, %3) : (!llvm.ptr, !llvm.ptr, i64) -> !llvm.struct<(ptr, ptr, i64)>
    llvm.store %4, %arg0 : !llvm.struct<(ptr, ptr, i64)>, !llvm.ptr
    llvm.return
  }
  llvm.func internal @circuit_0(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64) -> !llvm.struct<(ptr, ptr, i64)> attributes {diff_method = "parameter-shift", qnode} {
    %0 = llvm.mlir.constant(0 : index) : i64
    %1 = llvm.mlir.constant(64 : index) : i64
    %2 = llvm.mlir.constant(1 : index) : i64
    %3 = llvm.mlir.constant(3 : i64) : i64
    %4 = llvm.mlir.zero : !llvm.ptr
    %5 = llvm.mlir.constant(2 : i64) : i64
    %6 = llvm.mlir.constant(false) : i1
    %7 = llvm.mlir.addressof @"{'device_type': 'braket.local.qubit', 'backend': 'braket_sv'}" : !llvm.ptr
    %8 = llvm.mlir.addressof @OpenQasmDevice : !llvm.ptr
    %9 = llvm.mlir.addressof @"/home/ubuntu/catalyst/frontend/catalyst/utils/../../../runtime/build/lib/librtd_openqasm.so" : !llvm.ptr
    %10 = llvm.mlir.constant(0 : i64) : i64
    %11 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64)>
    %12 = llvm.getelementptr inbounds %9[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<92 x i8>
    %13 = llvm.getelementptr inbounds %8[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<15 x i8>
    %14 = llvm.getelementptr inbounds %7[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<62 x i8>
    llvm.call @__catalyst__rt__device_init(%12, %13, %14, %10, %6) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i1) -> ()
    %15 = llvm.call @__catalyst__rt__qubit_allocate_array(%5) : (i64) -> !llvm.ptr
    %16 = llvm.call @__catalyst__rt__array_get_element_ptr_1d(%15, %10) : (!llvm.ptr, i64) -> !llvm.ptr
    %17 = llvm.load %16 : !llvm.ptr -> !llvm.ptr
    llvm.call @__catalyst__qis__Hadamard(%17, %4) : (!llvm.ptr, !llvm.ptr) -> ()
    %18 = llvm.load %arg1 : !llvm.ptr -> f64
    llvm.call @__catalyst__qis__RX(%18, %17, %4) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %19 = llvm.call @__catalyst__qis__NamedObs(%3, %17) : (i64, !llvm.ptr) -> i64
    %20 = llvm.call @__catalyst__qis__Expval(%19) : (i64) -> f64
    %21 = llvm.getelementptr %4[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %22 = llvm.ptrtoint %21 : !llvm.ptr to i64
    %23 = llvm.add %22, %1 : i64
    %24 = llvm.call @_mlir_memref_to_llvm_alloc(%23) : (i64) -> !llvm.ptr
    %25 = llvm.ptrtoint %24 : !llvm.ptr to i64
    %26 = llvm.sub %1, %2 : i64
    %27 = llvm.add %25, %26 : i64
    %28 = llvm.urem %27, %1 : i64
    %29 = llvm.sub %27, %28 : i64
    %30 = llvm.inttoptr %29 : i64 to !llvm.ptr
    %31 = llvm.insertvalue %24, %11[0] : !llvm.struct<(ptr, ptr, i64)> 
    %32 = llvm.insertvalue %30, %31[1] : !llvm.struct<(ptr, ptr, i64)> 
    %33 = llvm.insertvalue %0, %32[2] : !llvm.struct<(ptr, ptr, i64)> 
    llvm.store %20, %30 : f64, !llvm.ptr
    llvm.call @__catalyst__rt__qubit_release_array(%15) : (!llvm.ptr) -> ()
    llvm.call @__catalyst__rt__device_release() : () -> ()
    llvm.return %33 : !llvm.struct<(ptr, ptr, i64)>
  }
  llvm.func @setup() {
    %0 = llvm.mlir.zero : !llvm.ptr
    llvm.call @__catalyst__rt__initialize(%0) : (!llvm.ptr) -> ()
    llvm.return
  }
  llvm.func @teardown() {
    llvm.call @__catalyst__rt__finalize() : () -> ()
    llvm.return
  }
}