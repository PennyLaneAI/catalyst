module @qfunc {
  llvm.func @__catalyst__rt__finalize()
  llvm.func @__catalyst__rt__initialize(!llvm.ptr)
  llvm.func @__catalyst__rt__device_release()
  llvm.func @__catalyst__rt__qubit_release_array(!llvm.ptr)
  llvm.func @__catalyst__qis__Expval(i64) -> f64
  llvm.func @__catalyst__qis__NamedObs(i64, !llvm.ptr) -> i64
  llvm.func @__catalyst__qis__T(!llvm.ptr, !llvm.ptr)
  llvm.func @__catalyst__qis__CNOT(!llvm.ptr, !llvm.ptr, !llvm.ptr)
  llvm.func @__catalyst__qis__Hadamard(!llvm.ptr, !llvm.ptr)
  llvm.func @__catalyst__rt__array_get_element_ptr_1d(!llvm.ptr, i64) -> !llvm.ptr
  llvm.func @__catalyst__rt__qubit_allocate_array(i64) -> !llvm.ptr
  llvm.mlir.global internal constant @"{'track_resources': False}"("{'track_resources': False}\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @NullQubit("NullQubit\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @"/Users/haider.sajjad/catalyst/frontend/catalyst/utils/../../../runtime/build/lib/librtd_null_qubit.dylib"("/Users/haider.sajjad/catalyst/frontend/catalyst/utils/../../../runtime/build/lib/librtd_null_qubit.dylib\00") {addr_space = 0 : i32}
  llvm.func @__catalyst__rt__device_init(!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i1)
  llvm.func @_mlir_memref_to_llvm_alloc(i64) -> !llvm.ptr
  llvm.mlir.global private constant @__constant_3xi64(dense<[0, 1, 2]> : tensor<3xi64>) {addr_space = 0 : i32, alignment = 64 : i64} : !llvm.array<3 x i64>
  llvm.func @jit_qfunc() -> !llvm.struct<(ptr, ptr, i64)> attributes {llvm.copy_memref, llvm.emit_c_interface} {
    %0 = llvm.mlir.constant(0 : index) : i64
    %1 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64)>
    %2 = llvm.mlir.zero : !llvm.ptr
    %3 = llvm.mlir.constant(1 : index) : i64
    %4 = llvm.mlir.constant(3735928559 : index) : i64
    %5 = llvm.call @qfunc_0() : () -> !llvm.struct<(ptr, ptr, i64)>
    %6 = llvm.extractvalue %5[0] : !llvm.struct<(ptr, ptr, i64)> 
    %7 = llvm.ptrtoint %6 : !llvm.ptr to i64
    %8 = llvm.icmp "eq" %4, %7 : i64
    llvm.cond_br %8, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    %9 = llvm.getelementptr %2[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %10 = llvm.ptrtoint %9 : !llvm.ptr to i64
    %11 = llvm.call @_mlir_memref_to_llvm_alloc(%10) : (i64) -> !llvm.ptr
    %12 = llvm.insertvalue %11, %1[0] : !llvm.struct<(ptr, ptr, i64)> 
    %13 = llvm.insertvalue %11, %12[1] : !llvm.struct<(ptr, ptr, i64)> 
    %14 = llvm.insertvalue %0, %13[2] : !llvm.struct<(ptr, ptr, i64)> 
    %15 = llvm.getelementptr %2[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %16 = llvm.ptrtoint %15 : !llvm.ptr to i64
    %17 = llvm.mul %16, %3 : i64
    %18 = llvm.extractvalue %5[1] : !llvm.struct<(ptr, ptr, i64)> 
    %19 = llvm.extractvalue %5[2] : !llvm.struct<(ptr, ptr, i64)> 
    %20 = llvm.getelementptr %18[%19] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    "llvm.intr.memcpy"(%11, %20, %17) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.br ^bb3(%14 : !llvm.struct<(ptr, ptr, i64)>)
  ^bb2:  // pred: ^bb0
    llvm.br ^bb3(%5 : !llvm.struct<(ptr, ptr, i64)>)
  ^bb3(%21: !llvm.struct<(ptr, ptr, i64)>):  // 2 preds: ^bb1, ^bb2
    llvm.br ^bb4
  ^bb4:  // pred: ^bb3
    llvm.return %21 : !llvm.struct<(ptr, ptr, i64)>
  }
  llvm.func @_catalyst_pyface_jit_qfunc(%arg0: !llvm.ptr, %arg1: !llvm.ptr) {
    llvm.call @_catalyst_ciface_jit_qfunc(%arg0) : (!llvm.ptr) -> ()
    llvm.return
  }
  llvm.func @_catalyst_ciface_jit_qfunc(%arg0: !llvm.ptr) attributes {llvm.copy_memref, llvm.emit_c_interface} {
    %0 = llvm.call @jit_qfunc() : () -> !llvm.struct<(ptr, ptr, i64)>
    llvm.store %0, %arg0 : !llvm.struct<(ptr, ptr, i64)>, !llvm.ptr
    llvm.return
  }
  llvm.func internal @qfunc_0() -> !llvm.struct<(ptr, ptr, i64)> attributes {diff_method = "parameter-shift", qnode} {
    %0 = llvm.mlir.constant(64 : index) : i64
    %1 = llvm.mlir.constant(true) : i1
    %2 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64)>
    %3 = llvm.mlir.constant(3 : i64) : i64
    %4 = llvm.mlir.constant(false) : i1
    %5 = llvm.mlir.addressof @"{'track_resources': False}" : !llvm.ptr
    %6 = llvm.mlir.addressof @NullQubit : !llvm.ptr
    %7 = llvm.mlir.addressof @"/Users/haider.sajjad/catalyst/frontend/catalyst/utils/../../../runtime/build/lib/librtd_null_qubit.dylib" : !llvm.ptr
    %8 = llvm.mlir.constant(0 : index) : i64
    %9 = llvm.mlir.addressof @__constant_3xi64 : !llvm.ptr
    %10 = llvm.mlir.zero : !llvm.ptr
    %11 = llvm.mlir.constant(1 : index) : i64
    %12 = llvm.mlir.constant(0 : i64) : i64
    %13 = llvm.mlir.constant(1 : i64) : i64
    %14 = llvm.alloca %13 x !llvm.struct<(i1, i64, ptr, ptr)> : (i64) -> !llvm.ptr
    %15 = llvm.alloca %13 x !llvm.struct<(i1, i64, ptr, ptr)> : (i64) -> !llvm.ptr
    %16 = llvm.alloca %13 x !llvm.struct<(i1, i64, ptr, ptr)> : (i64) -> !llvm.ptr
    %17 = llvm.getelementptr %9[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<3 x i64>
    %18 = llvm.getelementptr inbounds %7[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<105 x i8>
    %19 = llvm.getelementptr inbounds %6[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<10 x i8>
    %20 = llvm.getelementptr inbounds %5[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<27 x i8>
    llvm.call @__catalyst__rt__device_init(%18, %19, %20, %12, %4) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i1) -> ()
    %21 = llvm.call @__catalyst__rt__qubit_allocate_array(%3) : (i64) -> !llvm.ptr
    %22 = llvm.getelementptr %17[2] : (!llvm.ptr) -> !llvm.ptr, i64
    %23 = llvm.load %22 : !llvm.ptr -> i64
    %24 = llvm.call @__catalyst__rt__array_get_element_ptr_1d(%21, %23) : (!llvm.ptr, i64) -> !llvm.ptr
    %25 = llvm.load %24 : !llvm.ptr -> !llvm.ptr
    llvm.call @__catalyst__qis__Hadamard(%25, %10) : (!llvm.ptr, !llvm.ptr) -> ()
    %26 = llvm.getelementptr %17[1] : (!llvm.ptr) -> !llvm.ptr, i64
    %27 = llvm.load %26 : !llvm.ptr -> i64
    %28 = llvm.call @__catalyst__rt__array_get_element_ptr_1d(%21, %27) : (!llvm.ptr, i64) -> !llvm.ptr
    %29 = llvm.load %28 : !llvm.ptr -> !llvm.ptr
    llvm.call @__catalyst__qis__CNOT(%29, %25, %10) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    %30 = llvm.getelementptr inbounds %16[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i1, i64, ptr, ptr)>
    %31 = llvm.getelementptr inbounds %16[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i1, i64, ptr, ptr)>
    %32 = llvm.getelementptr inbounds %16[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i1, i64, ptr, ptr)>
    %33 = llvm.getelementptr inbounds %16[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i1, i64, ptr, ptr)>
    llvm.store %1, %30 : i1, !llvm.ptr
    llvm.store %12, %31 : i64, !llvm.ptr
    llvm.store %10, %32 : !llvm.ptr, !llvm.ptr
    llvm.store %10, %33 : !llvm.ptr, !llvm.ptr
    llvm.call @__catalyst__qis__T(%25, %16) : (!llvm.ptr, !llvm.ptr) -> ()
    %34 = llvm.load %17 : !llvm.ptr -> i64
    %35 = llvm.call @__catalyst__rt__array_get_element_ptr_1d(%21, %34) : (!llvm.ptr, i64) -> !llvm.ptr
    %36 = llvm.load %35 : !llvm.ptr -> !llvm.ptr
    llvm.call @__catalyst__qis__CNOT(%36, %25, %10) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @__catalyst__qis__T(%25, %10) : (!llvm.ptr, !llvm.ptr) -> ()
    %37 = llvm.call @__catalyst__rt__array_get_element_ptr_1d(%21, %27) : (!llvm.ptr, i64) -> !llvm.ptr
    %38 = llvm.load %37 : !llvm.ptr -> !llvm.ptr
    llvm.call @__catalyst__qis__CNOT(%38, %25, %10) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    %39 = llvm.getelementptr inbounds %15[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i1, i64, ptr, ptr)>
    %40 = llvm.getelementptr inbounds %15[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i1, i64, ptr, ptr)>
    %41 = llvm.getelementptr inbounds %15[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i1, i64, ptr, ptr)>
    %42 = llvm.getelementptr inbounds %15[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i1, i64, ptr, ptr)>
    llvm.store %1, %39 : i1, !llvm.ptr
    llvm.store %12, %40 : i64, !llvm.ptr
    llvm.store %10, %41 : !llvm.ptr, !llvm.ptr
    llvm.store %10, %42 : !llvm.ptr, !llvm.ptr
    llvm.call @__catalyst__qis__T(%25, %15) : (!llvm.ptr, !llvm.ptr) -> ()
    %43 = llvm.call @__catalyst__rt__array_get_element_ptr_1d(%21, %34) : (!llvm.ptr, i64) -> !llvm.ptr
    %44 = llvm.load %43 : !llvm.ptr -> !llvm.ptr
    llvm.call @__catalyst__qis__CNOT(%44, %25, %10) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @__catalyst__qis__T(%25, %10) : (!llvm.ptr, !llvm.ptr) -> ()
    %45 = llvm.call @__catalyst__rt__array_get_element_ptr_1d(%21, %27) : (!llvm.ptr, i64) -> !llvm.ptr
    %46 = llvm.load %45 : !llvm.ptr -> !llvm.ptr
    llvm.call @__catalyst__qis__T(%46, %10) : (!llvm.ptr, !llvm.ptr) -> ()
    %47 = llvm.call @__catalyst__rt__array_get_element_ptr_1d(%21, %34) : (!llvm.ptr, i64) -> !llvm.ptr
    %48 = llvm.load %47 : !llvm.ptr -> !llvm.ptr
    llvm.call @__catalyst__qis__CNOT(%48, %46, %10) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    %49 = llvm.call @__catalyst__rt__array_get_element_ptr_1d(%21, %23) : (!llvm.ptr, i64) -> !llvm.ptr
    %50 = llvm.load %49 : !llvm.ptr -> !llvm.ptr
    llvm.call @__catalyst__qis__Hadamard(%50, %10) : (!llvm.ptr, !llvm.ptr) -> ()
    %51 = llvm.call @__catalyst__rt__array_get_element_ptr_1d(%21, %34) : (!llvm.ptr, i64) -> !llvm.ptr
    %52 = llvm.load %51 : !llvm.ptr -> !llvm.ptr
    llvm.call @__catalyst__qis__T(%52, %10) : (!llvm.ptr, !llvm.ptr) -> ()
    %53 = llvm.call @__catalyst__rt__array_get_element_ptr_1d(%21, %27) : (!llvm.ptr, i64) -> !llvm.ptr
    %54 = llvm.load %53 : !llvm.ptr -> !llvm.ptr
    %55 = llvm.getelementptr inbounds %14[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i1, i64, ptr, ptr)>
    %56 = llvm.getelementptr inbounds %14[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i1, i64, ptr, ptr)>
    %57 = llvm.getelementptr inbounds %14[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i1, i64, ptr, ptr)>
    %58 = llvm.getelementptr inbounds %14[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i1, i64, ptr, ptr)>
    llvm.store %1, %55 : i1, !llvm.ptr
    llvm.store %12, %56 : i64, !llvm.ptr
    llvm.store %10, %57 : !llvm.ptr, !llvm.ptr
    llvm.store %10, %58 : !llvm.ptr, !llvm.ptr
    llvm.call @__catalyst__qis__T(%54, %14) : (!llvm.ptr, !llvm.ptr) -> ()
    llvm.call @__catalyst__qis__CNOT(%52, %54, %10) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    %59 = llvm.call @__catalyst__rt__array_get_element_ptr_1d(%21, %12) : (!llvm.ptr, i64) -> !llvm.ptr
    %60 = llvm.load %59 : !llvm.ptr -> !llvm.ptr
    %61 = llvm.call @__catalyst__qis__NamedObs(%3, %60) : (i64, !llvm.ptr) -> i64
    %62 = llvm.call @__catalyst__qis__Expval(%61) : (i64) -> f64
    %63 = llvm.getelementptr %10[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %64 = llvm.ptrtoint %63 : !llvm.ptr to i64
    %65 = llvm.add %64, %0 : i64
    %66 = llvm.call @_mlir_memref_to_llvm_alloc(%65) : (i64) -> !llvm.ptr
    %67 = llvm.ptrtoint %66 : !llvm.ptr to i64
    %68 = llvm.sub %0, %11 : i64
    %69 = llvm.add %67, %68 : i64
    %70 = llvm.urem %69, %0 : i64
    %71 = llvm.sub %69, %70 : i64
    %72 = llvm.inttoptr %71 : i64 to !llvm.ptr
    %73 = llvm.insertvalue %66, %2[0] : !llvm.struct<(ptr, ptr, i64)> 
    %74 = llvm.insertvalue %72, %73[1] : !llvm.struct<(ptr, ptr, i64)> 
    %75 = llvm.insertvalue %8, %74[2] : !llvm.struct<(ptr, ptr, i64)> 
    llvm.store %62, %72 : f64, !llvm.ptr
    llvm.call @__catalyst__rt__qubit_release_array(%21) : (!llvm.ptr) -> ()
    llvm.call @__catalyst__rt__device_release() : () -> ()
    llvm.return %75 : !llvm.struct<(ptr, ptr, i64)>
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