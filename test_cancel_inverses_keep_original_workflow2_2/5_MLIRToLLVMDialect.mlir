module @test_cancel_inverses_keep_original_workflow2 {
  llvm.func @__catalyst__rt__finalize()
  llvm.func @__catalyst__rt__initialize(!llvm.ptr)
  llvm.func @__catalyst__rt__device_release()
  llvm.func @__catalyst__rt__qubit_release_array(!llvm.ptr)
  llvm.func @__catalyst__qis__Expval(i64) -> f64
  llvm.func @__catalyst__qis__NamedObs(i64, !llvm.ptr) -> i64
  llvm.func @__catalyst__qis__Hadamard(!llvm.ptr, !llvm.ptr)
  llvm.func @__catalyst__qis__RX(f64, !llvm.ptr, !llvm.ptr)
  llvm.func @__catalyst__rt__array_get_element_ptr_1d(!llvm.ptr, i64) -> !llvm.ptr
  llvm.func @__catalyst__rt__qubit_allocate_array(i64) -> !llvm.ptr
  llvm.mlir.global internal constant @"{'mcmc': False, 'num_burnin': 0, 'kernel_name': None}"("{'mcmc': False, 'num_burnin': 0, 'kernel_name': None}\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @LightningSimulator("LightningSimulator\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @"/Users/christina/Prog/catalyst_env/lib/python3.12/site-packages/pennylane_lightning/liblightning_qubit_catalyst.dylib"("/Users/christina/Prog/catalyst_env/lib/python3.12/site-packages/pennylane_lightning/liblightning_qubit_catalyst.dylib\00") {addr_space = 0 : i32}
  llvm.func @__catalyst__rt__device_init(!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i1)
  llvm.func @_mlir_memref_to_llvm_alloc(i64) -> !llvm.ptr
  llvm.mlir.global private constant @__constant_xf64(1.000000e+00 : f64) {addr_space = 0 : i32, alignment = 64 : i64} : f64
  llvm.func @jit_test_cancel_inverses_keep_original_workflow2() -> !llvm.struct<(struct<(ptr, ptr, i64)>, struct<(ptr, ptr, i64)>)> attributes {llvm.copy_memref, llvm.emit_c_interface} {
    %0 = llvm.mlir.poison : !llvm.struct<(struct<(ptr, ptr, i64)>, struct<(ptr, ptr, i64)>)>
    %1 = llvm.mlir.constant(0 : index) : i64
    %2 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64)>
    %3 = llvm.mlir.addressof @__constant_xf64 : !llvm.ptr
    %4 = llvm.mlir.constant(3735928559 : index) : i64
    %5 = llvm.mlir.constant(1 : index) : i64
    %6 = llvm.mlir.zero : !llvm.ptr
    %7 = llvm.inttoptr %4 : i64 to !llvm.ptr
    %8 = llvm.call @f_0(%7, %3, %1) : (!llvm.ptr, !llvm.ptr, i64) -> !llvm.struct<(ptr, ptr, i64)>
    %9 = llvm.call @f_1_0(%7, %3, %1) : (!llvm.ptr, !llvm.ptr, i64) -> !llvm.struct<(ptr, ptr, i64)>
    %10 = llvm.extractvalue %8[0] : !llvm.struct<(ptr, ptr, i64)> 
    %11 = llvm.ptrtoint %10 : !llvm.ptr to i64
    %12 = llvm.icmp "eq" %4, %11 : i64
    llvm.cond_br %12, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    %13 = llvm.getelementptr %6[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %14 = llvm.ptrtoint %13 : !llvm.ptr to i64
    %15 = llvm.call @_mlir_memref_to_llvm_alloc(%14) : (i64) -> !llvm.ptr
    %16 = llvm.insertvalue %15, %2[0] : !llvm.struct<(ptr, ptr, i64)> 
    %17 = llvm.insertvalue %15, %16[1] : !llvm.struct<(ptr, ptr, i64)> 
    %18 = llvm.insertvalue %1, %17[2] : !llvm.struct<(ptr, ptr, i64)> 
    %19 = llvm.getelementptr %6[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %20 = llvm.ptrtoint %19 : !llvm.ptr to i64
    %21 = llvm.mul %20, %5 : i64
    %22 = llvm.extractvalue %8[1] : !llvm.struct<(ptr, ptr, i64)> 
    %23 = llvm.extractvalue %8[2] : !llvm.struct<(ptr, ptr, i64)> 
    %24 = llvm.getelementptr inbounds %22[%23] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    "llvm.intr.memcpy"(%15, %24, %21) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.br ^bb3(%18 : !llvm.struct<(ptr, ptr, i64)>)
  ^bb2:  // pred: ^bb0
    llvm.br ^bb3(%8 : !llvm.struct<(ptr, ptr, i64)>)
  ^bb3(%25: !llvm.struct<(ptr, ptr, i64)>):  // 2 preds: ^bb1, ^bb2
    llvm.br ^bb4
  ^bb4:  // pred: ^bb3
    %26 = llvm.extractvalue %9[0] : !llvm.struct<(ptr, ptr, i64)> 
    %27 = llvm.ptrtoint %26 : !llvm.ptr to i64
    %28 = llvm.icmp "eq" %4, %27 : i64
    llvm.cond_br %28, ^bb5, ^bb6
  ^bb5:  // pred: ^bb4
    %29 = llvm.getelementptr %6[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %30 = llvm.ptrtoint %29 : !llvm.ptr to i64
    %31 = llvm.call @_mlir_memref_to_llvm_alloc(%30) : (i64) -> !llvm.ptr
    %32 = llvm.insertvalue %31, %2[0] : !llvm.struct<(ptr, ptr, i64)> 
    %33 = llvm.insertvalue %31, %32[1] : !llvm.struct<(ptr, ptr, i64)> 
    %34 = llvm.insertvalue %1, %33[2] : !llvm.struct<(ptr, ptr, i64)> 
    %35 = llvm.getelementptr %6[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %36 = llvm.ptrtoint %35 : !llvm.ptr to i64
    %37 = llvm.mul %36, %5 : i64
    %38 = llvm.extractvalue %9[1] : !llvm.struct<(ptr, ptr, i64)> 
    %39 = llvm.extractvalue %9[2] : !llvm.struct<(ptr, ptr, i64)> 
    %40 = llvm.getelementptr inbounds %38[%39] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    "llvm.intr.memcpy"(%31, %40, %37) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.br ^bb7(%34 : !llvm.struct<(ptr, ptr, i64)>)
  ^bb6:  // pred: ^bb4
    llvm.br ^bb7(%9 : !llvm.struct<(ptr, ptr, i64)>)
  ^bb7(%41: !llvm.struct<(ptr, ptr, i64)>):  // 2 preds: ^bb5, ^bb6
    llvm.br ^bb8
  ^bb8:  // pred: ^bb7
    %42 = llvm.insertvalue %25, %0[0] : !llvm.struct<(struct<(ptr, ptr, i64)>, struct<(ptr, ptr, i64)>)> 
    %43 = llvm.insertvalue %41, %42[1] : !llvm.struct<(struct<(ptr, ptr, i64)>, struct<(ptr, ptr, i64)>)> 
    llvm.return %43 : !llvm.struct<(struct<(ptr, ptr, i64)>, struct<(ptr, ptr, i64)>)>
  }
  llvm.func @_catalyst_pyface_jit_test_cancel_inverses_keep_original_workflow2(%arg0: !llvm.ptr, %arg1: !llvm.ptr) {
    llvm.call @_catalyst_ciface_jit_test_cancel_inverses_keep_original_workflow2(%arg0) : (!llvm.ptr) -> ()
    llvm.return
  }
  llvm.func @_catalyst_ciface_jit_test_cancel_inverses_keep_original_workflow2(%arg0: !llvm.ptr) attributes {llvm.copy_memref, llvm.emit_c_interface} {
    %0 = llvm.call @jit_test_cancel_inverses_keep_original_workflow2() : () -> !llvm.struct<(struct<(ptr, ptr, i64)>, struct<(ptr, ptr, i64)>)>
    llvm.store %0, %arg0 : !llvm.struct<(struct<(ptr, ptr, i64)>, struct<(ptr, ptr, i64)>)>, !llvm.ptr
    llvm.return
  }
  llvm.func internal @f_0(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64) -> !llvm.struct<(ptr, ptr, i64)> attributes {diff_method = "adjoint", qnode} {
    %0 = llvm.mlir.constant(0 : index) : i64
    %1 = llvm.mlir.constant(64 : index) : i64
    %2 = llvm.mlir.constant(1 : index) : i64
    %3 = llvm.mlir.constant(3 : i64) : i64
    %4 = llvm.mlir.zero : !llvm.ptr
    %5 = llvm.mlir.constant(1 : i64) : i64
    %6 = llvm.mlir.constant(false) : i1
    %7 = llvm.mlir.addressof @"{'mcmc': False, 'num_burnin': 0, 'kernel_name': None}" : !llvm.ptr
    %8 = llvm.mlir.addressof @LightningSimulator : !llvm.ptr
    %9 = llvm.mlir.addressof @"/Users/christina/Prog/catalyst_env/lib/python3.12/site-packages/pennylane_lightning/liblightning_qubit_catalyst.dylib" : !llvm.ptr
    %10 = llvm.mlir.constant(0 : i64) : i64
    %11 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64)>
    %12 = llvm.getelementptr inbounds %9[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<118 x i8>
    %13 = llvm.getelementptr inbounds %8[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<19 x i8>
    %14 = llvm.getelementptr inbounds %7[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<54 x i8>
    llvm.call @__catalyst__rt__device_init(%12, %13, %14, %10, %6) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i1) -> ()
    %15 = llvm.call @__catalyst__rt__qubit_allocate_array(%5) : (i64) -> !llvm.ptr
    %16 = llvm.call @__catalyst__rt__array_get_element_ptr_1d(%15, %10) : (!llvm.ptr, i64) -> !llvm.ptr
    %17 = llvm.load %16 : !llvm.ptr -> !llvm.ptr
    %18 = llvm.load %arg1 : !llvm.ptr -> f64
    llvm.call @__catalyst__qis__RX(%18, %17, %4) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @__catalyst__qis__Hadamard(%17, %4) : (!llvm.ptr, !llvm.ptr) -> ()
    llvm.call @__catalyst__qis__Hadamard(%17, %4) : (!llvm.ptr, !llvm.ptr) -> ()
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
  llvm.func internal @f_1_0(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64) -> !llvm.struct<(ptr, ptr, i64)> attributes {diff_method = "adjoint", qnode} {
    %0 = llvm.mlir.constant(0 : index) : i64
    %1 = llvm.mlir.constant(64 : index) : i64
    %2 = llvm.mlir.constant(1 : index) : i64
    %3 = llvm.mlir.constant(3 : i64) : i64
    %4 = llvm.mlir.zero : !llvm.ptr
    %5 = llvm.mlir.constant(1 : i64) : i64
    %6 = llvm.mlir.constant(false) : i1
    %7 = llvm.mlir.addressof @"{'mcmc': False, 'num_burnin': 0, 'kernel_name': None}" : !llvm.ptr
    %8 = llvm.mlir.addressof @LightningSimulator : !llvm.ptr
    %9 = llvm.mlir.addressof @"/Users/christina/Prog/catalyst_env/lib/python3.12/site-packages/pennylane_lightning/liblightning_qubit_catalyst.dylib" : !llvm.ptr
    %10 = llvm.mlir.constant(0 : i64) : i64
    %11 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64)>
    %12 = llvm.getelementptr inbounds %9[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<118 x i8>
    %13 = llvm.getelementptr inbounds %8[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<19 x i8>
    %14 = llvm.getelementptr inbounds %7[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<54 x i8>
    llvm.call @__catalyst__rt__device_init(%12, %13, %14, %10, %6) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i1) -> ()
    %15 = llvm.call @__catalyst__rt__qubit_allocate_array(%5) : (i64) -> !llvm.ptr
    %16 = llvm.call @__catalyst__rt__array_get_element_ptr_1d(%15, %10) : (!llvm.ptr, i64) -> !llvm.ptr
    %17 = llvm.load %16 : !llvm.ptr -> !llvm.ptr
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