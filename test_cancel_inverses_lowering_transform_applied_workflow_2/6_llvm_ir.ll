; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

@"{'mcmc': False, 'num_burnin': 0, 'kernel_name': None}" = internal constant [54 x i8] c"{'mcmc': False, 'num_burnin': 0, 'kernel_name': None}\00"
@LightningSimulator = internal constant [19 x i8] c"LightningSimulator\00"
@"/Users/christina/Prog/catalyst_env/lib/python3.12/site-packages/pennylane_lightning/liblightning_qubit_catalyst.dylib" = internal constant [118 x i8] c"/Users/christina/Prog/catalyst_env/lib/python3.12/site-packages/pennylane_lightning/liblightning_qubit_catalyst.dylib\00"

declare void @__catalyst__rt__finalize()

declare void @__catalyst__rt__initialize(ptr)

declare void @__catalyst__qis__Hadamard(ptr, ptr)

declare void @__catalyst__rt__device_release()

declare void @__catalyst__rt__qubit_release_array(ptr)

declare double @__catalyst__qis__Expval(i64)

declare i64 @__catalyst__qis__NamedObs(i64, ptr)

declare void @__catalyst__qis__RX(double, ptr, ptr)

declare ptr @__catalyst__rt__array_get_element_ptr_1d(ptr, i64)

declare ptr @__catalyst__rt__qubit_allocate_array(i64)

declare void @__catalyst__rt__device_init(ptr, ptr, ptr, i64, i1)

declare ptr @_mlir_memref_to_llvm_alloc(i64)

define { { ptr, ptr, i64 }, { ptr, ptr, i64 }, { ptr, ptr, i64 } } @jit_test_cancel_inverses_lowering_transform_applied_workflow(ptr %0, ptr %1, i64 %2) {
  %4 = call { ptr, ptr, i64 } @f_0(ptr %0, ptr %1, i64 %2)
  %5 = call { ptr, ptr, i64 } @g_0(ptr %0, ptr %1, i64 %2)
  %6 = call { ptr, ptr, i64 } @h_0(ptr %0, ptr %1, i64 %2)
  %7 = extractvalue { ptr, ptr, i64 } %4, 0
  %8 = ptrtoint ptr %7 to i64
  %9 = icmp eq i64 3735928559, %8
  br i1 %9, label %10, label %18

10:                                               ; preds = %3
  %11 = call ptr @_mlir_memref_to_llvm_alloc(i64 8)
  %12 = insertvalue { ptr, ptr, i64 } poison, ptr %11, 0
  %13 = insertvalue { ptr, ptr, i64 } %12, ptr %11, 1
  %14 = insertvalue { ptr, ptr, i64 } %13, i64 0, 2
  %15 = extractvalue { ptr, ptr, i64 } %4, 1
  %16 = extractvalue { ptr, ptr, i64 } %4, 2
  %17 = getelementptr inbounds double, ptr %15, i64 %16
  call void @llvm.memcpy.p0.p0.i64(ptr %11, ptr %17, i64 8, i1 false)
  br label %19

18:                                               ; preds = %3
  br label %19

19:                                               ; preds = %10, %18
  %20 = phi { ptr, ptr, i64 } [ %4, %18 ], [ %14, %10 ]
  br label %21

21:                                               ; preds = %19
  %22 = extractvalue { ptr, ptr, i64 } %5, 0
  %23 = ptrtoint ptr %22 to i64
  %24 = icmp eq i64 3735928559, %23
  br i1 %24, label %25, label %33

25:                                               ; preds = %21
  %26 = call ptr @_mlir_memref_to_llvm_alloc(i64 8)
  %27 = insertvalue { ptr, ptr, i64 } poison, ptr %26, 0
  %28 = insertvalue { ptr, ptr, i64 } %27, ptr %26, 1
  %29 = insertvalue { ptr, ptr, i64 } %28, i64 0, 2
  %30 = extractvalue { ptr, ptr, i64 } %5, 1
  %31 = extractvalue { ptr, ptr, i64 } %5, 2
  %32 = getelementptr inbounds double, ptr %30, i64 %31
  call void @llvm.memcpy.p0.p0.i64(ptr %26, ptr %32, i64 8, i1 false)
  br label %34

33:                                               ; preds = %21
  br label %34

34:                                               ; preds = %25, %33
  %35 = phi { ptr, ptr, i64 } [ %5, %33 ], [ %29, %25 ]
  br label %36

36:                                               ; preds = %34
  %37 = extractvalue { ptr, ptr, i64 } %6, 0
  %38 = ptrtoint ptr %37 to i64
  %39 = icmp eq i64 3735928559, %38
  br i1 %39, label %40, label %48

40:                                               ; preds = %36
  %41 = call ptr @_mlir_memref_to_llvm_alloc(i64 8)
  %42 = insertvalue { ptr, ptr, i64 } poison, ptr %41, 0
  %43 = insertvalue { ptr, ptr, i64 } %42, ptr %41, 1
  %44 = insertvalue { ptr, ptr, i64 } %43, i64 0, 2
  %45 = extractvalue { ptr, ptr, i64 } %6, 1
  %46 = extractvalue { ptr, ptr, i64 } %6, 2
  %47 = getelementptr inbounds double, ptr %45, i64 %46
  call void @llvm.memcpy.p0.p0.i64(ptr %41, ptr %47, i64 8, i1 false)
  br label %49

48:                                               ; preds = %36
  br label %49

49:                                               ; preds = %40, %48
  %50 = phi { ptr, ptr, i64 } [ %6, %48 ], [ %44, %40 ]
  br label %51

51:                                               ; preds = %49
  %52 = insertvalue { { ptr, ptr, i64 }, { ptr, ptr, i64 }, { ptr, ptr, i64 } } poison, { ptr, ptr, i64 } %20, 0
  %53 = insertvalue { { ptr, ptr, i64 }, { ptr, ptr, i64 }, { ptr, ptr, i64 } } %52, { ptr, ptr, i64 } %35, 1
  %54 = insertvalue { { ptr, ptr, i64 }, { ptr, ptr, i64 }, { ptr, ptr, i64 } } %53, { ptr, ptr, i64 } %50, 2
  ret { { ptr, ptr, i64 }, { ptr, ptr, i64 }, { ptr, ptr, i64 } } %54
}

define void @_catalyst_pyface_jit_test_cancel_inverses_lowering_transform_applied_workflow(ptr %0, ptr %1) {
  %3 = load { ptr, ptr }, ptr %1, align 8
  %4 = extractvalue { ptr, ptr } %3, 0
  call void @_catalyst_ciface_jit_test_cancel_inverses_lowering_transform_applied_workflow(ptr %0, ptr %4)
  ret void
}

define void @_catalyst_ciface_jit_test_cancel_inverses_lowering_transform_applied_workflow(ptr %0, ptr %1) {
  %3 = load { ptr, ptr, i64 }, ptr %1, align 8
  %4 = extractvalue { ptr, ptr, i64 } %3, 0
  %5 = extractvalue { ptr, ptr, i64 } %3, 1
  %6 = extractvalue { ptr, ptr, i64 } %3, 2
  %7 = call { { ptr, ptr, i64 }, { ptr, ptr, i64 }, { ptr, ptr, i64 } } @jit_test_cancel_inverses_lowering_transform_applied_workflow(ptr %4, ptr %5, i64 %6)
  store { { ptr, ptr, i64 }, { ptr, ptr, i64 }, { ptr, ptr, i64 } } %7, ptr %0, align 8
  ret void
}

define internal { ptr, ptr, i64 } @f_0(ptr %0, ptr %1, i64 %2) {
  call void @__catalyst__rt__device_init(ptr @"/Users/christina/Prog/catalyst_env/lib/python3.12/site-packages/pennylane_lightning/liblightning_qubit_catalyst.dylib", ptr @LightningSimulator, ptr @"{'mcmc': False, 'num_burnin': 0, 'kernel_name': None}", i64 0, i1 false)
  %4 = call ptr @__catalyst__rt__qubit_allocate_array(i64 1)
  %5 = call ptr @__catalyst__rt__array_get_element_ptr_1d(ptr %4, i64 0)
  %6 = load ptr, ptr %5, align 8
  %7 = load double, ptr %1, align 8
  call void @__catalyst__qis__RX(double %7, ptr %6, ptr null)
  %8 = call i64 @__catalyst__qis__NamedObs(i64 3, ptr %6)
  %9 = call double @__catalyst__qis__Expval(i64 %8)
  %10 = call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %11 = ptrtoint ptr %10 to i64
  %12 = add i64 %11, 63
  %13 = urem i64 %12, 64
  %14 = sub i64 %12, %13
  %15 = inttoptr i64 %14 to ptr
  %16 = insertvalue { ptr, ptr, i64 } poison, ptr %10, 0
  %17 = insertvalue { ptr, ptr, i64 } %16, ptr %15, 1
  %18 = insertvalue { ptr, ptr, i64 } %17, i64 0, 2
  store double %9, ptr %15, align 8
  call void @__catalyst__rt__qubit_release_array(ptr %4)
  call void @__catalyst__rt__device_release()
  ret { ptr, ptr, i64 } %18
}

define internal { ptr, ptr, i64 } @g_0(ptr %0, ptr %1, i64 %2) {
  call void @__catalyst__rt__device_init(ptr @"/Users/christina/Prog/catalyst_env/lib/python3.12/site-packages/pennylane_lightning/liblightning_qubit_catalyst.dylib", ptr @LightningSimulator, ptr @"{'mcmc': False, 'num_burnin': 0, 'kernel_name': None}", i64 0, i1 false)
  %4 = call ptr @__catalyst__rt__qubit_allocate_array(i64 1)
  %5 = call ptr @__catalyst__rt__array_get_element_ptr_1d(ptr %4, i64 0)
  %6 = load ptr, ptr %5, align 8
  %7 = load double, ptr %1, align 8
  call void @__catalyst__qis__RX(double %7, ptr %6, ptr null)
  call void @__catalyst__qis__Hadamard(ptr %6, ptr null)
  call void @__catalyst__qis__Hadamard(ptr %6, ptr null)
  %8 = call i64 @__catalyst__qis__NamedObs(i64 3, ptr %6)
  %9 = call double @__catalyst__qis__Expval(i64 %8)
  %10 = call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %11 = ptrtoint ptr %10 to i64
  %12 = add i64 %11, 63
  %13 = urem i64 %12, 64
  %14 = sub i64 %12, %13
  %15 = inttoptr i64 %14 to ptr
  %16 = insertvalue { ptr, ptr, i64 } poison, ptr %10, 0
  %17 = insertvalue { ptr, ptr, i64 } %16, ptr %15, 1
  %18 = insertvalue { ptr, ptr, i64 } %17, i64 0, 2
  store double %9, ptr %15, align 8
  call void @__catalyst__rt__qubit_release_array(ptr %4)
  call void @__catalyst__rt__device_release()
  ret { ptr, ptr, i64 } %18
}

define internal { ptr, ptr, i64 } @h_0(ptr %0, ptr %1, i64 %2) {
  call void @__catalyst__rt__device_init(ptr @"/Users/christina/Prog/catalyst_env/lib/python3.12/site-packages/pennylane_lightning/liblightning_qubit_catalyst.dylib", ptr @LightningSimulator, ptr @"{'mcmc': False, 'num_burnin': 0, 'kernel_name': None}", i64 0, i1 false)
  %4 = call ptr @__catalyst__rt__qubit_allocate_array(i64 1)
  %5 = call ptr @__catalyst__rt__array_get_element_ptr_1d(ptr %4, i64 0)
  %6 = load ptr, ptr %5, align 8
  call void @__catalyst__qis__Hadamard(ptr %6, ptr null)
  %7 = load double, ptr %1, align 8
  call void @__catalyst__qis__RX(double %7, ptr %6, ptr null)
  call void @__catalyst__qis__Hadamard(ptr %6, ptr null)
  %8 = call i64 @__catalyst__qis__NamedObs(i64 3, ptr %6)
  %9 = call double @__catalyst__qis__Expval(i64 %8)
  %10 = call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %11 = ptrtoint ptr %10 to i64
  %12 = add i64 %11, 63
  %13 = urem i64 %12, 64
  %14 = sub i64 %12, %13
  %15 = inttoptr i64 %14 to ptr
  %16 = insertvalue { ptr, ptr, i64 } poison, ptr %10, 0
  %17 = insertvalue { ptr, ptr, i64 } %16, ptr %15, 1
  %18 = insertvalue { ptr, ptr, i64 } %17, i64 0, 2
  store double %9, ptr %15, align 8
  call void @__catalyst__rt__qubit_release_array(ptr %4)
  call void @__catalyst__rt__device_release()
  ret { ptr, ptr, i64 } %18
}

define void @setup() {
  call void @__catalyst__rt__initialize(ptr null)
  ret void
}

define void @teardown() {
  call void @__catalyst__rt__finalize()
  ret void
}

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #0

attributes #0 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
