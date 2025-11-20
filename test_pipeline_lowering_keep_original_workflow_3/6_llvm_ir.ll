; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

@"{'mcmc': False, 'num_burnin': 0, 'kernel_name': None}" = internal constant [54 x i8] c"{'mcmc': False, 'num_burnin': 0, 'kernel_name': None}\00"
@LightningSimulator = internal constant [19 x i8] c"LightningSimulator\00"
@"/Users/christina/Prog/catalyst_env/lib/python3.12/site-packages/pennylane_lightning/liblightning_qubit_catalyst.dylib" = internal constant [118 x i8] c"/Users/christina/Prog/catalyst_env/lib/python3.12/site-packages/pennylane_lightning/liblightning_qubit_catalyst.dylib\00"

declare void @__catalyst__rt__finalize()

declare void @__catalyst__rt__initialize(ptr)

declare void @__catalyst__rt__device_release()

declare void @__catalyst__rt__qubit_release_array(ptr)

declare double @__catalyst__qis__Expval(i64)

declare i64 @__catalyst__qis__NamedObs(i64, ptr)

declare void @__catalyst__qis__Hadamard(ptr, ptr)

declare void @__catalyst__qis__RX(double, ptr, ptr)

declare ptr @__catalyst__rt__array_get_element_ptr_1d(ptr, i64)

declare ptr @__catalyst__rt__qubit_allocate_array(i64)

declare void @__catalyst__rt__device_init(ptr, ptr, ptr, i64, i1)

declare ptr @_mlir_memref_to_llvm_alloc(i64)

define { { ptr, ptr, i64 }, { ptr, ptr, i64 } } @jit_test_pipeline_lowering_keep_original_workflow(ptr %0, ptr %1, i64 %2) {
  %4 = call { ptr, ptr, i64 } @f_0(ptr %0, ptr %1, i64 %2)
  %5 = call { ptr, ptr, i64 } @f_1_0(ptr %0, ptr %1, i64 %2)
  %6 = extractvalue { ptr, ptr, i64 } %4, 0
  %7 = ptrtoint ptr %6 to i64
  %8 = icmp eq i64 3735928559, %7
  br i1 %8, label %9, label %17

9:                                                ; preds = %3
  %10 = call ptr @_mlir_memref_to_llvm_alloc(i64 8)
  %11 = insertvalue { ptr, ptr, i64 } poison, ptr %10, 0
  %12 = insertvalue { ptr, ptr, i64 } %11, ptr %10, 1
  %13 = insertvalue { ptr, ptr, i64 } %12, i64 0, 2
  %14 = extractvalue { ptr, ptr, i64 } %4, 1
  %15 = extractvalue { ptr, ptr, i64 } %4, 2
  %16 = getelementptr inbounds double, ptr %14, i64 %15
  call void @llvm.memcpy.p0.p0.i64(ptr %10, ptr %16, i64 8, i1 false)
  br label %18

17:                                               ; preds = %3
  br label %18

18:                                               ; preds = %9, %17
  %19 = phi { ptr, ptr, i64 } [ %4, %17 ], [ %13, %9 ]
  br label %20

20:                                               ; preds = %18
  %21 = extractvalue { ptr, ptr, i64 } %5, 0
  %22 = ptrtoint ptr %21 to i64
  %23 = icmp eq i64 3735928559, %22
  br i1 %23, label %24, label %32

24:                                               ; preds = %20
  %25 = call ptr @_mlir_memref_to_llvm_alloc(i64 8)
  %26 = insertvalue { ptr, ptr, i64 } poison, ptr %25, 0
  %27 = insertvalue { ptr, ptr, i64 } %26, ptr %25, 1
  %28 = insertvalue { ptr, ptr, i64 } %27, i64 0, 2
  %29 = extractvalue { ptr, ptr, i64 } %5, 1
  %30 = extractvalue { ptr, ptr, i64 } %5, 2
  %31 = getelementptr inbounds double, ptr %29, i64 %30
  call void @llvm.memcpy.p0.p0.i64(ptr %25, ptr %31, i64 8, i1 false)
  br label %33

32:                                               ; preds = %20
  br label %33

33:                                               ; preds = %24, %32
  %34 = phi { ptr, ptr, i64 } [ %5, %32 ], [ %28, %24 ]
  br label %35

35:                                               ; preds = %33
  %36 = insertvalue { { ptr, ptr, i64 }, { ptr, ptr, i64 } } poison, { ptr, ptr, i64 } %19, 0
  %37 = insertvalue { { ptr, ptr, i64 }, { ptr, ptr, i64 } } %36, { ptr, ptr, i64 } %34, 1
  ret { { ptr, ptr, i64 }, { ptr, ptr, i64 } } %37
}

define void @_catalyst_pyface_jit_test_pipeline_lowering_keep_original_workflow(ptr %0, ptr %1) {
  %3 = load { ptr, ptr }, ptr %1, align 8
  %4 = extractvalue { ptr, ptr } %3, 0
  call void @_catalyst_ciface_jit_test_pipeline_lowering_keep_original_workflow(ptr %0, ptr %4)
  ret void
}

define void @_catalyst_ciface_jit_test_pipeline_lowering_keep_original_workflow(ptr %0, ptr %1) {
  %3 = load { ptr, ptr, i64 }, ptr %1, align 8
  %4 = extractvalue { ptr, ptr, i64 } %3, 0
  %5 = extractvalue { ptr, ptr, i64 } %3, 1
  %6 = extractvalue { ptr, ptr, i64 } %3, 2
  %7 = call { { ptr, ptr, i64 }, { ptr, ptr, i64 } } @jit_test_pipeline_lowering_keep_original_workflow(ptr %4, ptr %5, i64 %6)
  store { { ptr, ptr, i64 }, { ptr, ptr, i64 } } %7, ptr %0, align 8
  ret void
}

define internal { ptr, ptr, i64 } @f_0(ptr %0, ptr %1, i64 %2) {
  call void @__catalyst__rt__device_init(ptr @"/Users/christina/Prog/catalyst_env/lib/python3.12/site-packages/pennylane_lightning/liblightning_qubit_catalyst.dylib", ptr @LightningSimulator, ptr @"{'mcmc': False, 'num_burnin': 0, 'kernel_name': None}", i64 0, i1 false)
  %4 = call ptr @__catalyst__rt__qubit_allocate_array(i64 2)
  %5 = call ptr @__catalyst__rt__array_get_element_ptr_1d(ptr %4, i64 0)
  %6 = load ptr, ptr %5, align 8
  %7 = load double, ptr %1, align 8
  call void @__catalyst__qis__RX(double %7, ptr %6, ptr null)
  %8 = call ptr @__catalyst__rt__array_get_element_ptr_1d(ptr %4, i64 1)
  %9 = load ptr, ptr %8, align 8
  call void @__catalyst__qis__Hadamard(ptr %9, ptr null)
  call void @__catalyst__qis__Hadamard(ptr %9, ptr null)
  %10 = call i64 @__catalyst__qis__NamedObs(i64 2, ptr %6)
  %11 = call double @__catalyst__qis__Expval(i64 %10)
  %12 = call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %13 = ptrtoint ptr %12 to i64
  %14 = add i64 %13, 63
  %15 = urem i64 %14, 64
  %16 = sub i64 %14, %15
  %17 = inttoptr i64 %16 to ptr
  %18 = insertvalue { ptr, ptr, i64 } poison, ptr %12, 0
  %19 = insertvalue { ptr, ptr, i64 } %18, ptr %17, 1
  %20 = insertvalue { ptr, ptr, i64 } %19, i64 0, 2
  store double %11, ptr %17, align 8
  call void @__catalyst__rt__qubit_release_array(ptr %4)
  call void @__catalyst__rt__device_release()
  ret { ptr, ptr, i64 } %20
}

define internal { ptr, ptr, i64 } @f_1_0(ptr %0, ptr %1, i64 %2) {
  call void @__catalyst__rt__device_init(ptr @"/Users/christina/Prog/catalyst_env/lib/python3.12/site-packages/pennylane_lightning/liblightning_qubit_catalyst.dylib", ptr @LightningSimulator, ptr @"{'mcmc': False, 'num_burnin': 0, 'kernel_name': None}", i64 0, i1 false)
  %4 = call ptr @__catalyst__rt__qubit_allocate_array(i64 2)
  %5 = call ptr @__catalyst__rt__array_get_element_ptr_1d(ptr %4, i64 0)
  %6 = load ptr, ptr %5, align 8
  %7 = load double, ptr %1, align 8
  call void @__catalyst__qis__RX(double %7, ptr %6, ptr null)
  %8 = call ptr @__catalyst__rt__array_get_element_ptr_1d(ptr %4, i64 1)
  %9 = call i64 @__catalyst__qis__NamedObs(i64 2, ptr %6)
  %10 = call double @__catalyst__qis__Expval(i64 %9)
  %11 = call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %12 = ptrtoint ptr %11 to i64
  %13 = add i64 %12, 63
  %14 = urem i64 %13, 64
  %15 = sub i64 %13, %14
  %16 = inttoptr i64 %15 to ptr
  %17 = insertvalue { ptr, ptr, i64 } poison, ptr %11, 0
  %18 = insertvalue { ptr, ptr, i64 } %17, ptr %16, 1
  %19 = insertvalue { ptr, ptr, i64 } %18, i64 0, 2
  store double %10, ptr %16, align 8
  call void @__catalyst__rt__qubit_release_array(ptr %4)
  call void @__catalyst__rt__device_release()
  ret { ptr, ptr, i64 } %19
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
