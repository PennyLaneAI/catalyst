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

declare void @__catalyst__qis__RX(double, ptr, ptr)

declare ptr @__catalyst__rt__array_get_element_ptr_1d(ptr, i64)

declare ptr @__catalyst__rt__qubit_allocate_array(i64)

declare void @__catalyst__rt__device_init(ptr, ptr, ptr, i64, i1)

declare ptr @_mlir_memref_to_llvm_alloc(i64)

define { ptr, ptr, i64 } @jit_test_pipeline_lowering_workflow(ptr %0, ptr %1, i64 %2) {
  %4 = call { ptr, ptr, i64 } @test_pipeline_lowering_workflow_0(ptr %0, ptr %1, i64 %2)
  %5 = extractvalue { ptr, ptr, i64 } %4, 0
  %6 = ptrtoint ptr %5 to i64
  %7 = icmp eq i64 3735928559, %6
  br i1 %7, label %8, label %16

8:                                                ; preds = %3
  %9 = call ptr @_mlir_memref_to_llvm_alloc(i64 8)
  %10 = insertvalue { ptr, ptr, i64 } poison, ptr %9, 0
  %11 = insertvalue { ptr, ptr, i64 } %10, ptr %9, 1
  %12 = insertvalue { ptr, ptr, i64 } %11, i64 0, 2
  %13 = extractvalue { ptr, ptr, i64 } %4, 1
  %14 = extractvalue { ptr, ptr, i64 } %4, 2
  %15 = getelementptr inbounds double, ptr %13, i64 %14
  call void @llvm.memcpy.p0.p0.i64(ptr %9, ptr %15, i64 8, i1 false)
  br label %17

16:                                               ; preds = %3
  br label %17

17:                                               ; preds = %8, %16
  %18 = phi { ptr, ptr, i64 } [ %4, %16 ], [ %12, %8 ]
  br label %19

19:                                               ; preds = %17
  ret { ptr, ptr, i64 } %18
}

define void @_catalyst_pyface_jit_test_pipeline_lowering_workflow(ptr %0, ptr %1) {
  %3 = load { ptr, ptr }, ptr %1, align 8
  %4 = extractvalue { ptr, ptr } %3, 0
  call void @_catalyst_ciface_jit_test_pipeline_lowering_workflow(ptr %0, ptr %4)
  ret void
}

define void @_catalyst_ciface_jit_test_pipeline_lowering_workflow(ptr %0, ptr %1) {
  %3 = load { ptr, ptr, i64 }, ptr %1, align 8
  %4 = extractvalue { ptr, ptr, i64 } %3, 0
  %5 = extractvalue { ptr, ptr, i64 } %3, 1
  %6 = extractvalue { ptr, ptr, i64 } %3, 2
  %7 = call { ptr, ptr, i64 } @jit_test_pipeline_lowering_workflow(ptr %4, ptr %5, i64 %6)
  store { ptr, ptr, i64 } %7, ptr %0, align 8
  ret void
}

define internal { ptr, ptr, i64 } @test_pipeline_lowering_workflow_0(ptr %0, ptr %1, i64 %2) {
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
