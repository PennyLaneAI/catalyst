; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

@upstate = internal constant [8 x i8] c"upstate\00"
@estate = internal constant [7 x i8] c"estate\00"
@downstate = internal constant [10 x i8] c"downstate\00"
@Yb171 = internal constant [6 x i8] c"Yb171\00"

declare void @__catalyst__oqd__greetings()

declare void @__catalyst__rt__finalize()

declare void @__catalyst__rt__initialize(ptr)

declare void @__catalyst__oqd__ion(ptr)

define void @ion_op() {
  %1 = alloca [3 x { ptr, i64, double, double, double, double, double, double, double }], i64 3, align 8
  store [3 x { ptr, i64, double, double, double, double, double, double, double }] [{ ptr, i64, double, double, double, double, double, double, double } { ptr @downstate, i64 6, double 4.000000e-01, double 5.000000e-01, double 6.000000e-01, double 8.000000e-01, double 9.000000e-01, double 1.000000e+00, double 0.000000e+00 }, { ptr, i64, double, double, double, double, double, double, double } { ptr @estate, i64 6, double 1.400000e+00, double 1.500000e+00, double 1.600000e+00, double 1.800000e+00, double 1.900000e+00, double 2.000000e+00, double 1.264300e+10 }, { ptr, i64, double, double, double, double, double, double, double } { ptr @upstate, i64 5, double 2.400000e+00, double 2.500000e+00, double 2.600000e+00, double 2.800000e+00, double 2.900000e+00, double 3.000000e+00, double 8.115200e+14 }], ptr %1, align 8
  %2 = alloca [3 x { ptr, ptr, double }], i64 3, align 8
  store [3 x { ptr, ptr, double }] [{ ptr, ptr, double } { ptr @estate, ptr @downstate, double 2.200000e+00 }, { ptr, ptr, double } { ptr @upstate, ptr @downstate, double 1.100000e+00 }, { ptr, ptr, double } { ptr @upstate, ptr @estate, double 3.300000e+00 }], ptr %2, align 8
  %3 = insertvalue { ptr, double, double, <3 x i64>, ptr, ptr } { ptr @Yb171, double 1.710000e+02, double 5.611000e+01, <3 x i64> <i64 0, i64 11, i64 3>, ptr undef, ptr undef }, ptr %1, 4
  %4 = insertvalue { ptr, double, double, <3 x i64>, ptr, ptr } %3, ptr %2, 5
  %5 = alloca { ptr, double, double, <3 x i64>, ptr, ptr }, i64 1, align 32
  store { ptr, double, double, <3 x i64>, ptr, ptr } %4, ptr %5, align 32
  call void @__catalyst__oqd__ion(ptr %5)
  ret void
}

define void @jit_f() {
  call void @f_0()
  call void @ion_op()
  ret void
}

define void @_catalyst_pyface_jit_f(ptr %0, ptr %1) {
  call void @_catalyst_ciface_jit_f(ptr %0)
  ret void
}

define void @_catalyst_ciface_jit_f(ptr %0) {
  call void @jit_f()
  ret void
}

define internal void @f_0() {
  call void @__catalyst__oqd__greetings()
  ret void
}

define void @setup() {
  call void @__catalyst__rt__initialize(ptr null)
  ret void
}

define void @teardown() {
  call void @__catalyst__rt__finalize()
  ret void
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
