; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

@"{'shots': 1000}" = internal constant [16 x i8] c"{'shots': 1000}\00"
@oqd = internal constant [4 x i8] c"oqd\00"
@"/home/paul.wang/catalyst_new/catalyst/frontend/catalyst/utils/../../catalyst/third_party/oqd/src/build/librtd_oqd.so" = internal constant [117 x i8] c"/home/paul.wang/catalyst_new/catalyst/frontend/catalyst/utils/../../catalyst/third_party/oqd/src/build/librtd_oqd.so\00"

declare void @__catalyst__oqd__greetings()

declare void @__catalyst__rt__finalize()

declare void @__catalyst__rt__initialize(ptr)

declare void @__catalyst__rt__device_release()

declare void @__catalyst__rt__device_init(ptr, ptr, ptr, i64)

define void @jit_f() {
  call void @f_0()
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

define void @f_0() {
  call void @__catalyst__rt__device_init(ptr @"/home/paul.wang/catalyst_new/catalyst/frontend/catalyst/utils/../../catalyst/third_party/oqd/src/build/librtd_oqd.so", ptr @oqd, ptr @"{'shots': 1000}", i64 1000)
  call void @__catalyst__oqd__greetings()
  call void @__catalyst__rt__device_release()
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

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg) #0

attributes #0 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
