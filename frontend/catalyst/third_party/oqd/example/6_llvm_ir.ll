; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

declare void @__catalyst__oqd__greetings()

declare void @__catalyst__rt__finalize()

declare void @__catalyst__rt__initialize(ptr)

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
