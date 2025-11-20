builtin.module @workflow {
  func.func public @jit_workflow() -> (tensor<16xcomplex<f64>>) attributes {llvm.emit_c_interface} {
    %0 = catalyst.launch_kernel @module_workflow::@workflow() : () -> tensor<16xcomplex<f64>>
    func.return %0 : tensor<16xcomplex<f64>>
  }
  builtin.module @module_workflow {
    builtin.module attributes {transform.with_named_sequence} {
      transform.named_sequence @__transform_main(%arg0 : !transform.op<"builtin.module">) {
        transform.yield
      }
    }
    func.func public @workflow() -> (tensor<16xcomplex<f64>>) attributes {diff_method = "adjoint", llvm.linkage = #llvm.linkage<internal>, qnode} {
      %0 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
      %1 = tensor.extract %0[] : tensor<i64>
      quantum.device shots(%1) ["/Users/mudit.pandey/.pyenv/versions/pennylane-xdsl/lib/python3.12/site-packages/pennylane_lightning/liblightning_qubit_catalyst.dylib", "LightningSimulator", "{'mcmc': False, 'num_burnin': 0, 'kernel_name': None}"]
      %2 = "stablehlo.constant"() <{value = dense<4> : tensor<i64>}> : () -> tensor<i64>
      %3 = quantum.alloc(4) : !quantum.reg
      %4 = tensor.extract %0[] : tensor<i64>
      %5 = quantum.extract %3[%4] : !quantum.reg -> !quantum.bit
      %6 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
      %7 = tensor.extract %6[] : tensor<i64>
      %8 = quantum.extract %3[%7] : !quantum.reg -> !quantum.bit
      %9, %10 = quantum.custom "CZ"() %5, %8 : !quantum.bit, !quantum.bit
      %11 = quantum.custom "PauliZ"() %10 : !quantum.bit
      %12 = quantum.custom "PauliX"() %9 : !quantum.bit
      %13 = quantum.custom "S"() %12 : !quantum.bit
      %14 = quantum.custom "Hadamard"() %13 : !quantum.bit
      %15 = "stablehlo.constant"() <{value = dense<2> : tensor<i64>}> : () -> tensor<i64>
      %16 = tensor.extract %15[] : tensor<i64>
      %17 = quantum.extract %3[%16] : !quantum.reg -> !quantum.bit
      %18, %19 = quantum.custom "CNOT"() %11, %17 : !quantum.bit, !quantum.bit
      %20 = quantum.custom "PauliY"() %19 : !quantum.bit
      %21 = "stablehlo.constant"() <{value = dense<3> : tensor<i64>}> : () -> tensor<i64>
      %22 = tensor.extract %21[] : tensor<i64>
      %23 = quantum.extract %3[%22] : !quantum.reg -> !quantum.bit
      %24 = quantum.custom "PauliX"() %23 : !quantum.bit
      %25 = tensor.extract %0[] : tensor<i64>
      %26 = quantum.insert %3[%25], %14 : !quantum.reg, !quantum.bit
      %27 = tensor.extract %6[] : tensor<i64>
      %28 = quantum.insert %26[%27], %18 : !quantum.reg, !quantum.bit
      %29 = tensor.extract %15[] : tensor<i64>
      %30 = quantum.insert %28[%29], %20 : !quantum.reg, !quantum.bit
      %31 = tensor.extract %21[] : tensor<i64>
      %32 = quantum.insert %30[%31], %24 : !quantum.reg, !quantum.bit
      %33 = quantum.compbasis qreg %32 : !quantum.obs
      %34 = quantum.state %33 : tensor<16xcomplex<f64>>
      quantum.dealloc %32 : !quantum.reg
      quantum.device_release
      func.return %34 : tensor<16xcomplex<f64>>
    }
  }
  func.func @setup() {
    quantum.init
    func.return
  }
  func.func @teardown() {
    quantum.finalize
    func.return
  }
}