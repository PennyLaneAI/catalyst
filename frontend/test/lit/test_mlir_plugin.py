# Copyright 2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# RUN: %PYTHON %s | FileCheck %s

"""
This test makes sure that we can use plugins from the compiler

Given the standalone-plugin in the MLIR repository, can we verify that
it works when loading it from python?

This test uses a lot of machinery that is not exposed to the user.
However, testing the standalone-plugin (as written in the LLVM repository)
is impractical. The standalone-plugin rewrites a symbols with the name
`bar` to symbols with the name `foo`. However, since the standalone
plugin is meant to be more of an example, it does not modify
the uses of symbol `bar` and change them to `foo`.

What this practically means for this test is that if we were to write
something like the following:

  ```python
  import pennylane as qml
  from pathlib import Path

  from catalyst.passes import apply_pass

  plugin = Path("./mlir/standalone/build/lib/StandalonePlugin.so")

  @apply_pass("standalone-switch-bar-foo")
  @qml.qnode(qml.device("lightning.qubit", wires=1))
  def bar():
    return qml.state()

  @qml.qjit(keep_intermediate=True, verbose=True, pass_plugins=[plugin], dialect_plugins=[plugin])
  def module():
    return bar()

  print(module.mlir)
  ```

It would succeed in generate correct MLIR during the lowering from JAXPR to MLIR.
However, after the `standalone-switch-bar-foo` pass, the verifier would fail
because it would see callsites to `@bar` but no definitions for `@bar`.

As such, this test is perhaps a bit more limited. It does not test the
apply_pass interface nor the pass_plugins directly. Instead, it tests
that the `standalone-switch-bar-foo` pass can be executed using lower level
APIs, like the Compiler and options.
"""

import platform
from pathlib import Path

from catalyst.compiler import CompileOptions, Compiler
from catalyst.utils.filesystem import WorkspaceManager
from catalyst.utils.runtime_environment import get_bin_path

# CHECK: module
mlir_module = """
module @module {
  func.func public @jit_module() -> tensor<1xcomplex<f64>> attributes {llvm.emit_c_interface} {
    %0 = catalyst.launch_kernel @module_qnode::@qnode() : () -> tensor<1xcomplex<f64>>
    return %0 : tensor<1xcomplex<f64>>
  }
  module @module_qnode {
    module attributes {transform.with_named_sequence} {
      transform.named_sequence @__transform_main(%arg0: !transform.op<"builtin.module">) {
        %0 = transform.apply_registered_pass "standalone-switch-bar-foo" to %arg0 : (!transform.op<"builtin.module">) -> !transform.op<"builtin.module">
        transform.yield 
      }
    }
    func.func public @qnode() -> tensor<1xcomplex<f64>> attributes {diff_method = "parameter-shift", llvm.linkage = #llvm.linkage<internal>, qnode} {
      quantum.device["/home/ubuntu/code/env/lib/python3.10/site-packages/pennylane_lightning/liblightning_qubit_catalyst.so", "LightningSimulator", "{'shots': 0, 'mcmc': False, 'num_burnin': 0, 'kernel_name': None}"]
      %c = stablehlo.constant dense<0> : tensor<i64>
      %0 = quantum.alloc( 1) : !quantum.reg
      %1 = call @baz() : () -> tensor<i64>
      %extracted = tensor.extract %1[] : tensor<i64>
      %2 = quantum.extract %0[%extracted] : !quantum.reg -> !quantum.bit
      %out_qubits = quantum.custom "Hadamard"() %2 : !quantum.bit
      %extracted_0 = tensor.extract %1[] : tensor<i64>
      %3 = quantum.insert %0[%extracted_0], %out_qubits : !quantum.reg, !quantum.bit
      %4 = quantum.compbasis  : !quantum.obs
      %5 = quantum.state %4 : tensor<1xcomplex<f64>>
      quantum.dealloc %3 : !quantum.reg
      quantum.device_release
      return %5 : tensor<1xcomplex<f64>>
    }
    func.func private @baz() -> (tensor<i64> {mhlo.layout_mode = "default"}) attributes {llvm.linkage = #llvm.linkage<internal>} {
      %c = stablehlo.constant dense<0> : tensor<i64>
      return %c : tensor<i64>
    }
    func.func private @bar() -> (tensor<i64> {mhlo.layout_mode = "default"}) attributes {llvm.linkage = #llvm.linkage<internal>} {
      %c = stablehlo.constant dense<0> : tensor<i64>
      return %c : tensor<i64>
    }
  }
  func.func @setup() {
    quantum.init
    return
  }
  func.func @teardown() {
    quantum.finalize
    return
  }
}
"""

ext = "so" if platform.system() == "Linux" else "dylib"
plugin_path = get_bin_path("cli", "CATALYST_BIN_DIR") + f"/../lib/StandalonePlugin.{ext}"
plugin = Path(plugin_path)
custom_pipeline = [("run_only_plugin", ["builtin.module(apply-transform-sequence)"])]
options = CompileOptions(
    pipelines=custom_pipeline, lower_to_llvm=False, pass_plugins=[plugin], dialect_plugins=[plugin]
)
workspace = WorkspaceManager.get_or_create_workspace("test", None)
custom_compiler = Compiler(options)
_, mlir_string = custom_compiler.run_from_ir(mlir_module, "test", workspace)
print(mlir_string)
