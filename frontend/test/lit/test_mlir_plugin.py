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
This test makes sure that we can use plugins from the compiler.

Given the standalone-plugin in the MLIR repository, can we verify that
it works when loading it from python?

This test uses a lot of machinery that is not exposed to the user.
However, testing the standalone-plugin (as written in the LLVM repository)
is impractical. The standalone-plugin rewrites all symbols with the name
`bar` to symbols with the name `foo`. However, since the standalone
plugin is meant to be more of an example, it does not modify
the uses of symbol `bar` and change them to `foo`.

What this practically means for this test is that if we were to write
something like the following:

  ```python
  import pennylane as qml
  from catalyst import qjit
  from pathlib import Path

  from catalyst.passes import apply_pass

  plugin = Path("./mlir/standalone/build/lib/StandalonePlugin.so")

  @apply_pass("standalone-switch-bar-foo")
  @qml.qnode(qml.device("lightning.qubit", wires=1))
  def bar():
    return qml.state()

  @qjit(keep_intermediate=True, verbose=True, pass_plugins=[plugin], dialect_plugins=[plugin])
  def module():
    return bar()

  print(module.mlir)
  ```

It would succeed at generating correct MLIR during the lowering from JAXPR to MLIR.
However, after the `standalone-switch-bar-foo` pass, the verifier would fail
because it would see callsites to `@bar` but no definitions for `@bar`.

As such, this test is perhaps a bit more limited. It does not test the
apply_pass interface nor the pass_plugins directly. Instead, it tests
that the `standalone-switch-bar-foo` pass can be executed using lower level
APIs, like the Compiler and options.
"""

import platform
from pathlib import Path

import pennylane as qml

import catalyst
from catalyst import qjit
from catalyst.compiler import _quantum_opt
from catalyst.utils.runtime_environment import get_bin_path

mlir_module = """
module @module {
  module @module_qnode {
    module attributes {transform.with_named_sequence} {
      transform.named_sequence @__transform_main(%arg0: !transform.op<"builtin.module">) {
        %0 = transform.apply_registered_pass "standalone-switch-bar-foo" to %arg0 : (!transform.op<"builtin.module">) -> !transform.op<"builtin.module">
        transform.yield 
      }
    }
    // CHECK: func.func private @foo()
    func.func private @bar() -> (tensor<i64>) {
      %c = stablehlo.constant dense<0> : tensor<i64>
      return %c : tensor<i64>
    }
  }
}
"""

ext = "so" if platform.system() == "Linux" else "dylib"
plugin_path = get_bin_path("cli", "CATALYST_BIN_DIR") + f"/../lib/StandalonePlugin.{ext}"
plugin = Path(plugin_path)
custom_pipeline = [("run_only_plugin", ["builtin.module(apply-transform-sequence)"])]
mlir_string = _quantum_opt(
    ("--load-pass-plugin", plugin),
    ("--load-dialect-plugin", plugin),
    ("--pass-pipeline", "builtin.module(apply-transform-sequence)"),
    stdin=mlir_module,
)
print(mlir_string)


def test_pass_options():
    """Is the option in the generated MLIR?"""

    @qjit(target="mlir")
    # CHECK: options = "an-option maxValue=1"
    @catalyst.passes.apply_pass("some-pass", "an-option", maxValue=1)
    @qml.qnode(qml.device("null.qubit", wires=1))
    def example():
        return qml.state()

    print(example.mlir)


test_pass_options()
