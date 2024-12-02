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
"""
This file contains a couple of tests for the capture of measurement primitives into jaxpr.
"""

import jax
import pennylane as qml

import catalyst
from catalyst.debug import get_compilation_stage, replace_ir
from catalyst.jax_primitives import (
    compbasis_p,
    expval_p,
    probs_p,
    state_p,
    var_p,
)


def test_dynamic_sample_backend_functionality():
    """Test that a `sample` program with dynamic shots can be executed correctly."""

    @catalyst.qjit(keep_intermediate=True)
    def workflow_dyn_sample(shots):
        # qml.device still needs concrete shots
        device = qml.device("lightning.qubit", wires=1, shots=10)

        @qml.qnode(device)
        def circuit():
            qml.RX(1.5, 0)
            return qml.sample()

        return circuit()

    workflow_dyn_sample(10)
    old_ir = get_compilation_stage(workflow_dyn_sample, "mlir")
    workflow_dyn_sample.workspace.cleanup()

    new_ir = old_ir.replace(
        "catalyst.launch_kernel @module_circuit::@circuit() : () -> tensor<10x1xi64>",
        "catalyst.launch_kernel @module_circuit::@circuit(%arg0) : (tensor<i64>) -> tensor<?x1xi64>",
    )
    new_ir = new_ir.replace(
        "func.func public @circuit() -> tensor<10x1xi64>",
        "func.func public @circuit(%arg0: tensor<i64>) -> tensor<?x1xi64>",
    )
    new_ir = new_ir.replace(
        "quantum.device shots(%extracted) [",
        """%shots = tensor.extract %arg0[] : tensor<i64>
      quantum.device shots(%shots) [""",
    )
    new_ir = new_ir.replace("tensor<10x1x", "tensor<?x1x")

    replace_ir(workflow_dyn_sample, "mlir", new_ir)
    res = workflow_dyn_sample(37)
    assert len(res) == 37

    workflow_dyn_sample.workspace.cleanup()

    # Save for WIP. Remove when work is done.
    _new_ir = """
module @workflow {
  func.func public @jit_workflow(%arg0: tensor<i64>) -> tensor<?x1xi64> attributes {llvm.emit_c_interface} {
    %0 = catalyst.launch_kernel @module_circuit::@circuit(%arg0) : (tensor<i64>) -> tensor<?x1xi64>
    return %0 : tensor<?x1xi64>
  }
  module @module_circuit {
    func.func public @circuit(%arg0: tensor<i64>) -> tensor<?x1xi64> attributes {diff_method = "parameter-shift", llvm.linkage = #llvm.linkage<internal>, qnode} {
      %shots = tensor.extract %arg0[] : tensor<i64>
      quantum.device["/home/paul.wang/.local/lib/python3.10/site-packages/pennylane_lightning/liblightning_qubit_catalyst.so", "LightningSimulator", "{'shots': 10, 'mcmc': False, 'num_burnin': 0, 'kernel_name': None}"] shots %shots
      %c = stablehlo.constant dense<1> : tensor<i64>
      %0 = quantum.alloc( 1) : !quantum.reg
      %c_0 = stablehlo.constant dense<0> : tensor<i64>
      %extracted = tensor.extract %c_0[] : tensor<i64>
      %1 = quantum.extract %0[%extracted] : !quantum.reg -> !quantum.bit
      %cst = stablehlo.constant dense<1.500000e+00> : tensor<f64>
      %extracted_1 = tensor.extract %cst[] : tensor<f64>
      %out_qubits = quantum.custom "RX"(%extracted_1) %1 : !quantum.bit
      %2 = quantum.compbasis %out_qubits : !quantum.obs
      %3 = quantum.sample %2 : tensor<?x1xf64>
      %4 = stablehlo.convert %3 : (tensor<?x1xf64>) -> tensor<?x1xi64>
      %c_4 = stablehlo.constant dense<0> : tensor<i64>
      %extracted_5 = tensor.extract %c_4[] : tensor<i64>
      %5 = quantum.insert %0[%extracted_5], %out_qubits : !quantum.reg, !quantum.bit
      quantum.dealloc %5 : !quantum.reg
      quantum.device_release
      return %4 : tensor<?x1xi64>
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


def test_dynamic_counts_backend_functionality():
    """Test that a `counts` program with dynamic shots can be executed correctly."""

    @catalyst.qjit(keep_intermediate=True)
    def workflow_dyn_counts(shots):
        # qml.device still needs concrete shots
        device = qml.device("lightning.qubit", wires=1, shots=10)

        @qml.qnode(device)
        def circuit():
            qml.RX(1.5, 0)
            return qml.counts()

        return circuit()

    workflow_dyn_counts(10)
    old_ir = get_compilation_stage(workflow_dyn_counts, "mlir")
    workflow_dyn_counts.workspace.cleanup()

    new_ir = old_ir.replace(
        "catalyst.launch_kernel @module_circuit::@circuit() : () ->",
        "catalyst.launch_kernel @module_circuit::@circuit(%arg0) : (tensor<i64>) ->",
    )
    new_ir = new_ir.replace(
        "func.func public @circuit() ->", "func.func public @circuit(%arg0: tensor<i64>) ->"
    )
    new_ir = new_ir.replace(
        "quantum.device shots(%extracted) [",
        """%shots = tensor.extract %arg0[] : tensor<i64>
      quantum.device shots(%shots) [""",
    )

    replace_ir(workflow_dyn_counts, "mlir", new_ir)
    res = workflow_dyn_counts(4000)
    print("after: ", res)
    assert res[1][0] + res[1][1] == 4000

    workflow_dyn_counts.workspace.cleanup()

    # Save for WIP. Remove when work is done.
    _new_ir = """module @workflow {
  func.func public @jit_workflow(%arg0: tensor<i64>) -> (tensor<2xi64>, tensor<2xi64>) attributes {llvm.emit_c_interface} {
    %1:2 = catalyst.launch_kernel @module_circuit::@circuit(%arg0) : (tensor<i64>) -> (tensor<2xi64>, tensor<2xi64>)
    return %1#0, %1#1 : tensor<2xi64>, tensor<2xi64>
  }
  module @module_circuit {
    func.func public @circuit(%arg0: tensor<i64>) -> (tensor<2xi64>, tensor<2xi64>) attributes {diff_method = "parameter-shift", llvm.linkage = #llvm.linkage<internal>, qnode} {
      %shots = tensor.extract %arg0[] : tensor<i64>
      quantum.device["/home/paul.wang/.local/lib/python3.10/site-packages/pennylane_lightning/liblightning_qubit_catalyst.so", "LightningSimulator", "{'mcmc': False, 'num_burnin': 0, 'kernel_name': None}"] shots %shots
      %c = stablehlo.constant dense<1> : tensor<i64>
      %0 = quantum.alloc( 1) : !quantum.reg
      %c_0 = stablehlo.constant dense<0> : tensor<i64>
      %extracted = tensor.extract %c_0[] : tensor<i64>
      %cst = stablehlo.constant dense<15.70000e-01> : tensor<f64>
      %extracted_1 = tensor.extract %cst[] : tensor<f64>
      %1 = quantum.extract %0[%extracted] : !quantum.reg -> !quantum.bit
      %out_qubits = quantum.custom "RX"(%extracted_1) %1 : !quantum.bit
      %4 = quantum.compbasis %out_qubits : !quantum.obs
      %c_2 = stablehlo.constant dense<10> : tensor<i64>
      %extracted_3 = tensor.extract %c_2[] : tensor<i64>
      %dyn_shots = tensor.extract %arg0[] : tensor<i64>
      %eigvals, %counts = quantum.counts %4 : tensor<2xf64>, tensor<2xi64>
      %5 = stablehlo.convert %eigvals : (tensor<2xf64>) -> tensor<2xi64>
      %c_4 = stablehlo.constant dense<0> : tensor<i64>
      %extracted_5 = tensor.extract %c_4[] : tensor<i64>
      %6 = quantum.insert %0[%extracted_5], %out_qubits : !quantum.reg, !quantum.bit
      quantum.dealloc %6 : !quantum.reg
      quantum.device_release
      return %5, %counts : tensor<2xi64>, tensor<2xi64>
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
}"""


def test_expval():
    """Test that the expval primitive can be captured by jaxpr."""

    def f():
        obs = compbasis_p.bind()
        return expval_p.bind(obs, shots=5, shape=(1,))

    jaxpr = jax.make_jaxpr(f)()
    assert jaxpr.eqns[1].primitive == expval_p
    assert jaxpr.eqns[1].params == {"shape": (1,), "shots": 5}
    assert jaxpr.eqns[1].outvars[0].aval.shape == ()


def test_var():
    """Test that the var primitive can be captured by jaxpr."""

    def f():
        obs = compbasis_p.bind()
        return var_p.bind(obs, shots=5, shape=(1,))

    jaxpr = jax.make_jaxpr(f)()
    assert jaxpr.eqns[1].primitive == var_p
    assert jaxpr.eqns[1].params == {"shape": (1,), "shots": 5}
    assert jaxpr.eqns[1].outvars[0].aval.shape == ()


def test_probs():
    """Test that the var primitive can be captured by jaxpr."""

    def f():
        obs = compbasis_p.bind()
        return probs_p.bind(obs, shots=5, shape=(1,))

    jaxpr = jax.make_jaxpr(f)()
    assert jaxpr.eqns[1].primitive == probs_p
    assert jaxpr.eqns[1].params == {"shape": (1,), "shots": 5}
    assert jaxpr.eqns[1].outvars[0].aval.shape == (1,)


def test_state():
    """Test that the state primitive can be captured by jaxpr."""

    def f():
        obs = compbasis_p.bind()
        return state_p.bind(obs, shots=5, shape=(1,))

    jaxpr = jax.make_jaxpr(f)()
    assert jaxpr.eqns[1].primitive == state_p
    assert jaxpr.eqns[1].params == {"shape": (1,), "shots": 5}
    assert jaxpr.eqns[1].outvars[0].aval.shape == (1,)
