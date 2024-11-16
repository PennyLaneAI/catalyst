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

import catalyst
from catalyst.jax_primitives import (
    compbasis_p,
    counts_p,
    expval_p,
    probs_p,
    sample_p,
    state_p,
    var_p,
)
from catalyst.jax_tracer import lower_jaxpr_to_mlir


def test_sample():
    """Test that the sample primitive can be captured into jaxpr."""

    def f():
        obs = compbasis_p.bind()
        return sample_p.bind(obs, shots=5, shape=(5, 0))

    jaxpr = jax.make_jaxpr(f)().jaxpr
    #breakpoint()
    mlir = lower_jaxpr_to_mlir(jax.make_jaxpr(f)(), "foo")[0]
    #breakpoint()
    assert jaxpr == """
{ lambda ; . let
    a:AbstractObs(num_qubits=0,primitive=compbasis) = compbasis
    b:f64[5,0] = sample[shape=(5, 0) shots=5] a
  in (b,) }
"""

    assert mlir == """
module @foo {
  func.func public @jit_foo() -> tensor<5x0xf64> {
    %0 = "quantum.compbasis"() : () -> !quantum.obs
    %c5_i64 = arith.constant 5 : i64
    %1 = "quantum.sample"(%0, %c5_i64) : (!quantum.obs, i64) -> tensor<5x0xf64>
    return %1 : tensor<5x0xf64>
  }
}
"""

    #assert jaxpr.eqns[1].primitive == sample_p
    #assert jaxpr.eqns[1].params == {"shape": (5, 0), "shots": 5}
    #assert jaxpr.eqns[1].outvars[0].aval.shape == (5, 0)


def test_sample_dynamic_shape():
    """Test that the sample primitive with dynamic shape can be captured into jaxpr."""

    def f(shots):
        obs = compbasis_p.bind()
        x = shots+1
        # Note that in `primitive.bind(args, kwargs)`, args are treated as jaxpr primitive's
        # proper arguments, and kwargs are treated as primitive's `params`
        # Proper primitive arguments are propagated as jaxpr variables,
        # whereas primitive params are tracers.
        return sample_p.bind(obs, x, shape=(x, 0))

    jaxpr = jax.make_jaxpr(f)(5).jaxpr
    #breakpoint()
    mlir = lower_jaxpr_to_mlir(jax.make_jaxpr(f)(5), "foo")[0]
    breakpoint()
    assert jaxpr == """
{ lambda ; a:i64[]. let
    b:AbstractObs(num_qubits=0,primitive=compbasis) = compbasis
    c:i64[] = add a 1
    d:f64[c,0] = sample[
      shape=(Traced<ShapedArray(int64[], weak_type=True)>with<DynamicJaxprTrace(level=1/0)>, 0)
    ] b c
  in (c, d) }
"""

    assert mlir == """
module @foo {
  func.func public @jit_foo(%arg0: tensor<i64>) -> (tensor<i64>, tensor<?x0xf64>) {
    %0 = "quantum.compbasis"() : () -> !quantum.obs
    %c = stablehlo.constant dense<1> : tensor<i64>
    %1 = stablehlo.add %arg0, %c : tensor<i64>
    %2 = "quantum.sample"(%0, %1) : (!quantum.obs, tensor<i64>) -> tensor<?x0xf64>
    return %1, %2 : tensor<i64>, tensor<?x0xf64>
  }
}
"""

    #assert jaxpr.eqns[1].primitive == sample_p
    #assert jaxpr.eqns[1].params == {"shape": (5, 0), "shots": 5}
    #assert jaxpr.eqns[1].outvars[0].aval.shape == (5, 0)



def test_new_sampleop_still_good_with_backend():
    from catalyst.debug import replace_ir
    import pennylane as qml

    @catalyst.qjit
    def workflow(shots):
        # qml.device still needs concrete shots
        device = qml.device("lightning.qubit", wires=1, shots=10)
        #breakpoint()
        @qml.qnode(device)
        def circuit():
            qml.RX(0.1, 0)
            #return qml.expval(qml.PauliZ(wires=0))
            return qml.sample()

        return circuit()

    new_ir = """
    module @workflow {
      func.func public @jit_workflow(%arg0: tensor<i64>) -> tensor<10x1xi64> attributes {llvm.emit_c_interface} {
        %0 = catalyst.launch_kernel @module_circuit::@circuit(%arg0) : (tensor<i64>) -> tensor<10x1xi64>
        return %0 : tensor<10x1xi64>
      }
      module attributes {transform.with_named_sequence} {
        transform.named_sequence @__transform_main(%arg0: !transform.op<"builtin.module">) {
          transform.yield
        }
      }
      module @module_circuit {
        //func.func public @circuit() -> tensor<10x1xi64> attributes {diff_method = "parameter-shift", llvm.linkage = #llvm.linkage<internal>, qnode} {
        func.func public @circuit(%shots: tensor<i64>) -> tensor<10x1xi64> attributes {diff_method = "parameter-shift", llvm.linkage = #llvm.linkage<internal>, qnode} {
          %c10_i64 = tensor.extract %shots[] : tensor<i64> // newline
          quantum.device["/home/paul.wang/catalyst_new/catalyst/frontend/catalyst/utils/../../../runtime/build/lib/librtd_lightning.so", "LightningSimulator", "{'shots': 10, 'mcmc': False, 'num_burnin': 0, 'kernel_name': None}"]
          //quantum.device["/home/paul.wang/catalyst_new/catalyst/frontend/catalyst/utils/../../../runtime/build/lib/librtd_lightning.so", "LightningSimulator", "{'shots': %c10_i64, 'mcmc': False, 'num_burnin': 0, 'kernel_name': None}"]
          %c = stablehlo.constant dense<1> : tensor<i64>
          %0 = quantum.alloc( 1) : !quantum.reg
          %c_0 = stablehlo.constant dense<0> : tensor<i64>
          %extracted = tensor.extract %c_0[] : tensor<i64>
          %1 = quantum.extract %0[%extracted] : !quantum.reg -> !quantum.bit
          %out_qubits = quantum.custom "Hadamard"() %1 : !quantum.bit
          %2 = quantum.compbasis %out_qubits : !quantum.obs
          //%c10_i64 = arith.constant 10 : i64
          %3 = quantum.sample %2 %c10_i64 : tensor<10x1xf64>
          %4 = stablehlo.convert %3 : (tensor<10x1xf64>) -> tensor<10x1xi64>
          %c_1 = stablehlo.constant dense<0> : tensor<i64>
          %extracted_2 = tensor.extract %c_1[] : tensor<i64>
          %5 = quantum.insert %0[%extracted_2], %out_qubits : !quantum.reg, !quantum.bit
          quantum.dealloc %5 : !quantum.reg
          quantum.device_release
          return %4 : tensor<10x1xi64>
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
    #breakpoint()
    replace_ir(workflow, "mlir", new_ir)
    print("after: ", workflow(10))
'''
>> pytest test_measurement_primitives.py -k new_sample -s
    test_measurement_primitives.py after:  [[1]
 [0]
 [0]
 [1]
 [0]
 [0]
 [1]
 [0]
 [1]
 [1]]

'''

def test_counts():
    """Test that the counts primitive can be captured by jaxpr."""

    def f():
        obs = compbasis_p.bind()
        return counts_p.bind(obs, shots=5, shape=(1,))

    jaxpr = jax.make_jaxpr(f)()
    assert jaxpr.eqns[1].primitive == counts_p
    assert jaxpr.eqns[1].params == {"shape": (1,), "shots": 5}
    assert jaxpr.eqns[1].outvars[0].aval.shape == (1,)
    assert jaxpr.eqns[1].outvars[1].aval.shape == (1,)


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
