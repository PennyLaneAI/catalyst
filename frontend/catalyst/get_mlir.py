# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from copy import deepcopy
from functools import wraps

from pennylane.workflow import get_compile_pipeline

from .jit import QJIT


def get_mlir(workflow: QJIT,  level : int | str = "user"):
    """Extract the optimized MLIR after a certain number of user transforms.
    
    Args:
        workflow (QJIT): a qjitted workflow.
        level (int | str): an indication of which user transforms to apply.

    Returns:
        function: a function with the same call signature as ``workflow`` that returns
            a string for the MLIR.

    .. code-block:: python

        @qp.qjit(capture=True)
        @qp.transforms.merge_rotations
        @qp.qnode(qp.device('lightning.qubit', wires=10))
        def c(x):
            qp.RX(x, 0)
            qp.RX(x, 0)
            return qp.expval(qp.Z(0))

    If we use the default ``level="user"``, we can see that the two ``RX`` gates got merged
    together.

    >>> from catalyst.get_mlir import get_mlir
    >>> print(get_mlir(c)(0.5))
    module @c {
        func.func public @jit_c(%arg0: tensor<f64>) -> tensor<f64> attributes {llvm.emit_c_interface} {
            %0 = call @c_0(%arg0) : (tensor<f64>) -> tensor<f64>
            return %0 : tensor<f64>
        }
        func.func public @c_0(%arg0: tensor<f64>) -> tensor<f64> attributes {diff_method = "adjoint", llvm.linkage = #llvm.linkage<internal>, qnode} {
            %c0_i64 = arith.constant 0 : i64
            quantum.device shots(%c0_i64) ["/Users/christina/Prog/catalyst_env/lib/python3.13/site-packages/pennylane_lightning/liblightning_qubit_catalyst.dylib", "LightningSimulator", "{'mcmc': False, 'num_burnin': 0, 'kernel_name': None}"]
            %0 = quantum.alloc( 10) : !quantum.reg
            %extracted = tensor.extract %arg0[] : tensor<f64>
            %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
            %extracted_0 = tensor.extract %arg0[] : tensor<f64>
            %2 = arith.addf %extracted, %extracted_0 : f64
            %out_qubits = quantum.custom "RX"(%2) %1 : !quantum.bit
            %3 = quantum.namedobs %out_qubits[ PauliZ] : !quantum.obs
            %4 = quantum.insert %0[ 0], %out_qubits : !quantum.reg, !quantum.bit
            %5 = quantum.expval %3 : f64
            %from_elements = tensor.from_elements %5 : tensor<f64>
            quantum.dealloc %4 : !quantum.reg
            quantum.device_release
            return %from_elements : tensor<f64>
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

    We can also get the mlir before the transform has been applied:

    >>> print(get_mlir(c, level=0)(0.5))
    ...
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %out_qubits = quantum.custom "RX"(%extracted) %1 : !quantum.bit
    %extracted_0 = tensor.extract %arg0[] : tensor<f64>
    %out_qubits_1 = quantum.custom "RX"(%extracted_0) %out_qubits : !quantum.bit
    %2 = quantum.namedobs %out_qubits_1[ PauliZ] : !quantum.obs
    ...





    """
    if not isinstance(workflow, QJIT):
        raise ValueError(f"get_mlir accepts a QJIT object. Got {workflow}.")
    @wraps(workflow)
    def wrapper(*args, **kwargs) -> str:
        w = deepcopy(workflow)
        qnode = w.user_function
        # pylint: disable=protected-access
        qnode._compile_pipeline = get_compile_pipeline(w, level)(*args, **kwargs)
        
        w.compile_options.target = "mlir"
        w.compile_options.lower_to_llvm = False
        w.compile_options.pipelines = [("pipe", ["quantum-compilation-stage"])]

        w.workspace = w._get_workspace()
        w.jaxed_function = None
        w.jaxpr, w.out_type, w.out_treedef, w.c_sig = w.capture(args, **kwargs)
        w.mlir_module = w.generate_ir()
        return str(w.mlir_opt)
    return wrapper

        

