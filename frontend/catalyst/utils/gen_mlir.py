# Copyright 2022-2023 Xanadu Quantum Technologies Inc.

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
Utility code for adding common functions to MLIR modules.
"""

from jax.interpreters.mlir import ir
from jaxlib.mlir.dialects._func_ops_gen import FuncOp


def gen_setup(ctx):
    """
    This function returns an MLIR module with the "setup" function. The setup
    function is a function that needs to be called before calling a
    JIT-compiled function. It initializes the global device context in the runtime.
    """
    txt = """
func.func @setup() -> () {
    "quantum.init"() : () -> ()
    return
}
"""
    return ir.Module.parse(txt, ctx)


def gen_teardown(ctx):
    """
    This function returns an MLIR module with the "teardown" function. The
    teardown function is a function that needs to be called after calling a
    JIT-compiled function. It destroys the global device context in the runtime.
    """
    txt = """
func.func @teardown () -> () {
    "quantum.finalize"() : () -> ()
    return
}
"""
    return ir.Module.parse(txt, ctx)


def inject_functions(module, ctx):
    """
    This function appends functions to the input module.
    """
    # Add C interface for the quantum function.
    module.body.operations[0].attributes["llvm.emit_c_interface"] = ir.UnitAttr.get(context=ctx)

    setup_module = gen_setup(ctx)
    setup_func = setup_module.body.operations[0]
    module.body.append(setup_func)

    teardown_module = gen_teardown(ctx)
    teardown_func = teardown_module.body.operations[0]
    module.body.append(teardown_func)

    mlir_qfunc = module.body.operations[0]
    assert isinstance(mlir_qfunc, FuncOp)
