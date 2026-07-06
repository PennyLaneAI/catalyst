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

        

