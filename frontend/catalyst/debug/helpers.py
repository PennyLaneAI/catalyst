# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import functools

from catalyst.jit import QJIT
from catalyst.compiler import CompileOptions

def qjit_for_lit_tests(fn=None,
    *,
    autograph=False,
    autograph_include=(),
    async_qnodes=False,
    target="binary",
    keep_intermediate=False,
    verbose=False,
    logfile=None,
    pipelines=None,
    static_argnums=None,
    static_argnames=None,
    abstracted_axes=None,
    disable_assertions=False,
    seed=None,
    experimental_capture=False,
    circuit_transform_pipeline=None,
    pass_plugins=None,
    dialect_plugins=None,
    ):
    
    kwargs = copy.copy(locals())
    kwargs.pop("fn")
    if fn is None:
        return functools.partial(qjit_for_lit_tests, **kwargs)
    
    return QJITForLitTests(fn, CompileOptions(**kwargs))

class QJITForLitTests(QJIT):
    def __init__(self, *args, **kwargs):
        compile_options = args[1]
        compile_options.keep_intermediate = True
        compile_options.target = "mlir"
        super().__init__(*(args[0], compile_options), **kwargs)

    def __del__(self):
        self.workspace.cleanup()

