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
"""Core API for registering xDSL transforms for use with PennyLane and Catalyst."""

from pennylane.transforms.core.transform_dispatcher import TransformDispatcher
from xdsl.passes import ModulePass

from .apply_transform_sequence import register_pass


class PassDispatcher(TransformDispatcher):
    """Wrapper class for applying passes to QJIT-ed workflows."""

    module_pass: ModulePass

    def __init__(self, module_pass: ModulePass):
        self.module_pass = module_pass
        super().__init__(pass_name=module_pass.name)


def compiler_transform(module_pass: ModulePass) -> PassDispatcher:
    """Wrapper function to register xDSL passes to use with QJIT-ed workflows."""
    dispatcher = PassDispatcher(module_pass)

    # Registration for apply-transform-sequence interpreter
    def get_pass_cls():
        return module_pass

    register_pass(module_pass.name, get_pass_cls)
    return dispatcher
