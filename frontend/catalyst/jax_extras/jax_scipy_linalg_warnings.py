# Copyright 2022-2024 Xanadu Quantum Technologies Inc.

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
This module contains warnings for using jax.scipy.linalg functions inside qjit.
Due to improperly linked lapack symbols, occasionally these functions give wrong
numerical results when used in a qjit context. 
As for now, we warn users to wrap all of these with a catalyst.accelerate() callback. 
This patch should be removed when we have proper linkage to lapack.
See:
    https://app.shortcut.com/xanaduai/story/70899/find-a-system-to-automatically-create-a-custom-call-library-from-the-one-in-jax
    https://github.com/PennyLaneAI/catalyst/issues/753
    https://github.com/PennyLaneAI/catalyst/issues/1071
"""

import warnings

import jax

from catalyst.tracing.contexts import AccelerateContext


class JaxLinalgWarner:
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, *args, **kwargs):
        if not AccelerateContext.am_inside_accelerate():
            warnings.warn(
                f"""
    jax.scipy.linalg.{self.fn.__name__} occasionally gives wrong numerical results
    when used within a qjit-compiled function.
    See https://github.com/PennyLaneAI/catalyst/issues/1071.
    In the meantime, we recommend catalyst.accelerate to call
    the underlying {self.fn.__name__} function directly:

    @qjit
    def f(A):
        return catalyst.accelerate(jax.scipy.linalg.{self.fn.__name__})(A)

    See https://docs.pennylane.ai/projects/catalyst/en/latest/code/api/catalyst.accelerate.html
    """
            )
        return (self.fn)(*args, **kwargs)
