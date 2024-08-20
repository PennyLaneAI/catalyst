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

"""Tests JAX configuration utilities."""

import jax

from catalyst.jax_extras import transient_jax_config


def test_transient_jax_config():
    """Test that the ``transient_jax_config()`` context manager updates and
    restores the value of the JAX dynamic shapes option.
    """
    jax.config.update("jax_dynamic_shapes", False)

    with transient_jax_config({"jax_dynamic_shapes": True}):
        assert jax.config.jax_dynamic_shapes is True  # type: ignore

    assert jax.config.jax_dynamic_shapes is False  # type: ignore
