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

"""Unit tests for the JAX primitives module."""

import pytest

from catalyst.jax_primitives import AbstractQbit, qinst_p


class TestQinstPrim:
    """Test the quantum instruction primitive."""

    def test_abstract_eval_no_len(self):
        """Test that the number of qubits is properly deduced when not set automatically."""

        qb0, qb1 = (AbstractQbit(),) * 2
        result = qinst_p.abstract_eval(qb0, qb1, op="GarbageOp")[0]

        assert len(result) == 2
        assert all(isinstance(r, AbstractQbit) for r in result)


if __name__ == "__main__":
    pytest.main(["-x", __file__])
