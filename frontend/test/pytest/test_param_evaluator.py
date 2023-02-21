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

import pytest

import jax
from jax.tree_util import tree_flatten

from catalyst.param_evaluator import ParamEvaluator


class TestParamEvaluator:
    def test_basic_jax_fn(self):
        def fn(x):
            return x**2

        # input and expected output
        args = [5.0]
        expected_out = [25.0]

        jaxpr = jax.make_jaxpr(fn)(*args)

        pe = ParamEvaluator(jaxpr, [tree_flatten(o)[1] for o in expected_out])
        pe.send_partial_input(5.0)
        out = pe.get_partial_return_value()

        assert out == 25.0

    def test_constant_in_jax_fn(self):
        def fn(x):
            return 3.0, x**2

        # input and expected output
        args = [5.0]
        expected_out = [3.0, 25.0]

        jaxpr = jax.make_jaxpr(fn)(*args)

        pe = ParamEvaluator(jaxpr, [tree_flatten(o)[1] for o in expected_out])

        assert pe.get_partial_return_value() == 3.0
        pe.send_partial_input(5.0)
        assert pe.get_partial_return_value() == 25.0

    def test_outputs(self):
        def fn(x, y):
            return 3.0, x**2, 3 * x * y

        args = [5.0, 2.0]
        expected_out = [3.0, 25.0, 30.0]

        jaxpr = jax.make_jaxpr(fn)(*args)

        pe = ParamEvaluator(jaxpr, [tree_flatten(o)[1] for o in expected_out])

        assert pe.get_partial_return_value() == expected_out[0]
        pe.send_partial_input(args[0])
        assert pe.get_partial_return_value() == expected_out[1]
        pe.send_partial_input(args[1])
        assert pe.get_partial_return_value() == expected_out[2]

    def test_out_of_order(self):
        def fn(x):
            return x**2

        args = [5.0]
        expected_out = [25.0]

        jaxpr = jax.make_jaxpr(fn)(*args)

        pe = ParamEvaluator(jaxpr, [tree_flatten(o)[1] for o in expected_out])

        with pytest.raises(KeyError):
            pe.get_partial_return_value()


if __name__ == "__main__":
    pytest.main(["-x", __file__])
