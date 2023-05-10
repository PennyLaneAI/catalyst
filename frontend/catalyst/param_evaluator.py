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
Parameter Evaluator
"""

import jax
from jax.interpreters.partial_eval import partial_eval_jaxpr_nounits
from jax.tree_util import tree_flatten, tree_unflatten


class ParamEvaluator:
    """
    A class which can evaluate a jaxpr object one argument at a time.

    Suppose we have a function

    >>> def f(x, y):
    ...     return 5., x, x * y

    We would like to be able to get the first output of this function (``5.``)
    since it does not depend on ``x`` and ``y``.

    So using this class we could do

    >>> args = [6., 2.]
    >>> expected_out = [5., 6., 12.]
    >>> jaxpr = jax.make_jaxpr(fn)(*args)
    >>> pe = ParamEvaluator(jaxpr, [tree_flatten(o)[1] for o in expected_out])

    (Note we need the output shapes given by pytrees.)

    Then, to get the first output:

    >>> pe.get_partial_return_value()
    5.

    From there we can get the second output by providing only the ``x`` value:

    >>> pe.send_partial(6.)
    >>> pe.get_partial_return_value()
    6.

    and then one more time to get the final output

    >>> pe.send_partial(2.)
    >>> pe.get_partial_return_value()
    12.

    Note that, for a more complicated function return type,

    >>> def f(x):
    ...     return x**2, [x, x + 1], { "my_val": x ** 5 }

    then the output trees would look like, in JAX notation,

    .. code-block:: text

        [(*,), (*, *), { "my_val": * }]
    """

    def __init__(self, c_jaxpr, output_trees):
        self.output_trees = output_trees
        known, self.unknown, bools, _ = partial_eval_jaxpr_nounits(
            c_jaxpr, [True] * len(c_jaxpr.in_avals), instantiate=False
        )

        # the known jaxpr's outputs are ordered out1, out2, ..., carry1, carry2 ...
        # where the carrys are partial results that need to be fed into the unknown jaxpr
        outs_plus_carry_through = jax.core.eval_jaxpr(known.jaxpr, known.literals)
        num_known = bools.count(False)
        known_indices = [i for i, b in enumerate(bools) if not b]
        out = outs_plus_carry_through[:num_known]
        self.out_ordered = dict(zip(known_indices, out))
        self.indices_unknown = [i for i, b in enumerate(bools) if b]
        self.carry_through = outs_plus_carry_through[num_known:]
        self.cur_index = 0
        self.tree_index = 0

    def get_partial_return_value(self):
        """
        Get the next return value.
        """
        tree = self.output_trees[self.tree_index]
        self.tree_index += 1

        tree_list = [self._get_partial_return_value() for _ in range(tree.num_leaves)]
        return tree_unflatten(tree, tree_list)

    def _get_partial_return_value(self):
        """
        Return the next known output value from the function. If no additional known output value
        exists throw ``ValueError``.

        Returns:
            the next output value of the original jaxpr
        """
        try:
            return_val = self.out_ordered[self.cur_index]
            self.cur_index += 1
            return return_val
        except IndexError as exc:
            raise ValueError("no additional known outputs given the inputs") from exc

    def send_partial_input(self, val):
        """
        Send the next argument of the original jaxpr.

        Args:
            val: the next argument of the jaxpr
        """
        flat_val, _ = tree_flatten(val)

        known_parameters = [False] * (len(self.carry_through) + len(flat_val))
        unknown_parameters = [True] * (
            len(self.unknown.in_avals) - len(self.carry_through) - len(flat_val)
        )
        bool_array = known_parameters + unknown_parameters

        known, self.unknown, bools, _ = partial_eval_jaxpr_nounits(
            self.unknown, bool_array, instantiate=False
        )
        outs_plus_carry_through = jax.core.eval_jaxpr(
            known.jaxpr, known.literals, *[*self.carry_through, *flat_val]
        )
        num_known = bools.count(False)
        out = outs_plus_carry_through[:num_known]

        # remove now known indices
        new_indices_known = list(
            map(lambda v: v[0], filter(lambda v: not v[1], zip(self.indices_unknown, bools)))
        )
        new_indices_unknown = list(
            map(lambda v: v[0], filter(lambda v: v[1], zip(self.indices_unknown, bools)))
        )
        for i, k in enumerate(new_indices_known):
            self.out_ordered[k] = out[i]
        self.indices_unknown = new_indices_unknown
        self.carry_through = outs_plus_carry_through[num_known:]
