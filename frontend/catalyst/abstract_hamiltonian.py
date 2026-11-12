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

"""
A representation-agnostic Hamiltonian abstraction for Catalyst.

This module is the frontend (PennyLane) half of the abstract-Hamiltonian feature. It defines
a single canonical container, :class:`AbstractHamiltonian`, that can hold any of the supported
Hamiltonian representations, and :func:`convert_h`, which normalizes the various user-facing
inputs into that container.

The design follows the same "opaque handle + numeric leaves + static structure" split that
Catalyst already uses for observables (see ``AbstractObs`` / ``HamiltonianOp`` in
``catalyst.jax_primitives``):

* the *numeric leaves* (coefficients, factor tensors, ...) flow through tracing / lowering as
  ordinary array values, and
* the *structure* (which representation, Pauli words, tensor ranks, ...) rides along as static,
  hashable metadata.

Because :class:`AbstractHamiltonian` is registered as a JAX pytree whose children are exactly
the numeric leaves and whose auxiliary data is the (hashable) structure, an instance can be
passed as a runtime argument to a ``qjit`` / captured ``QNode``: the leaves are traced while the
structure is preserved as pytree metadata.

Supported representations (the ``kind`` discriminator):

* ``"pauli"`` -- a mapping of Pauli words to coefficients, e.g. ``{"XX": 50, "Z": 51}``.
* ``"lcu"``   -- a linear combination of operators, e.g. ``qp.dot(coeffs, ops)``.
* ``"cdf"``   -- a compressed double-factorized Hamiltonian given by ``core_tensors``,
  ``leaf_tensors`` and a scalar ``nuc_constant``.
"""

import numpy as np
import pennylane as qp
from jax.tree_util import register_pytree_node_class
from pennylane.operation import Operator

__all__ = ("AbstractHamiltonian", "convert_h", "CustomHOp")


@register_pytree_node_class
class AbstractHamiltonian:
    """Canonical, representation-agnostic container for a Hamiltonian.

    Args:
        kind (str): the representation discriminator, one of ``"pauli"``, ``"lcu"`` or ``"cdf"``.
        leaves (Sequence): the numeric leaves of the Hamiltonian (arrays). These are the values
            that are traced / lowered.
        structure (tuple): static, hashable metadata describing the structure. Stored as a tuple
            of ``(key, value)`` pairs so that it can be used as JAX pytree auxiliary data.

    The class is a JAX pytree: ``leaves`` are the children and ``(kind, structure)`` is the
    auxiliary data, so an instance can be passed as a runtime argument to a captured ``QNode``.
    """

    def __init__(self, kind, leaves, structure):
        self.kind = kind
        self.leaves = tuple(leaves)
        self.structure = tuple(structure)

    @property
    def structure_dict(self):
        """Return the structure as a plain dictionary for convenient inspection."""
        return dict(self.structure)

    def tree_flatten(self):
        """Children are the numeric leaves; aux data is the (static) kind and structure."""
        return (self.leaves, (self.kind, self.structure))

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Rebuild from traced leaves and static aux data."""
        kind, structure = aux_data
        return cls(kind, children, structure)

    def __repr__(self):
        return f"AbstractHamiltonian(kind={self.kind!r}, structure={self.structure_dict!r})"


def _convert_pauli(mapping):
    """Normalize a ``{pauli_word: coefficient}`` mapping into an AbstractHamiltonian."""
    words = tuple(mapping.keys())
    coeffs = np.array([mapping[w] for w in words], dtype=float)
    return AbstractHamiltonian("pauli", (coeffs,), (("words", words),))


def _convert_cdf(mapping):
    """Normalize a compressed double-factorized dict into an AbstractHamiltonian.

    Expects keys ``"core_tensors"`` (shape ``(L, M, M, N, N)``), ``"leaf_tensors"`` (shape
    ``(L, M, N, N)``) and a scalar ``"nuc_constant"``.
    """
    core = np.asarray(mapping["core_tensors"], dtype=float)
    leaf = np.asarray(mapping["leaf_tensors"], dtype=float)
    nuc = np.asarray(mapping["nuc_constant"], dtype=float)
    L, M, _, N, _ = core.shape
    structure = (("L", int(L)), ("M", int(M)), ("N", int(N)))
    return AbstractHamiltonian("cdf", (core, leaf, nuc), structure)


def _convert_lcu(op):
    """Normalize a PennyLane operator (e.g. ``qp.dot(coeffs, ops)``) into an AbstractHamiltonian.

    The coefficients are the numeric leaf; the per-term operator descriptions are stored as
    static structure.
    """
    coeffs, ops = op.terms()
    coeffs = np.array(coeffs, dtype=float)
    terms = tuple(str(o) for o in ops)
    return AbstractHamiltonian("lcu", (coeffs,), (("terms", terms),))


def convert_h(H):
    """Normalize a user-supplied Hamiltonian into the canonical :class:`AbstractHamiltonian`.

    Supported inputs:

    * ``dict`` with ``"core_tensors"`` -> a compressed double-factorized (``"cdf"``) Hamiltonian.
    * any other ``dict`` -> a Pauli-word (``"pauli"``) Hamiltonian, mapping words to coefficients.
    * a PennyLane operator exposing ``.terms()`` (e.g. ``qp.dot(coeffs, ops)``) -> an ``"lcu"``
      Hamiltonian.

    Args:
        H: the Hamiltonian in one of the supported representations.

    Returns:
        AbstractHamiltonian: the canonical container.

    Raises:
        TypeError: if ``H`` is not a supported representation.
    """
    if isinstance(H, AbstractHamiltonian):
        return H
    if isinstance(H, dict):
        if "core_tensors" in H:
            return _convert_cdf(H)
        return _convert_pauli(H)
    if isinstance(H, Operator) or hasattr(H, "terms"):
        return _convert_lcu(H)
    raise TypeError(f"Unsupported Hamiltonian representation: {type(H)!r}")


class CustomHOp(Operator):
    """A representation-agnostic Hamiltonian-evolution operation.

    This operation consumes an :class:`AbstractHamiltonian` together with an evolution ``time``
    and a pair of integer hyperparameters. The Hamiltonian's numeric leaves become the operation's
    trainable/traced data, while its ``kind`` and ``structure`` ride along as static
    hyperparameters. This mirrors how Catalyst's ``HamiltonianOp`` carries coefficients as SSA
    tensor values and structure as attributes.

    Args:
        H (AbstractHamiltonian): the Hamiltonian to evolve under.
        time: the evolution time (traced).
        n (int): a static integer hyperparameter.
        m (int): a static integer hyperparameter.
        wires: the wires the operation acts on.
    """

    def __init__(self, H, time, n, m, wires):
        H = convert_h(H)
        self._hyperparameters = {
            "kind": H.kind,
            "structure": H.structure,
            "n": n,
            "m": m,
        }
        super().__init__(*H.leaves, time, wires=wires)

    @property
    def kind(self):
        """The Hamiltonian representation discriminator."""
        return self.hyperparameters["kind"]

    @property
    def structure(self):
        """The static structure metadata as a dictionary."""
        return dict(self.hyperparameters["structure"])
