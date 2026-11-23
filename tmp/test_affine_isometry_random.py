"""One-off checker for the affine Clifford isometry in PartialUnaryStatePreparation.

The isometry should send each support basis state |ℓ⟩ to a unique subspace
label |f(ℓ)⟩ tensored with |0…0⟩ on the remainder wires, using only X, CNOT,
and SWAP. State prep then runs this circuit backwards.

This script is not part of the pytest suite. Run with:

    python tmp/test_affine_isometry_random.py
"""

import math
import random

from pennylane.templates.state_preparations.partial_unary import (
    _find_affine_subspace_isometry,
    _is_affine_subspace,
    _reduced_binary_row_basis,
)


def apply_circuit(states, circuit, n):
    """Apply the recorded Clifford circuit to packed bitstrings.

    Integers use MSB-first packing: bit n-1 is wire 0, bit 0 is wire n-1.
    X / CNOT / SWAP are all involutions on computational basis states.
    """
    states = list(map(int, states))
    for op_type, *data in circuit:
        if op_type == "X":
            # Flip the bit that sits on wire data[0].
            mask = 1 << (n - 1 - data[0])
            states = [s ^ mask for s in states]
        elif op_type == "CNOT":
            c, t = data
            cm, tm = 1 << (n - 1 - c), 1 << (n - 1 - t)
            # If control is 1, XOR the target bit.
            states = [s ^ tm if s & cm else s for s in states]
        elif op_type == "SWAP":
            w0, w1 = data[0]
            m0, m1 = 1 << (n - 1 - w0), 1 << (n - 1 - w1)
            # Swap two bits only when they differ.
            states = [s ^ m0 ^ m1 if bool(s & m0) != bool(s & m1) else s for s in states]
        else:
            raise AssertionError(op_type)
    return states


def check(states, n, n_subspace, tag, failures):
    """Verify one (states, n, n_subspace) instance, or record a failure.

    If the support is not affine of width n_subspace, the finder must return
    None. Otherwise the circuit must:

    - zero the remainder (low n - n_subspace bits)
    - report a bijection that matches those high bits
    - map distinct inputs to distinct subspace labels
    - invert: running the circuit backwards recovers the original states
    """
    res = _find_affine_subspace_isometry(states, n, n_subspace)
    affine = _is_affine_subspace(states, n_subspace)
    if not affine:
        if res is not None:
            failures.append((tag, "expected None"))
        return
    if res is None:
        failures.append((tag, "unexpected None", states, n, n_subspace))
        return
    circuit, bijection = res
    rem = (1 << (n - n_subspace)) - 1  # mask of remainder bits
    mapped = apply_circuit(states, circuit, n)
    if any(s & rem for s in mapped):
        failures.append((tag, "remainder not zero", states, mapped, circuit))
        return
    # Removes the remainder bits from the mapped states.
    got = [s >> (n - n_subspace) for s in mapped]
    want = [bijection[i] for i in range(len(states))]
    if got != want:
        failures.append((tag, "bijection mismatch", got, want))
        return
    if len(set(got)) != len(states):
        failures.append((tag, "not injective", got))
        return
    # X, CNOT, SWAP are self-inverse, so reverse(circuit) is the adjoint.
    back = apply_circuit(mapped, list(reversed(circuit)), n)
    if back != list(map(int, states)):
        failures.append((tag, "inverse failed", states, mapped, back))


def main():
    random.seed(0)
    failures = []

    # Remainder bits already 0: identity isometry, empty circuit.
    check([0b1100, 0b1000, 0b0100, 0b0000], 4, 2, "already-subspace", failures)
    # Docs example: span of 0011 and 1100, a 2-dimensional affine set.
    check([0b0000, 0b0011, 0b1100, 0b1111], 4, 2, "docs", failures)
    # Affine rank 1 < n_subspace=3: extra subspace wires should stay 0.
    check([0b0011, 0b1100], 4, 3, "rank-lt-subspace", failures)
    # Three of the four points of a 2-dim affine plane (not a full subspace).
    check([0b0000, 0b0011, 0b1100], 4, 2, "partial-3", failures)
    # Three independent bits, rank 3 > width 2: must fall back (None).
    check([0, 1, 2, 4], 4, 2, "non-affine", failures)
    # Independent bits sit in the remainder; SWAPs must move them up.
    check([0b0001, 0b0010, 0b0011, 0b0000], 4, 2, "low-pivots", failures)
    check([0b010101, 0b101010, 0b111111, 0b000000], 6, 2, "alternating", failures)

    # Random full affine spaces: pick k independent generators, XOR with an
    # origin (anchor), and take all 2^k combinations. Then also test a random
    # subset, using n_subspace = ceil(log2(|subset|)) as PUI would.
    n_ok = 0
    for trial in range(400):
        n = random.randint(3, 12)
        k = random.randint(1, min(5, n - 1))
        used = []
        for _try in range(50):
            if len(used) >= k:
                break
            v = random.randrange(1, 2**n)
            # Keep v only if it increases the GF(2) rank by one.
            b = _reduced_binary_row_basis(used + [v], k)
            if b is not None and len(b) == len(used) + 1:
                used.append(v)
        if len(used) != k:
            continue
        anchor = random.randrange(2**n)
        states = []
        for mask in range(2**k):
            s = anchor
            for j, vec in enumerate(used):
                if (mask >> j) & 1:
                    s ^= vec
            states.append(s)
        if len(set(states)) != len(states):
            continue
        check(states, n, k, f"full-{trial}", failures)
        n_ok += 1
        m = random.randint(2, 2**k)
        subset = random.sample(states, m)
        n_sub = max(math.ceil(math.log2(m)), 1)
        check(subset, n, n_sub, f"subset-{trial}", failures)

    print("full affine spaces checked", n_ok)
    print("failures", len(failures))
    for f in failures[:10]:
        print(f)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
