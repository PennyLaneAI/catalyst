"""Script to refactor _resources function signatures across all template files.
For each operator:
1. Remove resource_keys override
2. Remove resource_params property override
3. Update _resources function signatures to match __init__ params
4. Add num_* derivation lines where applicable
"""
import re

def process_file(filepath, changes):
    """Process a file with a list of changes.
    Each change is a dict with keys:
    - 'remove_resource_keys': exact string to remove
    - 'remove_resource_params': exact string to remove
    - 'sig_old': old function signature
    - 'sig_new': new function signature (may include derivation line)
    """
    with open(filepath, 'r') as f:
        content = f.read()

    for change in changes:
        if 'remove_resource_keys' in change:
            old = change['remove_resource_keys']
            if old in content:
                content = content.replace(old, '')
                print(f"  ✓ Removed resource_keys in {filepath.split('/')[-1]}")
            else:
                print(f"  ✗ resource_keys NOT FOUND in {filepath.split('/')[-1]}: {old[:50]}...")

        if 'remove_resource_params' in change:
            old = change['remove_resource_params']
            if old in content:
                content = content.replace(old, '')
                print(f"  ✓ Removed resource_params in {filepath.split('/')[-1]}")
            else:
                print(f"  ✗ resource_params NOT FOUND in {filepath.split('/')[-1]}: {old[:50]}...")

        if 'sig_old' in change and 'sig_new' in change:
            old = change['sig_old']
            new = change['sig_new']
            if old in content:
                content = content.replace(old, new)
                print(f"  ✓ Updated signature: {old[:60]}...")
            else:
                print(f"  ✗ Signature NOT FOUND: {old[:60]}...")

    with open(filepath, 'w') as f:
        f.write(content)


# Use regex-based removal for resource_keys and resource_params
def remove_resource_keys_and_params(filepath):
    """Remove resource_keys = ... and resource_params property."""
    with open(filepath, 'r') as f:
        content = f.read()

    original = content

    # Remove resource_keys = set()
    content = re.sub(r'\n    resource_keys = set\(\)\n', '\n', content)

    # Remove resource_keys = {"..."}  (single-line)
    content = re.sub(r'\n    resource_keys = \{[^}]+\}\n', '\n', content)

    # Remove resource_keys = frozenset({"..."}) (single-line)
    content = re.sub(r'\n    resource_keys = frozenset\(\{[^}]+\}\)\n', '\n', content)

    # Remove multi-line resource_keys (for QROM, QRAM etc.)
    content = re.sub(
        r'\n    resource_keys = \{\n(?:.*\n)*?    \}\n',
        '\n',
        content
    )

    # Remove resource_params property that returns {}
    content = re.sub(
        r'\n    @property\n    def resource_params\(self\) -> dict:\n        return \{\}\n',
        '\n',
        content
    )

    # Remove resource_params property (multi-line, various patterns)
    # This catches all resource_params property blocks
    content = re.sub(
        r'\n    @property\n(?:    @handle_recursion_error\n)?    def resource_params\(self\)(?:\s*-> dict)?:\n(?:        .*\n)*?(?=\n    [a-zA-Z@]|\nclass |\ndef )',
        '\n',
        content
    )

    if content != original:
        print(f"  ✓ Removed resource_keys and/or resource_params from {filepath.split('/')[-1]}")
    else:
        print(f"  - No resource_keys/resource_params found in {filepath.split('/')[-1]}")

    with open(filepath, 'w') as f:
        f.write(content)

    return content


# ============================================================
# Process each file with signature changes and derivation lines
# ============================================================

base = "../pennylane/templates"
ops_base = "../pennylane/ops"

# ---- SUBROUTINES ----

# QFT: __init__(self, wires, id=None)
# _qft_decomposition_resources(num_wires) -> _qft_decomposition_resources(wires)
# Add: num_wires = len(wires)
print("\n=== QFT ===")
remove_resource_keys_and_params(f"{base}/subroutines/qft.py")
process_file(f"{base}/subroutines/qft.py", [{
    'sig_old': 'def _qft_decomposition_resources(num_wires):',
    'sig_new': 'def _qft_decomposition_resources(wires):\n    num_wires = len(wires)',
}])

# AQFT: __init__(self, order, wires, *, id=None)
# _AQFT_resources(num_wires, order) -> _AQFT_resources(order, wires)
# Add: num_wires = len(wires)
print("\n=== AQFT ===")
remove_resource_keys_and_params(f"{base}/subroutines/aqft.py")
process_file(f"{base}/subroutines/aqft.py", [{
    'sig_old': 'def _AQFT_resources(num_wires, order):',
    'sig_new': 'def _AQFT_resources(order, wires):\n    num_wires = len(wires)',
}])

# QROM: __init__(self, bitstrings, control_wires, target_wires, work_wires, clean=True, id=None)
# Current: _qrom_decomposition_resources(num_bitstrings, num_control_wires, num_target_wires, num_work_wires, clean)
# New: _qrom_decomposition_resources(bitstrings, control_wires, target_wires, work_wires, clean)
print("\n=== QROM ===")
remove_resource_keys_and_params(f"{base}/subroutines/qrom.py")
with open(f"{base}/subroutines/qrom.py", 'r') as f:
    content = f.read()
content = content.replace(
    """def _qrom_decomposition_resources(
    num_bitstrings, num_control_wires, num_target_wires, num_work_wires, clean
):""",
    """def _qrom_decomposition_resources(
    bitstrings, control_wires, target_wires, work_wires, clean
):
    num_bitstrings = len(bitstrings)
    num_control_wires = len(control_wires)
    num_target_wires = len(target_wires)
    num_work_wires = len(work_wires)"""
)
with open(f"{base}/subroutines/qrom.py", 'w') as f:
    f.write(content)
print("  ✓ Updated QROM signature")

# Permute: __init__(self, permutation, wires, id=None)
# _permute_resources(wires, permutation) already matches! But order differs from __init__
# __init__ is (permutation, wires) so change to (permutation, wires)
print("\n=== Permute ===")
remove_resource_keys_and_params(f"{base}/subroutines/permute.py")
process_file(f"{base}/subroutines/permute.py", [{
    'sig_old': 'def _permute_resources(wires, permutation):',
    'sig_new': 'def _permute_resources(permutation, wires):',
}])

# PrepSelPrep: __init__(self, lcu, control, id=None)
# _prepselprep_resources(op_reps, num_control) -> _prepselprep_resources(lcu, control)
# num_control = len(control)
print("\n=== PrepSelPrep ===")
remove_resource_keys_and_params(f"{base}/subroutines/prepselprep.py")
process_file(f"{base}/subroutines/prepselprep.py", [{
    'sig_old': 'def _prepselprep_resources(op_reps, num_control):',
    'sig_new': 'def _prepselprep_resources(lcu, control):\n    num_control = len(control)',
}])

# AmplitudeAmplification: __init__(self, U, O, iters, fixed_point=False, work_wire=None, p_min=0, id=None)
# _amplitude_amplification_resources(fixed_point, O, iters, num_reflection_wires, U)
# -> (U, O, iters, fixed_point, work_wire, p_min) per __init__ order
print("\n=== AmplitudeAmplification ===")
remove_resource_keys_and_params(f"{base}/subroutines/amplitude_amplification.py")
process_file(f"{base}/subroutines/amplitude_amplification.py", [{
    'sig_old': 'def _amplitude_amplification_resources(fixed_point, O, iters, num_reflection_wires, U):',
    'sig_new': 'def _amplitude_amplification_resources(U, O, iters, fixed_point, work_wire, p_min):',
}])

# Reflection: __init__(self, U, alpha=np.pi, reflection_wires=None, id=None)
# _reflection_decomposition_resources(base_class, base_params, num_wires, num_reflection_wires)
# Keep base_class, base_params at front per Category D
# -> (base_class, base_params, U, alpha, reflection_wires)
print("\n=== Reflection ===")
remove_resource_keys_and_params(f"{base}/subroutines/reflection.py")
# Read current signature
with open(f"{base}/subroutines/reflection.py", 'r') as f:
    content = f.read()
content = content.replace(
    """def _reflection_decomposition_resources(
    base_class, base_params, num_wires, num_reflection_wires
):""",
    """def _reflection_decomposition_resources(
    base_class, base_params, U, alpha, reflection_wires
):
    num_wires = len(U.wires) if hasattr(U, 'wires') else 0
    num_reflection_wires = len(reflection_wires) if reflection_wires is not None else 0"""
)
with open(f"{base}/subroutines/reflection.py", 'w') as f:
    f.write(content)
print("  ✓ Updated Reflection signature")

# GroverOperator: __init__(self, wires, work_wires=(), id=None)
# _grover_operator_resources(num_wires, num_work_wires) -> (wires, work_wires)
print("\n=== GroverOperator ===")
remove_resource_keys_and_params(f"{base}/subroutines/grover.py")
process_file(f"{base}/subroutines/grover.py", [{
    'sig_old': 'def _grover_operator_resources(num_wires, num_work_wires):',
    'sig_new': 'def _grover_operator_resources(wires, work_wires):\n    num_wires = len(wires)\n    num_work_wires = len(work_wires)',
}])

# Select: __init__(self, ops, control, work_wires=None, partial=False, id=None)
# _select_multi_control_work_wire_resources(op_reps, num_control_wires, num_work_wires, partial)
# -> (ops, control, work_wires, partial)
print("\n=== Select ===")
remove_resource_keys_and_params(f"{base}/subroutines/select.py")
process_file(f"{base}/subroutines/select.py", [{
    'sig_old': 'def _select_multi_control_work_wire_resources(op_reps, num_control_wires, num_work_wires, partial):',
    'sig_new': 'def _select_multi_control_work_wire_resources(ops, control, work_wires, partial):\n    num_control_wires = len(control)\n    num_work_wires = len(work_wires) if work_wires is not None else 0',
}])

# Qubitization: __init__(self, hamiltonian, control, id=None)
# _qubitization_resources(num_control_wires, hamiltonian) -> (hamiltonian, control)
print("\n=== Qubitization ===")
remove_resource_keys_and_params(f"{base}/subroutines/qubitization.py")
process_file(f"{base}/subroutines/qubitization.py", [{
    'sig_old': 'def _qubitization_resources(num_control_wires, hamiltonian):',
    'sig_new': 'def _qubitization_resources(hamiltonian, control):\n    num_control_wires = len(control)',
}])

# ControlledSequence: __init__(self, base, control, id=None)
# _ctrl_seq_decomposition_resources(base_class, base_params, num_control_wires) -> (base_class, base_params, base, control)
print("\n=== ControlledSequence ===")
remove_resource_keys_and_params(f"{base}/subroutines/controlled_sequence.py")
process_file(f"{base}/subroutines/controlled_sequence.py", [{
    'sig_old': 'def _ctrl_seq_decomposition_resources(base_class, base_params, num_control_wires) -> dict:',
    'sig_new': 'def _ctrl_seq_decomposition_resources(base_class, base_params, base, control) -> dict:\n    num_control_wires = len(control)',
}])

# QSVT: __init__(self, UA, projectors, id=None)
# _QSVT_resources(projectors, UA) -> (UA, projectors) to match __init__ order
print("\n=== QSVT ===")
remove_resource_keys_and_params(f"{base}/subroutines/qsvt.py")
process_file(f"{base}/subroutines/qsvt.py", [{
    'sig_old': 'def _QSVT_resources(projectors, UA):',
    'sig_new': 'def _QSVT_resources(UA, projectors):',
}])

# GQSP: __init__(self, unitary, angles, control, id=None)
# _GQSP_resources(unitary, num_iters) -> (unitary, angles, control)
print("\n=== GQSP ===")
remove_resource_keys_and_params(f"{base}/subroutines/gqsp.py")
process_file(f"{base}/subroutines/gqsp.py", [{
    'sig_old': 'def _GQSP_resources(unitary, num_iters):',
    'sig_new': 'def _GQSP_resources(unitary, angles, control):',
}])

# QuantumMonteCarlo: __init__(self, probs, func, target_wires, estimation_wires, id=None)
# _quantum_monte_carlo_resources(num_target_wires, num_estimation_wires, q_resource_rep)
# -> (probs, func, target_wires, estimation_wires)
print("\n=== QuantumMonteCarlo ===")
remove_resource_keys_and_params(f"{base}/subroutines/qmc.py")
process_file(f"{base}/subroutines/qmc.py", [{
    'sig_old': 'def _quantum_monte_carlo_resources(num_target_wires, num_estimation_wires, q_resource_rep):',
    'sig_new': 'def _quantum_monte_carlo_resources(probs, func, target_wires, estimation_wires):\n    num_target_wires = len(target_wires)\n    num_estimation_wires = len(estimation_wires)',
}])

# HilbertSchmidt: __init__(self, v_params, v_function, v_wires, u_tape, id=None)
# _hilbert_schmidt_resources(num_wires, u_reps, v_wires)
# LocalHilbertSchmidt: same init, different resource function
print("\n=== HilbertSchmidt ===")
remove_resource_keys_and_params(f"{base}/subroutines/hilbert_schmidt.py")
process_file(f"{base}/subroutines/hilbert_schmidt.py", [{
    'sig_old': 'def _hilbert_schmidt_resources(\n    num_wires, u_reps, v_wires\n):',
    'sig_new': 'def _hilbert_schmidt_resources(\n    v_params, v_function, v_wires, u_tape\n):',
}, {
    'sig_old': 'def _local_hilbert_schmidt_resources(\n    num_wires, u_reps, v_wires\n):',
    'sig_new': 'def _local_hilbert_schmidt_resources(\n    v_params, v_function, v_wires, u_tape\n):',
}])

# FlipSign: __init__(self, n, wires, id=None)
# _flip_sign_resources(num_wires, arr_bin) -> (n, wires)
print("\n=== FlipSign ===")
remove_resource_keys_and_params(f"{base}/subroutines/flip_sign.py")
process_file(f"{base}/subroutines/flip_sign.py", [{
    'sig_old': 'def _flip_sign_resources(num_wires, arr_bin):',
    'sig_new': 'def _flip_sign_resources(n, wires):\n    num_wires = len(wires)',
}])

# FABLE: __init__(self, input_matrix, wires, tol=0, id=None)
# _fable_resources(wires, thetas, control_wires, tol) -> (input_matrix, wires, tol)
print("\n=== FABLE ===")
remove_resource_keys_and_params(f"{base}/subroutines/fable.py")
process_file(f"{base}/subroutines/fable.py", [{
    'sig_old': 'def _fable_resources(wires, thetas, control_wires, tol):',
    'sig_new': 'def _fable_resources(input_matrix, wires, tol):',
}])

# ArbitraryUnitary: __init__(self, weights, wires, id=None)
# _arbitrary_unitary_resources(num_wires) -> (weights, wires)
print("\n=== ArbitraryUnitary ===")
remove_resource_keys_and_params(f"{base}/subroutines/arbitrary_unitary.py")
process_file(f"{base}/subroutines/arbitrary_unitary.py", [{
    'sig_old': 'def _arbitrary_unitary_resources(num_wires: int) -> dict:',
    'sig_new': 'def _arbitrary_unitary_resources(weights, wires) -> dict:\n    num_wires = len(wires)',
}])

# IQP (subroutine): __init__(self, n_layers, pattern, wires, id=None)
# _instantaneous_quantum_polynomial_resources(spin_sym, pattern, num_wires) -> (n_layers, pattern, wires)
print("\n=== IQP (subroutine) ===")
remove_resource_keys_and_params(f"{base}/subroutines/iqp.py")
process_file(f"{base}/subroutines/iqp.py", [{
    'sig_old': 'def _instantaneous_quantum_polynomial_resources(spin_sym, pattern, num_wires):',
    'sig_new': 'def _instantaneous_quantum_polynomial_resources(n_layers, pattern, wires):\n    num_wires = len(wires)',
}])

# QRAM files (multiple operators)
print("\n=== QRAM ===")
remove_resource_keys_and_params(f"{base}/subroutines/qram.py")
with open(f"{base}/subroutines/qram.py", 'r') as f:
    content = f.read()

# BucketBrigadeQRAM: __init__(self, bitstrings, target_wires, control_wires=(), work_wires=(), id=None)
# _bucket_brigade_qram_resources(num_controls, num_target_wires) -> (bitstrings, target_wires, control_wires, work_wires)
content = content.replace(
    'def _bucket_brigade_qram_resources(num_controls, num_target_wires):',
    'def _bucket_brigade_qram_resources(bitstrings, target_wires, control_wires, work_wires):\n    num_controls = len(control_wires)\n    num_target_wires = len(target_wires)'
)

# HybridQRAM: __init__(self, bitstrings, target_wires, select_wires, tree_control_wires=(), work_wires=(), id=None)
# _hybrid_qram_resources(num_target_wires, num_select_wires, num_tree_control_wires)
content = content.replace(
    'def _hybrid_qram_resources(num_target_wires, num_select_wires, num_tree_control_wires):',
    'def _hybrid_qram_resources(bitstrings, target_wires, select_wires, tree_control_wires, work_wires):\n    num_target_wires = len(target_wires)\n    num_select_wires = len(select_wires)\n    num_tree_control_wires = len(tree_control_wires)'
)

# SelectOnlyQRAM: __init__(self, bitstrings, target_wires, control_wires, clean=True, work_wires=(), id=None)
# _select_only_qram_resources(num_control_wires, num_target_wires, num_bitstrings, num_work_wires, clean)
content = content.replace(
    """def _select_only_qram_resources(
    num_control_wires, num_target_wires, num_bitstrings, num_work_wires, clean
):""",
    """def _select_only_qram_resources(
    bitstrings, target_wires, control_wires, clean, work_wires
):
    num_control_wires = len(control_wires)
    num_target_wires = len(target_wires)
    num_bitstrings = len(bitstrings)
    num_work_wires = len(work_wires)"""
)

with open(f"{base}/subroutines/qram.py", 'w') as f:
    f.write(content)
print("  ✓ Updated QRAM signatures")

# ---- ARITHMETIC ----

# Adder: __init__(self, k, x_wires, mod=None, work_wires=(), id=None)
# _adder_decomposition_resources(num_x_wires, mod) -> (k, x_wires, mod, work_wires)
print("\n=== Adder ===")
remove_resource_keys_and_params(f"{base}/subroutines/arithmetic/adder.py")
process_file(f"{base}/subroutines/arithmetic/adder.py", [{
    'sig_old': 'def _adder_decomposition_resources(num_x_wires, mod) -> dict:',
    'sig_new': 'def _adder_decomposition_resources(k, x_wires, mod, work_wires) -> dict:\n    num_x_wires = len(x_wires)',
}])

# ModExp: __init__(self, x_wires, output_wires, base, mod, work_wires, id=None)
# _mod_exp_decomposition_resources(num_x_wires, num_output_wires, mod, num_work_wires) -> (x_wires, output_wires, base, mod, work_wires)
print("\n=== ModExp ===")
remove_resource_keys_and_params(f"{base}/subroutines/arithmetic/mod_exp.py")
process_file(f"{base}/subroutines/arithmetic/mod_exp.py", [{
    'sig_old': 'def _mod_exp_decomposition_resources(num_x_wires, num_output_wires, mod, num_work_wires) -> dict:',
    'sig_new': 'def _mod_exp_decomposition_resources(x_wires, output_wires, base, mod, work_wires) -> dict:\n    num_x_wires = len(x_wires)\n    num_output_wires = len(output_wires)\n    num_work_wires = len(work_wires)',
}])

# Multiplier: __init__(self, k, x_wires, mod=None, work_wires=(), id=None)
# _multiplier_decomposition_resources(num_x_wires, mod, num_work_wires)
print("\n=== Multiplier ===")
remove_resource_keys_and_params(f"{base}/subroutines/arithmetic/multiplier.py")
with open(f"{base}/subroutines/arithmetic/multiplier.py", 'r') as f:
    content = f.read()
content = content.replace(
    """def _multiplier_decomposition_resources(
    num_x_wires, mod, num_work_wires
) -> dict:""",
    """def _multiplier_decomposition_resources(
    k, x_wires, mod, work_wires
) -> dict:
    num_x_wires = len(x_wires)
    num_work_wires = len(work_wires)"""
)
with open(f"{base}/subroutines/arithmetic/multiplier.py", 'w') as f:
    f.write(content)
print("  ✓ Updated Multiplier signature")

# OutAdder: __init__(self, x_wires, y_wires, output_wires, mod=None, work_wires=(), id=None)
# _out_adder_decomposition_resources(num_output_wires, num_x_wires, num_y_wires, mod)
print("\n=== OutAdder ===")
remove_resource_keys_and_params(f"{base}/subroutines/arithmetic/out_adder.py")
process_file(f"{base}/subroutines/arithmetic/out_adder.py", [{
    'sig_old': 'def _out_adder_decomposition_resources(num_output_wires, num_x_wires, num_y_wires, mod) -> dict:',
    'sig_new': 'def _out_adder_decomposition_resources(x_wires, y_wires, output_wires, mod, work_wires) -> dict:\n    num_output_wires = len(output_wires)\n    num_x_wires = len(x_wires)\n    num_y_wires = len(y_wires)',
}])

# OutMultiplier: __init__(self, x_wires, y_wires, output_wires, mod=None, work_wires=(), id=None)
# _out_multiplier_decomposition_resources(num_output_wires, num_x_wires, num_y_wires, mod)
print("\n=== OutMultiplier ===")
remove_resource_keys_and_params(f"{base}/subroutines/arithmetic/out_multiplier.py")
with open(f"{base}/subroutines/arithmetic/out_multiplier.py", 'r') as f:
    content = f.read()
content = content.replace(
    """def _out_multiplier_decomposition_resources(
    num_output_wires, num_x_wires, num_y_wires, mod
) -> dict:""",
    """def _out_multiplier_decomposition_resources(
    x_wires, y_wires, output_wires, mod, work_wires
) -> dict:
    num_output_wires = len(output_wires)
    num_x_wires = len(x_wires)
    num_y_wires = len(y_wires)"""
)
with open(f"{base}/subroutines/arithmetic/out_multiplier.py", 'w') as f:
    f.write(content)
print("  ✓ Updated OutMultiplier signature")

# OutPoly: __init__(self, f, input_registers, output_wires, mod=None, work_wires=(), id=None)
# _out_poly_decomposition_resources(num_output_wires, num_work_wires, mod, coeffs_list)
print("\n=== OutPoly ===")
remove_resource_keys_and_params(f"{base}/subroutines/arithmetic/out_poly.py")
process_file(f"{base}/subroutines/arithmetic/out_poly.py", [{
    'sig_old': 'def _out_poly_decomposition_resources(num_output_wires, num_work_wires, mod, coeffs_list) -> dict:',
    'sig_new': 'def _out_poly_decomposition_resources(f, input_registers, output_wires, mod, work_wires) -> dict:\n    num_output_wires = len(output_wires)\n    num_work_wires = len(work_wires)',
}])

# PhaseAdder: __init__(self, k, x_wires, mod=None, work_wire=(), id=None)
# _phase_adder_decomposition_resources(num_x_wires, mod) -> (k, x_wires, mod, work_wire)
print("\n=== PhaseAdder ===")
remove_resource_keys_and_params(f"{base}/subroutines/arithmetic/phase_adder.py")
process_file(f"{base}/subroutines/arithmetic/phase_adder.py", [{
    'sig_old': 'def _phase_adder_decomposition_resources(num_x_wires, mod) -> dict:',
    'sig_new': 'def _phase_adder_decomposition_resources(k, x_wires, mod, work_wire) -> dict:\n    num_x_wires = len(x_wires)',
}])

# SemiAdder: __init__(self, x_wires, y_wires, output_wire, carry_wire, id=None)
# _semiadder_resources(num_y_wires) -> (x_wires, y_wires, output_wire, carry_wire)
print("\n=== SemiAdder ===")
remove_resource_keys_and_params(f"{base}/subroutines/arithmetic/semi_adder.py")
process_file(f"{base}/subroutines/arithmetic/semi_adder.py", [{
    'sig_old': 'def _semiadder_resources(num_y_wires):',
    'sig_new': 'def _semiadder_resources(x_wires, y_wires, output_wire, carry_wire):\n    num_y_wires = len(y_wires)',
}])

# TemporaryAnd: __init__(self, wires, control_values=(1,1), id=None)
# _temporary_and_resources() -> (wires, control_values)
# _adjoint_temporary_and_resources(base_class=None, base_params=None) -> keep as-is (Category D)
print("\n=== TemporaryAnd ===")
remove_resource_keys_and_params(f"{base}/subroutines/arithmetic/temporary_and.py")
process_file(f"{base}/subroutines/arithmetic/temporary_and.py", [{
    'sig_old': 'def _temporary_and_resources():',
    'sig_new': 'def _temporary_and_resources(wires, control_values):',
}])

# ---- TIME EVOLUTION ----

# ApproxTimeEvolution: __init__(self, hamiltonian, time, n, id=None)
# _approx_time_evolution_resources(words, n) -> (hamiltonian, time, n)
print("\n=== ApproxTimeEvolution ===")
remove_resource_keys_and_params(f"{base}/subroutines/time_evolution/approx_time_evolution.py")
process_file(f"{base}/subroutines/time_evolution/approx_time_evolution.py", [{
    'sig_old': 'def _approx_time_evolution_resources(words: tuple[PauliWord], n: int):',
    'sig_new': 'def _approx_time_evolution_resources(hamiltonian, time, n):',
}])

# CommutingEvolution: __init__(self, hamiltonian, time, frequencies=None, shifts=None, id=None)
# _commuting_evolution_resources(words) -> (hamiltonian, time, frequencies, shifts)
print("\n=== CommutingEvolution ===")
remove_resource_keys_and_params(f"{base}/subroutines/time_evolution/commuting_evolution.py")
process_file(f"{base}/subroutines/time_evolution/commuting_evolution.py", [{
    'sig_old': 'def _commuting_evolution_resources(words: tuple[PauliWord]):',
    'sig_new': 'def _commuting_evolution_resources(hamiltonian, time, frequencies, shifts):',
}])

# TrotterProduct: __init__(self, hamiltonian, time, n=1, order=2, check_hermitian=True, id=None)
# _trotter_product_decomposition_resources(n, order, ops) -> (hamiltonian, time, n, order, check_hermitian)
print("\n=== TrotterProduct ===")
remove_resource_keys_and_params(f"{base}/subroutines/time_evolution/trotter.py")
process_file(f"{base}/subroutines/time_evolution/trotter.py", [{
    'sig_old': 'def _trotter_product_decomposition_resources(n, order, ops):',
    'sig_new': 'def _trotter_product_decomposition_resources(hamiltonian, time, n, order, check_hermitian):',
}])

# ---- QCHEM TEMPLATES ----

# AllSinglesDoubles: __init__(self, weights, wires, hf_state, singles=None, doubles=None, id=None)
# No _resources function exists, but has resource_keys
print("\n=== AllSinglesDoubles ===")
remove_resource_keys_and_params(f"{base}/subroutines/qchem/all_singles_doubles.py")

# BasisRotation: __init__(self, wires, unitary_matrix, check=False, id=None)
# _basis_rotation_decomp_resources(dim, is_real) -> (wires, unitary_matrix, check)
print("\n=== BasisRotation ===")
remove_resource_keys_and_params(f"{base}/subroutines/qchem/basis_rotation.py")
process_file(f"{base}/subroutines/qchem/basis_rotation.py", [{
    'sig_old': 'def _basis_rotation_decomp_resources(dim, is_real):',
    'sig_new': 'def _basis_rotation_decomp_resources(wires, unitary_matrix, check):',
}])

# FermionicDoubleExcitation: __init__(self, weight, wires1, wires2, *, id=None)
# _fermionic_double_excitation_resources(num_wires_1, num_wires_2) -> (weight, wires1, wires2)
print("\n=== FermionicDoubleExcitation ===")
remove_resource_keys_and_params(f"{base}/subroutines/qchem/fermionic_double_excitation.py")
process_file(f"{base}/subroutines/qchem/fermionic_double_excitation.py", [{
    'sig_old': 'def _fermionic_double_excitation_resources(num_wires_1, num_wires_2):',
    'sig_new': 'def _fermionic_double_excitation_resources(weight, wires1, wires2):\n    num_wires_1 = len(wires1)\n    num_wires_2 = len(wires2)',
}])

# FermionicSingleExcitation: __init__(self, weight, wires, *, id=None)
# _fermionic_single_excitation_resources(num_wires) -> (weight, wires)
print("\n=== FermionicSingleExcitation ===")
remove_resource_keys_and_params(f"{base}/subroutines/qchem/fermionic_single_excitation.py")
process_file(f"{base}/subroutines/qchem/fermionic_single_excitation.py", [{
    'sig_old': 'def _fermionic_single_excitation_resources(num_wires):',
    'sig_new': 'def _fermionic_single_excitation_resources(weight, wires):\n    num_wires = len(wires)',
}])

# kUpCCGSD: __init__(self, weights, wires, k=1, delta_sz=0, init_state=None, id=None)
# _kupccgsd_resources(num_wires, k, d_wires, s_wires) -> (weights, wires, k, delta_sz, init_state)
print("\n=== kUpCCGSD ===")
remove_resource_keys_and_params(f"{base}/subroutines/qchem/kupccgsd.py")
process_file(f"{base}/subroutines/qchem/kupccgsd.py", [{
    'sig_old': 'def _kupccgsd_resources(num_wires: int, k: int, d_wires: list, s_wires: list):',
    'sig_new': 'def _kupccgsd_resources(weights, wires, k, delta_sz, init_state):\n    num_wires = len(wires)',
}])

# UCCSD: __init__(self, weights, wires, s_wires=None, d_wires=None, init_state=None, n_repeats=1, id=None)
# _UCCSD_resources(num_wires, n_repeats, num_d_wires, num_s_wires)
print("\n=== UCCSD ===")
remove_resource_keys_and_params(f"{base}/subroutines/qchem/uccsd.py")
process_file(f"{base}/subroutines/qchem/uccsd.py", [{
    'sig_old': 'def _UCCSD_resources(num_wires, n_repeats, num_d_wires, num_s_wires):',
    'sig_new': 'def _UCCSD_resources(weights, wires, s_wires, d_wires, init_state, n_repeats):\n    num_wires = len(wires)\n    num_d_wires = len(d_wires) if d_wires is not None else 0\n    num_s_wires = len(s_wires) if s_wires is not None else 0',
}])

# ---- STATE PREPARATIONS ----

# ArbitraryStatePreparation: __init__(self, weights, wires, id=None)
# _arbitrary_state_preparation_resources(num_wires) -> (weights, wires)
print("\n=== ArbitraryStatePreparation ===")
remove_resource_keys_and_params(f"{base}/state_preparations/arbitrary_state_preparation.py")
process_file(f"{base}/state_preparations/arbitrary_state_preparation.py", [{
    'sig_old': 'def _arbitrary_state_preparation_resources(num_wires):',
    'sig_new': 'def _arbitrary_state_preparation_resources(weights, wires):\n    num_wires = len(wires)',
}])

# CosineWindow: __init__(self, wires, id=None)
# _cosine_window_resources(num_wires) -> (wires)
print("\n=== CosineWindow ===")
remove_resource_keys_and_params(f"{base}/state_preparations/cosine_window.py")
process_file(f"{base}/state_preparations/cosine_window.py", [{
    'sig_old': 'def _cosine_window_resources(num_wires):',
    'sig_new': 'def _cosine_window_resources(wires):\n    num_wires = len(wires)',
}])

# MottonenStatePreparation: __init__(self, state_vector, wires, id=None)
# _mottonen_resources(num_wires) -> (state_vector, wires)
print("\n=== MottonenStatePreparation ===")
remove_resource_keys_and_params(f"{base}/state_preparations/mottonen.py")
process_file(f"{base}/state_preparations/mottonen.py", [{
    'sig_old': 'def _mottonen_resources(num_wires):',
    'sig_new': 'def _mottonen_resources(state_vector, wires):\n    num_wires = len(wires)',
}])

# MultiplexerStatePrep: __init__(self, state_vector, wires, id=None)
# _multiplexer_state_prep_decomposition_resources(num_wires) -> (state_vector, wires)
print("\n=== MultiplexerStatePrep ===")
remove_resource_keys_and_params(f"{base}/state_preparations/multiplexer_state_prep.py")
process_file(f"{base}/state_preparations/multiplexer_state_prep.py", [{
    'sig_old': 'def _multiplexer_state_prep_decomposition_resources(num_wires) -> dict:',
    'sig_new': 'def _multiplexer_state_prep_decomposition_resources(state_vector, wires) -> dict:\n    num_wires = len(wires)',
}])

# MPSPrep: __init__(self, mps, wires, work_wires=(), id=None)
# _mps_prep_decomposition_resources(bond_dimensions, num_sites, num_work_wires)
print("\n=== MPSPrep ===")
remove_resource_keys_and_params(f"{base}/state_preparations/state_prep_mps.py")
with open(f"{base}/state_preparations/state_prep_mps.py", 'r') as f:
    content = f.read()
content = content.replace(
    """def _mps_prep_decomposition_resources(
    bond_dimensions, num_sites, num_work_wires
):""",
    """def _mps_prep_decomposition_resources(
    mps, wires, work_wires
):
    num_work_wires = len(work_wires)"""
)
with open(f"{base}/state_preparations/state_prep_mps.py", 'w') as f:
    f.write(content)
print("  ✓ Updated MPSPrep signature")

# Superposition: __init__(self, bases, coeffs, wires, work_wires=(), id=None)
# _superposition_resources(num_wires, num_coeffs, bases) -> (bases, coeffs, wires, work_wires)
print("\n=== Superposition ===")
remove_resource_keys_and_params(f"{base}/state_preparations/superposition.py")
process_file(f"{base}/state_preparations/superposition.py", [{
    'sig_old': 'def _superposition_resources(num_wires, num_coeffs, bases):',
    'sig_new': 'def _superposition_resources(bases, coeffs, wires, work_wires):\n    num_wires = len(wires)\n    num_coeffs = len(coeffs)',
}])

# ---- EMBEDDINGS ----

# AmplitudeEmbedding: __init__(self, features, wires, pad_with=None, normalize=False, validate_norm=True, id=None)
# No _resources function, just resource_keys
print("\n=== AmplitudeEmbedding ===")
remove_resource_keys_and_params(f"{base}/embeddings/amplitude.py")

# AngleEmbedding: __init__(self, features, wires, rotation="X", id=None)
# _angle_embedding_resources(rotation, num_wires) -> (features, wires, rotation)
print("\n=== AngleEmbedding ===")
remove_resource_keys_and_params(f"{base}/embeddings/angle.py")
process_file(f"{base}/embeddings/angle.py", [{
    'sig_old': 'def _angle_embedding_resources(rotation: Operation, num_wires: int) -> dict:',
    'sig_new': 'def _angle_embedding_resources(features, wires, rotation) -> dict:\n    num_wires = len(wires)',
}])

# IQPEmbedding: __init__(self, features, wires, n_repeats=1, pattern=None, id=None)
# _iqp_embedding_resources(pattern_size, n_repeats, num_wires) -> (features, wires, n_repeats, pattern)
print("\n=== IQPEmbedding ===")
remove_resource_keys_and_params(f"{base}/embeddings/iqp.py")
process_file(f"{base}/embeddings/iqp.py", [{
    'sig_old': 'def _iqp_embedding_resources(pattern_size, n_repeats, num_wires):',
    'sig_new': 'def _iqp_embedding_resources(features, wires, n_repeats, pattern):\n    num_wires = len(wires)',
}])

# QAOAEmbedding: __init__(self, features, weights, wires, local_field="Y", id=None)
# _qaoa_embedding_resources(repeat, n_features, num_wires, local_field) -> (features, weights, wires, local_field)
print("\n=== QAOAEmbedding ===")
remove_resource_keys_and_params(f"{base}/embeddings/qaoaembedding.py")
process_file(f"{base}/embeddings/qaoaembedding.py", [{
    'sig_old': 'def _qaoa_embedding_resources(repeat, n_features, num_wires, local_field):',
    'sig_new': 'def _qaoa_embedding_resources(features, weights, wires, local_field):\n    num_wires = len(wires)',
}])

# ---- LAYERS ----

# BasicEntanglerLayers: __init__(self, weights, wires=None, rotation=None, id=None)
# _basic_entangler_resources(repeat, num_wires, rotation) -> (weights, wires, rotation)
print("\n=== BasicEntanglerLayers ===")
remove_resource_keys_and_params(f"{base}/layers/basic_entangler.py")
process_file(f"{base}/layers/basic_entangler.py", [{
    'sig_old': 'def _basic_entangler_resources(repeat, num_wires, rotation):',
    'sig_new': 'def _basic_entangler_resources(weights, wires, rotation):\n    num_wires = len(wires)',
}])

# GateFabric: __init__(self, weights, wires, init_state, include_pi=False, id=None)
# _gate_fabric_resources(n_layers, num_wires, len_wire_pattern, include_pi) -> (weights, wires, init_state, include_pi)
print("\n=== GateFabric ===")
remove_resource_keys_and_params(f"{base}/layers/gate_fabric.py")
process_file(f"{base}/layers/gate_fabric.py", [{
    'sig_old': 'def _gate_fabric_resources(n_layers, num_wires, len_wire_pattern, include_pi):',
    'sig_new': 'def _gate_fabric_resources(weights, wires, init_state, include_pi):\n    num_wires = len(wires)',
}])

# ParticleConservingU1: __init__(self, weights, wires, init_state=None, id=None)
# _particle_conserving_u1_resources(n_layers, num_wires) -> (weights, wires, init_state)
print("\n=== ParticleConservingU1 ===")
remove_resource_keys_and_params(f"{base}/layers/particle_conserving_u1.py")
process_file(f"{base}/layers/particle_conserving_u1.py", [{
    'sig_old': 'def _particle_conserving_u1_resources(n_layers: int, num_wires: int):',
    'sig_new': 'def _particle_conserving_u1_resources(weights, wires, init_state):\n    num_wires = len(wires)',
}])

# ParticleConservingU2: __init__(self, weights, wires, init_state=None, id=None)
# _particle_conserving_u2_resources(num_wires, n_layers) -> (weights, wires, init_state)
print("\n=== ParticleConservingU2 ===")
remove_resource_keys_and_params(f"{base}/layers/particle_conserving_u2.py")
process_file(f"{base}/layers/particle_conserving_u2.py", [{
    'sig_old': 'def _particle_conserving_u2_resources(num_wires: int, n_layers: int):',
    'sig_new': 'def _particle_conserving_u2_resources(weights, wires, init_state):\n    num_wires = len(wires)',
}])

# SimplifiedTwoDesign: __init__(self, initial_layer_weights, weights, wires, id=None)
# _simplified_two_design_resources(n_layers, num_wires) -> (initial_layer_weights, weights, wires)
print("\n=== SimplifiedTwoDesign ===")
remove_resource_keys_and_params(f"{base}/layers/simplified_two_design.py")
process_file(f"{base}/layers/simplified_two_design.py", [{
    'sig_old': 'def _simplified_two_design_resources(n_layers, num_wires):',
    'sig_new': 'def _simplified_two_design_resources(initial_layer_weights, weights, wires):\n    num_wires = len(wires)',
}])

# StronglyEntanglingLayers: __init__(self, weights, wires, ranges=None, imprimitive=CNOT, id=None)
# _strongly_entangling_resources(imprimitive, n_wires, n_layers) -> (weights, wires, ranges, imprimitive)
print("\n=== StronglyEntanglingLayers ===")
remove_resource_keys_and_params(f"{base}/layers/strongly_entangling.py")
process_file(f"{base}/layers/strongly_entangling.py", [{
    'sig_old': 'def _strongly_entangling_resources(imprimitive, n_wires, n_layers):',
    'sig_new': 'def _strongly_entangling_resources(weights, wires, ranges, imprimitive):\n    n_wires = len(wires)',
}])

print("\n✅ All template files refactored!")
