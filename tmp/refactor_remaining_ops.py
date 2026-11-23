"""Script to refactor _resources function signatures across multiple ops files"""

def apply_replacements(filepath, replacements, removals=None):
    with open(filepath, 'r') as f:
        content = f.read()

    for old, new in replacements:
        if old in content:
            content = content.replace(old, new)
            print(f"  ✓ Replaced: {old[:60]}...")
        else:
            print(f"  ✗ NOT FOUND: {old[:60]}...")

    if removals:
        for old, new in removals:
            if old in content:
                content = content.replace(old, new)
                print(f"  ✓ Removed block: {old[:40]}...")
            else:
                print(f"  ✗ NOT FOUND block: {old[:40]}...")

    with open(filepath, 'w') as f:
        f.write(content)


# ============================================================
# qchem_ops.py
# All ops have __init__(self, phi, wires, id=None)
# ============================================================
print("\n=== qchem_ops.py ===")
filepath = "../pennylane/ops/qubit/qchem_ops.py"

with open(filepath, 'r') as f:
    content = f.read()

# Update _resources signatures
sig_replacements = [
    ("def _single_excitation_resources():", "def _single_excitation_resources(phi, wires):"),
    ("def _single_excitation_minus_decomp_resources():", "def _single_excitation_minus_decomp_resources(phi, wires):"),
    ("def _single_excitation_plus_decomp_resources():", "def _single_excitation_plus_decomp_resources(phi, wires):"),
    ("def _orbital_rotation_decomposition_resources():", "def _orbital_rotation_decomposition_resources(phi, wires):"),
    ("def _fermionic_swap_decomp_resources():", "def _fermionic_swap_decomp_resources(phi, wires):"),
]

for old, new in sig_replacements:
    if old in content:
        content = content.replace(old, new)
        print(f"  ✓ Replaced: {old}")
    else:
        print(f"  ✗ NOT FOUND: {old}")

# Remove all resource_keys = set() and resource_params overrides
# Pattern: resource_keys = set() followed by resource_params property
import re

# Remove resource_keys = set() lines
content = re.sub(r'\n    resource_keys = set\(\)\n', '\n', content)
print("  ✓ Removed all resource_keys = set()")

# Remove resource_params property blocks (return {})
content = re.sub(
    r'\n    @property\n    def resource_params\(self\) -> dict:\n        return \{\}\n',
    '\n',
    content
)
print("  ✓ Removed all resource_params returning {}")

with open(filepath, 'w') as f:
    f.write(content)


# ============================================================
# arithmetic_ops.py
# QubitCarry and QubitSum have __init__(self, wires, id=None)
# ============================================================
print("\n=== arithmetic_ops.py ===")
filepath = "../pennylane/ops/qubit/arithmetic_ops.py"

with open(filepath, 'r') as f:
    content = f.read()

sig_replacements = [
    ("def _qubitcarry_to_cnot_toffoli_resources():", "def _qubitcarry_to_cnot_toffoli_resources(wires):"),
    ("def _qubitsum_to_cnots_resources():", "def _qubitsum_to_cnots_resources(wires):"),
]

for old, new in sig_replacements:
    if old in content:
        content = content.replace(old, new)
        print(f"  ✓ Replaced: {old}")
    else:
        print(f"  ✗ NOT FOUND: {old}")

with open(filepath, 'w') as f:
    f.write(content)


# ============================================================
# special_unitary.py
# SpecialUnitary has __init__(self, theta, wires, id=None)
# It has resource_keys = {"pauli_word"} — this is a Category E param
# _tmp_paulirot_decomp_resources(pauli_word: str)
# ============================================================
print("\n=== special_unitary.py ===")
filepath = "../pennylane/ops/qubit/special_unitary.py"

with open(filepath, 'r') as f:
    content = f.read()

# Remove resource_keys and resource_params from SpecialUnitary
content = content.replace(
    """    resource_keys = {
        "pauli_word",
    }

    @property
    def resource_params(self) -> dict:
        return {"pauli_word": self.hyperparameters["pauli_word"]}""",
    ""
)
print("  ✓ Removed SpecialUnitary resource_keys and resource_params")

# Update _tmp_paulirot_decomp_resources signature
# SpecialUnitary.__init__(self, theta, wires, id=None)
# The _tmp_paulirot_decomp_resources(pauli_word) is used for individual PauliRot sub-decompositions
# Let's keep it as-is since it's not a direct Operator resources function
# Actually wait - let me re-read this. The function name suggests it's for internal use.
# Let me leave this one since it corresponds to PauliRot decomps, not SpecialUnitary directly
print("  Note: _tmp_paulirot_decomp_resources(pauli_word) kept as-is (internal PauliRot helper)")

with open(filepath, 'w') as f:
    f.write(content)


# ============================================================
# state_preparation.py
# BasisState: __init__(self, state, wires, id=None), resource_keys={"num_wires"}
# StatePrep: __init__(self, state, wires, pad_with=None, normalize=False, validate_norm=True)
#   resource_keys=frozenset({"num_wires"})
# ============================================================
print("\n=== state_preparation.py ===")
filepath = "../pennylane/ops/qubit/state_preparation.py"

with open(filepath, 'r') as f:
    content = f.read()

# BasisState: __init__(self, state, wires, id=None)
# _basis_state_decomp_resources(num_wires) -> _basis_state_decomp_resources(state, wires)
# Add: num_wires = len(wires)
content = content.replace(
    """    resource_keys = {"num_wires"}

    @property
    def resource_params(self) -> dict:
        return {"num_wires": len(self.wires)}

    def __init__(self, state, wires: WiresLike, id=None):""",
    """    def __init__(self, state, wires: WiresLike, id=None):"""
)
print("  ✓ Removed BasisState resource_keys and resource_params")

content = content.replace(
    "def _basis_state_decomp_resources(num_wires):",
    "def _basis_state_decomp_resources(state, wires):\n    num_wires = len(wires)"
)
print("  ✓ Updated _basis_state_decomp_resources signature with derivation")

# StatePrep: __init__(self, state, wires, pad_with=None, normalize=False, validate_norm=True)
# _stateprep_resources(num_wires) -> _stateprep_resources(state, wires, pad_with, normalize, validate_norm)
# Add: num_wires = len(wires)
content = content.replace(
    """    resource_keys = frozenset({"num_wires"})

    @property
    def resource_params(self):
        return {"num_wires": len(self.wires)}""",
    ""
)
print("  ✓ Removed StatePrep resource_keys and resource_params")

content = content.replace(
    "def _stateprep_resources(num_wires):",
    "def _stateprep_resources(state, wires, pad_with, normalize, validate_norm):\n    num_wires = len(wires)"
)
print("  ✓ Updated _stateprep_resources signature with derivation")

with open(filepath, 'w') as f:
    f.write(content)

print("\n✅ All ops files refactored")
