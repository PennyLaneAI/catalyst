"""Script to refactor _resources function signatures in non_parametric_ops.py"""

filepath = "../pennylane/ops/qubit/non_parametric_ops.py"

with open(filepath, "r") as f:
    content = f.read()

# All non-parametric ops have __init__(self, wires, id=None)
# So _resources() -> _resources(wires)
replacements = [
    # Hadamard
    ("def _hadamard_rz_rx_resources():", "def _hadamard_rz_rx_resources(wires):"),
    ("def _hadamard_rz_ry_resources():", "def _hadamard_rz_ry_resources(wires):"),
    # PauliX
    ("def _paulix_to_rx_resources():", "def _paulix_to_rx_resources(wires):"),
    # PauliY
    ("def _pauliy_to_ry_gp_resources():", "def _pauliy_to_ry_gp_resources(wires):"),
    # PauliZ
    ("def _pauliz_to_ps_resources():", "def _pauliz_to_ps_resources(wires):"),
    # S
    ("def _s_phaseshift_resources():", "def _s_phaseshift_resources(wires):"),
    # T
    ("def _t_phaseshift_resources():", "def _t_phaseshift_resources(wires):"),
    # SX
    ("def _sx_to_rx_resources():", "def _sx_to_rx_resources(wires):"),
    # SWAP
    ("def _swap_to_cnot_resources():", "def _swap_to_cnot_resources(wires):"),
    # ECR
    ("def _ecr_decomp_resources():", "def _ecr_decomp_resources(wires):"),
    # ISWAP
    ("def _iswap_decomp_resources():", "def _iswap_decomp_resources(wires):"),
    # SISWAP
    ("def _siswap_decomp_resources():", "def _siswap_decomp_resources(wires):"),
]

for old, new in replacements:
    if old in content:
        content = content.replace(old, new)
        print(f"✓ Replaced: {old}")
    else:
        print(f"✗ NOT FOUND: {old}")

with open(filepath, "w") as f:
    f.write(content)

print("\n✅ All changes applied to non_parametric_ops.py")
