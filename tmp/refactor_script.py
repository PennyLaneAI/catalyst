"""Script to refactor _resources function signatures in parametric_ops_single_qubit.py"""
import re

filepath = "../pennylane/ops/qubit/parametric_ops_single_qubit.py"

with open(filepath, 'r') as f:
    content = f.read()

# Map of function name pattern -> new signature params
# For RX, RY, RZ: __init__(self, phi, wires, id=None) -> (phi, wires)
# For PhaseShift: __init__(self, phi, wires, id=None) -> (phi, wires)
# For Rot: __init__(self, phi, theta, omega, wires, id=None) -> (phi, theta, omega, wires)
# For U1: __init__(self, phi, wires, id=None) -> (phi, wires)
# For U2: __init__(self, phi, delta, wires, id=None) -> (phi, delta, wires)
# For U3: __init__(self, theta, phi, delta, wires, id=None) -> (theta, phi, delta, wires)

replacements = [
    # RZ resources (already done RX, RY above via Edit tool)
    ("def _rz_to_ps_resources():", "def _rz_to_ps_resources(phi, wires):"),
    ("def _rz_to_rot_resources():", "def _rz_to_rot_resources(phi, wires):"),
    ("def _rz_to_ry_rx_resources():", "def _rz_to_ry_rx_resources(phi, wires):"),
    ("def _rz_to_rx_cliff_resources():", "def _rz_to_rx_cliff_resources(phi, wires):"),
    ("def _rz_to_ry_cliff_resources():", "def _rz_to_ry_cliff_resources(phi, wires):"),
    ("def _rz_to_ppr_resources():", "def _rz_to_ppr_resources(phi, wires):"),

    # PhaseShift resources
    ("def _phaseshift_to_rz_gp_resources():", "def _phaseshift_to_rz_gp_resources(phi, wires):"),

    # Rot resources
    ("def _rot_to_rz_ry_rz_resources():", "def _rot_to_rz_ry_rz_resources(phi, theta, omega, wires):"),

    # U1 resources
    ("def _u1_phaseshift_resources():", "def _u1_phaseshift_resources(phi, wires):"),

    # U2 resources
    ("def _u2_phaseshift_rot_resources():", "def _u2_phaseshift_rot_resources(phi, delta, wires):"),

    # U3 resources
    ("def _u3_phaseshift_rot_resources():", "def _u3_phaseshift_rot_resources(theta, phi, delta, wires):"),
]

for old, new in replacements:
    if old in content:
        content = content.replace(old, new)
        print(f"✓ Replaced: {old}")
    else:
        print(f"✗ NOT FOUND: {old}")

# Remove resource_keys = set() and resource_params for PhaseShift
# PhaseShift has resource_keys = set() and resource_params returning {}
content = content.replace(
    """    resource_keys = set()

    basis = "Z"
    grad_method = "A"
    parameter_frequencies = [(1,)]

    @property
    def resource_params(self) -> dict:
        return {}

    def generator(self) -> "qml.Projector":
        return qml.Projector(np.array([1]), wires=self.wires)

    def __init__(self, phi: TensorLike, wires: WiresLike, id: str | None = None):
        super().__init__(phi, wires=wires, id=id)""",
    """    basis = "Z"
    grad_method = "A"
    parameter_frequencies = [(1,)]

    def generator(self) -> "qml.Projector":
        return qml.Projector(np.array([1]), wires=self.wires)

    def __init__(self, phi: TensorLike, wires: WiresLike, id: str | None = None):
        super().__init__(phi, wires=wires, id=id)"""
)
print("✓ Removed PhaseShift resource_keys and resource_params")

# Remove Rot resource_keys and resource_params
content = content.replace(
    """    resource_keys = set()

    grad_method = "A"
    parameter_frequencies = [(1,), (1,), (1,)]

    resource_keys = set()

    # pylint: disable=too-many-positional-arguments
    def __init__(
        self,
        phi: TensorLike,
        theta: TensorLike,
        omega: TensorLike,
        wires: WiresLike,
        id: str | None = None,
    ):
        super().__init__(phi, theta, omega, wires=wires, id=id)

    @property
    def resource_params(self) -> dict:
        return {}""",
    """    grad_method = "A"
    parameter_frequencies = [(1,), (1,), (1,)]

    # pylint: disable=too-many-positional-arguments
    def __init__(
        self,
        phi: TensorLike,
        theta: TensorLike,
        omega: TensorLike,
        wires: WiresLike,
        id: str | None = None,
    ):
        super().__init__(phi, theta, omega, wires=wires, id=id)"""
)
print("✓ Removed Rot resource_keys and resource_params")

# Remove U1 resource_keys and resource_params
content = content.replace(
    """    resource_keys = set()

    def generator(self) -> "qml.Projector":
        return qml.Projector(np.array([1]), wires=self.wires)

    def __init__(self, phi: TensorLike, wires: WiresLike, id: str | None = None):
        super().__init__(phi, wires=wires, id=id)

    @property
    def resource_params(self) -> dict:
        return {}""",
    """    def generator(self) -> "qml.Projector":
        return qml.Projector(np.array([1]), wires=self.wires)

    def __init__(self, phi: TensorLike, wires: WiresLike, id: str | None = None):
        super().__init__(phi, wires=wires, id=id)"""
)
print("✓ Removed U1 resource_keys and resource_params")

# Remove U2 resource_keys and resource_params
content = content.replace(
    """    resource_keys = set()

    def __init__(self, phi: TensorLike, delta: TensorLike, wires: WiresLike, id: str | None = None):
        super().__init__(phi, delta, wires=wires, id=id)

    @property
    def resource_params(self) -> dict:
        return {}""",
    """    def __init__(self, phi: TensorLike, delta: TensorLike, wires: WiresLike, id: str | None = None):
        super().__init__(phi, delta, wires=wires, id=id)"""
)
print("✓ Removed U2 resource_keys and resource_params")

# Remove U3 resource_keys and resource_params
content = content.replace(
    """    resource_keys = set()

    # pylint: disable=too-many-positional-arguments
    def __init__(
        self,
        theta: TensorLike,
        phi: TensorLike,
        delta: TensorLike,
        wires: WiresLike,
        id: str | None = None,
    ):
        super().__init__(theta, phi, delta, wires=wires, id=id)

    @property
    def resource_params(self) -> dict:
        return {}""",
    """    # pylint: disable=too-many-positional-arguments
    def __init__(
        self,
        theta: TensorLike,
        phi: TensorLike,
        delta: TensorLike,
        wires: WiresLike,
        id: str | None = None,
    ):
        super().__init__(theta, phi, delta, wires=wires, id=id)"""
)
print("✓ Removed U3 resource_keys and resource_params")

with open(filepath, 'w') as f:
    f.write(content)

print("\n✅ All changes applied to parametric_ops_single_qubit.py")
