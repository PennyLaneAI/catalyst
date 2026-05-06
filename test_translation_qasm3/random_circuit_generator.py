"""
Random Circuit Generator for QASM 3 Translation Benchmarking

Provides utilities for generating various types of random quantum circuits
for benchmarking the Qiskit -> MLIR -> QASM3 translation pipeline.
"""

import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from qiskit import QuantumCircuit
from qiskit.circuit.random import random_circuit

# Import QuantumVolume circuit class
try:
    from qiskit.circuit.library import QuantumVolume

    HAS_QUANTUM_VOLUME = True
except ImportError:
    HAS_QUANTUM_VOLUME = False


class RandomCircuitConfig:
    """Configuration for random circuit generation."""

    def __init__(
        self,
        num_qubits: int = 3,
        depth: int = 10,
        max_operands: int = 2,
        measure: bool = True,
        conditional: bool = False,
        reset: bool = False,
        seed: Optional[int] = None,
        gate_set: Optional[List[str]] = None,
    ):
        """
        Initialize circuit configuration.

        Args:
            num_qubits: Number of qubits in the circuit
            depth: Number of gate layers
            max_operands: Maximum number of qubits per gate
            measure: Whether to add measurements
            conditional: Whether to include conditional operations
            reset: Whether to include reset operations
            seed: Random seed for reproducibility
            gate_set: Custom gate set (if None, uses standard gates)
        """
        self.num_qubits = num_qubits
        self.depth = depth
        self.max_operands = max_operands
        self.measure = measure
        self.conditional = conditional
        self.reset = reset
        self.seed = seed
        self.gate_set = gate_set


class RandomCircuitGenerator:
    """Generator for various types of random quantum circuits."""

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the generator.

        Args:
            seed: Global random seed for reproducibility
        """
        self.global_seed = seed
        if seed is not None:
            random.seed(seed)

    def generate_standard_random(self, config: RandomCircuitConfig) -> QuantumCircuit:
        """
        Generate a standard random circuit using Qiskit's random_circuit.

        Args:
            config: Circuit configuration

        Returns:
            Random QuantumCircuit
        """
        seed = config.seed if config.seed is not None else self.global_seed

        circuit = random_circuit(
            num_qubits=config.num_qubits,
            depth=config.depth,
            max_operands=config.max_operands,
            measure=config.measure,
            conditional=config.conditional,
            reset=config.reset,
            seed=seed,
        )

        return circuit

    def generate_quantum_volume(
        self, num_qubits: int, depth: Optional[int] = None, seed: Optional[int] = None
    ) -> QuantumCircuit:
        """
        Generate a quantum volume circuit.

        Quantum volume circuits consist of random SU(4) unitaries applied
        to random pairs of qubits, useful for benchmarking quantum hardware.

        Args:
            num_qubits: Number of qubits
            depth: Circuit depth (if None, defaults to num_qubits)
            seed: Random seed

        Returns:
            Quantum volume circuit with measurements
        """
        if depth is None:
            depth = num_qubits

        seed = seed if seed is not None else self.global_seed

        if not HAS_QUANTUM_VOLUME:
            # Fallback: create a random circuit as a proxy for QV
            print("Warning: QuantumVolume not available, using random circuit as fallback")
            qv_circuit = random_circuit(num_qubits, depth, max_operands=2, seed=seed)
        else:
            # Generate quantum volume circuit
            qv_circuit = QuantumVolume(num_qubits, depth, seed=seed)

        # Add measurements if not already present
        if qv_circuit.num_clbits == 0:
            qv_circuit.measure_all()

        return qv_circuit

    def generate_clifford_random(
        self, num_qubits: int, num_gates: int, seed: Optional[int] = None
    ) -> QuantumCircuit:
        """
        Generate a random Clifford circuit.

        Clifford circuits are composed of gates from the Clifford group
        (H, S, CNOT), which are efficiently simulable classically but
        useful for testing.

        Args:
            num_qubits: Number of qubits
            num_gates: Number of Clifford gates
            seed: Random seed

        Returns:
            Random Clifford circuit
        """
        try:
            from qiskit.circuit.random import random_clifford_circuit

            seed = seed if seed is not None else self.global_seed
            circuit = random_clifford_circuit(num_qubits, num_gates, seed=seed)
            circuit.measure_all()
            return circuit
        except ImportError:
            # Fallback: manually create Clifford circuit
            return self._generate_clifford_fallback(num_qubits, num_gates, seed)

    def _generate_clifford_fallback(
        self, num_qubits: int, num_gates: int, seed: Optional[int] = None
    ) -> QuantumCircuit:
        """Fallback method to generate Clifford circuits manually."""
        if seed is not None:
            random.seed(seed)

        qc = QuantumCircuit(num_qubits)
        clifford_gates_1q = ["h", "s", "sdg", "x", "y", "z"]

        for _ in range(num_gates):
            # Randomly choose between 1-qubit and 2-qubit gates
            if random.random() < 0.7 or num_qubits == 1:  # 70% single-qubit
                gate = random.choice(clifford_gates_1q)
                qubit = random.randint(0, num_qubits - 1)
                getattr(qc, gate)(qubit)
            else:  # 30% two-qubit (CNOT)
                q1, q2 = random.sample(range(num_qubits), 2)
                qc.cx(q1, q2)

        qc.measure_all()
        return qc

    def generate_custom_gate_set(
        self, num_qubits: int, depth: int, gate_set: List[str], seed: Optional[int] = None
    ) -> QuantumCircuit:
        """
        Generate a random circuit with a custom gate set.

        Args:
            num_qubits: Number of qubits
            depth: Circuit depth
            gate_set: List of gate names (e.g., ['h', 'rx', 'ry', 'cx'])
            seed: Random seed

        Returns:
            Random circuit using only specified gates
        """
        if seed is not None:
            random.seed(seed)

        qc = QuantumCircuit(num_qubits)

        # Categorize gates
        single_qubit_gates = []
        two_qubit_gates = []
        parameterized_gates = []

        for gate in gate_set:
            if gate in ["cx", "cz", "cy", "swap", "cnot"]:
                two_qubit_gates.append(gate)
            elif gate in ["rx", "ry", "rz", "u1", "u2", "u3", "cp", "crx", "cry", "crz"]:
                parameterized_gates.append(gate)
            else:
                single_qubit_gates.append(gate)

        for _ in range(depth):
            # Choose gate type
            available_types = []
            if single_qubit_gates:
                available_types.append("1q")
            if two_qubit_gates and num_qubits > 1:
                available_types.append("2q")
            if parameterized_gates:
                available_types.append("param")

            if not available_types:
                break

            gate_type = random.choice(available_types)

            if gate_type == "1q":
                gate = random.choice(single_qubit_gates)
                qubit = random.randint(0, num_qubits - 1)
                getattr(qc, gate)(qubit)

            elif gate_type == "2q":
                gate = random.choice(two_qubit_gates)
                q1, q2 = random.sample(range(num_qubits), 2)
                if gate == "cnot":
                    gate = "cx"
                getattr(qc, gate)(q1, q2)

            elif gate_type == "param":
                gate = random.choice(parameterized_gates)
                param = random.uniform(0, 2 * 3.14159)  # 0 to 2π

                if gate in ["rx", "ry", "rz", "u1"]:
                    qubit = random.randint(0, num_qubits - 1)
                    getattr(qc, gate)(param, qubit)
                elif gate in ["crx", "cry", "crz", "cp"]:
                    q1, q2 = random.sample(range(num_qubits), 2)
                    getattr(qc, gate)(param, q1, q2)
                elif gate == "u2":
                    qubit = random.randint(0, num_qubits - 1)
                    phi = random.uniform(0, 2 * 3.14159)
                    lam = random.uniform(0, 2 * 3.14159)
                    qc.u(3.14159 / 2, phi, lam, qubit)  # U2 is U(π/2, φ, λ)
                elif gate == "u3":
                    qubit = random.randint(0, num_qubits - 1)
                    theta = random.uniform(0, 3.14159)
                    phi = random.uniform(0, 2 * 3.14159)
                    lam = random.uniform(0, 2 * 3.14159)
                    qc.u(theta, phi, lam, qubit)

        qc.measure_all()
        return qc

    def generate_batch(
        self, config: RandomCircuitConfig, count: int, circuit_type: str = "standard"
    ) -> List[QuantumCircuit]:
        """
        Generate a batch of random circuits.

        Args:
            config: Circuit configuration
            count: Number of circuits to generate
            circuit_type: Type of circuit ('standard', 'qv', 'clifford', 'custom')

        Returns:
            List of random circuits
        """
        circuits = []

        for i in range(count):
            # Use different seed for each circuit
            seed = config.seed + i if config.seed is not None else None

            if circuit_type == "standard":
                config_copy = RandomCircuitConfig(
                    num_qubits=config.num_qubits,
                    depth=config.depth,
                    max_operands=config.max_operands,
                    measure=config.measure,
                    conditional=config.conditional,
                    reset=config.reset,
                    seed=seed,
                    gate_set=config.gate_set,
                )
                circuit = self.generate_standard_random(config_copy)

            elif circuit_type == "qv":
                circuit = self.generate_quantum_volume(config.num_qubits, config.depth, seed=seed)

            elif circuit_type == "clifford":
                circuit = self.generate_clifford_random(config.num_qubits, config.depth, seed=seed)

            elif circuit_type == "custom" and config.gate_set:
                circuit = self.generate_custom_gate_set(
                    config.num_qubits, config.depth, config.gate_set, seed=seed
                )

            else:
                raise ValueError(f"Unknown circuit type: {circuit_type}")

            circuits.append(circuit)

        return circuits

    def save_circuit(self, circuit: QuantumCircuit, filepath: Path) -> None:
        """
        Save a circuit to a QASM 2.0 file.

        Args:
            circuit: Circuit to save
            filepath: Output file path
        """
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Try newer qasm2 module first, fallback to deprecated qasm()
        try:
            from qiskit import qasm2

            qasm_str = qasm2.dumps(circuit)
        except (ImportError, AttributeError):
            # Fallback to older method
            qasm_str = circuit.qasm() if hasattr(circuit, "qasm") else str(circuit)

        with open(filepath, "w") as f:
            f.write(qasm_str)

    def save_batch(
        self, circuits: List[QuantumCircuit], output_dir: Path, prefix: str = "random"
    ) -> List[Path]:
        """
        Save a batch of circuits to QASM files.

        Args:
            circuits: List of circuits
            output_dir: Output directory
            prefix: Filename prefix

        Returns:
            List of saved file paths
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        filepaths = []

        for i, circuit in enumerate(circuits):
            filename = f"{prefix}_{i:04d}.qasm"
            filepath = output_dir / filename
            self.save_circuit(circuit, filepath)
            filepaths.append(filepath)

        return filepaths


def create_benchmark_circuits(
    output_dir: Optional[Path] = None, seed: int = 42
) -> Dict[str, List[Path]]:
    """
    Create a standard set of benchmark circuits for testing.

    Args:
        output_dir: Directory to save circuits (if None, returns circuits only)
        seed: Random seed

    Returns:
        Dictionary mapping circuit category to list of file paths
    """
    generator = RandomCircuitGenerator(seed=seed)
    results = {}

    # Small circuits (2-4 qubits, depth 5-10)
    small_config = RandomCircuitConfig(num_qubits=3, depth=5, seed=seed)
    small_circuits = generator.generate_batch(small_config, count=10, circuit_type="standard")

    # Medium circuits (5-7 qubits, depth 10-20)
    medium_config = RandomCircuitConfig(num_qubits=5, depth=15, seed=seed)
    medium_circuits = generator.generate_batch(medium_config, count=10, circuit_type="standard")

    # Quantum volume circuits
    qv_config = RandomCircuitConfig(num_qubits=4, depth=4, seed=seed)
    qv_circuits = generator.generate_batch(qv_config, count=5, circuit_type="qv")

    # Clifford circuits
    clifford_config = RandomCircuitConfig(num_qubits=3, depth=10, seed=seed)
    clifford_circuits = generator.generate_batch(clifford_config, count=5, circuit_type="clifford")

    if output_dir:
        results["small"] = generator.save_batch(
            small_circuits, output_dir / "small", "small_random"
        )
        results["medium"] = generator.save_batch(
            medium_circuits, output_dir / "medium", "medium_random"
        )
        results["qv"] = generator.save_batch(qv_circuits, output_dir / "qv", "qv")
        results["clifford"] = generator.save_batch(
            clifford_circuits, output_dir / "clifford", "clifford"
        )
    else:
        results["small"] = small_circuits
        results["medium"] = medium_circuits
        results["qv"] = qv_circuits
        results["clifford"] = clifford_circuits

    return results


if __name__ == "__main__":
    # Example usage
    import sys
    from pathlib import Path

    script_dir = Path(__file__).parent
    output_dir = script_dir / "random_circuits"

    print("Generating benchmark circuits...")
    results = create_benchmark_circuits(output_dir=output_dir, seed=42)

    print("\nGenerated circuits:")
    for category, paths in results.items():
        print(f"  {category}: {len(paths)} circuits")

    print(f"\nCircuits saved to: {output_dir}")
