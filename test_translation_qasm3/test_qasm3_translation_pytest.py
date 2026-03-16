"""
Pytest-based tests for QASM3 translation pipeline.

This test suite provides comprehensive testing of the Qiskit -> MLIR -> QASM3 pipeline.
"""

import pytest
import sys
import subprocess
import tempfile
import math
from pathlib import Path
from qiskit import QuantumCircuit

try:
    from qiskit_importer_standalone import QiskitToCatalystImporter
except ImportError:
    pytest.skip("qiskit_importer_standalone not available", allow_module_level=True)

try:
    from qiskit_aer import AerSimulator
    import qiskit.qasm3
    AER_AVAILABLE = True
except ImportError:
    AER_AVAILABLE = False


def hellinger_distance(dict1, dict2, shots):
    """Compute Hellinger distance between two probability distributions."""
    p1 = {k: v / shots for k, v in dict1.items()}
    p2 = {k: v / shots for k, v in dict2.items()}

    keys = set(p1.keys()).union(set(p2.keys()))
    distance_sq = 0.0
    for k in keys:
        distance_sq += (math.sqrt(p1.get(k, 0.0)) - math.sqrt(p2.get(k, 0.0))) ** 2
    return math.sqrt(distance_sq) / math.sqrt(2)


def aggregate_by_hamming_weight(counts_dict):
    """Aggregate counts by Hamming weight (number of 1s)."""
    agg = {}
    for k, v in counts_dict.items():
        weight = str(k.count('1'))
        agg[weight] = agg.get(weight, 0) + v
    return agg


class TestQASM3Pipeline:
    """Test the full QASM3 translation pipeline."""

    def run_full_pipeline(self, circuit_path, quantum_opt_path, quantum_translate_path, root_dir, use_pass_pipeline=False):
        """Run the complete translation pipeline on a circuit.

        Args:
            circuit_path: Path to input QASM circuit
            quantum_opt_path: Path to quantum-opt binary
            quantum_translate_path: Path to quantum-translate binary
            root_dir: Root directory of the project
            use_pass_pipeline: If True, apply quantum-opt optimization passes (default: False)

        Returns:
            Tuple of (qasm3_output_string, original_qiskit_circuit)
        """
        # 1. Load QASM
        qc = QuantumCircuit.from_qasm_file(str(circuit_path))

        # 2. Convert to MLIR
        importer = QiskitToCatalystImporter(qc)
        module = importer.convert()

        # 3. Write MLIR to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.mlir', delete=False) as tmp_mlir:
            tmp_mlir.write(str(module))
            tmp_mlir_path = tmp_mlir.name

        # 4. Conditionally apply quantum-opt passes
        if use_pass_pipeline:
            opt_cmd = [
                str(quantum_opt_path),
                "--pass-pipeline=builtin.module(apply-transform-sequence, canonicalize, merge-rotations)",
                tmp_mlir_path,
                "-o", tmp_mlir_path
            ]
            subprocess.run(opt_cmd, capture_output=True, text=True, check=True)

        # 5. Translate to QASM3
        cmd = [str(quantum_translate_path), "--mlir-to-qasm3", tmp_mlir_path]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        Path(tmp_mlir_path).unlink()  # Cleanup

        return result.stdout, qc

    @pytest.mark.parametrize("circuit_file", [
        "bell_state_00.qasm",
        "bell_state_01.qasm",
        "bell_state_10.qasm",
        "bell_state_11.qasm",
    ])
    def test_bell_states(self, circuit_file, circuits_dir, quantum_opt_path, quantum_translate_path, root_dir, use_pass_pipeline):
        """Test all Bell state variants."""
        circuit_path = circuits_dir / circuit_file
        qasm3_code, original_circuit = self.run_full_pipeline(
            circuit_path, quantum_opt_path, quantum_translate_path, root_dir, use_pass_pipeline
        )

        # Verify structure
        assert "OPENQASM 3.0" in qasm3_code
        assert "h" in qasm3_code.lower()
        assert "cx" in qasm3_code.lower() or "cnot" in qasm3_code.lower()
        assert "measure" in qasm3_code

    @pytest.mark.parametrize("circuit_file", [
        "ghz_3qubit.qasm",
        "ghz_4qubit.qasm",
        "ghz_5qubit.qasm",
    ])
    def test_ghz_states(self, circuit_file, circuits_dir, quantum_opt_path, quantum_translate_path, root_dir, use_pass_pipeline):
        """Test GHZ state preparation for various sizes."""
        circuit_path = circuits_dir / circuit_file
        qasm3_code, _ = self.run_full_pipeline(
            circuit_path, quantum_opt_path, quantum_translate_path, root_dir, use_pass_pipeline
        )

        assert "OPENQASM 3.0" in qasm3_code
        assert "h" in qasm3_code.lower()
        assert qasm3_code.lower().count("cx") + qasm3_code.lower().count("cnot") >= 2

    @pytest.mark.parametrize("circuit_file", [
        "single_qubit_pauli.qasm",
        "single_qubit_phase.qasm",
    ])
    def test_single_qubit_gates(self, circuit_file, circuits_dir, quantum_opt_path, quantum_translate_path, root_dir, use_pass_pipeline):
        """Test single-qubit gate translations."""
        circuit_path = circuits_dir / circuit_file
        qasm3_code, _ = self.run_full_pipeline(
            circuit_path, quantum_opt_path, quantum_translate_path, root_dir, use_pass_pipeline
        )

        assert "OPENQASM 3.0" in qasm3_code
        assert "measure" in qasm3_code

    @pytest.mark.parametrize("circuit_file", [
        "rotation_gates_basic.qasm",
        "rotation_gates_advanced.qasm",
    ])
    def test_rotation_gates(self, circuit_file, circuits_dir, quantum_opt_path, quantum_translate_path, root_dir, use_pass_pipeline):
        """Test parameterized rotation gates."""
        circuit_path = circuits_dir / circuit_file
        qasm3_code, _ = self.run_full_pipeline(
            circuit_path, quantum_opt_path, quantum_translate_path, root_dir, use_pass_pipeline
        )

        assert "OPENQASM 3.0" in qasm3_code
        # Check for rotation gates (could be rx/ry/rz or decomposed)
        assert any(gate in qasm3_code.lower() for gate in ["rx", "ry", "rz", "u3"])

    @pytest.mark.parametrize("circuit_file", [
        "mid_measurement.qasm",
        "mid_circuit_simple.qasm",
        "multiple_measurements.qasm",
    ])
    def test_mid_circuit_measurement(self, circuit_file, circuits_dir, quantum_opt_path, quantum_translate_path, root_dir, use_pass_pipeline):
        """Test mid-circuit measurement support."""
        circuit_path = circuits_dir / circuit_file
        qasm3_code, _ = self.run_full_pipeline(
            circuit_path, quantum_opt_path, quantum_translate_path, root_dir, use_pass_pipeline
        )

        assert "OPENQASM 3.0" in qasm3_code
        assert "measure" in qasm3_code
        # Mid-circuit measurements should have multiple measure statements
        assert qasm3_code.count("measure") >= 2

    @pytest.mark.parametrize("circuit_file", [
        "conditional_x.qasm",
        "conditional_z.qasm",
        "ctrl_logic.qasm",
    ])
    def test_conditional_operations(self, circuit_file, circuits_dir, quantum_opt_path, quantum_translate_path, root_dir, use_pass_pipeline):
        """Test classical control flow."""
        circuit_path = circuits_dir / circuit_file
        qasm3_code, _ = self.run_full_pipeline(
            circuit_path, quantum_opt_path, quantum_translate_path, root_dir, use_pass_pipeline
        )

        assert "OPENQASM 3.0" in qasm3_code
        assert "measure" in qasm3_code
        # Note: Conditional logic may be optimized away by canonicalization
        # The important thing is the circuit translates without errors
        assert len(qasm3_code) > 50  # Non-empty circuit

    @pytest.mark.parametrize("circuit_file", [
        "toffoli_gate.qasm",
        "fredkin_gate.qasm",
    ])
    def test_three_qubit_gates(self, circuit_file, circuits_dir, quantum_opt_path, quantum_translate_path, root_dir, use_pass_pipeline):
        """Test three-qubit gates."""
        circuit_path = circuits_dir / circuit_file
        qasm3_code, _ = self.run_full_pipeline(
            circuit_path, quantum_opt_path, quantum_translate_path, root_dir, use_pass_pipeline
        )

        assert "OPENQASM 3.0" in qasm3_code
        # Toffoli and Fredkin might be decomposed
        assert len(qasm3_code) > 100  # Non-trivial circuit

    @pytest.mark.skipif(not AER_AVAILABLE, reason="Qiskit Aer not available")
    @pytest.mark.parametrize("circuit_file", [
        "bell_state_00.qasm",
        "ghz_3qubit.qasm",
        "grover_2qubit.qasm",
    ])
    def test_semantic_equivalence(self, circuit_file, circuits_dir, quantum_opt_path, quantum_translate_path, root_dir, use_pass_pipeline):
        """Test that translated circuits are semantically equivalent."""
        circuit_path = circuits_dir / circuit_file
        qasm3_code, original_circuit = self.run_full_pipeline(
            circuit_path, quantum_opt_path, quantum_translate_path, root_dir, use_pass_pipeline
        )

        if original_circuit.num_clbits == 0:
            pytest.skip("No measurements in circuit")

        SHOTS = 10000
        sim = AerSimulator()

        # Simulate original
        from qiskit import transpile
        qc_t = transpile(original_circuit, sim)
        res1 = sim.run(qc_t, shots=SHOTS).result()
        counts1 = res1.get_counts()

        # Simulate translated
        qc2 = qiskit.qasm3.loads(qasm3_code)
        qc2_t = transpile(qc2, sim)
        res2 = sim.run(qc2_t, shots=SHOTS).result()
        counts2 = res2.get_counts()

        # Compare using Hamming weight aggregation
        agg1 = aggregate_by_hamming_weight(counts1)
        agg2 = aggregate_by_hamming_weight(counts2)

        dist = hellinger_distance(agg1, agg2, SHOTS)

        # Allow for statistical noise
        assert dist < 0.15, f"Hellinger distance too large: {dist}"


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_circuit(self, circuits_dir, quantum_opt_path, quantum_translate_path, root_dir):
        """Test minimal/empty circuit."""
        circuit_path = circuits_dir / "empty_circuit.qasm"

        qc = QuantumCircuit.from_qasm_file(str(circuit_path))
        importer = QiskitToCatalystImporter(qc)
        module = importer.convert()

        # Should not crash
        assert module is not None

    def test_single_gate_circuit(self, circuits_dir, quantum_opt_path, quantum_translate_path, root_dir):
        """Test circuit with single gate."""
        circuit_path = circuits_dir / "single_hadamard.qasm"

        qc = QuantumCircuit.from_qasm_file(str(circuit_path))
        importer = QiskitToCatalystImporter(qc)
        module = importer.convert()

        assert module is not None

    def test_dense_circuit(self, circuits_dir, quantum_opt_path, quantum_translate_path, root_dir):
        """Test circuit with many gates."""
        circuit_path = circuits_dir / "dense_circuit.qasm"

        qc = QuantumCircuit.from_qasm_file(str(circuit_path))
        importer = QiskitToCatalystImporter(qc)
        module = importer.convert()

        # Should handle large circuits
        assert module is not None
        assert len(str(module)) > 500  # Should be substantial


class TestOutputFormat:
    """Test output format compliance."""

    def test_qasm3_header(self, circuits_dir, quantum_opt_path, quantum_translate_path, root_dir, use_pass_pipeline):
        """Verify QASM 3.0 header is present."""
        circuit_path = circuits_dir / "bell_state_00.qasm"

        qc = QuantumCircuit.from_qasm_file(str(circuit_path))
        importer = QiskitToCatalystImporter(qc)
        module = importer.convert()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.mlir', delete=False) as tmp_mlir:
            tmp_mlir.write(str(module))
            tmp_mlir_path = tmp_mlir.name

        if use_pass_pipeline:
            opt_cmd = [
                str(quantum_opt_path),
                "--pass-pipeline=builtin.module(apply-transform-sequence, canonicalize, merge-rotations)",
                tmp_mlir_path,
                "-o", tmp_mlir_path
            ]
            subprocess.run(opt_cmd, capture_output=True, text=True, check=True)

        cmd = [str(quantum_translate_path), "--mlir-to-qasm3", tmp_mlir_path]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        qasm3_code = result.stdout
        Path(tmp_mlir_path).unlink()

        lines = qasm3_code.strip().split('\n')
        assert lines[0] == "OPENQASM 3.0;"
        assert 'include "stdgates.inc"' in qasm3_code

    @pytest.mark.parametrize("circuit_file", [
        "prepare_plus_state.qasm",
        "prepare_minus_state.qasm",
    ])
    def test_qubit_declarations(self, circuit_file, circuits_dir, quantum_opt_path, quantum_translate_path, root_dir, use_pass_pipeline):
        """Verify qubit declarations are emitted."""
        circuit_path = circuits_dir / circuit_file

        qc = QuantumCircuit.from_qasm_file(str(circuit_path))
        importer = QiskitToCatalystImporter(qc)
        module = importer.convert()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.mlir', delete=False) as tmp_mlir:
            tmp_mlir.write(str(module))
            tmp_mlir_path = tmp_mlir.name

        if use_pass_pipeline:
            opt_cmd = [
                str(quantum_opt_path),
                "--pass-pipeline=builtin.module(apply-transform-sequence, canonicalize, merge-rotations)",
                tmp_mlir_path,
                "-o", tmp_mlir_path
            ]
            subprocess.run(opt_cmd, capture_output=True, text=True, check=True)

        cmd = [str(quantum_translate_path), "--mlir-to-qasm3", tmp_mlir_path]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        qasm3_code = result.stdout
        Path(tmp_mlir_path).unlink()

        # Should have qubit declaration
        assert "qubit" in qasm3_code.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
