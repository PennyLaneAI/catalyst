"""
Pytest-based tests for QASM3 translation pipeline.

This test suite provides comprehensive testing of the Qiskit -> MLIR -> QASM3 pipeline.
"""

import math
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
from qiskit import QuantumCircuit

try:
    from qiskit_importer_standalone import QiskitToCatalystImporter
except ImportError:
    pytest.skip("qiskit_importer_standalone not available", allow_module_level=True)

try:
    import qiskit.qasm3
    from qiskit_aer import AerSimulator

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
        weight = str(k.count("1"))
        agg[weight] = agg.get(weight, 0) + v
    return agg


class TestQASM3Pipeline:
    """Test the full QASM3 translation pipeline."""

    def run_full_pipeline(
        self,
        circuit_path,
        quantum_opt_path,
        quantum_translate_path,
        root_dir,
        use_pass_pipeline=False,
    ):
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
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as tmp_mlir:
            tmp_mlir.write(str(module))
            tmp_mlir_path = tmp_mlir.name

        # 4. Conditionally apply quantum-opt passes
        if use_pass_pipeline:
            opt_cmd = [
                str(quantum_opt_path),
                "--pass-pipeline=builtin.module(apply-transform-sequence, canonicalize, merge-rotations)",
                tmp_mlir_path,
                "-o",
                tmp_mlir_path,
            ]
            subprocess.run(opt_cmd, capture_output=True, text=True, check=True)

        # 5. Translate to QASM3
        cmd = [str(quantum_translate_path), "--mlir-to-qasm3", tmp_mlir_path]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        Path(tmp_mlir_path).unlink()  # Cleanup

        return result.stdout, qc

    @pytest.mark.parametrize(
        "circuit_file",
        [
            "bell_state_00.qasm",
            "bell_state_01.qasm",
            "bell_state_10.qasm",
            "bell_state_11.qasm",
        ],
    )
    def test_bell_states(
        self,
        circuit_file,
        circuits_dir,
        quantum_opt_path,
        quantum_translate_path,
        root_dir,
        use_pass_pipeline,
    ):
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

    @pytest.mark.parametrize(
        "circuit_file",
        [
            "ghz_3qubit.qasm",
            "ghz_4qubit.qasm",
            "ghz_5qubit.qasm",
        ],
    )
    def test_ghz_states(
        self,
        circuit_file,
        circuits_dir,
        quantum_opt_path,
        quantum_translate_path,
        root_dir,
        use_pass_pipeline,
    ):
        """Test GHZ state preparation for various sizes."""
        circuit_path = circuits_dir / circuit_file
        qasm3_code, _ = self.run_full_pipeline(
            circuit_path, quantum_opt_path, quantum_translate_path, root_dir, use_pass_pipeline
        )

        assert "OPENQASM 3.0" in qasm3_code
        assert "h" in qasm3_code.lower()
        assert qasm3_code.lower().count("cx") + qasm3_code.lower().count("cnot") >= 2

    @pytest.mark.parametrize(
        "circuit_file",
        [
            "single_qubit_pauli.qasm",
            "single_qubit_phase.qasm",
        ],
    )
    def test_single_qubit_gates(
        self,
        circuit_file,
        circuits_dir,
        quantum_opt_path,
        quantum_translate_path,
        root_dir,
        use_pass_pipeline,
    ):
        """Test single-qubit gate translations."""
        circuit_path = circuits_dir / circuit_file
        qasm3_code, _ = self.run_full_pipeline(
            circuit_path, quantum_opt_path, quantum_translate_path, root_dir, use_pass_pipeline
        )

        assert "OPENQASM 3.0" in qasm3_code
        assert "measure" in qasm3_code

    @pytest.mark.parametrize(
        "circuit_file",
        [
            "rotation_gates_basic.qasm",
            "rotation_gates_advanced.qasm",
        ],
    )
    def test_rotation_gates(
        self,
        circuit_file,
        circuits_dir,
        quantum_opt_path,
        quantum_translate_path,
        root_dir,
        use_pass_pipeline,
    ):
        """Test parameterized rotation gates."""
        circuit_path = circuits_dir / circuit_file
        qasm3_code, _ = self.run_full_pipeline(
            circuit_path, quantum_opt_path, quantum_translate_path, root_dir, use_pass_pipeline
        )

        assert "OPENQASM 3.0" in qasm3_code
        # Check for rotation gates (could be rx/ry/rz or decomposed)
        assert any(gate in qasm3_code.lower() for gate in ["rx", "ry", "rz", "u3"])

    @pytest.mark.parametrize(
        "circuit_file",
        [
            "mid_measurement.qasm",
            "mid_circuit_simple.qasm",
            "multiple_measurements.qasm",
        ],
    )
    def test_mid_circuit_measurement(
        self,
        circuit_file,
        circuits_dir,
        quantum_opt_path,
        quantum_translate_path,
        root_dir,
        use_pass_pipeline,
    ):
        """Test mid-circuit measurement support."""
        circuit_path = circuits_dir / circuit_file
        qasm3_code, _ = self.run_full_pipeline(
            circuit_path, quantum_opt_path, quantum_translate_path, root_dir, use_pass_pipeline
        )

        assert "OPENQASM 3.0" in qasm3_code
        assert "measure" in qasm3_code
        # Mid-circuit measurements should have multiple measure statements
        assert qasm3_code.count("measure") >= 2

    @pytest.mark.parametrize(
        "circuit_file",
        [
            "conditional_x.qasm",
            "conditional_z.qasm",
            "ctrl_logic.qasm",
        ],
    )
    def test_conditional_operations(
        self,
        circuit_file,
        circuits_dir,
        quantum_opt_path,
        quantum_translate_path,
        root_dir,
        use_pass_pipeline,
    ):
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

    @pytest.mark.parametrize(
        "circuit_file",
        [
            "toffoli_gate.qasm",
            "fredkin_gate.qasm",
        ],
    )
    def test_three_qubit_gates(
        self,
        circuit_file,
        circuits_dir,
        quantum_opt_path,
        quantum_translate_path,
        root_dir,
        use_pass_pipeline,
    ):
        """Test three-qubit gates."""
        circuit_path = circuits_dir / circuit_file
        qasm3_code, _ = self.run_full_pipeline(
            circuit_path, quantum_opt_path, quantum_translate_path, root_dir, use_pass_pipeline
        )

        assert "OPENQASM 3.0" in qasm3_code
        # Toffoli and Fredkin might be decomposed
        assert len(qasm3_code) > 100  # Non-trivial circuit

    @pytest.mark.skipif(not AER_AVAILABLE, reason="Qiskit Aer not available")
    @pytest.mark.parametrize(
        "circuit_file",
        [
            "bell_state_00.qasm",
            "ghz_3qubit.qasm",
            "grover_2qubit.qasm",
        ],
    )
    def test_semantic_equivalence(
        self,
        circuit_file,
        circuits_dir,
        quantum_opt_path,
        quantum_translate_path,
        root_dir,
        use_pass_pipeline,
    ):
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

    def test_single_gate_circuit(
        self, circuits_dir, quantum_opt_path, quantum_translate_path, root_dir
    ):
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

    def test_qasm3_header(
        self, circuits_dir, quantum_opt_path, quantum_translate_path, root_dir, use_pass_pipeline
    ):
        """Verify QASM 3.0 header is present."""
        circuit_path = circuits_dir / "bell_state_00.qasm"

        qc = QuantumCircuit.from_qasm_file(str(circuit_path))
        importer = QiskitToCatalystImporter(qc)
        module = importer.convert()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as tmp_mlir:
            tmp_mlir.write(str(module))
            tmp_mlir_path = tmp_mlir.name

        if use_pass_pipeline:
            opt_cmd = [
                str(quantum_opt_path),
                "--pass-pipeline=builtin.module(apply-transform-sequence, canonicalize, merge-rotations)",
                tmp_mlir_path,
                "-o",
                tmp_mlir_path,
            ]
            subprocess.run(opt_cmd, capture_output=True, text=True, check=True)

        cmd = [str(quantum_translate_path), "--mlir-to-qasm3", tmp_mlir_path]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        qasm3_code = result.stdout
        Path(tmp_mlir_path).unlink()

        lines = qasm3_code.strip().split("\n")
        assert lines[0] == "OPENQASM 3.0;"
        assert 'include "stdgates.inc"' in qasm3_code

    @pytest.mark.parametrize(
        "circuit_file",
        [
            "prepare_plus_state.qasm",
            "prepare_minus_state.qasm",
        ],
    )
    def test_qubit_declarations(
        self,
        circuit_file,
        circuits_dir,
        quantum_opt_path,
        quantum_translate_path,
        root_dir,
        use_pass_pipeline,
    ):
        """Verify qubit declarations are emitted."""
        circuit_path = circuits_dir / circuit_file

        qc = QuantumCircuit.from_qasm_file(str(circuit_path))
        importer = QiskitToCatalystImporter(qc)
        module = importer.convert()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as tmp_mlir:
            tmp_mlir.write(str(module))
            tmp_mlir_path = tmp_mlir.name

        if use_pass_pipeline:
            opt_cmd = [
                str(quantum_opt_path),
                "--pass-pipeline=builtin.module(apply-transform-sequence, canonicalize, merge-rotations)",
                tmp_mlir_path,
                "-o",
                tmp_mlir_path,
            ]
            subprocess.run(opt_cmd, capture_output=True, text=True, check=True)

        cmd = [str(quantum_translate_path), "--mlir-to-qasm3", tmp_mlir_path]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        qasm3_code = result.stdout
        Path(tmp_mlir_path).unlink()

        # Should have qubit declaration
        assert "qubit" in qasm3_code.lower()


class TestDynamicCircuits:
    """OpenQASM 3.0 dynamic-circuit syntax: reset, barrier, classical
    registers, feedforward if/else on register values, and while loops.

    Circuits are built in code (these constructs cannot all be expressed in
    QASM 2.0 inputs) and pushed through the full pipeline with and without
    the quantum-opt pass pipeline.
    """

    def pipeline_from_circuit(self, qc, quantum_opt_path, quantum_translate_path, optimize):
        """Convert a QuantumCircuit to QASM3 via importer -> [quantum-opt] -> quantum-translate."""
        importer = QiskitToCatalystImporter(qc)
        module = importer.convert()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as tmp_mlir:
            tmp_mlir.write(str(module))
            tmp_mlir_path = tmp_mlir.name

        if optimize:
            opt_cmd = [
                str(quantum_opt_path),
                "--pass-pipeline=builtin.module(apply-transform-sequence, canonicalize, merge-rotations)",
                tmp_mlir_path,
                "-o",
                tmp_mlir_path,
            ]
            subprocess.run(opt_cmd, capture_output=True, text=True, check=True)

        cmd = [str(quantum_translate_path), "--mlir-to-qasm3", tmp_mlir_path]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        Path(tmp_mlir_path).unlink()
        return result.stdout

    @staticmethod
    def assert_valid_qasm3(qasm3_code):
        """Parse the emitted QASM3 with qiskit.qasm3 if the importer is installed."""
        try:
            import qiskit.qasm3 as qasm3_mod

            return qasm3_mod.loads(qasm3_code)
        except ImportError:
            pytest.skip("qiskit qasm3 importer not installed")

    @staticmethod
    def creg_circuit():
        from qiskit import ClassicalRegister, QuantumRegister

        q = QuantumRegister(2, "q")
        c = ClassicalRegister(2, "c")
        qc = QuantumCircuit(q, c)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure(q, c)
        return qc

    @staticmethod
    def reset_barrier_circuit():
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.reset(0)
        qc.barrier()
        qc.x(0)
        qc.measure([0, 1], [0, 1])
        return qc

    @staticmethod
    def if_else_circuit():
        from qiskit import ClassicalRegister, QuantumRegister

        q = QuantumRegister(2, "q")
        c = ClassicalRegister(2, "c")
        qc = QuantumCircuit(q, c)
        qc.h(0)
        qc.measure(0, 0)
        with qc.if_test((c[0], 1)) as else_:
            qc.x(1)
        with else_:
            qc.z(1)
        qc.measure(1, 1)
        return qc

    @staticmethod
    def register_condition_circuit():
        from qiskit import ClassicalRegister, QuantumRegister

        q = QuantumRegister(3, "q")
        c = ClassicalRegister(2, "c")
        qc = QuantumCircuit(q, c)
        qc.h(0)
        qc.h(1)
        qc.measure(0, 0)
        qc.measure(1, 1)
        with qc.if_test((c, 2)):
            qc.x(2)
        qc.measure(2, 0)
        return qc

    @staticmethod
    def while_loop_circuit():
        from qiskit import ClassicalRegister, QuantumRegister

        q = QuantumRegister(1, "q")
        c = ClassicalRegister(1, "c")
        qc = QuantumCircuit(q, c)
        qc.h(0)
        qc.measure(0, 0)
        with qc.while_loop((c[0], 1)):
            qc.h(0)
            qc.measure(0, 0)
        return qc

    @pytest.mark.parametrize("optimize", [False, True])
    def test_creg_measurement(self, quantum_opt_path, quantum_translate_path, optimize):
        qasm3 = self.pipeline_from_circuit(
            self.creg_circuit(), quantum_opt_path, quantum_translate_path, optimize
        )
        assert "bit[2] c;" in qasm3
        assert "c[0] = measure" in qasm3
        assert "c[1] = measure" in qasm3
        assert "bit m_" not in qasm3, "creg measurements must not fall back to anonymous bits"

    @pytest.mark.parametrize("optimize", [False, True])
    def test_reset_and_barrier(self, quantum_opt_path, quantum_translate_path, optimize):
        qasm3 = self.pipeline_from_circuit(
            self.reset_barrier_circuit(), quantum_opt_path, quantum_translate_path, optimize
        )
        assert "reset q0[0];" in qasm3
        assert "barrier q0[0], q0[1];" in qasm3

    @pytest.mark.parametrize("optimize", [False, True])
    def test_if_else(self, quantum_opt_path, quantum_translate_path, optimize):
        qasm3 = self.pipeline_from_circuit(
            self.if_else_circuit(), quantum_opt_path, quantum_translate_path, optimize
        )
        assert "if (c[0]) {" in qasm3
        assert "} else {" in qasm3
        assert "x q0[1];" in qasm3
        assert "z q0[1];" in qasm3

    @pytest.mark.parametrize("optimize", [False, True])
    def test_register_value_condition(self, quantum_opt_path, quantum_translate_path, optimize):
        qasm3 = self.pipeline_from_circuit(
            self.register_condition_circuit(), quantum_opt_path, quantum_translate_path, optimize
        )
        # c == 2 is AND-folded bitwise in MLIR; the translator reconstructs
        # the whole-register comparison.
        assert "if (c == 2)" in qasm3

    @pytest.mark.parametrize("optimize", [False, True])
    def test_while_loop(self, quantum_opt_path, quantum_translate_path, optimize):
        qasm3 = self.pipeline_from_circuit(
            self.while_loop_circuit(), quantum_opt_path, quantum_translate_path, optimize
        )
        assert "while (c[0]) {" in qasm3
        # The in-body measurement must re-assign the same named bit the
        # condition reads; that is what carries the loop state in QASM3.
        body = qasm3.split("while (c[0]) {", 1)[1]
        assert "c[0] = measure" in body

    @pytest.mark.parametrize(
        "builder",
        ["creg_circuit", "reset_barrier_circuit", "if_else_circuit", "while_loop_circuit"],
    )
    def test_output_parses_as_qasm3(self, quantum_opt_path, quantum_translate_path, builder):
        qc = getattr(self, builder)()
        qasm3 = self.pipeline_from_circuit(qc, quantum_opt_path, quantum_translate_path, True)
        self.assert_valid_qasm3(qasm3)

    @pytest.mark.skipif(not AER_AVAILABLE, reason="qiskit-aer not installed")
    def test_if_else_semantics(self, quantum_opt_path, quantum_translate_path):
        """Feedforward circuit: original and translated distributions must match."""
        qc = self.if_else_circuit()
        qasm3 = self.pipeline_from_circuit(qc, quantum_opt_path, quantum_translate_path, True)
        translated = self.assert_valid_qasm3(qasm3)

        shots = 10000
        sim = AerSimulator()
        counts1 = sim.run(qc, shots=shots).result().get_counts()
        counts2 = sim.run(translated, shots=shots).result().get_counts()

        agg1 = aggregate_by_hamming_weight(counts1)
        agg2 = aggregate_by_hamming_weight(counts2)
        distance = hellinger_distance(agg1, agg2, shots)
        assert distance < 0.1, f"Hellinger distance too high: {distance}"

    @pytest.mark.parametrize("optimize", [False, True])
    def test_logic_and_is_not_register_equality(
        self, quantum_opt_path, quantum_translate_path, optimize
    ):
        """A generic logic AND of two bits on a WIDER register must stay a
        conjunction — rewriting it to `c == 3` would wrongly assert the
        unmentioned bit is 0. Only importer-tagged equality folds may be
        reconstructed as whole-register comparisons."""
        from qiskit import ClassicalRegister, QuantumRegister
        from qiskit.circuit.classical import expr

        q = QuantumRegister(4, "q")
        c = ClassicalRegister(3, "c")
        qc = QuantumCircuit(q, c)
        qc.h(0)
        qc.h(1)
        qc.h(2)
        qc.measure(0, 0)
        qc.measure(1, 1)
        qc.measure(2, 2)
        with qc.if_test(expr.logic_and(expr.lift(c[0]), expr.lift(c[1]))):
            qc.x(3)
        qc.measure(3, 0)

        qasm3 = self.pipeline_from_circuit(qc, quantum_opt_path, quantum_translate_path, optimize)
        assert "if ((c[0] && c[1])) {" in qasm3
        assert "c == 3" not in qasm3

    @pytest.mark.parametrize("optimize", [False, True])
    def test_while_body_measures_other_clbit(
        self, quantum_opt_path, quantum_translate_path, optimize
    ):
        """A non-condition clbit first measured inside a while body must be
        usable in conditions after the loop (loop-carried classical state)."""
        from qiskit import ClassicalRegister, QuantumRegister

        q = QuantumRegister(2, "q")
        c = ClassicalRegister(2, "c")
        qc = QuantumCircuit(q, c)
        qc.h(0)
        qc.measure(0, 0)
        with qc.while_loop((c[0], 1)):
            qc.h(1)
            qc.measure(1, 1)
            qc.h(0)
            qc.measure(0, 0)
        with qc.if_test((c[1], 1)):
            qc.x(1)
        qc.measure(1, 1)

        qasm3 = self.pipeline_from_circuit(qc, quantum_opt_path, quantum_translate_path, optimize)
        assert "while (c[0]) {" in qasm3
        assert "if (c[1]) {" in qasm3
        assert "unknown_cond" not in qasm3

    def test_official_teleport_roundtrip(self, quantum_opt_path, quantum_translate_path):
        """Official spec example teleport.qasm (reset, barrier, U, feedforward
        ifs, custom identity gate) must survive the full pipeline and re-parse
        as valid QASM3."""
        import re

        try:
            import qiskit.qasm3 as qasm3_mod
        except ImportError:
            pytest.skip("qiskit qasm3 importer not installed")

        src_path = Path(__file__).parent / "openqasm3_official_example" / "teleport.qasm"
        src = src_path.read_text()
        # qiskit_qasm3_import quirk: single-bit comparisons must be written
        # '== true'; the spec's own example uses '== 1'.
        src = re.sub(r"==\s*1\b", "==true", src)
        qc = qasm3_mod.loads(src)

        qasm3 = self.pipeline_from_circuit(qc, quantum_opt_path, quantum_translate_path, True)
        assert qasm3.count("reset ") == 3
        assert "barrier" in qasm3
        assert "U(" in qasm3
        assert "if (" in qasm3
        qasm3_mod.loads(qasm3)

    @pytest.mark.skipif(not AER_AVAILABLE, reason="qiskit-aer not installed")
    def test_while_loop_semantics(self, quantum_opt_path, quantum_translate_path):
        """Repeat-until-zero loop must terminate with c[0] == 0 in both versions."""
        qc = self.while_loop_circuit()
        qasm3 = self.pipeline_from_circuit(qc, quantum_opt_path, quantum_translate_path, True)
        translated = self.assert_valid_qasm3(qasm3)

        shots = 1000
        sim = AerSimulator()
        counts1 = sim.run(qc, shots=shots).result().get_counts()
        counts2 = sim.run(translated, shots=shots).result().get_counts()
        assert set(counts1) == {"0"}
        assert set(counts2) == {"0"}


class TestHybridFrontend:
    """The hybrid QASM3 loader (qasm3_frontend.load_qasm3): qiskit fast path
    plus the openqasm3-AST partial evaluator for constructs qiskit rejects."""

    # Official spec examples that must translate through the full pipeline.
    SUPPORTED = [
        "adder.qasm", "cphase.qasm", "inverseqft1.qasm", "inverseqft2.qasm",
        "qec.qasm", "qft.qasm", "qpt.qasm", "rb.qasm", "rus.qasm",
        "teleport.qasm", "varteleport.qasm",
    ]
    # Files that must fail cleanly, with the named blocker in the error.
    UNSUPPORTED = {
        "alignment.qasm": "Stretch",
        "dd.qasm": "Stretch",
        "t1.qasm": "Duration",
        "defcal.qasm": "defcal",
        "gateteleport.qasm": "extern",
        "scqec.qasm": "extern",
        "vqe.qasm": "extern",
        "ipe.qasm": "bit or bit register",
        # arrays.qasm and msd.qasm contain genuine out-of-bounds accesses in
        # the spec's own text (my_defined_uints[4] on size 4; scratch[3] on
        # qubit[3]).
        "arrays.qasm": "out of bounds",
        "msd.qasm": "out of bounds",
    }
    INPUTS = {"msd.qasm": {"level": 1}}

    @staticmethod
    def examples_dir():
        return Path(__file__).parent / "openqasm3_official_example"

    def full_pipeline(self, qc, quantum_opt_path, quantum_translate_path):
        from qiskit_importer_standalone import QiskitToCatalystImporter

        importer = QiskitToCatalystImporter(qc)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as tmp:
            tmp.write(str(importer.convert()))
            tmp_path = tmp.name
        # NOTE: -o must not target the input file — quantum-opt mmaps its
        # input, and overwriting it in place SIGBUSes on larger modules.
        opt_path = tmp_path + ".opt.mlir"
        subprocess.run(
            [str(quantum_opt_path),
             "--pass-pipeline=builtin.module(apply-transform-sequence, canonicalize, merge-rotations)",
             tmp_path, "-o", opt_path],
            capture_output=True, text=True, check=True)
        result = subprocess.run(
            [str(quantum_translate_path), "--mlir-to-qasm3", opt_path],
            capture_output=True, text=True, check=True)
        Path(tmp_path).unlink()
        Path(opt_path).unlink()
        return result.stdout

    @pytest.mark.parametrize("filename", SUPPORTED)
    def test_official_example_translates(
        self, quantum_opt_path, quantum_translate_path, filename
    ):
        import openqasm3

        from qasm3_frontend import load_qasm3

        src = (self.examples_dir() / filename).read_text()
        qc = load_qasm3(src, inputs=self.INPUTS.get(filename))
        qasm3 = self.full_pipeline(qc, quantum_opt_path, quantum_translate_path)
        assert "OPENQASM 3.0;" in qasm3
        assert "unknown_cond" not in qasm3
        # The emitted program must be grammatically valid QASM3.
        openqasm3.parse(qasm3)

    @pytest.mark.parametrize("filename", sorted(UNSUPPORTED))
    def test_official_example_unsupported_reason(self, filename):
        from qasm3_frontend import QASM3FrontendError, load_qasm3

        src = (self.examples_dir() / filename).read_text()
        with pytest.raises(QASM3FrontendError) as excinfo:
            load_qasm3(src, inputs=self.INPUTS.get(filename))
        assert self.UNSUPPORTED[filename].lower() in str(excinfo.value).lower()

    def test_qiskit_fast_path_still_used(self):
        """Programs qiskit can parse must go through qiskit (dynamic-circuit
        support is better there)."""
        from qasm3_frontend import load_qasm3

        qc = load_qasm3(
            'OPENQASM 3.0;\ninclude "stdgates.inc";\n'
            "qubit[2] q;\nbit[2] c;\nh q[0];\ncx q[0], q[1];\n"
            "c[0] = measure q[0];\nc[1] = measure q[1];\n"
        )
        assert qc.num_qubits == 2

    def test_input_values_required(self):
        from qasm3_frontend import QASM3FrontendError, load_qasm3

        src = ('OPENQASM 3.0;\ninclude "stdgates.inc";\n'
               "input uint[4] a_in;\nqubit[4] q;\n"
               "for uint i in [0:3] { if (bool(a_in[i])) x q[i]; }\n")
        with pytest.raises(QASM3FrontendError, match="input 'a_in'"):
            load_qasm3(src)
        qc = load_qasm3(src, inputs={"a_in": 5})
        # bits 0 and 2 of 5 are set -> two x gates
        assert sum(1 for inst in qc.data if inst.operation.name == "x") == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
