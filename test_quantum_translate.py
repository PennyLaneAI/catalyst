import unittest
import os
import subprocess
import tempfile
import sys
from qiskit import QuantumCircuit
try:
    from qiskit_importer_standalone import QiskitToCatalystImporter
except ImportError:
    # Attempt to setup PYTHONPATH automatically for developer convenience
    import sys
    from pathlib import Path
    
    # Current script directory
    script_dir = Path(__file__).parent.resolve()
    
    # Expected relative path to mlir_core
    # /home/ubuntu/catalyst/mlir/llvm-project/build/tools/mlir/python_packages/mlir_core
    mlir_core_path = script_dir / "mlir" / "llvm-project" / "build" / "tools" / "mlir" / "python_packages" / "mlir_core"
    
    if mlir_core_path.exists():
        print(f"Adding {mlir_core_path} to sys.path")
        sys.path.append(str(mlir_core_path))
    else:
        print(f"Warning: Could not find mlir_core at {mlir_core_path}")

    # Also add current directory to path to find qiskit_importer_standalone
    sys.path.append(str(script_dir))
    
    try:
        from qiskit_importer_standalone import QiskitToCatalystImporter
    except ImportError as e:
        # If it still fails, it might be due to mlir dependency inside the importer
        # The importer itself raises ImportError if mlir is missing
        print(f"Error importing modules: {e}")
        # We might need to retry import inside the test or ensure dependencies are met
        # functionality is verified in main block usually, but here we depend on it globally?
        # Actually QiskitToCatalystImporter import might fail because IT imports mlir.
        pass

from mlir.ir import Module

class TestQuantumTranslate(unittest.TestCase):
    def run_pipeline(self, circuit: QuantumCircuit) -> str:
        # Convert Qiskit circuit to MLIR
        importer = QiskitToCatalystImporter(circuit)
        module = importer.convert()
        
        # Write MLIR to a temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.mlir', delete=False) as tmp_mlir:
            tmp_mlir.write(str(module))
            tmp_mlir_path = tmp_mlir.name
        
        try:
            # Run quantum-translate
            cmd = ["./mlir/build/bin/quantum-translate", "--mlir-to-qasm3", tmp_mlir_path]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return result.stdout
        finally:
            os.remove(tmp_mlir_path)

    def test_basic_gates(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.x(1)
        qc.cx(0, 1)
        
        qasm_output = self.run_pipeline(qc)
        
        # Verify essential QASM elements
        self.assertIn("OPENQASM 3.0;", qasm_output)
        self.assertIn("include \"stdgates.inc\";", qasm_output)
        self.assertIn("qubit", qasm_output) # qubit declaration might vary, usually qubit[n] or similar
        self.assertIn("h ", qasm_output)
        self.assertIn("x ", qasm_output) # x gate might be mapped to something else or standard x
        self.assertIn("cnot ", qasm_output) # cnot might be cx

    def test_measurement(self):
        qc = QuantumCircuit(1, 1)
        qc.h(0)
        qc.measure(0, 0)
        
        qasm_output = self.run_pipeline(qc)
        
        self.assertIn("measure ", qasm_output)

    def test_rotation_gates(self):
        # Depending on what qiskit_importer supports. 
        # Based on my read, it supports 'custom' for unknown gates, let's see how they translate.
        qc = QuantumCircuit(1)
        qc.rx(3.14, 0)
        
        qasm_output = self.run_pipeline(qc)
        # Verify it at least runs without error and produces something
        self.assertTrue(len(qasm_output) > 0)
        
if __name__ == '__main__':
    unittest.main()
