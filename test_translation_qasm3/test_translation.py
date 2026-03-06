import os
import sys
import tempfile
import subprocess
from pathlib import Path
from qiskit import QuantumCircuit

# Setup environment to find mlir_core without setting PYTHONPATH manually
script_dir = Path(__file__).parent.resolve()
root_dir = script_dir.parent
sys.path.append(str(root_dir)) # For qiskit_importer_standalone

# Helper to find mlir_core location similar to test_quantum_translate.py
mlir_core_path = root_dir / "mlir" / "llvm-project" / "build" / "tools" / "mlir" / "python_packages" / "mlir_core"
if mlir_core_path.exists():
    sys.path.append(str(mlir_core_path))
else:
    print(f"Warning: Could not find mlir_core at {mlir_core_path}")

try:
    from qiskit_importer_standalone import QiskitToCatalystImporter
except ImportError as e:
    print(f"Failed to import QiskitToCatalystImporter: {e}")
    sys.exit(1)

def run_pipeline(circuit_path):
    print(f"Testing {circuit_path.name}...")
    try:
        # 1. Load QASM into Qiskit Circuit
        qc = QuantumCircuit.from_qasm_file(str(circuit_path))
    except Exception as e:
        print(f"  [Error] Failed to load QASM: {e}")
        return False

    try:
        # 2. Convert to Catalyst MLIR
        importer = QiskitToCatalystImporter(qc)
        module = importer.convert()
    except Exception as e:
        print(f"  [Error] Failed to convert to MLIR: {e}")
        return False

    # 3. Apply Catalyst passes (decomposition, optimization)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.mlir', delete=False) as tmp_mlir:
        tmp_mlir.write(str(module))
        tmp_mlir_path = tmp_mlir.name
    
    try:
        opt_cmd = [
            str(root_dir / "mlir" / "build" / "bin" / "quantum-opt"),
            "--pass-pipeline=builtin.module(apply-transform-sequence, canonicalize, merge-rotations)",
            tmp_mlir_path,
            "-o", tmp_mlir_path
        ]
        result = subprocess.run(opt_cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"  [Error] quantum-opt failed: {e.stderr}")
        os.remove(tmp_mlir_path)
        return False
    except Exception as e:
        print(f"  [Error] quantum-opt failed: {e}")
        os.remove(tmp_mlir_path)
        return False

    # 4. Translate to OpenQASM 3
    try:
        cmd = [str(root_dir / "mlir" / "build" / "bin" / "quantum-translate"), "--mlir-to-qasm3", tmp_mlir_path]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        qasm3_code = result.stdout
    except subprocess.CalledProcessError as e:
        print(f"  [Error] quantum-translate failed: {e.stderr}")
        os.remove(tmp_mlir_path)
        return False
    finally:
        if os.path.exists(tmp_mlir_path):
            os.remove(tmp_mlir_path)

    # 4. Validate Output
    # Basic validation: check for expected instructions based on input file name/content
    
    filename = circuit_path.name
    
    if "mid_measurement" in filename:
        if "measure" not in qasm3_code:
             print("  [Failure] Missing 'measure' instruction")
             return False

    elif "teleportation" in filename:
        if "measure" not in qasm3_code:
             print("  [Failure] Missing 'measure' instruction")
             return False

    elif "cascade_measure" in filename:
        if "measure" not in qasm3_code:
             print("  [Failure] Missing 'measure' instruction")
             return False

    elif "gate_library" in filename:
        # Check for parameterized gates, they might be rx/ry/rz or decomposed into u3
        if "rx" not in qasm3_code and "u3" not in qasm3_code and "ry" not in qasm3_code and "rz" not in qasm3_code:
             print("  [Failure] Missing parameterized gates (rx/ry/rz/u3)")
             return False

    elif "ctrl_logic" in filename:
        # If statements can be optimized out. Just ensure we generated QASM
        if "OPENQASM 3.0" not in qasm3_code:
             print("  [Failure] Missing QASM header")
             return False
             
    elif "reused_qubit" in filename:
        pass # Placeholder

    print("  [Success] Generated OpenQASM 3 code.")
    print(qasm3_code) 
    return True

def main():
    circuits_dir = script_dir / "qasm3_circuits"
    if not circuits_dir.exists():
        print(f"Directory not found: {circuits_dir}")
        sys.exit(1)
    
    success_count = 0
    total_count = 0
    
    for qasm_file in circuits_dir.glob("*.qasm"):
        total_count += 1
        if run_pipeline(qasm_file):
            success_count += 1
            
    print(f"\nTest Summary: {success_count}/{total_count} passed.")
    
    if success_count < total_count:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()
