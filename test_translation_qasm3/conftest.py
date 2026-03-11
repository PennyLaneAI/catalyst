"""
Pytest configuration and fixtures for QASM3 translation tests.
"""

import sys
import pytest
from pathlib import Path

# Setup paths
ROOT_DIR = Path(__file__).parent.parent.resolve()
sys.path.append(str(ROOT_DIR))

# Setup MLIR path
mlir_core_path = ROOT_DIR / "mlir" / "llvm-project" / "build" / "tools" / "mlir" / "python_packages" / "mlir_core"
if mlir_core_path.exists():
    sys.path.append(str(mlir_core_path))


@pytest.fixture(scope="session")
def root_dir():
    """Root directory of the Catalyst project."""
    return ROOT_DIR


@pytest.fixture(scope="session")
def quantum_opt_path(root_dir):
    """Path to quantum-opt binary."""
    return root_dir / "mlir" / "build" / "bin" / "quantum-opt"


@pytest.fixture(scope="session")
def quantum_translate_path(root_dir):
    """Path to quantum-translate binary."""
    return root_dir / "mlir" / "build" / "bin" / "quantum-translate"


@pytest.fixture(scope="session")
def circuits_dir():
    """Directory containing test circuits."""
    return Path(__file__).parent / "qasm3_circuits"


@pytest.fixture
def temp_mlir_file(tmp_path):
    """Temporary MLIR file for testing."""
    return tmp_path / "test.mlir"
