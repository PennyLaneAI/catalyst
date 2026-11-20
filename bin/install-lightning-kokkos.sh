#!/usr/bin/env bash
# Published Kokkos packages still use Classic PennyLane. Installing the Lightning
# branch for Qubit does not install Kokkos, which is a separate package.
# Build Kokkos from the same companion branch until PennyLane2-compatible
# Kokkos nightlies are available, then replace this helper with a nightly install.
# Stop on command failures, unset variables, or failed pipeline steps.
set -euo pipefail

# Use the Python selected by CI, or python3 when run locally.
PYTHON=${PYTHON:-python3}
# Read the same Lightning dependency that Catalyst uses.
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
lightning_source=$(sed -n 's/^lightning=//p' "$repo_root/.dep-versions")

# A numbered version can be installed directly from the package index.
if [[ "$lightning_source" != git+* ]]; then
    "$PYTHON" -m pip install "pennylane-lightning-kokkos==$lightning_source"
    exit
fi

# Build in a temporary checkout and remove it on exit, even if the build fails.
build_dir=$(mktemp -d)
trap 'rm -rf "$build_dir"' EXIT

# Split the Git dependency into its repository URL and branch name.
lightning_repo=${lightning_source%@*}
lightning_ref=${lightning_source##*@}
git clone --depth 1 --branch "$lightning_ref" "${lightning_repo#git+}" "$build_dir"

# The backend configuration script needs tomlkit to update package metadata.
"$PYTHON" -m pip install tomlkit
cd "$build_dir"
# Build the separate Kokkos package from this Lightning checkout.
PL_BACKEND=lightning_kokkos "$PYTHON" scripts/configure_pyproject_toml.py

# Enable the serial CPU backend while preserving any existing CMake options.
# --no-deps keeps Catalyst's selected PennyLane2 and Lightning-Qubit dependencies;
# pip can still install the tools needed to build the package.
CMAKE_ARGS="${CMAKE_ARGS:-} -DKokkos_ENABLE_SERIAL=ON" \
    "$PYTHON" -m pip install --no-deps .
