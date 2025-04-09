import re
import os
import sys
import requests

version_file_path = os.path.join(os.path.dirname(__file__), "../../frontend/catalyst/_version.py")

assert os.path.isfile(version_file_path)

def get_base_version():
    with open(version_file_path, "r") as f:
        lines = f.readlines()

        version_line = lines[-1]
        assert "__version__ = " in version_line

    pattern = r"(\d+).(\d+).(\d+)"
    match = re.search(pattern, version_line)
    assert match

    major, minor, bug = match.groups()

    return f"{major}.{minor}.{bug}"


def get_latest_rc_version(base_version):
    # Query TestPyPI for all versions of the package
    package_name = "PennyLane-Catalyst"  # Adjust if needed
    url = f"https://test.pypi.org/pypi/{package_name}/json"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise an exception for bad status codes
        data = response.json()
        versions = data.get("releases", {}).keys()
    except requests.exceptions.RequestException as e:
        print(f"Error querying TestPyPI: {e}", file=sys.stderr)
        sys.exit(1)

    # Filter versions that start with our base version and have .rc suffix
    rc_versions = []
    for version in versions:
        if version.startswith(base_version) and ".rc" in version:
            rc_versions.append(version)

    if not rc_versions:
        # If no existing RC versions are found, start with rc0
        return f"{base_version}.rc0"

    # Sort versions and get the highest rc number
    rc_versions.sort()
    latest_rc = rc_versions[-1]
    rc_match = re.search(r"\.rc(\d+)$", latest_rc)
    if not rc_match:
        print(f"Error: Unexpected version format: {latest_rc}", file=sys.stderr)
        sys.exit(1)

    rc_number = int(rc_match.group(1))
    new_rc_number = rc_number + 1
    new_version = f"{base_version}.rc{new_rc_number}"
    return new_version

def backup_version_file():
    """Create a backup of the version file."""
    backup_path = f"{version_file_path}.bak"
    with open(version_file_path, "r", encoding="UTF-8") as src, open(backup_path, "w", encoding="UTF-8") as dst:
        dst.write(src.read())
    return backup_path

def update_version_file(new_version):
    """Update the version file with the new version."""
    with open(version_file_path, "r", encoding="UTF-8") as f:
        lines = f.readlines()
        version_line = lines[-1]
        assert "__version__ = " in version_line
        lines[-1] = f'__version__ = "{new_version}"\n'
    with open(version_file_path, "w", encoding="UTF-8") as f:
        f.writelines(lines)

def main():
    base_version = get_base_version()
    next_rc_version = get_latest_rc_version(base_version)
    
    # Note that we only want to update the version file temporarily,
    # and we want to be able to revert to the original version after the
    # release candidate build is finished.
    backup_path = backup_version_file()
    update_version_file(next_rc_version)
    
    print(f"Backed up version file to: {backup_path}")
    print(f"Updated version to: {next_rc_version}")

if __name__ == "__main__":
    main() 
