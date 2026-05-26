"""
Patch to catalyst/device/qjit_device.py
========================================

This shows the exact modifications needed to integrate the PennyLane Python
compatibility backend into Catalyst's frontend device resolution.

Apply by inserting into catalyst/device/qjit_device.py at the indicated locations.
"""

# ===========================================================================
# CHANGE 1: Add to SUPPORTED_RT_DEVICES dict (line ~134)
# ===========================================================================

# Before:
# SUPPORTED_RT_DEVICES = {
#     "null.qubit": ("NullQubit", "librtd_null_qubit"),
#     "braket.aws.qubit": ("OpenQasmDevice", "librtd_openqasm"),
#     "braket.local.qubit": ("OpenQasmDevice", "librtd_openqasm"),
# }

# After:
SUPPORTED_RT_DEVICES_NEW = {
    "null.qubit": ("NullQubit", "librtd_null_qubit"),
    "braket.aws.qubit": ("OpenQasmDevice", "librtd_openqasm"),
    "braket.local.qubit": ("OpenQasmDevice", "librtd_openqasm"),
    "pennylane.python": ("PLPythonDevice", "librtd_pennylane_python"),
}


# ===========================================================================
# CHANGE 2: Modify extract_backend_info to auto-wrap unknown devices (line ~180)
# ===========================================================================

# Replace the else clause that raises CompileError with a fallback:
#
# Before:
#     elif hasattr(device, "get_c_interface"):
#         # Support third party devices with `get_c_interface`
#         device_name, device_lpath = device.get_c_interface()
#     else:
#         raise CompileError(f"The {dname} device does not provide C interface for compilation.")
#
# After:

def extract_backend_info_patched(device):
    """Patched version of extract_backend_info with PLPython fallback."""
    import pennylane as qp
    from catalyst.utils.exceptions import CompileError
    from catalyst.utils.runtime_environment import get_lib_path

    dname = device.name
    if isinstance(device, qp.devices.LegacyDeviceFacade):
        dname = device.target_device.short_name

    device_name = ""
    device_lpath = ""
    device_kwargs = {}

    if dname in SUPPORTED_RT_DEVICES_NEW:
        device_name = SUPPORTED_RT_DEVICES_NEW[dname][0]
        device_lpath = get_lib_path("runtime", "RUNTIME_LIB_DIR")
        import platform, os
        sys_platform = platform.system()
        if sys_platform == "Linux":
            device_lpath = os.path.join(device_lpath, SUPPORTED_RT_DEVICES_NEW[dname][1] + ".so")
        elif sys_platform == "Darwin":
            device_lpath = os.path.join(device_lpath, SUPPORTED_RT_DEVICES_NEW[dname][1] + ".dylib")
        else:
            raise NotImplementedError(f"Platform not supported: {sys_platform}")

    elif hasattr(device, "get_c_interface"):
        device_name, device_lpath = device.get_c_interface()

    else:
        # === NEW FALLBACK: wrap any PennyLane device with the Python compat layer ===
        # Instead of raising CompileError, we wrap the device automatically.
        import warnings
        warnings.warn(
            f"Device '{dname}' does not provide a native C interface. "
            f"Using the PennyLane Python compatibility layer (pennylane.python). "
            f"This executes quantum instructions via Python callback and is slower "
            f"than native backends.",
            stacklevel=2,
        )
        device_name = "PLPythonDevice"
        device_lpath = get_lib_path("runtime", "RUNTIME_LIB_DIR")
        import platform, os
        sys_platform = platform.system()
        if sys_platform == "Linux":
            device_lpath = os.path.join(device_lpath, "librtd_pennylane_python.so")
        elif sys_platform == "Darwin":
            device_lpath = os.path.join(device_lpath, "librtd_pennylane_python.dylib")
        else:
            raise NotImplementedError(f"Platform not supported: {sys_platform}")

        # Pass device info as kwargs so the C++ runtime can reconstruct it
        device_kwargs["pl_device_name"] = dname
        device_kwargs["wires"] = str(device.num_wires)
        device_kwargs["shots"] = str(device.shots.total_shots if device.shots else 0)

    # ... rest of the function continues as before ...
    return device_name, device_lpath, device_kwargs


# ===========================================================================
# CHANGE 3: Add to runtime/lib/backend/CMakeLists.txt
# ===========================================================================
# add_subdirectory(pennylane_python)
