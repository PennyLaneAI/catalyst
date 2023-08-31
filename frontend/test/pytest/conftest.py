try:
    import tensorflow as tf
except (ImportError, ModuleNotFoundError) as e:
    tf_available = False
else:
    tf_available = True

def pytest_runtest_setup(item):
    """Automatically skip tests if interfaces are not installed"""
    # Autograd is assumed to be installed
    interfaces = {"tf"}
    available_interfaces = {
        "tf": tf_available,
    }

    allowed_interfaces = [
        allowed_interface
        for allowed_interface in interfaces
        if available_interfaces[allowed_interface] is True
    ]

    # load the marker specifying what the interface is
    all_interfaces = {"tf"}
    marks = {mark.name for mark in item.iter_markers() if mark.name in all_interfaces}

    for b in marks:
        if b not in allowed_interfaces:
            pytest.skip(
                f"\nTest {item.nodeid} only runs with {allowed_interfaces} interfaces(s) but {b} interface provided",
            )
