from setuptools import setup

entry_points = {
    "catalyst.passes_resolution": [
        "standalone.passes = standalone_plugin",
    ],
}

setup(
    name="standalone_plugin",
    version="0.1.0",
    description="The standalone plugin example as a python package",
    packages=["standalone_plugin"],
    entry_points=entry_points,
    include_package_data=True,
    install_requires=["PennyLane", "PennyLane-Catalyst"],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Science/Research",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3.10",
    ],
)
