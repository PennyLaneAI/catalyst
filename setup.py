from os import path

from setuptools import setup, find_namespace_packages

with open(path.join("frontend", "catalyst", "_version.py")) as f:
    version = f.readlines()[-1].split()[-1].strip("\"'")

with open(".dep-versions") as f:
    jax_version = [line[4:].strip() for line in f.readlines() if "jax=" in line][0]

requirements = [
    "pennylane>=0.28",
    f"jax=={jax_version}",
    f"jaxlib=={jax_version}",
]

classifiers = [
    "Environment :: Console",
    "Natural Language :: English",
    "Intended Audience :: Science/Research",
    "Development Status :: 3 - Alpha",
    "Operating System :: POSIX",
    "Operating System :: POSIX :: Linux",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3 :: Only",
]

setup(
    classifiers=classifiers,
    name="catalyst",
    version=version,
    python_requires=">=3.8",
    install_requires=requirements,
    packages=find_namespace_packages(
        where="frontend",
        include=["catalyst", "catalyst.*", "mlir_quantum"],
    ),
    package_dir={"": "frontend"},
    include_package_data=True,
)
