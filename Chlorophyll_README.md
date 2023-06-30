# Chlorophyll

[Brief Presentation Slides](https://docs.google.com/presentation/d/1Z2Im4KdymWeJ3aO2LewoNmYIuOLrDEbgNzXtrWWuFxs/edit?usp=sharing)

Chlorophyll is a (hugely) experimental Catalyst frontend that uses a combination of reflection and AST traversal to capture PennyLane Python programs.

It was originally developed for the Xanadu Q2 Hack Week 2023 by Jacob Mai Peng.

## Reproducing the benchmarks

Unfortunately, the Python configurations for JAX-based Catalyst and Chlorophyll are incompatible due to patches applied to the MLIR Python bindings during the build process.

Chlorophyll requires that the MLIR Python bindings are unmodified, which can be done after building Catalyst normally by running:

```sh
cmake --build mlir/build
```

Chlorophyll also currently requires that programs use PennyLane's NumPy, though support for other NumPy wrappers (and vanilla NumPy) should be possible.

These factors make a nice, reproducible setup challenging for Chlorophyll as it currently stands.
