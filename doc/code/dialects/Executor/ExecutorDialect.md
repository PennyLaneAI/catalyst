<!-- Manually maintained; modified from mlir-tblgen output -->

# 'executor' Dialect

_A executor dialect for dispatching kernels and calling symbols._

The executor dialect models a small set of operations that describe how a host program
interacts with an executor service:

* Opening a session with an executor endpoint
* Shipping a cross-compiled kernel object file (or other assets) to that endpoint
* Dispatching a previously-shipped qnode kernel
* Invoking an arbitrary symbol in a shared library
