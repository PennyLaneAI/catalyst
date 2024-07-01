---
title: 'Catalyst: a Python JIT compiler for auto-differentiable hybrid quantum programs'
tags:
  - Python
  - quantum computing
  - machine learning
  - automatic differentiation
  - GPU
  - high-performance computing
  - JAX
authors:
  - name: David Ittah
    orcid: 0000-0003-0975-6448
    equal-contrib: true
    affiliation: 1
  - name: Ali Asadi
    equal-contrib: true
    affiliation: 1
  - name: Erick Ochoa Lopez
    equal-contrib: true
    affiliation: 1
  - name: Sergei Mironov
    equal-contrib: true
    affiliation: 1
  - name: Samuel Banning
    equal-contrib: true
    affiliation: 1
  - name: Romain Moyard
    orcid: 0000-0003-0837-6451
    equal-contrib: true
    affiliation: 1
  - name: Mai Jacob Peng
    orcid: 0000-0002-2377-264X
    equal-contrib: true
    affiliation: 1
  - name: Josh Izaac
    orcid: 0000-0003-2640-0734
    corresponding: true
    affiliation: 1
affiliations:
 - name: Xanadu, Toronto, ON, M5G 2C8, Canada
   index: 1
date: 01 July 2023
bibliography: paper.bib
---

# Summary

Catalyst is a software package for capturing Python-based hybrid quantum programs (that is, programs
that contain both quantum and classical instructions), and just-in-time (JIT) compiling them down
to an MLIR and LLVM representation and generating binary code. As a result, Catalyst enables the
ability to rapidly prototype quantum algorithms in Python alongside efficient compilation,
optimization, and execution of the program on classical and quantum accelerators. In addition,
Catalyst allows for advanced quantum programming features essential for fault-tolerant hardware
support and advanced algorithm design, such as mid-circuit measurement with arbitrary
post-processing, support for classical control flow in and around quantum algorithms, built-in
measurement statistics, and hardware-compatible automatic differentiation (AD).

# Statement of need

The rapid development and availability of quantum software and hardware has had significant
influence on quantum algorithm development and general quantum computing research. Through direct
access to full-featured quantum programming SDKs [@qiskit; @cirq; @pennylane], high-performance
simulators [@pennylane; @cuquantum], and near-term noisy hardware
[@borealis; @braket; @ibmq], researchers have new tools available to prototype, execute, analyze,
and iterate during algorithm development. Notably, this has resulted in the development and
exploration of new categories of quantum algorithms that take advantage of the strong integration
with advanced classical tooling; an example being auto-differentiation and variational quantum
algorithms [@mcclean2016theory; @farhi2014quantum; @delgado2021variational].

However, as our hardware capabilities scale, it is becoming clear that we cannot separate classical
and quantum parts of the program; classical processing remains essential for processing quantum
input and output, as well as mid-circuit processing that must be performant enough to keep up with
the quantum execution. Furthermore, we must support this while retaining the ability to rapidly
prototype, integrate with classical tooling and accelerators, and provide efficient optimization
and compilation of both quantum and classical instructions.

One of the core goals of Catalyst is to provide a unified representation for hybrid programs with
which to drive optimization, device compilation, automatic differentiation, and many other types of
transformations in a scalable way. Moreover, Catalyst is being developed to support next-generation
quantum programming paradigms, such as dynamic circuit generation with classical control flow,
real-time measurement feedback, qubit reuse, and dynamic quantum memory management. Most
importantly, Catalyst provides a way to transform large scale user workflows from Python into
low-level binary code for accelerated execution in heterogeneous environments.

Catalyst is divided into three core components: a *frontend* which captures and lowers hybrid Python
programs, a *compiler* which applies quantum and classical optimizations and transformations, and
a *runtime*, which allows the compiled binary to call into quantum devices for execution.

## Frontend

The Catalyst frontend, built in Python and C++, directly integrates with both PennyLane [@pennylane]
(a Python framework for differentiable quantum programming) and JAX [@jax] (a Python framework for
accelerated auto-differentiation) to capture hybrid quantum programs. As a result, by decorating
hybrid programs with the `@qjit` decorator, the Catalyst frontend is able to capture and
ahead-of-time or just-in-time compile (from within Python) the quantum and classical instructions
provided by PennyLane and JAX. In addition, Catalyst provides high-level functions for compact and
dynamic circuit representation (`for_loop`, `cond`, `while_loop`, `measure`) as well as
auto-differentiation (`grad`, `jacobian`, `vjp`, `jvp`). Preliminary support for AutoGraph
[@moldovan2018autograph] also allows users to write hybrid quantum programs using native Python
control flow; all branches of the computation will be represented in the captured hybrid program.

## Compiler

Building on the LLVM [@llvm] and MLIR [@mlir] compiler frameworks, and the QIR project
[@qir], compilation is then performed on the MLIR-based representation for hybrid quantum programs
defined by Catalyst. The compiler invokes a sequence of transformations that lowers the hybrid
program to a lower level of abstraction, outputting LLVM IR with QIR syntax. In addition to the
lowering process, various transformations take place, including quantum optimizations
(adjoint cancellation, operator fusion), classical optimizations (code elimination), and automatic
differentiation. In the latter case, classical auto-differentiation is provided via the Enzyme
[@enzyme] framework, while hardware-compatible quantum gradient methods (such as the parameter-shift
rule [@schuld2018gradients; @wierichs2021general]) are provided as Catalyst compiler transforms.

## Runtime

The Catalyst Runtime is designed to enable Catalyst’s highly dynamic execution model. As such, it
generally assumes direct communication between a quantum device and its classical controller or
host, although it also supports more restrictive execution models. Execution of the user program
proceeds on the host’s native architecture, while the runtime provides an abstract communication
API for quantum devices that the user program is free to invoke at any time during its execution.
Currently, Catalyst provides runtime integration for the high-performance PennyLane Lightning suite
of simulators [@lightning], as well as an OpenQASM3 pipeline with
Amazon Braket simulator and hardware support.

# Examples

The following example highlights the capabilities of the Catalyst frontend, enabling scalable and
high-performance quantum computing from a feature-rich interactive Python environment.

First, we utilize Catalyst to just-in-time compile a complex function involving a mix of classical
and quantum processing. Note that, through the AutoGraph feature, native Python control flow is
automatically captured, allowing both branches to be represented in the compiled program.

```python
import pennylane as qml
from catalyst import qjit, measure
from jax import numpy as jnp

dev = qml.device("lightning.qubit", wires=2)

@qjit(autograph=True)
def hybrid_function(x):

    @qml.qnode(dev, diff_method="parameter-shift")
    def circuit(x):
        qml.RX(x, wires=0)
        qml.RY(x ** 2, wires=1)
        qml.CNOT(wires=[0, 1])

        for i in range(0, 10):
            m = measure(wires=0)

            if m == 1:
                qml.CRX(x * jnp.exp(- x ** 2), wires=[0, 1])

            x = x * 0.2

        return qml.expval(qml.PauliZ(0))

    return jnp.sin(circuit(x)) ** 2
```

```pycon
>>> hybrid_function(0.543)
array(0.70807342)
```

We can also consider an example that includes a classical optimization loop, such as optimizing a
quantum computer to find the ground state energy of a molecule:

```python
import pennylane as qml
from catalyst import grad, for_loop, qjit
import jaxopt
from jax import numpy as jnp

mol = qml.data.load("qchem", molname="H3+")[0]
n_qubits = len(mol.hamiltonian.wires)

dev = qml.device("lightning.qubit", wires=n_qubits)

@qjit
@qml.qnode(dev)
def cost(params):
    qml.BasisState(jnp.array(mol.hf_state), wires=range(n_qubits))
    qml.DoubleExcitation(params[0], wires=[0, 1, 2, 3])
    qml.DoubleExcitation(params[1], wires=[0, 1, 4, 5])
    return qml.expval(mol.hamiltonian)

@qjit
def optimization(params):
    loss = lambda x: (cost(x), grad(cost)(x))

    # set up optimizer and define optimization step
    opt = jaxopt.GradientDescent(loss, stepsize=0.3, value_and_grad=True)
    update_step = lambda step, args: tuple(opt.update(*args))

    # gradient descent parameter update loop using jit-compatible for-loop
    state = opt.init_state(params)
    (params, _) = for_loop(0, 10, step=1)(update_step)((params, state))
    return params
```

```pycon
>>> params = jnp.array([0.54, 0.3154])
>>> final_params = optimization(params)
>>> cost(final_params)  # optimized energy of H3+
-1.2621179827928877
>>> mol.vqe_energy  # expected energy of H3+
-1.2613407428534986
```

Here, we are using the JAXopt gradient optimization library [@blondel2022efficient] alongside the
built-in auto-differentiation capabilities of Catalyst, to compile the entire optimization
workflow. For this small toy example on 6 qubits, we can time the execution after compilation on
the same system, as a non-rigorous demonstration of the advantage of performing this for loop
outside of Python:

```python
>>> %timeit optimization(params)
599 ms ± 96 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
```

Comparing this to a non-compiled workflow, where the `@qjit` decorator has
been removed:

```python
@qml.qnode(dev)
def no_qjit_cost(params):
    qml.BasisState(jnp.array(mol.hf_state), wires=range(n_qubits))
    qml.DoubleExcitation(params[0], wires=[0, 1, 2, 3])
    qml.DoubleExcitation(params[1], wires=[0, 1, 4, 5])
    return qml.expval(mol.hamiltonian)

def no_qjit_optimization(params):
    # set up optimizer
    opt = jaxopt.GradientDescent(no_qjit_cost, stepsize=0.3, jit=False)
    state = opt.init_state(params)

    for i in range(15):
        (params, state) = opt.update(params, state)

    return params
```

```pycon
>>> %timeit no_qjit_optimization(params)
3.73 s ± 522 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
```

For more code examples, please see the Catalyst documentation\footnote
{https://docs.pennylane.ai/projects/catalyst}.

# Installation and contribution

The Catalyst source code is available under the Apache 2.0 license on GitHub\footnote
{https://github.com/PennyLaneAI/catalyst}, and binaries are available for `pip` installation on
Linux and macOS\footnote{https://pypi.org/project/pennylane-catalyst}. Contributions to
Catalyst --- via feedback, issues, or pull requests on GitHub --- are welcomed. Additional Catalyst
documentation and tutorials are available on our online documentation\footnote
{https://docs.pennylane.ai/projects/catalyst}.

# Discussion and future work

The Catalyst hybrid compilation stack as presented here provides an end-to-end infrastructure
to explore next-generation dynamic hybrid quantum-classical algorithms, by allowing for workflows
that support compressed representation of large, highly structured quantum algorithms,
as well as mid-circuit measurements with arbitrary classical processing and feedforward.

The Catalyst software stack will continue to be developed alongside research, algorith, and hardware
needs, with potential future work including support for quantum hardware control systems, building
out a library of MLIR quantum compilation passes for optimizing quantum circuits (without unrolling
classical control structure), and explorations of dynamic quantum error mitigation and proof-of-concept
error correction experiments.

Quantum software is driving many new results and ideas in quantum computing research, and the
PennyLane framework has already been used in a number of scientific publications [@delgado2021variational; @wierichs2021general]
and educational materials [@demos]. By enabling researchers to scale up their ideas and
algorithms, and execute on both near-term and future quantum hardware, the software presented here
will help drive future research in quantum computing.

# Acknowledgements

We acknowledge contributions from Lee J O’Riordan, Nathan Killoran, and Olivia Di Matteo during the
genesis of this project.

# References
