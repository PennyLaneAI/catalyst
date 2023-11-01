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
simulators[@pennylane; @cuquantum], and near-term noisy hardware
[@borealis; @braket; @ibmq], researchers have new tools available to prototype, execute, analyze,
and iterate during algorithm development. Notably, this has resulted in the development and
exploration of new categories of quantum algorithms that take advantage of the strong integration
with advanced classical tooling; an example being auto-differentiation and variational quantum
algorithms[@mcclean2016theory; @farhi2014quantum; @delgado2021variational].

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

The Catalyst frontend, built in Python and C++, directly integrates with both PennyLane[@pennylane]
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
[@enzyme] framework, while hardware-compatible quantum gradient methods(such as the parameter-shift
rule[@schuld2018gradients; @wierichs2021general]) are provided as Catalyst compiler transforms.

## Runtime

The Catalyst Runtime is designed to enable Catalyst’s highly dynamic execution model. As such, it
generally assumes direct communication between a quantum device and its classical controller or
host, although it also supports more restrictive execution models. Execution of the user program
proceeds on the host’s native architecture, while the runtime provides an abstract communication
API for quantum devices that the user program is free to invoke at any time during its execution.
Currently, Catalyst provides runtime integration for the high-performance PennyLane Lightning suite
of simulators[URL ref may be acceptable as a citation here], as well as an OpenQASM3 pipeline with
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

We can also consider an example that includes a classical optimization loop, such as optimizing a
quantum computer to find the ground state energy of a molecule:

```python
import jaxopt
from catalyst import grad

molecule = qml.data.load("qchem", molname="H3+")[0]
n_qubits = len(molecule.hamiltonian.wires)

dev = qml.device("lightning.qubit", wires=n_qubits)

@qjit
@qml.qnode(dev)
def cost(params):
    qml.BasisState(molecule.hf_state, wires=range(n_qubits))
    qml.DoubleExcitation(params[0], wires=[0, 1, 2, 3])
    qml.DoubleExcitation(params[1], wires=[0, 1, 4, 5])
    return qml.expval(molecule.hamiltonian)

@qjit
def workflow(init_params):
    loss = lambda x: (cost(params), grad(cost)(params)[0])
    opt = jaxopt.GradientDescent(loss, stepsize=0.4, value_and_grad=True)

    update_step = lambda step, args: tuple(opt.update(*args))

    params = init_params
    state = opt.init_state(params)
    (param, _) = for_loop(0, 10, step=1)(update_step)((params, state))
    return param

params = jnp.array([0.54, 0.3154])
workflow(params)
```

Here, we are using the JAXopt gradient optimization library[@blondel2022efficient] alongside the
built-in auto-differentiation capabilities of Catalyst, to compile the entire optimization
workflow. For this small toy example on 6 qubits, we can time the execution after compilation on
the same system, as a non-rigorous demonstration of the advantage of performing this for loop
outside of Python:

```python
>>> %timeit workflow(params)
62 ms ± 254 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
```

Comparing this to a non-compiled workflow, where the `@qjit` decorator has
been removed, and the Catalyst gradient function has been replaced with
`jax.grad`:

```python
>>> %timeit nojit_workflow(params)
440 ms ± 10.1 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
```

For more code examples, please see the Catalyst documentation\footnote
{https://docs.pennylane.ai/projects/catalyst}.

# Installation and contribution

The Catalyst source code is available under the Apache 2.0 license on GitHub\footnote{\url
{https://github.com/PennyLaneAI/catalyst}}, and binaries are available for `pip` installation on
Linux and macOS\footnote{https://pypi.org/project/pennylane-catalyst}. Contributions to
Catalyst --- via feedback, issues, or pull requests on GitHub --- are welcomed. Additional Catalyst
documentation and tutorials are available on our online documentation\footnote
{https://docs.pennylane.ai/projects/catalyst}.

Quantum software is driving many new results and ideas in quantum computing research, and the
PennyLane framework has already been used in a number of scientific publications [citation needed]
and educational materials[citation needed]. By enabling researchers to scale up their ideas and
algorithms, and execute on both near-term and future quantum hardware, the software presented here
will help drive future research in quantum computing.

# Acknowledgements

We acknowledge contributions from Lee J O’Riordan, Nathan Killoran, and Olivia Di Matteo during the
genesis of this project.

# References

@online{ibmq,
    title = {{IBM Quantum Experience}},
    author = {{IBM Corporation}},
    year = 2016,
    url = {https://quantumexperience.ng.bluemix.net/},
    urldate = {2018-11-01}
}

@online{qiskit,
    title = {{Qiskit}},
    author = {{IBM Corporation}},
    year = 2016,
    url = {https://qiskit.org/},
    urldate = {2018-11-01}
}

@online{cirq,
    title = {{Cirq}},
    author = {{Google Inc.}},
    year = 2018,
    url = {https://cirq.readthedocs.io/en/latest/},
    urldate = {2018-11-01}
}

@website{braket,
    title = {{Amazon Braket}},
    author = {{Amazon Web Services}},
    year = 2020,
    url = {https://aws.amazon.com/braket/}
}

@article{pennylane,
    title = {Pennylane: Automatic differentiation of hybrid quantum-classical computations},
    author = {
        Bergholm, Ville and Izaac, Josh and Schuld, Maria and Gogolin, Christian and Ahmed,
        Shahnawaz and Ajith, Vishnu and Alam, M Sohaib and Alonso-Linaje, Guillermo and
        AkashNarayanan, B and Asadi, Ali and others
    },
    year = 2018,
    journal = {arXiv preprint arXiv:1811.04968}
}

@software{cuquantum,
    title = {NVIDIA/cuQuantum: cuQuantum v22.03.0},
    author = {NVIDIA cuQuantum team},
    year = 2022,
    month = mar,
    publisher = {Zenodo},
    doi = {10.5281/zenodo.6385575},
    url = {https://doi.org/10.5281/zenodo.6385575},
    version = {v22.03.0}
}

@article{mcclean2016theory,
    title = {The theory of variational hybrid quantum-classical algorithms},
    author = {McClean, Jarrod R and Romero, Jonathan and Babbush, Ryan and Aspuru-Guzik, Al{\'a}n},
    year = 2016,
    journal = {New Journal of Physics},
    publisher = {IOP Publishing},
    volume = 18,
    number = 2,
    pages = {023023},
    doi = {10.1088/1367-2630/18/2/023023}
}

@article{farhi2014quantum,
    title = {A quantum approximate optimization algorithm},
    author = {Farhi, Edward and Goldstone, Jeffrey and Gutmann, Sam},
    year = 2014,
    journal = {arXiv preprint arXiv:1411.4028}
}

@article{delgado2021variational,
    title = {Variational quantum algorithm for molecular geometry optimization},
    author = {
        Delgado, Alain and Arrazola, Juan Miguel and Jahangiri, Soran and Niu, Zeyue and Izaac,
        Josh and Roberts, Chase and Killoran, Nathan
    },
    year = 2021,
    journal = {Physical Review A},
    publisher = {APS},
    volume = 104,
    number = 5,
    pages = {052402}
}

@software{jax,
    title = {{JAX}: composable transformations of {P}ython+{N}um{P}y programs},
    author = {
        James Bradbury and Roy Frostig and Peter Hawkins and Matthew James Johnson and Chris Leary
        and Dougal Maclaurin and George Necula and Adam Paszke and Jake Vander{P}las and Skye
        Wanderman-{M}ilne and Qiao Zhang
    },
    year = 2018,
    url = {http://github.com/google/jax},
    version = {0.3.13}
}

@inproceedings{llvm,
    title = {{LLVM}: A Compilation Framework for Lifelong Program Analysis and Transformation},
    author = {Chris Lattner and Vikram Adve},
    year = 2004,
    month = {Mar},
    booktitle = CGO,
    address = {San Jose, CA, USA},
    pages = {75--88}
}

@inproceedings{mlir,
    title = {{{MLIR}}: Scaling Compiler Infrastructure for Domain Specific Computation},
    author = {
        Lattner, Chris and Amini, Mehdi and Bondhugula, Uday and Cohen, Albert and Davis, Andy and
        Pienaar, Jacques and Riddle, River and Shpeisman, Tatiana and Vasilache, Nicolas and
        Zinenko, Oleksandr
    },
    year = 2021,
    booktitle = {2021 {{IEEE/ACM}} International Symposium on Code Generation and Optimization (CGO)},
    volume = {},
    number = {},
    pages = {2--14},
    doi = {10.1109/CGO51591.2021.9370308}
}

@article{borealis,
    title = {Quantum computational advantage with a programmable photonic processor},
    author = {
        Madsen, Lars S and Laudenbach, Fabian and Askarani, Mohsen Falamarzi and Rortais, Fabien
        and Vincent, Trevor and Bulmer, Jacob FF and Miatto, Filippo M and Neuhaus, Leonhard and
        Helt, Lukas G and Collins, Matthew J and others
    },
    year = 2022,
    journal = {Nature},
    publisher = {Nature Publishing Group UK London},
    volume = 606,
    number = 7912,
    pages = {75--81}
}

@inproceedings{enzyme,
    title = {
        Instead of Rewriting Foreign Code for Machine Learning, Automatically Synthesize Fast
        Gradients
    },
    author = {Moses, William and Churavy, Valentin},
    year = 2020,
    booktitle = {Advances in Neural Information Processing Systems},
    publisher = {Curran Associates, Inc.},
    volume = 33,
    pages = {12472--12485},
    url = {https://proceedings.neurips.cc/paper/2020/file/9332c513ef44b682e9347822c2e457ac-Paper.pdf},
    editor = {H. Larochelle and M. Ranzato and R. Hadsell and M. F. Balcan and H. Lin}
}

@manual{qir,
    title = {{QIR Specification}},
    author = {{QIR Alliance}},
    year = 2021,
    url = {https://github.com/qir-alliance/qir-spec},
    note = {Also see \url{https://qir-alliance.org}}
}

@article{wierichs2021general,
    title = {General parameter-shift rules for quantum gradients},
    author = {Wierichs, David and Izaac, Josh and Wang, Cody and Lin, Cedric Yen-Yu},
    year = 2022,
    month = mar,
    journal = {{Quantum}},
    publisher = {{Verein zur F{\"{o}}rderung des Open Access Publizierens in den Quantenwissenschaften}},
    volume = 6,
    pages = 677,
    doi = {10.22331/q-2022-03-30-677},
    issn = {2521-327X},
    url = {https://doi.org/10.22331/q-2022-03-30-677}
}

@article{schuld2018gradients,
    title = {Evaluating analytic gradients on quantum hardware},
    author = {Schuld, Maria and Bergholm, Ville and Gogolin, Christian and Izaac, Josh and Killoran, Nathan},
    year = 2019,
    journal = {Physical Review A},
    publisher = {APS},
    volume = 99,
    number = 3,
    pages = {032331}
}

@article{moldovan2018autograph,
    title = {AutoGraph: Imperative-style Coding with Graph-based Performance.(oct 2018)},
    author = {
        Moldovan, Dan and Decker, James M and Wang, Fei and Johnson, Andrew A and Lee, Brian K and
        Nado, Zachary and Sculley, D and Rompf, Tiark and Wiltschko, Alexander B
    },
    year = 2018,
    journal = {arXiv preprint arXiv:1810.08061}
}

@article{blondel2022efficient,
    title = {Efficient and modular implicit differentiation},
    author = {
        Blondel, Mathieu and Berthet, Quentin and Cuturi, Marco and Frostig, Roy and Hoyer, Stephan
        and Llinares-L{\'o}pez, Felipe and Pedregosa, Fabian and Vert, Jean-Philippe
    },
    year = 2022,
    journal = {Advances in neural information processing systems},
    volume = 35,
    pages = {5230--5242}
}




