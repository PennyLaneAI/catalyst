:og:description: Catalyst is an experimental package that enables just-in-time (JIT) compilation of PennyLane programs. Compile the entire quantum-classical workflow.

Catalyst
########

:Release: |release|

.. raw:: html

    <style>
        .breadcrumb {
            display: none;
        }
        h1 {
            text-align: center;
            margin-bottom: 15px;
        }
        p.lead.grey-text {
            margin-bottom: 30px;
        }
        .footer-relations {
            border-top: 0px;
        }
    </style>

    <div class="container mt-2 mb-2">
        <p class="lead grey-text">
            Catalyst is an experimental package that enables just-in-time (JIT)
            compilation of PennyLane programs. Compile the entire quantum-classical workflow.
        </p>
      <div class="row mt-3">

.. index-card::
    :name: Installation
    :link: dev/installation.html
    :description: Learn how to install Catalyst

.. index-card::
    :name: Quickstart
    :link: dev/quick_start.html
    :description: Get started using Catalyst with PennyLane

.. index-card::
    :name: GitHub
    :link: https://github.com/PennyLaneAI/catalyst
    :description: View the Catalyst source code on GitHub

.. raw:: html

      </div>
   </div>
   <br>

.. image:: _static/pl-catalyst-logo-lightmode.png
    :align: center
    :width: 700px
    :target: javascript:void(0);

.. mdinclude:: ../README.md
  :start-line: 20
  :end-line: 72

.. mdinclude:: ../README.md
  :start-line: 134
  :end-line: 167

.. toctree::
   :maxdepth: 2
   :caption: Getting Started
   :hidden:

   Installation <dev/installation>
   Quick Start <dev/quick_start>
   Devices <dev/devices>
   Autograph <dev/autograph>
   Sharp Bits <dev/sharp_bits>
   JAX Integration <dev/jax_integration>
   Callbacks <dev/callbacks>
   Release Notes <dev/release_notes>

.. toctree::
   :maxdepth: 2
   :caption: Development
   :hidden:

   Architecture <dev/architecture>
   PennyLane Frontend <modules/frontend>
   Compiler Core <modules/mlir>
   MLIR Dialects <dev/dialects>
   Compiler Passes <dev/transforms>
   Compiler Plugins <dev/plugins>
   Quantum Runtime <modules/runtime>
   Debugging <dev/debugging>
   Custom Devices <dev/custom_devices>
   Roadmap <dev/roadmap>

.. toctree::
   :maxdepth: 2
   :caption: Examples
   :hidden:

   Magic Distillation <demos/magic_distillation_demo>
   Adaptive Circuits <demos/adaptive_circuits_demo>
   Qubit Rotation Tutorial <demos/qml/tutorial_qubit_rotation>
   QML Optimization with Optax <demos/qml_optimization>

.. toctree::
   :maxdepth: 1
   :caption: API
   :hidden:

   Python API <code/__init__>
   Command Line Interface <catalyst-cli/catalyst-cli>
   Runtime Device Interface <api/structCatalyst_1_1Runtime_1_1QuantumDevice>
   MLIR Dialects API <code/dialects/__init__>