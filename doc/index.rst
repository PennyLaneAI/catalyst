:og:description: Catalyst is a package that enables just-in-time (JIT) compilation of PennyLane programs. Compile the entire quantum-classical workflow.

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
            Catalyst is a package that enables just-in-time (JIT)
            compilation of PennyLane programs. Compile the entire quantum-classical workflow.
        </p>
        <img src="_static/catalyst.png" style="max-width: 700px; width: 100%;">
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

.. mdinclude:: ../README.md
  :start-line: 20
  :end-line: 73

.. mdinclude:: ../README.md
  :start-line: 142
  :end-line: 163

.. toctree::
   :maxdepth: 2
   :titlesonly:
   :hidden:

   dev/installation
   dev/quick_start
   dev/devices
   dev/autograph
   dev/sharp_bits
   dev/jax_integration
   dev/callbacks
   dev/release_notes

.. toctree::
   :maxdepth: 2
   :caption: Development
   :hidden:

   dev/architecture
   PennyLane Frontend <modules/frontend>
   Compiler Core <modules/mlir>
   MLIR Dialects <dev/dialects>
   Compiler Passes <dev/transforms>
   Quantum Runtime <modules/runtime>
   dev/debugging
   dev/custom_devices
   dev/roadmap

.. toctree::
   :maxdepth: 2
   :caption: Examples
   :hidden:

   demos/magic_distillation_demo
   demos/adaptive_circuits_demo
   demos/qml/tutorial_qubit_rotation
   QML Optimization with Optax <demos/qml_optimization>

.. toctree::
   :maxdepth: 1
   :caption: API
   :hidden:

   code/__init__
   Runtime Device Interface <api/file_runtime_include_QuantumDevice.hpp>
   QIR C-API <api/file_runtime_include_RuntimeCAPI.h>
