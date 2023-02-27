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
    :name: API
    :link: code/__init__.html
    :description: Explore the Catalyst API

.. raw:: html

      </div>
   </div>
   <br>

.. mdinclude:: ../README.md
  :start-line: 19
  :end-line: 75

.. toctree::
   :maxdepth: 2
   :titlesonly:
   :hidden:

   dev/installation
   dev/quick_start
   dev/development
   dev/architecture
   dev/release_notes

.. toctree::
   :maxdepth: 2
   :caption: Modules
   :hidden:

   modules/frontend
   modules/mlir
   modules/runtime

.. toctree::
   :maxdepth: 2
   :caption: Examples
   :hidden:

   demos/magic_distillation_demo
   demos/adaptive_circuits_demo
   demos/tutorial_qubit_rotation
   demos/tutorial_qft_arithmetics

.. toctree::
   :maxdepth: 1
   :caption: API
   :hidden:

   code/__init__
   C++ API <api/library_root>
