Python API
==========

The following modules form the current forward-facing API. GLPBench remains an
alpha release, so interfaces may evolve before version 1.0.

Core contracts
--------------

The contract objects are structurally immutable: their fields and parameter
mappings cannot be reassigned, while contained PyTorch tensors and PyG data
objects retain their normal mutable behavior.

.. automodule:: glpbench.contracts
   :members:

Component interfaces
--------------------

These runtime-checkable protocols describe the intended extension boundaries
without requiring user components to inherit from GLPBench base classes.
Model authors can also subclass ``glpbench.models.BaseModelAdapter`` for shared
validation, loss, and scoring behavior. The bundled models expose adapters
implementing the model contract. The
``glpbench.models.LinkPrediction`` compatibility façade and current runners
resolve construction, loss computation, and scoring through these adapters.

.. automodule:: glpbench.interfaces
   :members:

Exceptions
----------

.. automodule:: glpbench.exceptions
   :members:

Configuration
-------------

Canonical configuration files are parsed strictly into frozen dataclasses.
Unknown fields and invalid nested values raise path-aware
``ConfigurationError`` exceptions.

.. automodule:: glpbench.configuration
   :members:
   :imported-members:

Datasets
--------

.. automodule:: glpbench.datasets
   :members:
   :imported-members:

Protocols
---------

.. automodule:: glpbench.protocols
   :members:
   :imported-members:

Preprocessing
-------------

.. automodule:: glpbench.preprocessing
   :members:
   :imported-members:

Models
------

.. automodule:: glpbench.models
   :members:
   :imported-members:

Testing extensions
------------------

Reusable conformance checks help external model authors verify the adapter
boundary without depending on GLPBench's internal test suite.

.. automodule:: glpbench.testing
   :members:
   :imported-members:

Evaluation
----------

.. automodule:: glpbench.evaluation
   :members:
   :imported-members:

Training
--------

.. automodule:: glpbench.training
   :members:
   :imported-members:

Logging
-------

.. automodule:: glpbench.logging
   :members:
   :imported-members:

Runners
-------

.. automodule:: glpbench.runners
   :members:
   :imported-members:
