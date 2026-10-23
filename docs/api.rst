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
without requiring user components to inherit from GLPBench base classes. The
bundled models expose adapters implementing the model contract, although the
current runners continue to use ``glpbench.models.LinkPrediction`` until the
registry migration is complete.

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
