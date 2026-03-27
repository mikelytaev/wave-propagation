============
API reference
============

.. toctree::
   :maxdepth: 2

   propagators

JAX-based modules (primary)
===========================

Helmholtz propagator
--------------------

.. automodule:: pywaveprop.experimental.helmholtz_jax
   :members:

Tropospheric radio wave propagation
------------------------------------

.. automodule:: pywaveprop.experimental.rwp_jax
   :members:

.. automodule:: pywaveprop.experimental.rwp_field
   :members:

Underwater acoustics
--------------------

.. automodule:: pywaveprop.experimental.uwa_jax
   :members:

Computational parameters
------------------------

.. automodule:: pywaveprop.experimental.helmholtz_common
   :members:

.. automodule:: pywaveprop.experimental.uwa_utils
   :members:

Legacy modules (deprecated)
===========================

.. warning::
   The following modules are deprecated. Use the JAX-based modules above instead.

.. automodule:: pywaveprop.propagators.sspade
   :members:
   :deprecated: