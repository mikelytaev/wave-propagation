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

.. automodule:: pywaveprop.helmholtz_jax
   :members:

Tropospheric radio wave propagation
------------------------------------

.. automodule:: pywaveprop.rwp_jax
   :members:

.. automodule:: pywaveprop.rwp_field
   :members:

Underwater acoustics
--------------------

.. automodule:: pywaveprop.uwa_jax
   :members:

Computational parameters
------------------------

.. automodule:: pywaveprop.helmholtz_common
   :members:

.. automodule:: pywaveprop.uwa_utils
   :members:

Legacy modules (deprecated)
===========================

.. warning::
   The following modules are deprecated. Use the JAX-based modules above instead.

.. automodule:: pywaveprop.propagators.sspade
   :members:
   :deprecated: