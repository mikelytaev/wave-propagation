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

Terrain-following rational propagator
-------------------------------------

.. automodule:: pywaveprop.helmholtz_tf_jax
   :members:

.. automodule:: pywaveprop.terrain_transform
   :members:

Split-step Fourier propagator
-----------------------------

.. automodule:: pywaveprop.helmholtz_ssf_jax
   :members:

Rank-adaptive split-step Fourier propagator
-------------------------------------------

.. automodule:: pywaveprop.helmholtz_rassf_jax
   :members:

Split-step wavelet propagator
-----------------------------

.. automodule:: pywaveprop.helmholtz_ssw_jax
   :members:

.. automodule:: pywaveprop.utils.wavelets
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