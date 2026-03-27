===================================
Migration guide: v1.x to v2.0 (JAX)
===================================

PyWaveProp 2.0 introduces a JAX-based computation backend as the primary implementation.
The old NumPy/Cython code is deprecated and will be removed in a future release.

Installation
============

Install JAX and its dependencies::

    pip install jax jaxlib lineax flax

For GPU support, install the CUDA-enabled JAX::

    pip install jax[cuda12]


Tropospheric Radio Wave Propagation (RWP)
==========================================

Old API (deprecated)
--------------------

.. code-block:: python

    from pywaveprop.rwp.sspade import rwp_ss_pade, RWPSSpadeComputationalParams
    from pywaveprop.rwp.antennas import GaussAntenna
    from pywaveprop.rwp.environment import Troposphere

    antenna = GaussAntenna(freq_hz=300e6, height=30, beam_width=15,
                           elevation_angle=0, polarz='H')
    env = Troposphere()
    # ... set up M_profile, terrain, etc.

    params = RWPSSpadeComputationalParams(
        max_range_m=10000,
        max_height_m=300,
    )
    field = rwp_ss_pade(antenna, env, params)
    pl = field.path_loss()


New API (JAX)
-------------

.. code-block:: python

    from pywaveprop.experimental.rwp_jax import (
        RWPGaussSourceModel,
        RWPComputationalParams,
        TroposphereModel,
        EvaporationDuctModel,
        rwp_forward_task,
    )

    src = RWPGaussSourceModel(
        freq_hz=300e6,
        height_m=30,
        beam_width_deg=15,
        elevation_angle_deg=0,
    )
    env = TroposphereModel(
        N_profile=EvaporationDuctModel(height_m=20.0),
    )
    params = RWPComputationalParams(
        max_range_m=10000,
        max_height_m=300,
        dx_m=50,
        dz_m=1,
    )
    field = rwp_forward_task(src, env, params)
    pl = field.path_loss()


Key differences
^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1

   * - Old API
     - New JAX API
   * - ``GaussAntenna``
     - ``RWPGaussSourceModel``
   * - ``Troposphere``
     - ``TroposphereModel``
   * - ``rwp_ss_pade()``
     - ``rwp_forward_task()``
   * - ``RWPSSpadeComputationalParams``
     - ``RWPComputationalParams``
   * - M-profile as function ``M_profile(x, z)``
     - ``AbstractNProfileModel`` objects (``EvaporationDuctModel``, ``PiecewiseLinearNProfileModel``, etc.)
   * - ``Terrain`` class with elevation function
     - ``PiecewiseLinearTerrainModel`` (from ``helmholtz_jax``)


Underwater Acoustics (UWA)
==========================

Old API (deprecated)
--------------------

.. code-block:: python

    from pywaveprop.uwa.sspade import uwa_ss_pade, UWASSpadeComputationalParams
    from pywaveprop.uwa.source import GaussSource
    from pywaveprop.uwa.environment import UnderwaterEnvironment, Bathymetry

    src = GaussSource(freq_hz=500, depth_m=50, beam_width_deg=30, elevation_angle_deg=0)
    env = UnderwaterEnvironment(
        sound_speed_profile_m_s=lambda x, z: 1500,
        bottom_profile=Bathymetry(ranges_m=[0], depths_m=[200]),
        bottom_sound_speed_m_s=1700,
        bottom_density_g_cm=1.8,
        bottom_attenuation_dm_lambda=0.5,
    )
    params = UWASSpadeComputationalParams(max_range_m=5000, max_depth_m=300)
    field = uwa_ss_pade(src, env, params)


New API (JAX)
-------------

.. code-block:: python

    from pywaveprop.experimental.uwa_jax import (
        UWAGaussSourceModel,
        UnderwaterLayerModel,
        UnderwaterEnvironmentModel,
        uwa_forward_task,
    )
    from pywaveprop.experimental.uwa_utils import UWAComputationalParams
    from pywaveprop.experimental.helmholtz_jax import ConstWaveSpeedModel

    src = UWAGaussSourceModel(freq_hz=500, depth_m=50, beam_width_deg=30)
    env = UnderwaterEnvironmentModel(layers=[
        UnderwaterLayerModel(
            height_m=200,
            sound_speed_profile_m_s=ConstWaveSpeedModel(c0=1500),
            density=1.0,
        ),
        UnderwaterLayerModel(
            height_m=100,
            sound_speed_profile_m_s=ConstWaveSpeedModel(c0=1700),
            density=1.8,
            attenuation_dm_lambda=0.5,
        ),
    ])
    params = UWAComputationalParams(max_range_m=5000, dx_m=50, dz_m=1)
    field = uwa_forward_task(src, env, params)


Key differences
^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1

   * - Old API
     - New JAX API
   * - ``GaussSource``
     - ``UWAGaussSourceModel``
   * - ``UnderwaterEnvironment`` (single bottom)
     - ``UnderwaterEnvironmentModel`` with ``UnderwaterLayerModel`` list (multi-layer)
   * - ``uwa_ss_pade()``
     - ``uwa_forward_task()``
   * - Sound speed as ``lambda x, z: ...``
     - ``AbstractWaveSpeedModel`` objects (``ConstWaveSpeedModel``, ``PiecewiseLinearWaveSpeedModel``, etc.)
   * - ``Bathymetry``
     - Layer heights define geometry


Helmholtz Propagator (low-level)
================================

.. list-table::
   :header-rows: 1

   * - Old API
     - New JAX API
   * - ``HelmholtzPadeSolver``
     - ``RationalHelmholtzPropagator``
   * - ``HelmholtzEnvironment``
     - ``AbstractWaveSpeedModel`` + ``AbstractTerrainModel``
   * - ``HelmholtzPropagatorComputationalParams``
     - ``HelmholtzMeshParams2D`` + ``RationalHelmholtzPropagator.create()``


JAX-specific features
=====================

The JAX implementation enables:

* **GPU acceleration**: Computations run on GPU when CUDA-enabled JAX is installed.
* **JIT compilation**: The propagator is JIT-compiled for optimal performance.
* **Automatic differentiation**: Use ``jax.grad`` to compute gradients through the propagation.
* **JAX PyTree compatibility**: All model classes are registered as JAX PyTrees for use with JAX transformations.
* **Composable models**: Wave speed and N-profile models support addition (``+``) and scalar multiplication (``*``).
