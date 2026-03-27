============
Installation
============

Requirements
============

* Python >= 3.10
* JAX >= 0.4
* lineax
* flax
* NumPy
* SciPy
* mpmath
* matplotlib (for visualization)

Install from source
===================

.. code-block:: bash

    git clone https://github.com/mikelytaev/wave-propagation.git
    cd wave-propagation
    python -m venv .venv
    source .venv/bin/activate
    pip install jax jaxlib lineax flax numpy scipy mpmath matplotlib

For GPU support (CUDA)::

    pip install jax[cuda12] lineax flax numpy scipy mpmath matplotlib

Then add the project to your Python path::

    export PYTHONPATH=/path/to/wave-propagation:$PYTHONPATH

Quick start
===========

Tropospheric radio wave propagation::

    from pywaveprop.experimental.rwp_jax import (
        RWPGaussSourceModel, RWPComputationalParams,
        TroposphereModel, EvaporationDuctModel, rwp_forward_task,
    )

    src = RWPGaussSourceModel(freq_hz=3e9, height_m=10, beam_width_deg=5)
    env = TroposphereModel(N_profile=EvaporationDuctModel(height_m=20))
    params = RWPComputationalParams(max_range_m=50000, dx_m=100, dz_m=0.5)
    field = rwp_forward_task(src, env, params)

Underwater acoustics::

    from pywaveprop.experimental.uwa_jax import (
        UWAGaussSourceModel, UnderwaterLayerModel,
        UnderwaterEnvironmentModel, uwa_forward_task,
    )
    from pywaveprop.experimental.uwa_utils import UWAComputationalParams
    from pywaveprop.experimental.helmholtz_jax import ConstWaveSpeedModel

    src = UWAGaussSourceModel(freq_hz=500, depth_m=50, beam_width_deg=30)
    env = UnderwaterEnvironmentModel(layers=[
        UnderwaterLayerModel(height_m=200, sound_speed_profile_m_s=ConstWaveSpeedModel(c0=1500)),
        UnderwaterLayerModel(height_m=100, sound_speed_profile_m_s=ConstWaveSpeedModel(c0=1700), density=1.8),
    ])
    params = UWAComputationalParams(max_range_m=10000, dx_m=50, dz_m=1)
    field = uwa_forward_task(src, env, params)
