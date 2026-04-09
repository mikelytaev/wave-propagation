import os
from setuptools import setup
from setuptools.extension import Extension

# Cython extensions are optional (legacy implementation)
USE_CYTHON = os.environ.get('PYWAVEPROP_USE_CYTHON', '0') == '1'

extensions = []
if USE_CYTHON:
    import numpy
    ext = '.pyx'
    extensions = [
        Extension("propagators.contfrac", ["propagators/contfrac"+ext], include_dirs=['.', numpy.get_include()]),
        Extension("propagators._cn_utils", ["propagators/_cn_utils"+ext], include_dirs=['.', numpy.get_include()]),
        Extension("propagators.dispersion_relations", ["propagators/dispersion_relations"+ext], include_dirs=['.'])
    ]
    from Cython.Build import cythonize
    extensions = cythonize(extensions, language_level="3")

setup(
    name='pywaveprop',
    version='2.0.0',
    url='https://github.com/mikelytaev/wave-propagation',
    license='MIT',
    author='Mikhail Lytaev',
    author_email='mikelytaev@gmail.com',
    description='Wave propagation framework (JAX-based)',
    long_description=open('README.md').read() if os.path.exists('README.md') else '',
    long_description_content_type='text/markdown',
    python_requires='>=3.10',
    setup_requires=[
        'numpy',
    ],
    install_requires=[
        'numpy',
        'scipy',
        'mpmath',
        'matplotlib',
        'jax',
        'jaxlib',
        'lineax',
        'flax',
    ],
    extras_require={
        'legacy': ['cython'],
        'gpu': ['jax[cuda12]'],
    },
    ext_modules=extensions,
    zip_safe=False,
    packages=[
        'pywaveprop',
        'pywaveprop.propagators',
        'pywaveprop.rwp',
        'pywaveprop.uwa',
        'pywaveprop.uwa._optimization',
        'pywaveprop.transforms',
        'pywaveprop.transforms.fcc_fourier',
        'pywaveprop.utils',
        'pywaveprop.experimental',
    ],
    package_data={
        '': ['*.pyx']
    },
    keywords=[
        "wave propagation", "parabolic equation", "troposphere",
        "underwater acoustics", "diffraction", "refraction", "jax",
    ],
)
