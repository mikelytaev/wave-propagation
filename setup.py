from setuptools import setup
from setuptools.extension import Extension
import numpy

USE_CYTHON = True
ext = '.pyx' if USE_CYTHON else '.c'
extensions = [Extension("example", ["example"+ext])]

extensions = [Extension("rwp.contfrac", ["rwp/contfrac"+ext], include_dirs=['.', numpy.get_include()]),
              Extension("rwp._cn_utils", ["rwp/_cn_utils"+ext], include_dirs=['.', numpy.get_include()])]

if USE_CYTHON:
    from Cython.Build import cythonize
    extensions = cythonize(extensions)

setup(
    name='rwp',
    version='1.0',
    url='https://github.com/mikelytaev/wave-propagation',
    license='MIT',
    author='Mikhail Lytaev',
    author_email='mikelytaev@gmail.com',
    description='Tropospheric radiowave propagation modelling',
    install_requires=[
        'numpy',
        'scipy',
        'mpmath',
        'matplotlib',
        'cython',
        'fcc_fourier'
    ],
    ext_modules=extensions,
    zip_safe=False,
    packages=['rwp'],
)
