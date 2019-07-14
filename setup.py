from setuptools import setup
from setuptools.extension import Extension
import numpy

USE_CYTHON = True
ext = '.pyx' if USE_CYTHON else '.c'
extensions = [Extension("example", ["example"+ext])]

extensions = [Extension("propagators.contfrac", ["propagators/contfrac"+ext], include_dirs=['.', numpy.get_include()]),
              Extension("propagators._cn_utils", ["propagators/_cn_utils"+ext], include_dirs=['.', numpy.get_include()])]

if USE_CYTHON:
    from Cython.Build import cythonize
    extensions = cythonize(extensions)

setup(
    name='rwp',
    version='1.1.4',
    url='https://github.com/mikelytaev/wave-propagation',
    license='MIT',
    author='Mikhail Lytaev',
    author_email='mikelytaev@gmail.com',
    description='Tropospheric radiowave propagation modelling',
    setup_requires=[
        'numpy'
    ],
    install_requires=[
        'numpy',
        'scipy',
        'mpmath',
        'matplotlib',
        'cython'
    ],
    ext_modules=extensions,
    zip_safe=False,
    packages=['propagators', 'rwp', 'uwa'],
)
