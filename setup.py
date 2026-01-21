from setuptools import setup, Extension

def build_extensions():
    from Cython.Build import cythonize
    import numpy
    
    np_inc = numpy.get_include()
    
    extensions = [
        Extension(
            "pywaveprop.propagators.contfrac",
            ["pywaveprop/propagators/contfrac.pyx"],
            include_dirs=['.', np_inc],
        ),
        Extension(
            "pywaveprop.propagators._cn_utils",
            ["pywaveprop/propagators/_cn_utils.pyx"],
            include_dirs=['.', np_inc],
        ),
        Extension(
            "pywaveprop.propagators.dispersion_relations",
            ["pywaveprop/propagators/dispersion_relations.pyx"],
            include_dirs=['.', np_inc],
        ),
    ]
    return cythonize(extensions, language_level="3")

setup(ext_modules=build_extensions())
