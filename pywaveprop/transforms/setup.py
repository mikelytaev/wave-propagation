import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="fcc_fourier",
    version="1.0",
    author="Mikhail Lytaev",
    author_email="mikelytaev@gmail.com",
    description="Adaptive Filon–Clenshaw–Curtis Fourier transform",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mikelytaev/wave-propagation/tree/master/transforms",
    packages=['fcc_fourier'],
    classifiers=[
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    install_requires=[
        'numpy',
        'scipy'
    ]
)