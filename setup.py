from setuptools import setup
from setuptools.extension import Extension
import os


def mpi_includes():
    import mpi4py

    config = mpi4py.get_config()
    cmd_compile = " ".join([config["mpicc"], "--showme:compile"])
    out_stream = os.popen(cmd_compile)
    compile_flags = out_stream.read().strip()

    include_dirs = [p[2:] for p in compile_flags.split()]
    include_dirs.append(mpi4py.get_include())
    return include_dirs


setup(
    name="netket",
    version="3.0",
    author="Giuseppe Carleo et al.",
    url="http://github.com/netket/netket",
    author_email="netket@netket.org",
    license="Apache 2.0",
    packages=[
        "netket",
        "netket.cython",
        "netket.graph",
        "netket.hilbert",
        "netket.logging",
        "netket.machine",
        "netket.machine.density_matrix",
        "netket.sampler",
        "netket.sampler.numpy",
        "netket.sampler.jax",
        "netket.stats",
        "netket.operator",
        "netket.optimizer",
        "netket.optimizer.numpy",
        "netket.optimizer.jax",
    ],
    ext_modules=[
        Extension(
            name="netket.cython.mpi_xla_bridge",
            sources=["netket/cython/mpi_xla_bridge.pyx"],
            include_dirs=mpi_includes(),
        ),
    ],
    long_description="""NetKet is an open - source project delivering cutting - edge
         methods for the study of many - body quantum systems with artificial
         neural networks and machine learning techniques.""",
    setup_requires=["setuptools>=18.0", "cython>=0.21", "mpi4py>=3.0.1",],
    install_requires=[
        "numpy>=1.16",
        "scipy>=1.2.1",
        "mpi4py>=3.0.1",
        "tqdm>=4.42.1",
        "numba>=0.48.0",
        "networkx>=2.4",
        "cython>=0.21",
    ],
    python_requires=">=3.6",
    extras_require={"dev": ["pytest", "python-igraph", "pre-commit", "black"],},
)
