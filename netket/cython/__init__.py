import pyximport
import mpi4py
import os

print("Using XLA-MPI interop")
# Get mpi compiler
config = mpi4py.get_config()
# Detect compile flags and MPI header files
cmd_compile = " ".join([config["mpicc"], "--showme:compile"])
out_stream = os.popen(cmd_compile)
out_compile = out_stream.read().strip()

compile_flags, _ = out_compile.communicate()

# Push all include flags in a list
include_dirs = [p.decode()[2:] for p in compile_flags.split()]
include_dirs.append(mpi4py.get_include())
print("End installing")

# mokeypatch
pyximport.install(setup_args={"include_dirs": include_dirs})

from . import mpi_xla_bridge
from jax.lib import xla_client

for name, fn in mpi_xla_bridge.cpu_custom_call_targets.items():
    xla_client.register_cpu_custom_call_target(name, fn)

print("XLA setup")
