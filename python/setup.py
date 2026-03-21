"""Build script for cylon-armada Python extensions (ContextTable bindings)."""

import os
import sys
import numpy as np
import pyarrow as pa
from Cython.Build import cythonize
from setuptools import Extension, setup

CYLON_PREFIX = os.environ.get('CYLON_PREFIX')
CYLON_ARMADA_PREFIX = os.environ.get('CYLON_ARMADA_PREFIX', CYLON_PREFIX)
ARROW_PREFIX = os.environ.get('ARROW_PREFIX', os.environ.get('CONDA_PREFIX'))

if not CYLON_PREFIX:
    print("CYLON_PREFIX must be set", file=sys.stderr)
    sys.exit(1)

pyarrow_location = os.path.dirname(pa.__file__)
pyarrow_include = os.path.join(pyarrow_location, "include")
arrow_include = os.path.join(ARROW_PREFIX, "include") if ARROW_PREFIX else pyarrow_include

include_dirs = [
    os.path.join(CYLON_PREFIX, "include"),
    os.path.join(CYLON_ARMADA_PREFIX, "include"),
    arrow_include,
    pyarrow_include,
    np.get_include(),
]

library_dirs = [
    os.path.join(CYLON_PREFIX, "lib"),
    os.path.join(CYLON_ARMADA_PREFIX, "lib"),
    os.path.join(ARROW_PREFIX, "lib") if ARROW_PREFIX else "",
]

libraries = ["cylon", "cylon_armada", "arrow"]

# MPI include/lib
try:
    import subprocess
    mpi_show = subprocess.check_output(["mpicc", "-show"]).decode().strip().split()
    for s in mpi_show:
        if s.startswith('-I'):
            include_dirs.append(s[2:])
        elif s.startswith('-L'):
            library_dirs.append(s[2:])
        elif s.startswith('-l'):
            libraries.append(s[2:])
except Exception:
    pass

extra_compile_args = ['-std=c++17', '-DOMPI_SKIP_MPICXX=1']

extensions = [
    Extension(
        "context.context_table",
        sources=["context/context_table.pyx"],
        include_dirs=include_dirs,
        language='c++',
        extra_compile_args=extra_compile_args,
        libraries=libraries,
        library_dirs=library_dirs,
    )
]

compile_time_env = {
    'CYTHON_REDIS': bool(os.environ.get('CYLON_REDIS')),
    'CYTHON_FMI': bool(os.environ.get('CYLON_FMI')),
    'CYTHON_GLOO': False,
    'CYTHON_UCX': False,
    'CYTHON_UCC': False,
    'CYTHON_LIBFABRIC': False,
}

setup(
    name="cylon-armada-context",
    version="0.1.0",
    ext_modules=cythonize(
        extensions,
        compiler_directives={"language_level": 3},
        compile_time_env=compile_time_env,
    ),
    packages=["context"],
    zip_safe=False,
)