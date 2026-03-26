"""Build script for cylon-armada Python extensions (ContextTable bindings)."""

import os
import sys
import numpy as np
import pyarrow as pa
from Cython.Build import cythonize
from setuptools import Extension, setup

CYLON_PREFIX = os.environ.get('CYLON_PREFIX')
CYLON_ARMADA_PREFIX = os.environ.get('CYLON_ARMADA_PREFIX', CYLON_PREFIX)
CYLON_HOME = os.environ.get('CYLON_HOME', '/cylon')
ARROW_PREFIX = os.environ.get('ARROW_PREFIX', os.environ.get('CONDA_PREFIX'))

if not CYLON_PREFIX:
    print("CYLON_PREFIX must be set", file=sys.stderr)
    sys.exit(1)

if not os.path.isdir(os.path.join(CYLON_HOME, "python", "pycylon")):
    print(f"CYLON_HOME={CYLON_HOME} does not contain python/pycylon. "
          "Set CYLON_HOME to the cylon source root.", file=sys.stderr)
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

# The pycylon .pxd files generate #include "../../../../cpp/src/cylon/..." in C++.
# These relative paths resolve from pycylon's source subdirectories (e.g. pycylon/common/).
# Add those directories so the C compiler can find the headers via -I.
_pycylon_src = os.path.join(CYLON_HOME, "python", "pycylon", "pycylon")
for subdir in ["common", "ctx", "api", "data", "net", "context", "checkpoint"]:
    p = os.path.join(_pycylon_src, subdir)
    if os.path.isdir(p):
        include_dirs.append(p)

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

extra_compile_args = ['-std=c++17', '-DOMPI_SKIP_MPICXX=1', '-D_GLIBCXX_USE_CXX11_ABI=1']

macros = []
if os.environ.get('CYLON_REDIS'):
    macros.append(('BUILD_CYLON_REDIS', '1'))
if os.environ.get('CYLON_FMI'):
    macros.append(('BUILD_CYLON_FMI', '1'))

extensions = [
    Extension(
        "cylon_armada.context_table",
        sources=["cylon_armada/context_table.pyx"],
        include_dirs=include_dirs,
        language='c++',
        extra_compile_args=extra_compile_args,
        libraries=libraries,
        library_dirs=library_dirs,
        define_macros=macros,
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
    packages=["cylon_armada"],
    zip_safe=False,
)