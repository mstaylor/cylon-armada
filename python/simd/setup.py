"""Build configuration for Cython batch similarity search.

Follows the Cylon pycylon build pattern (setuptools + Cython).
See cylon/python/pycylon/setup.py for reference.

Requires CYLON_PREFIX to point to the Cylon install directory
so that Cylon's C++ SIMD headers and library are available.

Build:
    cd target/shared/scripts/simd
    CYLON_PREFIX=/path/to/cylon/install python setup.py build_ext --inplace
"""

import os
import sys

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

# --- Cylon C++ integration (required) ---
cylon_prefix = os.environ.get("CYLON_PREFIX", "")
if not cylon_prefix or not os.path.isdir(cylon_prefix):
    print(
        "ERROR: CYLON_PREFIX must be set to the Cylon install directory.\n"
        "  CYLON_PREFIX=/path/to/cylon/install python setup.py build_ext --inplace",
        file=sys.stderr,
    )
    sys.exit(1)

cylon_include = os.path.join(cylon_prefix, "include")
cylon_lib = os.path.join(cylon_prefix, "lib")

include_dirs = [
    np.get_include(),
    cylon_include,
]

library_dirs = [cylon_lib]
libraries = ["cylon"]

extra_compile_args = ["-std=c++17", "-O3"]
extra_link_args = [f"-Wl,-rpath,{cylon_lib}"]

compiler_directives = {
    "profile": False,
    "language_level": 3,
    "embedsignature": True,
}

extensions = [
    Extension(
        "*",
        sources=["*.pyx"],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        language="c++",
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        define_macros=[("CYLON_AVAILABLE", "1")],
    ),
]

setup(
    name="cylon-armada-simd",
    ext_modules=cythonize(
        extensions,
        nthreads=1,
        compiler_directives=compiler_directives,
        force=True,
    ),
)