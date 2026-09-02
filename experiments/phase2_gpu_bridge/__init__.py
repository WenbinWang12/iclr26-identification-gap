"""Pure NumPy public surface for the Phase-2 planted bridge."""

import os


# The protocol requires these controls before NumPy, BLAS, or CUDA is loaded.
for _key, _value in {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "MKL_DYNAMIC": "FALSE",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
}.items():
    os.environ[_key] = _value

from .core_numpy import *
from .core_numpy import __all__ as _core_all

__all__ = list(_core_all)
