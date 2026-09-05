"""Batch cell performance — array-in / array-out over the Protos model.

Public surface is deliberately small: :func:`run_batch` plus the two
envelope models. The vendored single-design model lives in
:mod:`cell_performance_batch.cell_performance` and is importable
directly when you want one design without the array wrapper.
"""

from cell_performance_batch.batch import (
    COMPARISON_KEYS,
    MAX_WORKERS_CAP,
    BatchInput,
    DesignSpec,
    run_batch,
)

#: protos-v2 commit the vendored ``cell_performance.py`` was taken from.
#: Bump alongside the file itself — see that module's header.
UPSTREAM_COMMIT = "b5339610e0e44df648aa394e27165e68f5400ee2"

__version__ = "0.1.0"

__all__ = [
    "COMPARISON_KEYS",
    "MAX_WORKERS_CAP",
    "UPSTREAM_COMMIT",
    "BatchInput",
    "DesignSpec",
    "__version__",
    "run_batch",
]
