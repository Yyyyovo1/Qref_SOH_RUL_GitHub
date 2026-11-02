from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, BinaryIO


# The HUST files contain only NumPy scalar/array objects and Pandas DataFrames.
# Any other executable global is rejected rather than imported during unpickling.
ALLOWED_PICKLE_GLOBALS: frozenset[tuple[str, str]] = frozenset(
    {
        ("builtins", "slice"),
        ("numpy", "dtype"),
        ("numpy", "ndarray"),
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy.core.multiarray", "scalar"),
        ("numpy._core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "scalar"),
        ("pandas.core.frame", "DataFrame"),
        ("pandas.core.indexes.base", "Index"),
        ("pandas.core.indexes.base", "_new_Index"),
        ("pandas.core.indexes.range", "RangeIndex"),
        ("pandas.core.internals.managers", "BlockManager"),
    }
)


class RestrictedHUSTUnpickler(pickle.Unpickler):
    """Unpickler restricted to the inert NumPy/Pandas objects used by HUST."""

    def find_class(self, module: str, name: str) -> Any:
        if (module, name) not in ALLOWED_PICKLE_GLOBALS:
            raise pickle.UnpicklingError(
                f"Blocked unsafe or unexpected pickle global: {module}.{name}"
            )
        return super().find_class(module, name)


def restricted_load(handle: BinaryIO) -> Any:
    return RestrictedHUSTUnpickler(handle).load()


def load_hust_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return restricted_load(handle)


__all__ = [
    "ALLOWED_PICKLE_GLOBALS",
    "RestrictedHUSTUnpickler",
    "load_hust_pickle",
    "restricted_load",
]
