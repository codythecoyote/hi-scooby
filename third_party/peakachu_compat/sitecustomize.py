"""Compatibility aliases required by the unmodified Peakachu 2.2 release."""

import numpy as np

# Peakachu 2.2's clustering code still uses the alias removed in NumPy 1.24.
if not hasattr(np, "int"):
    np.int = int
