"""Compatibility alias for Peakachu 2.2 with current hic-straw wheels.

Peakachu imports the historical module name ``straw`` while hic-straw 1.3.1
exports the same API as ``hicstraw``. Keeping this shim outside the isolated
environment makes the compatibility adjustment explicit and reproducible.
"""

from hicstraw import *  # noqa: F401,F403
