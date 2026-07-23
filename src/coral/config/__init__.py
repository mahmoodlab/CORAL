"""Pydantic configuration models for CORAL.

Concepts:

- :class:`PatchConfig` — one patch extraction's parameters
  (size, stride, overlap, target_mpp, mode, tissue threshold).
- :class:`Subset` — a cohort channel + image selection. Its ``channels``
  glob (include/exclude) both seeds ``keep_for_analysis`` at ingest
  (``coral ingest --subset``) and selects a marker subset for one
  extraction run (``coral extract --subset``); ``images`` applies at
  ingest only. :class:`Selection` is one include/exclude glob pair.

Each model has a ``from_yaml(path)`` classmethod that loads + validates
a YAML file.
"""

from coral.config.patch import PatchConfig, PatchMode
from coral.config.subset import Selection, Subset

__all__ = [
    "PatchConfig",
    "PatchMode",
    "Selection",
    "Subset",
]
