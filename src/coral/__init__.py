"""CORAL — a spatial proteomics toolkit.

A spatial proteomics toolkit built around the KRONOS foundation
model. CORAL exposes clean, atomic building blocks for spatial
proteomics data ingestion, segmentation, feature extraction, and
downstream analysis.

See https://github.com/mahmoodlab/ESB-internal for the project home.
"""

from importlib.metadata import PackageNotFoundError, version

from coral.processor import CohortResult, CoralProcessor
from coral.slide import CoralSlide

try:
    __version__ = version("coral")
except PackageNotFoundError:  # a source checkout, not an installed dist
    __version__ = "0.1.0"

__all__ = ["CohortResult", "CoralProcessor", "CoralSlide", "__version__"]
