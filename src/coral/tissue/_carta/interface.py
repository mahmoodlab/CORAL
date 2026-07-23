"""Segmenter ABC (inference). Vendored + trimmed from CARTA segmenter/interface.py.

Only the ``Segmenter`` base class is kept; CARTA's ``build_segmenter`` /
``postprocess_mask`` / ``segment_tissue`` / ``dearray`` helpers (which pull in
otsu / dearray / postprocess config) are not part of the inference embed.
"""
from __future__ import annotations

import abc

import numpy as np


class Segmenter(abc.ABC):
    @abc.abstractmethod
    def segment(self, image: np.ndarray, source: str = None) -> np.ndarray:
        """(H, W) uint16/float DAPI -> (H, W) bool tissue mask."""
        ...
