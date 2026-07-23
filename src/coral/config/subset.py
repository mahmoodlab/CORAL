"""Ingest subset configuration.

A *subset* is the cohort-level selection applied at ingest: which
channels stay in the analysis set (glob include/exclude, seeding
``keep_for_analysis``) and which images are processed (exact-name
include/exclude). Both live in one YAML supplied via ``coral ingest
--subset``; a copy is saved to ``<job_dir>/subset.yaml`` for provenance.

The same YAML also drives ``coral extract --subset``: there its
``channels`` glob selects the marker subset for one extraction run
(``images`` applies at ingest only).
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class Selection(BaseModel):
    """An include/exclude pair of patterns.

    Attributes:
        include: Patterns to include. Empty means "all".
        exclude: Patterns to exclude. Empty means "exclude none".
            Applied after ``include``.

    Example:
        >>> sel = Selection(include=["CD*", "DAPI"], exclude=["IgG_*"])
        >>> sel.include
        ['CD*', 'DAPI']
    """

    include: list[str] = Field(default_factory=list)
    exclude: list[str] = Field(default_factory=list)


class Subset(BaseModel):
    """Cohort-level channel + image selection for ``coral ingest``.

    The ``channels`` selection uses glob patterns matched against marker
    names (seeding ``keep_for_analysis``); the ``images`` selection uses
    exact entry names (the filename with extension, or a per-channel
    directory name).

    Attributes:
        name: Identifier (e.g. ``"hnscc_immune"``).
        description: Optional human-readable description.
        channels: Marker glob include/exclude. Empty include = all
            markers kept.
        images: Image exact-name include/exclude. Empty include = all
            images processed.

    Example:
        A subset keeping the immune markers and skipping a control slide::

            name: hnscc_immune
            channels:
              include: ["CD*", "DAPI"]
              exclude: ["IgG_*"]
            images:
              exclude: ["control-1.ome.tiff"]
    """

    name: str = "subset"
    description: str | None = None
    channels: Selection = Field(default_factory=Selection)
    images: Selection = Field(default_factory=Selection)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Subset:
        """Load a subset from a YAML file.

        Args:
            path: Path to a YAML file with the subset schema.

        Returns:
            Validated ``Subset`` instance.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
            pydantic.ValidationError: If the YAML doesn't match the
                schema.
        """
        path = Path(path)
        if not path.exists():
            msg = f"Subset YAML not found: {path}"
            raise FileNotFoundError(msg)
        data = yaml.safe_load(path.read_text()) or {}
        return cls.model_validate(data)
