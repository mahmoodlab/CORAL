"""Cell phenotype label table — per-cell ``(cell_id, label_set, label)``.

Optional ground-truth phenotypes for user-imported cells.
``label_set`` names each annotation version (e.g. ``c14`` / ``c15``) so
multiple coexist; ``cell_id`` joins onto the stored instance mask.
Written as **CSV** (what CORAL reads and users open). Labels are optional —
CORAL works without them; they exist for cell-phenotyping FM benchmarking.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

__all__ = ["write_cell_labels"]


def write_cell_labels(
    df: pd.DataFrame, csv_path: str | Path, label_set: str
) -> None:
    """Append a named label set to ``cells/cell_labels.csv``.

    Output schema is ``(cell_id, label_set, label)``. An existing set
    with the same ``label_set`` is **replaced**; other sets are
    preserved, so multiple annotation versions coexist.

    Args:
        df: A table with at least ``cell_id`` + ``label`` columns.
        csv_path: Destination ``cell_labels.csv``.
        label_set: Name for this annotation version (e.g. ``"c14"``).

    Example:
        >>> import tempfile, pandas as pd
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as d:
        ...     p = Path(d) / "labels.csv"
        ...     write_cell_labels(
        ...         pd.DataFrame({"cell_id": [1, 2], "label": ["t", "b"]}),
        ...         p,
        ...         "c14",
        ...     )
        ...     list(pd.read_csv(p).columns)
        ['cell_id', 'label_set', 'label']
    """
    out = df[["cell_id", "label"]].copy()
    out.insert(1, "label_set", str(label_set))
    path = Path(csv_path)
    if path.exists():
        existing = pd.read_csv(path)
        existing = existing[existing["label_set"] != str(label_set)]
        out = pd.concat([existing, out], ignore_index=True)
    out.to_csv(path, index=False)
