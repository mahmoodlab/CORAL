"""``prepare_cohort_stats`` — the novel-marker prepare pass.

Before a KRONOS2 extract over a cohort with **novel markers** (channels
absent from the model's pretraining vocabulary), CORAL must fill the blank
``mean``/``std`` of ``additional_markers.csv`` from the data. This module is
the orchestration that does it, composing the pieces shipped in Tasks 1-3:

```
pre-flight  classify novel markers across slides; require a CSV row (text
            columns) for each; error early with the marker + path
prepare     per slide: prepare_slide computes (n, μ, s²) over the
            tissue-masked, dtype-scaled [0, 1] pixels; persist the partials
            in the slide store (reuse them when the mask is unchanged)
pool        pool each marker's per-slide partials → one (mean, std)
write       fill the CSV's blank stat cells (user-supplied values untouched)
register    hand the completed CSV to the model
```

Single image vs cohort is one code path — ``n = 1`` pooling reduces to that
slide's own ``(μ₁, s₁²)``. The extract pass itself is unchanged and runs
after this, in the caller (the ``coral extract`` CLI).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from coral.dtypes import scaling_factor
from coral.markers.additional import (
    SlideStat,
    markers_needing_stats,
    missing_marker_rows,
    pool_marker_stats,
    read_additional_markers,
    region_key,
    validate_additional_markers,
    write_marker_stats,
)
from coral.markers.normalize import normalize_key
from coral.utils.errors import CoralError

if TYPE_CHECKING:
    from coral.features.kronos2 import Kronos2Extractor
    from coral.slide.core import CoralSlide


def _slide_novel(
    extractor: Kronos2Extractor,
    slide: CoralSlide,
    channels: Any,  # noqa: ANN401
) -> tuple[list[int], list[str], list[str]]:
    """Resolve a slide's selected channels + the novel ones among them.

    Returns ``(idxs, used, novel)`` — the selected channel indices, their
    names, and the subset the model would z-score with default stats (so
    ``novel ⊆ used``).
    """
    idxs, used = slide._resolve_markers(channels)
    nuc = slide.nuclear_channel
    nuclear_marker = slide.markers[nuc] if nuc is not None else None
    novel = extractor.novel_markers(used, nuclear_marker)
    return idxs, used, novel


def _slide_partials(
    extractor: Kronos2Extractor,
    slide: CoralSlide,
    idxs: list[int],
    used: list[str],
    targets: list[str],
) -> dict[str, SlideStat]:
    """Compute (or reuse) one slide's ``(n, μ, s²)`` for ``targets``.

    Reuses the store's persisted partials when their region key matches the
    current tissue mask + scaling and they cover every target; otherwise
    runs :meth:`prepare_slide` and persists the fresh partials.
    """
    mask = slide._tissue_mask_np()  # raises if tissue not detected
    image = slide.image.isel(c=idxs)
    expected = region_key(mask, scaling_factor(np.dtype(image.dtype)))
    stored = slide.read_novel_marker_stats()
    if stored is not None:
        s_stats, s_key = stored
        if s_key == expected and all(t in s_stats for t in targets):
            return {t: s_stats[t] for t in targets}
    extractor.configure_stats(targets)
    extractor.prepare_slide(image, markers=used, tissue_mask=mask)
    stats = dict(extractor.last_slide_stats)
    slide.write_novel_marker_stats(stats, expected)
    return stats


def prepare_cohort_stats(
    extractor: Kronos2Extractor,
    slides: Sequence[CoralSlide],
    csv_path: str | Path | None = None,
    *,
    channels: Any = None,  # noqa: ANN401 — a Selection or None
) -> Path | None:
    """Fill ``additional_markers.csv`` blanks from cohort pixels + register.

    The full prepare pass (see the module docstring). A no-op (returns
    without reading or registering) when no selected channel is novel — so
    the caller can always invoke it. Pure CORAL: the partials are numpy
    reductions and the frozen model is untouched until
    :meth:`register_additional_markers`, so the extract that follows stays
    bit-exact.

    Args:
        extractor: A built KRONOS2 extractor (``supports_novel_markers``).
        slides: The cohort (one slide is the ``n = 1`` case). Each needs a
            tissue mask — run ``detect_tissue`` first.
        csv_path: The ``additional_markers.csv`` the user passed; its blank
            ``mean``/``std`` cells are filled in place. ``None`` is allowed
            only when nothing is novel — novel markers with no CSV raise.
        channels: Optional channel ``Selection`` (``None`` = every channel).

    Returns:
        The written ``csv_path``, or ``None`` when nothing was novel.

    Raises:
        CoralError: If a novel marker is present with no ``csv_path``, has no
            CSV row, or its row leaves a required text column blank (each
            names the offending marker).

    Example:
        >>> import tempfile, numpy as np, zarr
        >>> from pathlib import Path
        >>> from coral import CoralSlide
        >>> from coral.features.kronos2 import Kronos2Extractor
        >>> class _M:  # empty stats table → nothing is novel
        ...     _marker_stats: dict = {}
        ...     _marker_index: dict = {}
        >>> with tempfile.TemporaryDirectory() as d:
        ...     p = Path(d) / "s.zarr"
        ...     r = zarr.open_group(str(p), mode="w")
        ...     _ = r.create_dataset(
        ...         "0", data=np.ones((1, 4, 4), dtype="uint16")
        ...     )
        ...     r.attrs["channels"] = [
        ...         {"marker": "dapi", "raw": "DAPI", "match": "exact"}
        ...     ]
        ...     r.attrs["mpp"] = 0.5
        ...     ext = Kronos2Extractor(model=_M())
        ...     prepare_cohort_stats(ext, [CoralSlide.open(p)]) is None
        True
    """
    resolved = [(s, *_slide_novel(extractor, s, channels)) for s in slides]

    cohort_novel = list(
        dict.fromkeys(m for _, _, _, novel in resolved for m in novel)
    )
    if not cohort_novel:  # nothing novel — no stats, no register
        return Path(csv_path) if csv_path is not None else None
    if csv_path is None:
        msg = (
            f"novel marker(s) absent from KRONOS2's vocabulary: "
            f"{cohort_novel}. Pass --additional-markers <csv> with a row "
            f"for each so CORAL can compute their (mean, std)."
        )
        raise CoralError(msg)

    csv_path = Path(csv_path)
    df = read_additional_markers(csv_path)
    validate_additional_markers(df, csv_path)
    missing = missing_marker_rows(df, cohort_novel)
    if missing:
        msg = (
            f"{csv_path}: novel marker(s) with no additional_markers.csv "
            f"row: {missing}. Add a row (text columns) for each so KRONOS2 "
            f"can embed the marker name."
        )
        raise CoralError(msg)

    need_keys = {normalize_key(m) for m in markers_needing_stats(df)}
    partials: dict[str, list[SlideStat]] = {}
    for slide, idxs, used, novel in resolved:
        targets = [m for m in novel if normalize_key(m) in need_keys]
        if not targets:
            continue
        for marker, stat in _slide_partials(
            extractor, slide, idxs, used, targets
        ).items():
            partials.setdefault(normalize_key(marker), []).append(stat)

    pooled = {key: pool_marker_stats(v) for key, v in partials.items()}
    write_marker_stats(df, csv_path, pooled)
    extractor.register_additional_markers(csv_path)
    return csv_path
