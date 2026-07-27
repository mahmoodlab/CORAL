"""LoRA fine-tuning plumbing for the KRONOS2 cell-phenotyping tutorial.

The tutorial's subject is the *protocol*: reuse Tutorial 5's spatial
folds, rebuild the exact patch stream the frozen extractor saw, inject
LoRA adapters into the last few transformer blocks, train a linear head
on the CLS token, and compare against a frozen linear probe on identical
rows. The mechanics of running that protocol — a fold-aware
``Dataset``, a class-balanced sampler, the training loop, the metric
set — are ordinary PyTorch and get in the way of reading it, so they
live here instead of in the notebook, and in a training script
(``tutorials/finetune_cells.py``) so the hours-long fit runs detached
rather than in a notebook kernel.

Requires the ``kronos2`` extra (torch + transformers) and scikit-learn::

    uv sync --extra kronos2
    pip install scikit-learn
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
import zarr
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm.auto import tqdm

from coral.features.dataset import CoralDataset
from coral.features.kronos2 import Kronos2Extractor
from utils.lora import (
    LoRALinear,
    count_trainable_parameters,
    freeze_all,
    inject_lora_last_blocks,
    trainable_parameters,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from coral import CoralSlide
    from coral.config import PatchConfig
    from coral.config.subset import Selection

    #: A ``DataLoader`` collate: raw ``(patch, label)`` items -> tensors.
    CollateFn = Callable[
        [Sequence[tuple[np.ndarray, int]]],
        tuple[torch.Tensor, torch.Tensor],
    ]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FinetuneConfig:
    """Hyperparameters for one LoRA fine-tuning run.

    The defaults reproduce the reference KRONOS2 cell-phenotyping
    configuration, which is also what the notebook claims parity with,
    so a reader never has to open this file to learn what a run did.
    ``max_cells_per_class`` / ``max_valid_per_class`` of ``None`` use the
    fold CSVs exactly as Tutorial 5 wrote them (2000 train cells per
    class, the full validation split), which is what keeps the frozen-probe
    comparison controlled.

    Args:
        lora_rank: LoRA rank ``r`` (inner dimension of the update).
        lora_alpha: LoRA scaling numerator; effective scale is
            ``alpha / rank``.
        lora_dropout: Dropout applied to each adapter's input.
        num_lora_blocks: How many of the final transformer blocks to
            adapt.
        lr: AdamW learning rate.
        weight_decay: AdamW weight decay.
        epochs: Maximum epochs; early stopping usually ends sooner.
        batch_size: Patches per step (~13 GB VRAM at 128).
        patience: Epochs without validation improvement before stopping.
        min_delta: Minimum validation-loss improvement that counts.
        num_workers: ``DataLoader`` worker processes. The workload is
            data-loading bound, so workers are the main speed lever;
            gains plateau around 8.
        balanced_sampler: Draw each class with equal probability per
            batch (the classes are severely imbalanced).
        max_cells_per_class: Cap on training cells per class, or ``None``.
        max_valid_per_class: Cap on validation cells per class, or
            ``None``.
        seed: Seed for the per-class subsample and the run.

    Example:
        The defaults are the tutorial's, so ``FinetuneConfig()`` is the
        configuration the notebook documents; the training script
        overrides individual fields from its command-line flags.
    """

    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0
    num_lora_blocks: int = 6
    lr: float = 6e-4
    weight_decay: float = 0.01
    epochs: int = 20
    batch_size: int = 128
    patience: int = 5
    min_delta: float = 0.0
    num_workers: int = 8
    balanced_sampler: bool = True
    max_cells_per_class: int | None = None
    max_valid_per_class: int | None = None
    seed: int = 42


def cap_per_class(
    rows: np.ndarray,
    codes: np.ndarray,
    max_per_class: int | None,
    *,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Subsample to at most ``max_per_class`` rows of each class.

    Caps the training budget so an abundant class cannot dominate the
    fit by sheer volume. Classes with fewer rows than the cap are kept
    whole, and the returned rows stay in ascending order.

    Args:
        rows: Row positions into the patch set.
        codes: Integer class code for each entry of ``rows``.
        max_per_class: Maximum rows to keep per class, or ``None`` to
            keep all of them.
        seed: Seed for the subsample.

    Returns:
        The retained ``(rows, codes)``, aligned and row-sorted.

    Example:
        >>> import numpy as np
        >>> rows = np.arange(10)
        >>> codes = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
        >>> r, c = cap_per_class(rows, codes, 2, seed=0)
        >>> np.bincount(c).tolist()
        [2, 2]
        >>> len(r)
        4
    """
    if max_per_class is None:
        return rows, codes
    rng = np.random.default_rng(seed)
    keep = np.concatenate(
        [
            rng.permutation(np.flatnonzero(codes == c))[:max_per_class]
            for c in np.unique(codes)
        ]
    )
    keep.sort()
    return rows[keep], codes[keep]


@dataclass
class FoldPatches:
    """Turn a fold CSV into a labeled, reproducible cell-patch stream.

    Bundles everything needed to map ``(cell_id, label)`` rows to the
    exact patches the frozen extractor saw: the patch-set coords and
    cell ids, the marker order and channel indices read off the stored
    feature array, the slide's nuclear marker, and the global class
    order shared across every fold. Build one with :meth:`from_slide`.

    The stored feature array is the source of truth for the panel: its
    ``markers_used`` attribute fixes the order the model receives marker
    names in, and its ``patch_slug`` names the patch set the rows index
    into — so nothing is re-derived and nothing can drift.

    Attributes:
        slide_path: Path to the slide's zarr store.
        coords: ``(n, 2)`` top-left coord of every patch in the set.
        cell_ids: ``(n,)`` cell id each patch is centred on.
        channel_idxs: Slide channel indices for ``markers``, in order.
        markers: Ordered marker names the encoder saw.
        nuclear: The slide's nuclear marker (KRONOS2's DAPI special
            case), or ``None``.
        patch_size: Patch side in level-0 pixels.
        class_names: Global class order, fixed across folds.
        frozen_features: ``(n, d)`` stored KRONOS2 features, for the
            frozen-probe baseline.
        folds_dir: Directory holding the fold CSVs.
    """

    slide_path: Path
    coords: np.ndarray
    cell_ids: np.ndarray
    channel_idxs: list[int]
    markers: list[str]
    nuclear: str | None
    patch_size: int
    class_names: list[str]
    frozen_features: np.ndarray
    folds_dir: Path

    @classmethod
    def from_slide(
        cls,
        slide: CoralSlide,
        cell_cfg: PatchConfig,
        panel: Selection,
        folds_dir: Path,
    ) -> FoldPatches:
        """Read the patch set, panel, and folds off ``slide`` and disk.

        Args:
            slide: The opened slide (Tutorial 1's ingest output).
            cell_cfg: The cell-centred patch config Tutorial 3 used.
            panel: The phenotypic-marker selection.
            folds_dir: Directory of Tutorial 5's fold CSVs.

        Returns:
            A ready-to-use :class:`FoldPatches`.

        Example:
            ``FoldPatches.from_slide(slide, cell_cfg, panel, folds_dir)``
            reads the ordered marker list and patch slug from the stored
            features, opens the patch-set coords/cell-ids, and collects
            the global class order from every fold CSV.
        """
        emb = slide.features("KRONOS2", cell_cfg, channels=panel)
        markers = list(emb.attrs["markers_used"])
        patch_slug = emb.attrs["patch_slug"]
        slide_markers = list(slide.markers)
        channel_idxs = [slide_markers.index(m) for m in markers]
        nuclear = (
            slide_markers[slide.nuclear_channel]
            if slide.nuclear_channel is not None
            else None
        )

        patch_dir = slide.path / "patches" / patch_slug
        coords = np.asarray(
            zarr.open_array(str(patch_dir / "coords"), mode="r")
        )
        cell_ids = np.asarray(
            zarr.open_array(str(patch_dir / "cell_ids"), mode="r")
        )

        fold_frames = [pd.read_csv(p) for p in sorted(folds_dir.glob("*.csv"))]
        class_names = sorted(pd.concat(fold_frames)["label"].unique())
        frozen = np.asarray(emb["features"].values, dtype=np.float32)

        return cls(
            slide_path=slide.path,
            coords=coords,
            cell_ids=cell_ids,
            channel_idxs=channel_idxs,
            markers=markers,
            nuclear=nuclear,
            patch_size=int(cell_cfg.patch_size),
            class_names=class_names,
            frozen_features=frozen,
            folds_dir=folds_dir,
        )

    @property
    def labels(self) -> np.ndarray:
        """Integer label set on the global class order."""
        return np.arange(len(self.class_names))

    def load_split(self, filename: str) -> tuple[np.ndarray, np.ndarray]:
        """Read a fold CSV into ``(patch-set rows, label strings)``.

        Args:
            filename: A fold CSV name inside ``folds_dir``, carrying
                ``cell_id`` and ``label`` columns.

        Returns:
            ``(rows, labels)`` — integer row positions into the patch
            set and the matching label strings.

        Raises:
            RuntimeError: If any fold ``cell_id`` is absent from the
                patch set (Tutorials 3 and 5 were run against different
                slide stores).

        Example:
            ``rows, labels = fold_patches.load_split("test_fold1.csv")``
            gives the row positions the dataset and the frozen probe both
            index, so the two models score identical cells.
        """
        pos = pd.Series(np.arange(len(self.cell_ids)), index=self.cell_ids)
        df = pd.read_csv(self.folds_dir / filename)
        rows = pos.reindex(df["cell_id"].to_numpy())
        if rows.isna().any():
            raise RuntimeError(
                f"{filename}: {int(rows.isna().sum())} cell_ids are absent "
                "from the patch set. Were Tutorial 3 and Tutorial 5 run "
                "against the same slide store?"
            )
        return rows.astype(int).to_numpy(), df["label"].to_numpy()

    def codes(self, labels: np.ndarray) -> np.ndarray:
        """Map label strings to integer codes on the global class order.

        Args:
            labels: Label strings, e.g. from :meth:`load_split`.

        Returns:
            ``int64`` class codes aligned to ``class_names``.

        Example:
            ``fold_patches.codes(labels)`` yields the same code for a
            class in every fold, so probability columns line up across
            folds and models.
        """
        return pd.Categorical(
            labels, categories=self.class_names
        ).codes.astype(np.int64)

    def reader(self, rows: np.ndarray) -> CoralDataset:
        """A :class:`CoralDataset` over the given patch-set rows.

        Each box is scaled to float32 ``[0, 1]`` by the image dtype and
        isolated to its own cell via the instance mask — the same reads
        the frozen extractor made.

        Args:
            rows: Row positions into the patch set.

        Returns:
            A lazy, worker-safe patch reader yielding ``(patch, coord)``.

        Example:
            ``fold_patches.reader(rows)[0]`` returns the first patch as a
            ``(len(markers), size, size)`` float32 array.
        """
        return CoralDataset(
            self.slide_path,
            self.coords[rows],
            self.patch_size,
            self.channel_idxs,
            scale=True,
            cell_ids=self.cell_ids[rows],
        )


class CellPatchDataset(Dataset):
    """Cell patches + integer labels for one fold split.

    The coral-native equivalent of the reference
    ``finetune_probe.CellPatchDataset``: it resolves a fold CSV to patch
    rows, wraps :meth:`FoldPatches.reader` over them, and pairs each
    patch with its class code. ``__getitem__`` returns the cell-isolated,
    dtype-scaled patch; the per-marker z-score is applied later, in the
    collate function, so it batches once and parallelizes with the reads.

    Args:
        fold_patches: The shared patch/label context.
        filename: A fold CSV name inside ``fold_patches.folds_dir``.
        max_per_class: Cap on rows per class, or ``None`` for all.
        seed: Seed for the per-class subsample.

    Example:
        ``CellPatchDataset(fold_patches, "train_2000_fold1.csv")`` yields
        ``(patch, label)`` pairs ready for a ``DataLoader``; passing
        ``max_per_class=500`` trims each class for a faster pass.
    """

    def __init__(
        self,
        fold_patches: FoldPatches,
        filename: str,
        *,
        max_per_class: int | None = None,
        seed: int = 42,
    ) -> None:
        """Resolve the split, cap it, and build the patch reader."""
        rows, labels = fold_patches.load_split(filename)
        codes = fold_patches.codes(labels)
        rows, codes = cap_per_class(rows, codes, max_per_class, seed=seed)
        self.reader = fold_patches.reader(rows)
        self.labels = codes
        self.num_classes = len(fold_patches.class_names)

    def __len__(self) -> int:
        """Number of cells in the split."""
        return len(self.labels)

    def __getitem__(self, i: int) -> tuple[np.ndarray, int]:
        """The ``i``-th ``(patch, label)`` pair."""
        patch, _ = self.reader[i]
        return patch, int(self.labels[i])


def make_collate(
    model: nn.Module,
    markers: list[str],
    nuclear: str | None,
) -> CollateFn:
    """Build a collate that z-scores a batch against the marker stats.

    The returned function stacks the raw patches, applies the model's
    per-marker normalization once for the whole batch, and returns
    ``(patches, labels)`` tensors. It runs inside the ``DataLoader``
    worker, so the normalization parallelizes with the reads.

    Args:
        model: The live KRONOS2 module (its ``preprocess`` holds the
            per-marker z-score statistics).
        markers: Ordered marker names, matching the patch channels.
        nuclear: The nuclear marker for KRONOS2's DAPI special case.

    Returns:
        A ``collate_fn`` for :class:`torch.utils.data.DataLoader`.

    Example:
        ``DataLoader(ds, collate_fn=make_collate(model, markers, nuclear))``
        yields normalized ``float32`` batches the backbone can consume
        directly.
    """

    def collate(
        batch: Sequence[tuple[np.ndarray, int]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        patches = np.stack([b[0] for b in batch])
        normed = model.preprocess(patches, markers, preferred_dapi=nuclear)
        return (
            torch.from_numpy(np.ascontiguousarray(normed)),
            torch.tensor([b[1] for b in batch], dtype=torch.long),
        )

    return collate


def make_loader(
    dataset: CellPatchDataset,
    *,
    train: bool,
    config: FinetuneConfig,
    collate: CollateFn,
) -> DataLoader:
    """A balanced-sampling train loader, or a sequential eval loader.

    Args:
        dataset: The split to load.
        train: Whether this is the training loader (enables the
            class-balanced sampler / shuffling).
        config: Supplies ``batch_size``, ``num_workers``, and whether the
            balanced sampler is used.
        collate: The collate function from :func:`make_collate`.

    Returns:
        A configured :class:`torch.utils.data.DataLoader`.

    Example:
        The training loader draws each class with roughly equal
        probability so a 100:1 imbalance does not swamp the rare types;
        the eval loader reads the split once, in order.
    """
    sampler = None
    if train and config.balanced_sampler:
        counts = np.bincount(dataset.labels, minlength=dataset.num_classes)
        weights = 1.0 / counts[dataset.labels]
        sampler = WeightedRandomSampler(
            torch.from_numpy(weights),
            num_samples=len(weights),
            replacement=True,
        )
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        shuffle=train and sampler is None,
        num_workers=config.num_workers,
        collate_fn=collate,
        pin_memory=True,
    )


def build_classifier(
    device: str,
    num_classes: int,
    config: FinetuneConfig,
) -> tuple[nn.Module, nn.Module, list[str], int]:
    """Frozen KRONOS2 + LoRA adapters + a linear head on the CLS token.

    Freezes the backbone first, then injects LoRA into the last
    ``num_lora_blocks`` blocks (order matters — freezing after injection
    would freeze the adapters too), and adds a linear classification head.

    Args:
        device: Torch device string.
        num_classes: Number of output classes (the head width).
        config: The LoRA structure to inject.

    Returns:
        ``(backbone, head, wrapped, n_trainable)`` — the adapted
        backbone, the head, the names of the wrapped layers, and the
        trainable-parameter count.

    Example:
        On ViT-Base, rank 8 over the last 6 blocks wraps 24 layers and
        costs ~602k trainable parameters — about 0.7% of the backbone.
    """
    # `_model` is private on purpose: CORAL's public feature API stops at
    # "give me the vectors", and reaching the live module for
    # gradient-requiring work is outside that contract. Fine-tuning needs
    # the module itself, so we reach through. If CORAL ever grows a public
    # accessor, this is the line to change.
    backbone = Kronos2Extractor.from_pretrained(device=device)._model.backbone
    freeze_all(backbone)
    wrapped = inject_lora_last_blocks(
        backbone,
        num_blocks=config.num_lora_blocks,
        rank=config.lora_rank,
        alpha=config.lora_alpha,
        dropout=config.lora_dropout,
    )
    head = nn.Linear(backbone.embed_dim, num_classes).to(device)

    n_trainable = count_trainable_parameters(backbone) + (
        count_trainable_parameters(head)
    )
    return backbone, head, wrapped, n_trainable


def forward_cls(
    backbone: nn.Module,
    head: nn.Module,
    x: torch.Tensor,
    markers: list[str],
) -> torch.Tensor:
    """CLS logits, with gradients.

    Calls the backbone's ``forward_features`` rather than its public
    ``forward``: the latter is decorated ``@torch.inference_mode()`` and
    cannot backpropagate. ``marker_names`` must be per-sample here — the
    public forward expands a single list for you, this path does not.

    Args:
        backbone: The LoRA-adapted backbone.
        head: The classification head.
        x: A normalized ``(batch, marker, y, x)`` patch tensor.
        markers: Ordered marker names for the batch.

    Returns:
        ``(batch, num_classes)`` logits.

    Example:
        ``forward_cls(backbone, head, x, markers)`` is the one forward
        pass shared by training and evaluation.
    """
    # `marker_names` must be per-sample: KRONOS2 allows a different panel
    # per sample, so the backbone wants one marker list per batch element
    # and asserts `len(marker_names) == B`. The public `forward` expands a
    # single flat list for you; `forward_features` does not, so we expand
    # it here (B references to one list, not B copies). Passing the flat
    # list would make the model misread the panel.
    out = backbone.forward_features(
        x, masks=None, marker_names=[markers] * len(x)
    )
    return head(out["x_norm_clstoken"])


def set_training(backbone: nn.Module, head: nn.Module, mode: bool) -> None:
    """Train mode for the head + adapters only; frozen blocks stay eval.

    The frozen blocks contain stochastic depth and dropout; putting them
    in train mode would inject noise into weights that are not learning.
    Parameters receive gradients by ``requires_grad``, not by module
    mode, so keeping them in ``eval`` costs nothing.

    Args:
        backbone: The adapted backbone.
        head: The classification head.
        mode: ``True`` for train mode, ``False`` for eval.

    Example:
        ``set_training(backbone, head, True)`` switches only the head and
        the LoRA adapters into train mode at the start of an epoch.
    """
    backbone.eval()
    head.train(mode)
    for module in backbone.modules():
        if isinstance(module, LoRALinear):
            module.train(mode)


@torch.no_grad()
def evaluate_loader(
    backbone: nn.Module,
    head: nn.Module,
    loader: DataLoader,
    device: str,
    markers: list[str],
    criterion: nn.Module | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Mean loss, per-cell probabilities, and truths over ``loader``.

    Args:
        backbone: The adapted backbone.
        head: The classification head.
        loader: A sequential eval loader.
        device: Torch device string.
        markers: Ordered marker names.
        criterion: Optional loss; when given, its mean over batches is
            returned as the first element.

    Returns:
        ``(mean_loss, probs, truths)`` — the mean loss (0.0 if no
        criterion), softmax probabilities ``(n, num_classes)``, and the
        integer truths ``(n,)``.

    Example:
        ``_, probs, truth = evaluate_loader(backbone, head, test_loader,
        device, markers)`` gives the held-out predictions to score.
    """
    set_training(backbone, head, False)
    total_loss, n_batches, probs, truths = 0.0, 0, [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = forward_cls(backbone, head, x, markers)
        if criterion is not None:
            total_loss += criterion(logits, y.to(device)).item()
            n_batches += 1
        probs.append(torch.softmax(logits, dim=1).float().cpu().numpy())
        truths.append(y.numpy())
    return (
        total_loss / max(n_batches, 1),
        np.concatenate(probs),
        np.concatenate(truths),
    )


def train_fold(
    fold_patches: FoldPatches,
    model: nn.Module,
    config: FinetuneConfig,
    device: str,
    fold: int,
    out_dir: Path,
) -> tuple[nn.Module, nn.Module, pd.DataFrame]:
    """LoRA fine-tune one fold; save and return the trained model.

    Trains the head + adapters with a class-balanced sampler, a cosine
    schedule, and early stopping on the held-out validation split. The
    best-validation state — not the last epoch's — is restored and
    saved; only the trainable tensors (LoRA factors + head, ~2.4 MB) are
    written, since the frozen weights are reproducible from the Hub.

    Args:
        fold_patches: The shared patch/label context.
        model: The live KRONOS2 module (for the collate's normalization).
        config: The run's hyperparameters.
        device: Torch device string.
        fold: Which spatial fold to train (1..4).
        out_dir: Directory to write ``lora_fold{fold}.pt`` into.

    Returns:
        ``(backbone, head, history)`` — the trained model and a per-epoch
        history frame.

    Example:
        ``train_fold(fold_patches, model, config, device, 1, out_dir)``
        trains fold 1 and writes its adapter checkpoint; the returned
        model is ready to score on the held-out quadrant.
    """
    markers, nuclear = fold_patches.markers, fold_patches.nuclear
    num_classes = len(fold_patches.class_names)
    collate = make_collate(model, markers, nuclear)

    train_ds = CellPatchDataset(
        fold_patches,
        f"train_2000_fold{fold}.csv",
        max_per_class=config.max_cells_per_class,
        seed=config.seed,
    )
    valid_ds = CellPatchDataset(
        fold_patches,
        f"val_fold{fold}.csv",
        max_per_class=config.max_valid_per_class,
        seed=config.seed,
    )
    train_loader = make_loader(
        train_ds, train=True, config=config, collate=collate
    )
    valid_loader = make_loader(
        valid_ds, train=False, config=config, collate=collate
    )
    logger.info(
        "fold %d: train %d cells (%d steps/epoch) | valid %d",
        fold,
        len(train_ds),
        len(train_loader),
        len(valid_ds),
    )

    backbone, head, _, _ = build_classifier(device, num_classes, config)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        trainable_parameters(backbone, head),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs
    )

    best_loss, best_epoch, stale = float("inf"), -1, 0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float]] = []
    started = time.perf_counter()

    for epoch in range(config.epochs):
        set_training(backbone, head, True)
        running, seen = 0.0, 0
        bar = tqdm(
            train_loader,
            desc=f"fold {fold} epoch {epoch + 1}/{config.epochs}",
            leave=False,
        )
        for x, y in bar:
            x, y = x.to(device, non_blocking=True), y.to(device)
            optimizer.zero_grad()
            loss = criterion(forward_cls(backbone, head, x, markers), y)
            loss.backward()
            optimizer.step()
            running += loss.item()
            seen += 1
            bar.set_postfix(
                loss=f"{running / seen:.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
            )
        scheduler.step()

        val_loss, val_probs, val_true = evaluate_loader(
            backbone, head, valid_loader, device, markers, criterion
        )
        val_bacc = balanced_accuracy_score(val_true, val_probs.argmax(1))
        improved = val_loss < best_loss - config.min_delta
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": running / max(seen, 1),
                "val_loss": val_loss,
                "val_balanced_acc": val_bacc,
            }
        )
        logger.info(
            "fold %d epoch %2d: train %.4f | val %.4f | val bal-acc %.4f%s",
            fold,
            epoch + 1,
            running / max(seen, 1),
            val_loss,
            val_bacc,
            "  <- best" if improved else "",
        )

        if improved:
            best_loss, best_epoch, stale = val_loss, epoch, 0
            best_state = _trainable_state(backbone, head)
        else:
            stale += 1
            if stale >= config.patience:
                logger.info(
                    "fold %d early stop at epoch %d (best was epoch %d)",
                    fold,
                    epoch + 1,
                    best_epoch + 1,
                )
                break

    if best_state is not None:
        _load_trainable_state(backbone, head, best_state)
    elapsed = (time.perf_counter() - started) / 60
    logger.info(
        "fold %d done in %.1f min (best epoch %d)",
        fold,
        elapsed,
        best_epoch + 1,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "trainable_state": best_state,
            "best_epoch": best_epoch + 1,
            "class_names": fold_patches.class_names,
        },
        out_dir / f"lora_fold{fold}.pt",
    )
    return backbone, head, pd.DataFrame(history)


def _trainable_state(
    backbone: nn.Module, head: nn.Module
) -> dict[str, torch.Tensor]:
    """Detached CPU copy of the trainable tensors (LoRA factors + head)."""
    named = list(backbone.named_parameters()) + [
        ("head." + n, p) for n, p in head.named_parameters()
    ]
    return {
        name: p.detach().cpu().clone() for name, p in named if p.requires_grad
    }


def _load_trainable_state(
    backbone: nn.Module,
    head: nn.Module,
    state: dict[str, torch.Tensor],
) -> None:
    """Restore a :func:`_trainable_state` checkpoint into the model."""
    backbone.load_state_dict(
        {k: v for k, v in state.items() if not k.startswith("head.")},
        strict=False,
    )
    head.load_state_dict(
        {
            k[len("head.") :]: v
            for k, v in state.items()
            if k.startswith("head.")
        },
        strict=False,
    )


def score(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float]:
    """The tutorial's four macro metrics on a fixed global class order.

    All four are macro-averaged, weighting every cell type equally
    regardless of abundance. ``labels`` pins the class order so a fold
    that never sees a rare class still aligns its probability columns.

    Args:
        y_true: Integer truths ``(n,)``.
        y_prob: Probabilities ``(n, num_classes)``.
        labels: Every class code in the global label set.

    Returns:
        Macro F1, balanced accuracy, average precision, and ROC AUC.

    Example:
        ``score(truth, probs, fold_patches.labels)`` returns metrics
        directly comparable across folds and against the frozen probe.
    """
    y_pred = y_prob.argmax(1)
    return {
        "F1-Score": f1_score(
            y_true, y_pred, average="macro", labels=labels, zero_division=0
        ),
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
        "Average Precision": average_precision_score(
            label_binarize(y_true, classes=labels), y_prob, average="macro"
        ),
        "ROC AUC": roc_auc_score(
            y_true, y_prob, average="macro", multi_class="ovr", labels=labels
        ),
    }


def frozen_probe(
    fold_patches: FoldPatches,
    best_c: float,
    fold: int,
    *,
    max_cells_per_class: int | None = None,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Tutorial 5's frozen linear probe, refit on this fold's rows.

    The controlled baseline for the LoRA comparison: same cells, same
    fold, same budget, same encoder weights, and the same ``C`` Tutorial
    5's Optuna search selected — the only difference is whether the
    adapters were allowed to move. It reads the stored KRONOS2 features,
    so it needs no GPU and takes seconds.

    Args:
        fold_patches: The shared patch/label context (holds the frozen
            features).
        best_c: The regularization strength inherited from Tutorial 5.
        fold: Which spatial fold to score.
        max_cells_per_class: Cap matching the LoRA side; keep ``None`` to
            use the fold CSVs as written.
        seed: Seed for the per-class cap.

    Returns:
        ``(truths, probs)`` on the held-out quadrant, with probabilities
        widened to the global class order.

    Example:
        ``truth, probs = frozen_probe(fold_patches, best_c, 1)`` produces
        the baseline predictions to compare against the LoRA run's.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    tr_rows, tr_labels = fold_patches.load_split(f"train_2000_fold{fold}.csv")
    te_rows, te_labels = fold_patches.load_split(f"test_fold{fold}.csv")
    tr_codes = fold_patches.codes(tr_labels)
    te_codes = fold_patches.codes(te_labels)
    tr_rows, tr_codes = cap_per_class(
        tr_rows, tr_codes, max_cells_per_class, seed=seed
    )

    x = fold_patches.frozen_features
    x_tr, x_te = x[tr_rows], x[te_rows]
    scaler = StandardScaler().fit(x_tr)
    clf = LogisticRegression(
        C=best_c, class_weight="balanced", max_iter=10_000
    )
    clf.fit(scaler.transform(x_tr), tr_codes)

    probs = np.zeros((len(x_te), len(fold_patches.labels)))
    probs[:, clf.classes_] = clf.predict_proba(scaler.transform(x_te))
    return te_codes, probs


def train_and_eval_folds(
    fold_patches: FoldPatches,
    model: nn.Module,
    config: FinetuneConfig,
    device: str,
    folds: Sequence[int],
    out_dir: Path,
) -> pd.DataFrame:
    """Train, evaluate, and persist one LoRA model per fold.

    For each fold: fine-tune, score the held-out quadrant, and write the
    adapter checkpoint (``lora_fold{f}.pt``), the test predictions
    (``preds_fold{f}.npz``), and the per-epoch history
    (``history_fold{f}.csv``). A combined ``lora_results.csv`` is written
    at the end. The model is released after each fold, so peak memory
    does not grow with the number of folds. This is the entry point the
    training script (``tutorials/finetune_cells.py``) calls.

    Args:
        fold_patches: The shared patch/label context.
        model: The live KRONOS2 module (for the collate's normalization).
        config: The run's hyperparameters.
        device: Torch device string.
        folds: The folds to run, e.g. ``(1,)`` or ``(1, 2, 3, 4)``.
        out_dir: Directory to write all artifacts into.

    Returns:
        Per-fold metrics indexed by fold.

    Example:
        ``train_and_eval_folds(fold_patches, model, config, device,
        (1, 2, 3, 4), out_dir)`` runs the full sweep the notebook then
        reads back from disk.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    markers = fold_patches.markers
    rows = []
    for fold in folds:
        logger.info("=== fold %d ===", fold)
        backbone, head, history = train_fold(
            fold_patches, model, config, device, fold, out_dir
        )
        history.to_csv(out_dir / f"history_fold{fold}.csv", index=False)

        collate = make_collate(model, markers, fold_patches.nuclear)
        test_ds = CellPatchDataset(fold_patches, f"test_fold{fold}.csv")
        test_loader = make_loader(
            test_ds, train=False, config=config, collate=collate
        )
        _, probs, truth = evaluate_loader(
            backbone, head, test_loader, device, markers
        )
        np.savez(out_dir / f"preds_fold{fold}.npz", truth=truth, probs=probs)

        metrics = score(truth, probs, fold_patches.labels)
        rows.append({"Fold": f"fold_{fold}", **metrics})
        logger.info(
            "fold %d test: %s",
            fold,
            ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()),
        )

        del backbone, head
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    results = pd.DataFrame(rows).set_index("Fold")
    results.to_csv(out_dir / "lora_results.csv")
    logger.info("wrote artifacts to %s", out_dir)
    return results
