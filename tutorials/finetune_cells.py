r"""LoRA fine-tune KRONOS2 for cell phenotyping — from the command line.

Tutorial 6 (``6-LoRA-Finetuning.ipynb``) explains the protocol and
verifies the data path, but the fit itself takes hours per fold — too
long to hold a notebook kernel open. Run it here instead, detached, so
it survives a dropped SSH session::

    # one fold, in a tmux / screen session:
    uv run python tutorials/finetune_cells.py --folds 1 --num-workers 8

    # or with nohup, logging to a file:
    nohup uv run python tutorials/finetune_cells.py --folds 1,2,3,4 \
        > lora.log 2>&1 &
    tail -f lora.log

Run from the repo root. ``uv sync`` installs into the project's
``.venv/``, not into whatever interpreter a bare ``python`` resolves to,
so dropping the ``uv run`` prefix fails with ``No module named
'torch'``. Activating the environment (``source .venv/bin/activate``)
once per shell works too.

For each fold it writes, into ``--out-dir``:

    lora_fold{f}.pt        adapter + head checkpoint (~2.4 MB)
    preds_fold{f}.npz      held-out truths + probabilities
    history_fold{f}.csv    per-epoch train/val loss and balanced accuracy
    lora_results.csv       combined per-fold metrics

The notebook then reads these back to compare against the frozen probe.

Prerequisites: Tutorials 0/1/3/5 (the slide store and the fold CSVs), a
GPU, the ``kronos2`` extra, and scikit-learn. See the notebook's step 0.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Make the tutorial's utils/ importable whether this is run from the repo
# root or from tutorials/ (the notebooks use the same trick).
TUTORIALS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TUTORIALS_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(name)s] %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger("finetune_cells")

# The 18 phenotypic markers Tutorial 3 extracted on (fixed for this slide).
PANEL = [
    "dapi", "cd11b", "cd11c", "cd15", "cd163", "cd20", "cd206", "cd30",
    "cd31", "cd4", "cd56", "cd68", "cd7", "cd8", "cytokeratin", "foxp3",
    "mct", "podoplanin",
]  # fmt: skip


def parse_args() -> argparse.Namespace:
    """Parse command-line flags for one fine-tuning run."""
    default_data = TUTORIALS_DIR / "example-data"
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- data locations (default to the tutorial's example-data layout) ---
    parser.add_argument(
        "--slide",
        type=Path,
        default=default_data / "processed" / "raw_image.zarr",
        help="Slide zarr store from Tutorial 1's ingest.",
    )
    parser.add_argument(
        "--folds-dir",
        type=Path,
        default=default_data / "cell-pheno-results" / "folds",
        help="Directory of Tutorial 5's fold CSVs.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=default_data / "cell-pheno-results" / "lora",
        help="Where to write checkpoints, predictions, and results.",
    )
    parser.add_argument(
        "--folds",
        type=str,
        default="1",
        help="Comma-separated folds to run, e.g. '1' or '1,2,3,4'.",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=64,
        help="Cell patch side in pixels (must match Tutorial 3).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device (e.g. 'cuda', 'cuda:1', 'cpu'); auto if unset.",
    )

    # --- hyperparameters (default to FinetuneConfig's tutorial values) ---
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--num-lora-blocks", type=int, default=6)
    parser.add_argument(
        "--max-cells-per-class",
        type=int,
        default=None,
        help="Cap on training cells per class (default: all, ~2000).",
    )
    parser.add_argument(
        "--max-valid-per-class",
        type=int,
        default=None,
        help="Cap on validation cells per class (default: the full split).",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    """Open the slide, load KRONOS2 once, and run the folds."""
    args = parse_args()

    # Imported here (not at module top) so ``--help`` works without the
    # heavy ML stack, and so the sys.path insert above is in effect.
    import numpy as np
    import torch
    from utils.lora_finetune import (
        FinetuneConfig,
        FoldPatches,
        train_and_eval_folds,
    )

    from coral import CoralSlide
    from coral.config import PatchConfig
    from coral.config.subset import Selection
    from coral.features.kronos2 import Kronos2Extractor
    from coral.utils import resolve_device

    if not args.folds_dir.exists():
        raise SystemExit(
            f"No folds at {args.folds_dir}. Run Tutorial 5 (Cell "
            "Phenotyping) first — it writes the train/val/test CSVs this "
            "script reuses."
        )

    folds = [int(f) for f in args.folds.split(",") if f.strip()]
    device = resolve_device(args.device)
    logger.info("device: %s", device)
    if device == "cpu":
        logger.warning("no GPU found — fine-tuning on CPU is impractical.")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    slide = CoralSlide.open(args.slide)
    cell_cfg = PatchConfig(patch_size=args.patch_size, mode="cell_centered")
    panel = Selection(include=PANEL)
    fold_patches = FoldPatches.from_slide(
        slide, cell_cfg, panel, args.folds_dir
    )
    logger.info(
        "%d classes, %d markers -> %s",
        len(fold_patches.class_names),
        len(fold_patches.markers),
        fold_patches.markers,
    )

    # One live model, reused for the collate's per-marker normalization
    # across every fold; each fold builds its own fresh backbone to train.
    model = Kronos2Extractor.from_pretrained(device=device)._model
    logger.info("KRONOS2 loaded")

    config = FinetuneConfig(
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        num_lora_blocks=args.num_lora_blocks,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        num_workers=args.num_workers,
        max_cells_per_class=args.max_cells_per_class,
        max_valid_per_class=args.max_valid_per_class,
        seed=args.seed,
    )

    results = train_and_eval_folds(
        fold_patches, model, config, device, folds, args.out_dir
    )
    logger.info("done. per-fold metrics:\n%s", results.round(4))


if __name__ == "__main__":
    main()
