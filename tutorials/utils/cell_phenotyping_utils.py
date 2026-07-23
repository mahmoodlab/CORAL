"""Linear-probe evaluation helpers for the cell-phenotyping tutorial.

The tutorial's subject is the *protocol* — spatial folds, a tuned
logistic-regression probe, and the metrics that make its score
comparable to a published benchmark. The mechanics of running that
protocol (looping folds, wiring Optuna, aligning probability columns to
a global class order) are ordinary and get in the way of reading it, so
they live here instead of in the notebook.

Every protocol knob is an explicit keyword argument rather than a
module constant, so the notebook keeps stating the numbers it is
claiming parity with. The defaults below reproduce the reference
protocol; the notebook passes them anyway, so a reader never has to
open this file to learn what the run actually did.

Nothing here is ESB-specific: it takes a feature matrix and a label
vector and returns scores. ESB's job ends at the feature matrix.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import optuna
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler, label_binarize

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

#: Optuna trials per fold in the reference protocol.
DEFAULT_N_TRIALS = 15

#: Log-uniform search bounds on the regularization strength ``C``.
DEFAULT_C_RANGE = (1e-4, 1e2)

#: Training cells per class; ``None`` disables the cap.
DEFAULT_MAX_CELLS_PER_CLASS = 2000

#: Solver iteration budget. High enough that convergence is not the
#: variable under study.
DEFAULT_MAX_ITER = 10000

# Optuna narrates every trial at INFO; the notebook shows a progress bar
# instead. Convergence warnings from the solver are routed through
# logging so they can be filtered rather than printed per fit.
optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.captureWarnings(True)


def split_fold(
    quadrant: np.ndarray,
    fold_id: int,
    *,
    valid_frac: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split row positions into train/valid/test, holding out one quadrant.

    The held-out quadrant becomes the test set, so train and test cells
    are never spatial neighbours. The validation split is drawn from the
    remaining quadrants and is used only to tune ``C``.

    Args:
        quadrant: Per-cell quadrant id in 1..4, one entry per row of the
            feature matrix.
        fold_id: Quadrant to hold out as the test set.
        valid_frac: Fraction of the training rows reserved for
            validation.
        random_state: Seed for the validation draw.

    Returns:
        Row positions as ``(train, valid, test)`` integer arrays.

    Example:
        With four quadrants and ``fold_id=1``, every cell in quadrant 1
        lands in ``test`` and the other three quadrants are split 80/20
        into ``train`` and ``valid``.
    """
    pos = pd.Series(np.arange(len(quadrant)))
    test_pos = pos[quadrant == fold_id]

    train_pos = pd.concat(
        [pos[quadrant == q] for q in (1, 2, 3, 4) if q != fold_id]
    )
    valid_pos = train_pos.sample(frac=valid_frac, random_state=random_state)
    train_pos = train_pos.drop(valid_pos.index)

    return train_pos.values, valid_pos.values, test_pos.values


def cap_per_class(
    idx: np.ndarray,
    y: np.ndarray,
    max_per_class: int | None,
    *,
    random_state: int = 42,
) -> np.ndarray:
    """Subsample row positions to at most ``max_per_class`` per class.

    Caps the training budget so an abundant class cannot dominate the
    fit purely by volume. Classes with fewer rows than the cap are kept
    whole.

    Args:
        idx: Row positions to subsample.
        y: Integer class codes for every row of the feature matrix.
        max_per_class: Maximum rows to keep per class, or ``None`` to
            keep all of them.
        random_state: Seed for the subsample.

    Returns:
        The retained row positions.

    Example:
        Capping at 2000 leaves a 50,000-cell tumour class with 2000
        rows and a 300-cell mast class untouched.
    """
    if max_per_class is None:
        return idx
    grouped = pd.DataFrame({"pos": idx, "cls": y[idx]}).groupby(
        "cls", group_keys=False
    )
    capped = grouped.apply(
        lambda g: g.sample(min(len(g), max_per_class), random_state=random_state)
    )
    return capped["pos"].values


def fit_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    *,
    n_trials: int = DEFAULT_N_TRIALS,
    c_range: tuple[float, float] = DEFAULT_C_RANGE,
    max_iter: int = DEFAULT_MAX_ITER,
    seed: int | None = 42,
    verbose: bool = True,
) -> tuple[StandardScaler, LogisticRegression, float, float, pd.DataFrame]:
    """Tune ``C`` with Optuna, then refit the probe on the best value.

    Standardizes the features, searches ``C`` with a TPE sampler scored
    on validation macro F1, then refits from scratch on the winning
    value. The refit is deliberate: nothing is warm-started, so it is
    the same fit as the best trial's, just made explicit.

    Args:
        X_train: Training features, shape ``(n_cells, n_features)``.
        y_train: Training class codes.
        X_valid: Validation features used to score each trial.
        y_valid: Validation class codes.
        n_trials: Optuna trials to run.
        c_range: Log-uniform ``(low, high)`` bounds on ``C``.
        max_iter: Solver iteration budget.
        seed: Sampler seed. Pass ``None`` for an unseeded sampler, as
            the published protocol used.
        verbose: Show the trial progress bar and print the winner.

    Returns:
        ``(scaler, model, best_C, best_valid_f1, trials)``, where
        ``trials`` is a frame of every sampled ``C`` and its validation
        score — the raw material for a search-trajectory plot.

    Example:
        Fitting on one fold's training rows returns a fitted scaler and
        probe ready to hand to :func:`evaluate`, plus the ``C`` that
        won and the score it won with.
    """
    scaler = StandardScaler().fit(X_train)
    Xt, Xv = scaler.transform(X_train), scaler.transform(X_valid)
    c_low, c_high = c_range

    def objective(trial: optuna.Trial) -> float:
        model = LogisticRegression(
            C=trial.suggest_float("C", low=c_low, high=c_high, log=True),
            class_weight="balanced",
            max_iter=max_iter,
        )
        model.fit(Xt, y_train)
        return f1_score(y_valid, model.predict(Xv), average="macro")

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=verbose)

    best_c = study.best_trial.params["C"]
    if verbose:
        print(
            f"    -> best C={best_c:.3e}, "
            f"valid macro-F1={study.best_trial.value:.4f}"
        )

    model = LogisticRegression(
        C=best_c, class_weight="balanced", max_iter=max_iter
    )
    model.fit(Xt, y_train)

    trials = pd.DataFrame(
        {
            "C": [t.params["C"] for t in study.trials],
            "valid_f1": [t.value for t in study.trials],
        }
    )
    return scaler, model, best_c, study.best_trial.value, trials


def evaluate(
    clf: LogisticRegression,
    scaler: StandardScaler,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    labels: np.ndarray,
) -> tuple[dict[str, float], np.ndarray]:
    """Score a fitted probe on held-out cells.

    All four metrics are macro-averaged, which weights every cell type
    equally regardless of how rare it is — the right choice when the
    rare classes are the interesting ones.

    Args:
        clf: Probe fitted on the standardized training features.
        scaler: The scaler that standardized them.
        X_test: Held-out features, unscaled.
        y_test: Held-out class codes.
        labels: Every class code in the global label set. Needed because
            a fold's training rows may not contain all classes, so
            ``clf.classes_`` can be a subset.

    Returns:
        ``(metrics, y_pred)`` — a dict of macro F1, balanced accuracy,
        average precision, and ROC AUC, plus the per-cell predictions.

    Example:
        Passing a fold's fitted probe and its test rows returns scores
        directly comparable across folds, since ``labels`` pins the
        class order.
    """
    Xs = scaler.transform(X_test)
    y_pred = clf.predict(Xs)

    # Widen the predicted probabilities to the global class order: a
    # fold missing a rare class would otherwise emit narrower columns
    # and misalign against the one-hot truth.
    y_prob = np.zeros((len(Xs), len(labels)), dtype=float)
    y_prob[:, clf.classes_] = clf.predict_proba(Xs)

    y_onehot = label_binarize(y_test, classes=labels)
    metrics = {
        "F1-Score": f1_score(
            y_test, y_pred, average="macro", labels=labels, zero_division=0
        ),
        "Balanced Accuracy": balanced_accuracy_score(y_test, y_pred),
        "Average Precision": average_precision_score(
            y_onehot, y_prob, average="macro"
        ),
        "ROC AUC": roc_auc_score(
            y_test, y_prob, average="macro", multi_class="ovr", labels=labels
        ),
    }
    return metrics, y_pred


def run_cross_validation(
    X: np.ndarray,
    y: np.ndarray,
    quadrant: np.ndarray,
    *,
    name: str,
    labels: np.ndarray,
    n_trials: int = DEFAULT_N_TRIALS,
    c_range: tuple[float, float] = DEFAULT_C_RANGE,
    max_cells_per_class: int | None = DEFAULT_MAX_CELLS_PER_CLASS,
    max_iter: int = DEFAULT_MAX_ITER,
    seed: int = 42,
    sampler_seed: int | None = 42,
    verbose: bool = True,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Fit and score the probe once per spatial fold.

    Runs the full protocol: for each of the four quadrants, hold it out,
    cap the training budget, tune and fit the probe, and score it on the
    held-out cells. Because every cell is tested exactly once, the
    out-of-fold predictions form a complete prediction map over the
    slide.

    Args:
        X: Feature matrix, shape ``(n_cells, n_features)``.
        y: Integer class codes, one per row.
        quadrant: Per-cell quadrant id in 1..4.
        name: Label for the progress lines, e.g. ``"KRONOS2"``.
        labels: Every class code in the global label set.
        n_trials: Optuna trials per fold.
        c_range: Log-uniform ``(low, high)`` bounds on ``C``.
        max_cells_per_class: Training cells per class, or ``None``.
        max_iter: Solver iteration budget.
        seed: Seed for the fold splits and the per-class cap.
        sampler_seed: Seed for the Optuna sampler. Kept separate from
            ``seed`` so the search can be left unseeded, as the
            published protocol had it, without also unpinning the
            splits.
        verbose: Print per-fold progress.

    Returns:
        ``(results, oof, trials)`` — per-fold metrics indexed by fold,
        out-of-fold predictions aligned to ``y``, and every Optuna trial
        tagged with the fold it came from.

    Example:
        Comparing two feature sets means calling this twice with the
        same ``y`` and ``quadrant`` and reading the difference in the
        returned scores.
    """
    rows = []
    oof = np.full(len(y), -1, dtype=np.int64)
    all_trials = []

    for fold_id in (1, 2, 3, 4):
        if verbose:
            print(f"[{name}] fold {fold_id} — {n_trials} Optuna trials")
        train_idx, valid_idx, test_idx = split_fold(
            quadrant, fold_id, random_state=seed
        )
        train_idx = cap_per_class(
            train_idx, y, max_cells_per_class, random_state=seed
        )

        scaler, clf, best_c, _, trials = fit_probe(
            X[train_idx],
            y[train_idx],
            X[valid_idx],
            y[valid_idx],
            n_trials=n_trials,
            c_range=c_range,
            max_iter=max_iter,
            seed=sampler_seed,
            verbose=verbose,
        )
        metrics, y_pred = evaluate(
            clf, scaler, X[test_idx], y[test_idx], labels=labels
        )
        oof[test_idx] = y_pred

        rows.append({"Fold": f"fold_{fold_id}", "C": best_c, **metrics})
        trials["Fold"] = f"fold_{fold_id}"
        all_trials.append(trials)
        if verbose:
            print(
                "    test:",
                ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()),
            )

    return (
        pd.DataFrame(rows).set_index("Fold"),
        oof,
        pd.concat(all_trials, ignore_index=True),
    )


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """Append mean and standard-deviation rows to a per-fold table.

    Drops the ``C`` column first: averaging a tuned hyperparameter
    across folds describes nothing, since each fold's ``C`` is only
    meaningful against its own training set.

    Args:
        results: Per-fold metrics as returned by
            :func:`run_cross_validation`.

    Returns:
        The metric columns with ``Mean`` and ``Std Dev`` rows appended,
        rounded to four decimals.

    Example:
        The ``Std Dev`` row is the one to read first — a large spread
        across folds means the score depends on which region was held
        out, and the mean alone would hide that.
    """
    metrics = results.drop(columns="C")
    summary = pd.DataFrame(
        [metrics.mean(), metrics.std(ddof=0)], index=["Mean", "Std Dev"]
    )
    return pd.concat([metrics, summary]).round(4)


def save_split(
    idx: np.ndarray,
    filename: str,
    *,
    cell_ids: np.ndarray,
    cell_labels: np.ndarray,
    out_dir: Path,
) -> int:
    """Write one split's ``cell_id,label`` rows to CSV.

    Persisting the splits lets another notebook score a different model
    on exactly these cells, which is what makes two runs comparable.

    Args:
        idx: Row positions belonging to this split.
        filename: File to write inside ``out_dir``.
        cell_ids: Cell ids for every row of the feature matrix.
        cell_labels: Class names for every row of the feature matrix.
        out_dir: Directory to write into. Must already exist.

    Returns:
        The number of rows written.

    Example:
        Writing ``train_2000_fold1.csv`` records both which cells were
        trained on and, via the filename, the budget they were drawn
        under.
    """
    pd.DataFrame(
        {"cell_id": cell_ids[idx], "label": cell_labels[idx]}
    ).to_csv(out_dir / filename, index=False)
    return len(idx)
