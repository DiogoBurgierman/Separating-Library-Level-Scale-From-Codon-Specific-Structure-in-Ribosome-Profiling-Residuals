"""Validation criteria of the latent stage, plus the complementary criteria described
in the article's Materials and Methods.

  (a) residual variance reduction
  (b) reduction of correlation between residuals
  (c) generalisation on subsets not used for inference
  (d) stability of Z_c under resampling

and: latent-geometry stability under perturbation,
sensitivity to normalisation/protocol choice, and internal coherence among
conditions of similar experimental design.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .models import LatentFit, centre, fit_pca


# ---------------------------------------------------------------------------
# (a) residual variance reduction
# ---------------------------------------------------------------------------


def variance_accounting(matrix: np.ndarray, fit: LatentFit) -> dict:
    """Every denominator the criterion could be read against, reported side by side."""
    x = matrix.T
    xc, offset = centre(matrix)

    if fit.name == "vae":
        fitted_centred = fit.extra["fitted_centred"]
    else:
        fitted_centred = fit.latent_term()

    residual_after = xc - fitted_centred

    raw_mean = float(x.mean())
    var_raw = float(np.mean((x - raw_mean) ** 2))
    mse_raw = float(np.mean(x**2))

    mse_offset_only = float(np.mean(xc**2))
    var_offset_only = float(np.mean((xc - xc.mean()) ** 2))

    mse_latent = float(np.mean(residual_after**2))
    var_latent = float(np.mean((residual_after - residual_after.mean()) ** 2))

    return {
        "d": fit.d,
        "estimator": fit.name,
        # frozen baseline quantities
        "mse_R_raw": mse_raw,
        "var_R_raw": var_raw,
        "mean_R_raw": raw_mean,
        # after the codon offset m_i only (this is the honest comparator)
        "mse_after_offset_only": mse_offset_only,
        "var_after_offset_only": var_offset_only,
        # after m_i + g(Z_c)
        "mse_after_latent": mse_latent,
        "var_after_latent": var_latent,
        # gains
        "latent_gain_vs_offset_only": (
            1.0 - mse_latent / mse_offset_only if mse_offset_only > 0 else None
        ),
        "total_gain_vs_var_R_raw": (
            1.0 - var_latent / var_raw if var_raw > 0 else None
        ),
        "offset_only_gain_vs_var_R_raw": (
            1.0 - var_offset_only / var_raw if var_raw > 0 else None
        ),
        "criterion_var_reduced": bool(var_latent < var_raw),
        "criterion_var_reduced_beyond_offset": bool(mse_latent < mse_offset_only),
    }


# ---------------------------------------------------------------------------
# (b) residual correlation structure
# ---------------------------------------------------------------------------


def residual_correlation_reduction(matrix: np.ndarray, fit: LatentFit) -> dict:
    xc, _ = centre(matrix)
    fitted = fit.extra["fitted_centred"] if fit.name == "vae" else fit.latent_term()
    after = xc - fitted

    def _offdiag_stats(block: np.ndarray) -> tuple[float, float]:
        corr = np.corrcoef(block)
        n = corr.shape[0]
        iu = np.triu_indices(n, k=1)
        vals = corr[iu]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return float("nan"), float("nan")
        return float(vals.mean()), float(np.abs(vals).mean())

    before_mean, before_abs = _offdiag_stats(xc)
    after_mean, after_abs = _offdiag_stats(after)

    return {
        "d": fit.d,
        "estimator": fit.name,
        "mean_offdiagonal_correlation_before": before_mean,
        "mean_absolute_offdiagonal_correlation_before": before_abs,
        "mean_offdiagonal_correlation_after": after_mean,
        "mean_absolute_offdiagonal_correlation_after": after_abs,
        "absolute_correlation_reduction": before_abs - after_abs,
        "relative_absolute_correlation_reduction": (
            (before_abs - after_abs) / before_abs if before_abs else None
        ),
    }


# ---------------------------------------------------------------------------
# (c) generalisation - bi-cross-validation
# ---------------------------------------------------------------------------


def bi_cross_validation(
    matrix: np.ndarray,
    d_grid,
    n_splits: int,
    seed: int,
    row_holdout: float = 0.5,
    col_holdout: int = 4,
) -> pd.DataFrame:
    """Owen-Perry style bi-cross-validation.

    Codons are split into train rows A and held-out rows B; conditions into train
    columns C1 and held-out columns C2. The rank-d structure is fitted on block
    (A, C1) only and used to predict block (B, C2), which shares no row and no
    column with the fitted block. The codon offset m_i is computed from training
    columns only, so no held-out cell influences its own prediction.
    """
    n_codons, n_conditions = matrix.shape
    rng = np.random.default_rng(seed)
    records = []

    for split in range(n_splits):
        rows = rng.permutation(n_codons)
        n_train_rows = int(round(n_codons * row_holdout))
        rows_a, rows_b = rows[:n_train_rows], rows[n_train_rows:]

        cols = rng.permutation(n_conditions)
        cols_c2, cols_c1 = cols[:col_holdout], cols[col_holdout:]
        if rows_a.size < 2 or rows_b.size < 2 or cols_c1.size < 2 or cols_c2.size < 1:
            continue

        offset = matrix[:, cols_c1].mean(axis=1, keepdims=True)
        centred = matrix - offset

        e_a1 = centred[np.ix_(rows_a, cols_c1)]
        e_a2 = centred[np.ix_(rows_a, cols_c2)]
        e_b1 = centred[np.ix_(rows_b, cols_c1)]
        e_b2 = centred[np.ix_(rows_b, cols_c2)]

        baseline_mse = float(np.mean(e_b2**2))

        u, s, vt = np.linalg.svd(e_a1, full_matrices=False)
        for d in d_grid:
            if d > s.size:
                continue
            s_inv = np.zeros_like(s)
            nonzero = s[:d] > 1e-12
            s_inv[:d][nonzero] = 1.0 / s[:d][nonzero]
            pinv_d = vt.T @ np.diag(s_inv) @ u.T
            prediction = e_b1 @ pinv_d @ e_a2
            mse = float(np.mean((e_b2 - prediction) ** 2))
            records.append(
                {
                    "split": split,
                    "d": d,
                    "heldout_mse": mse,
                    "offset_only_mse": baseline_mse,
                    "skill_vs_offset_only": 1.0 - mse / baseline_mse if baseline_mse > 0 else None,
                }
            )

    frame = pd.DataFrame(records)
    if frame.empty:
        return frame
    summary = (
        frame.groupby("d")
        .agg(
            n_splits=("split", "count"),
            mean_heldout_mse=("heldout_mse", "mean"),
            mean_offset_only_mse=("offset_only_mse", "mean"),
            mean_skill=("skill_vs_offset_only", "mean"),
            std_skill=("skill_vs_offset_only", "std"),
            median_skill=("skill_vs_offset_only", "median"),
            p05_skill=("skill_vs_offset_only", lambda v: float(np.quantile(v, 0.05))),
            p95_skill=("skill_vs_offset_only", lambda v: float(np.quantile(v, 0.95))),
            fraction_splits_with_positive_skill=(
                "skill_vs_offset_only",
                lambda v: float(np.mean(np.asarray(v) > 0)),
            ),
        )
        .reset_index()
    )
    summary["se_skill"] = summary["std_skill"] / np.sqrt(summary["n_splits"])
    return summary


# ---------------------------------------------------------------------------
# (d) robustness of Z_c under resampling
# ---------------------------------------------------------------------------


def _procrustes_align(reference: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Rotate `target` onto `reference` (both N x d); latent axes are unidentifiable
    up to an orthogonal transform, so geometry is compared after alignment."""
    u, _, vt = np.linalg.svd(target.T @ reference)
    return target @ (u @ vt)


def bootstrap_stability(
    matrix: np.ndarray,
    d: int,
    n_bootstrap: int,
    seed: int,
) -> tuple[dict, pd.DataFrame]:
    """Resample codons with replacement, refit, and compare latent geometry."""
    reference = fit_pca(matrix, d)
    ref_scores = reference.scores
    ref_loadings = reference.loadings

    n_codons = matrix.shape[0]
    rng = np.random.default_rng(seed)

    per_condition = np.zeros((n_bootstrap, matrix.shape[1]))
    subspace_cos = np.zeros(n_bootstrap)
    pc1_share = np.zeros(n_bootstrap)
    geometry_corr = np.zeros(n_bootstrap)

    for b in range(n_bootstrap):
        idx = rng.integers(0, n_codons, size=n_codons)
        boot = matrix[idx, :]
        fit = fit_pca(boot, d)
        aligned = _procrustes_align(ref_scores, fit.scores)

        norms_ref = np.linalg.norm(ref_scores, axis=1)
        norms_boot = np.linalg.norm(aligned, axis=1)
        denom = norms_ref * norms_boot
        cos = np.where(denom > 0, np.sum(ref_scores * aligned, axis=1) / np.where(denom > 0, denom, 1), np.nan)
        per_condition[b] = cos

        # principal angles between loading subspaces, restricted to the resampled rows
        q_ref, _ = np.linalg.qr(ref_loadings[idx, :])
        q_boot, _ = np.linalg.qr(fit.loadings)
        sv = np.linalg.svd(q_ref.T @ q_boot, compute_uv=False)
        subspace_cos[b] = float(np.mean(np.clip(sv, -1.0, 1.0)))

        ratios = fit.extra["explained_variance_ratio"]
        pc1_share[b] = ratios[0]

        flat_ref = ref_scores.reshape(-1)
        flat_boot = aligned.reshape(-1)
        if np.std(flat_ref) > 0 and np.std(flat_boot) > 0:
            geometry_corr[b] = float(np.corrcoef(flat_ref, flat_boot)[0, 1])
        else:
            geometry_corr[b] = np.nan

    condition_frame = pd.DataFrame(
        {
            "mean_cosine_to_reference": np.nanmean(per_condition, axis=0),
            "p05_cosine_to_reference": np.nanquantile(per_condition, 0.05, axis=0),
            "p95_cosine_to_reference": np.nanquantile(per_condition, 0.95, axis=0),
        }
    )

    summary = {
        "d": d,
        "n_bootstrap": n_bootstrap,
        "mean_subspace_cosine": float(np.mean(subspace_cos)),
        "p05_subspace_cosine": float(np.quantile(subspace_cos, 0.05)),
        "mean_geometry_correlation": float(np.nanmean(geometry_corr)),
        "p05_geometry_correlation": float(np.nanquantile(geometry_corr, 0.05)),
        "mean_pc1_explained_variance_ratio": float(np.mean(pc1_share)),
        "p05_pc1_explained_variance_ratio": float(np.quantile(pc1_share, 0.05)),
        "p95_pc1_explained_variance_ratio": float(np.quantile(pc1_share, 0.95)),
        "min_condition_mean_cosine": float(np.nanmin(condition_frame["mean_cosine_to_reference"])),
    }
    return summary, condition_frame


# ---------------------------------------------------------------------------
# complementary: internal coherence among conditions of similar design
# ---------------------------------------------------------------------------


def design_coherence(
    scores: np.ndarray,
    design: pd.DataFrame,
    groupings: list[str],
    n_permutations: int,
    seed: int,
) -> pd.DataFrame:
    """Within-group vs between-group distance in Z_c space, with a label-permutation test."""
    n = scores.shape[0]
    diff = scores[:, None, :] - scores[None, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    iu = np.triu_indices(n, k=1)
    pair_dist = dist[iu]

    rng = np.random.default_rng(seed)
    rows = []
    for grouping in groupings:
        labels = design[grouping].to_numpy()
        same = labels[iu[0]] == labels[iu[1]]
        if same.sum() == 0 or (~same).sum() == 0:
            continue
        observed = float(pair_dist[same].mean() / pair_dist[~same].mean())

        null = np.empty(n_permutations)
        for b in range(n_permutations):
            permuted = rng.permutation(labels)
            s = permuted[iu[0]] == permuted[iu[1]]
            if s.sum() == 0 or (~s).sum() == 0:
                null[b] = np.nan
                continue
            null[b] = pair_dist[s].mean() / pair_dist[~s].mean()
        null = null[np.isfinite(null)]
        p_value = float((np.sum(null <= observed) + 1) / (null.size + 1))

        rows.append(
            {
                "grouping": grouping,
                "n_groups": int(pd.unique(labels).size),
                "mean_within_group_distance": float(pair_dist[same].mean()),
                "mean_between_group_distance": float(pair_dist[~same].mean()),
                "within_over_between_ratio": observed,
                "permutation_null_mean_ratio": float(null.mean()),
                "permutation_p_value": p_value,
                "coherent_at_0.05": bool(p_value < 0.05),
            }
        )
    return pd.DataFrame(rows)
