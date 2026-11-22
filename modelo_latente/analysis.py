"""Residual analysis run before any Z_c is fitted.

Three questions, in order:
  (i)   do the residuals depend on the condition c?
  (ii)  is there low-dimensional structure?
  (iii) what are the intra- and inter-condition correlations?

The null used throughout permutes cells independently within each codon row.
That preserves every codon's marginal residual distribution while destroying the
alignment of conditions across codons, so it isolates exactly the coherence a
condition-level latent variable would have to explain.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .models import centre, singular_spectrum


def _permute_within_rows(matrix: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = np.empty_like(matrix)
    for i in range(matrix.shape[0]):
        out[i] = rng.permutation(matrix[i])
    return out


def condition_dependence(
    matrix: np.ndarray, n_permutations: int, seed: int
) -> dict:
    """(i) Share of residual variance carried by the condition main effect."""
    grand = float(matrix.mean())
    ss_total = float(np.sum((matrix - grand) ** 2))
    n_codons, n_conditions = matrix.shape

    column_means = matrix.mean(axis=0)
    ss_condition = float(n_codons * np.sum((column_means - grand) ** 2))
    row_means = matrix.mean(axis=1)
    ss_codon = float(n_conditions * np.sum((row_means - grand) ** 2))

    observed = ss_condition / ss_total if ss_total > 0 else 0.0

    rng = np.random.default_rng(seed)
    null = np.empty(n_permutations)
    for b in range(n_permutations):
        perm = _permute_within_rows(matrix, rng)
        g = float(perm.mean())
        tot = float(np.sum((perm - g) ** 2))
        cm = perm.mean(axis=0)
        null[b] = (n_codons * float(np.sum((cm - g) ** 2))) / tot if tot > 0 else 0.0

    p_value = float((np.sum(null >= observed) + 1) / (n_permutations + 1))

    return {
        "ss_total": ss_total,
        "ss_condition_main_effect": ss_condition,
        "ss_codon_main_effect": ss_codon,
        "fraction_variance_condition_main_effect": observed,
        "fraction_variance_codon_main_effect": ss_codon / ss_total if ss_total else 0.0,
        "permutation_null_mean": float(null.mean()),
        "permutation_null_p95": float(np.quantile(null, 0.95)),
        "permutation_p_value": p_value,
        "n_permutations": n_permutations,
    }


def low_dimensional_structure(
    matrix: np.ndarray, n_permutations: int, seed: int, max_components: int
) -> tuple[dict, pd.DataFrame]:
    """(ii) Parallel analysis on the singular spectrum after removing m_i."""
    observed = singular_spectrum(matrix)
    k = min(max_components, observed.size)
    observed = observed[:k]
    obs_var = observed**2
    total = float(np.sum(obs_var))

    rng = np.random.default_rng(seed)
    null = np.empty((n_permutations, k))
    for b in range(n_permutations):
        perm = _permute_within_rows(matrix, rng)
        s = singular_spectrum(perm)[:k]
        null[b] = s

    p_values = (np.sum(null >= observed, axis=0) + 1) / (n_permutations + 1)
    null_p95 = np.quantile(null, 0.95, axis=0)

    rows = []
    cumulative = 0.0
    n_significant = 0
    still_significant = True
    for idx in range(k):
        share = float(obs_var[idx] / total) if total > 0 else 0.0
        cumulative += share
        significant = bool(p_values[idx] < 0.05)
        if significant and still_significant:
            n_significant += 1
        else:
            still_significant = False
        rows.append(
            {
                "component": idx + 1,
                "singular_value": float(observed[idx]),
                "explained_variance_ratio": share,
                "cumulative_explained_variance_ratio": cumulative,
                "permutation_null_mean_singular_value": float(null[:, idx].mean()),
                "permutation_null_p95_singular_value": float(null_p95[idx]),
                "permutation_p_value": float(p_values[idx]),
                "significant_at_0.05": significant,
            }
        )

    spectrum = pd.DataFrame(rows)
    summary = {
        "n_components_evaluated": k,
        "n_components_above_permutation_null": n_significant,
        "explained_variance_ratio_pc1": rows[0]["explained_variance_ratio"],
        "cumulative_explained_variance_first_two": (
            rows[1]["cumulative_explained_variance_ratio"] if k > 1 else rows[0]["cumulative_explained_variance_ratio"]
        ),
        "n_permutations": n_permutations,
    }
    return summary, spectrum


def condition_correlations(matrix: np.ndarray, design: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """(iii) Correlation between conditions, computed over the 61 codons."""
    xc, _ = centre(matrix)  # (n_conditions, n_codons), m_i removed
    corr = np.corrcoef(xc)
    ids = design["condition_id"].tolist()
    frame = pd.DataFrame(corr, index=ids, columns=ids)

    n = len(ids)
    iu = np.triu_indices(n, k=1)
    pairs = corr[iu]

    dataset = design["dataset"].to_numpy()
    family = design["protocol_family"].to_numpy()
    harvest = design["harvest"].to_numpy()

    same_dataset = dataset[iu[0]] == dataset[iu[1]]
    same_family = family[iu[0]] == family[iu[1]]
    same_harvest = harvest[iu[0]] == harvest[iu[1]]

    def _mean(mask: np.ndarray) -> float | None:
        return float(pairs[mask].mean()) if mask.any() else None

    summary = {
        "mean_pairwise_correlation": float(pairs.mean()),
        "mean_absolute_pairwise_correlation": float(np.abs(pairs).mean()),
        "mean_correlation_within_dataset": _mean(same_dataset),
        "mean_correlation_between_datasets": _mean(~same_dataset),
        "mean_correlation_within_protocol_family": _mean(same_family),
        "mean_correlation_between_protocol_families": _mean(~same_family),
        "mean_correlation_same_harvest_method": _mean(same_harvest),
        "mean_correlation_different_harvest_method": _mean(~same_harvest),
        "n_condition_pairs": int(pairs.size),
    }
    return summary, frame
