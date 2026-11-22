from collections import defaultdict

import numpy as np
import pandas as pd

from . import config


def empty_accumulators():
    return {
        "sum_k": defaultdict(float),
        "sumsq_k": defaultdict(float),
        "n_k": defaultdict(int),
        "sum_kc": defaultdict(float),
        "n_kc": defaultdict(int),
        "sum_rho_kc": defaultdict(float),
        "n_rho_kc": defaultdict(int),
        "sum_all": 0.0,
        "sumsq_all": 0.0,
        "n_all": 0,
    }


def add_gene_observations(acc, codons: list[str], rho: np.ndarray, k_obs: np.ndarray, condition_id: str):
    drop_start = config.EXCLUDE_START_NT // 3
    interior_codons = codons[drop_start : drop_start + len(rho)]
    if len(interior_codons) != len(k_obs):
        raise ValueError("codon/k_obs length mismatch")
    for codon, rho_i, k_i in zip(interior_codons, rho, k_obs):
        if codon in config.STOP_CODONS:
            continue
        if len(codon) != 3 or codon not in config.STANDARD_GENETIC_CODE:
            continue
        if np.isfinite(rho_i):
            acc["sum_rho_kc"][(codon, condition_id)] += float(rho_i)
            acc["n_rho_kc"][(codon, condition_id)] += 1
        if not np.isfinite(k_i):
            continue
        acc["sum_k"][codon] += float(k_i)
        acc["sumsq_k"][codon] += float(k_i) * float(k_i)
        acc["n_k"][codon] += 1
        acc["sum_kc"][(codon, condition_id)] += float(k_i)
        acc["n_kc"][(codon, condition_id)] += 1
        acc["sum_all"] += float(k_i)
        acc["sumsq_all"] += float(k_i) * float(k_i)
        acc["n_all"] += 1


def fit_additive_codon_model(acc) -> dict:
    if acc["n_all"] == 0:
        raise RuntimeError("no observations to fit k_seq")
    mu = acc["sum_all"] / acc["n_all"]
    k_seq = {}
    beta = {}
    n_obs = {}
    for codon, n in acc["n_k"].items():
        mean_codon = acc["sum_k"][codon] / n
        k_seq[codon] = mean_codon
        beta[codon] = mean_codon - mu
        n_obs[codon] = n

    ss_tot = acc["sumsq_all"] - acc["n_all"] * mu * mu
    ss_res = 0.0
    for codon, n in acc["n_k"].items():
        mean_codon = k_seq[codon]
        ss_res += acc["sumsq_k"][codon] - n * mean_codon * mean_codon
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    var_residual = ss_res / acc["n_all"]
    var_obs = ss_tot / acc["n_all"]

    effects_rows = []
    for codon in sorted(k_seq):
        effects_rows.append(
            {
                "codon": codon,
                "amino_acid": config.STANDARD_GENETIC_CODE[codon],
                "n_obs": n_obs[codon],
                "k_seq": k_seq[codon],
                "mu": mu,
                "beta_codon": beta[codon],
            }
        )
    effects = pd.DataFrame(effects_rows)

    condition_ids = sorted({cond for _, cond in acc["n_kc"]})
    matrix_rows = []
    for codon in sorted(k_seq):
        row = {"codon": codon, "amino_acid": config.STANDARD_GENETIC_CODE[codon]}
        for cond in condition_ids:
            n_kc = acc["n_kc"].get((codon, cond), 0)
            if n_kc == 0:
                k_obs_c = np.nan
                rho_c = np.nan
            else:
                k_obs_c = acc["sum_kc"][(codon, cond)] / n_kc
                n_rho = acc["n_rho_kc"].get((codon, cond), 0)
                rho_c = acc["sum_rho_kc"][(codon, cond)] / n_rho if n_rho else np.nan
            row[f"k_obs::{cond}"] = k_obs_c
            row[f"R::{cond}"] = k_obs_c - k_seq[codon]
            row[f"rho::{cond}"] = rho_c
            row[f"n::{cond}"] = n_kc
        matrix_rows.append(row)
    matrix = pd.DataFrame(matrix_rows)

    codon_cond_k = []
    codon_cond_fitted = []
    for codon in k_seq:
        for cond in condition_ids:
            n_kc = acc["n_kc"].get((codon, cond), 0)
            if n_kc == 0:
                continue
            codon_cond_k.append(acc["sum_kc"][(codon, cond)] / n_kc)
            codon_cond_fitted.append(k_seq[codon])
    codon_cond_k = np.asarray(codon_cond_k)
    codon_cond_fitted = np.asarray(codon_cond_fitted)
    cc_mean = float(codon_cond_k.mean())
    cc_ss_tot = float(np.sum((codon_cond_k - cc_mean) ** 2))
    cc_ss_res = float(np.sum((codon_cond_k - codon_cond_fitted) ** 2))
    r2_codon_condition = 1.0 - cc_ss_res / cc_ss_tot if cc_ss_tot > 0 else 0.0

    # k_seq is fitted by OLS weighted by position-level n, but this table averages
    # cells unweighted, so the codon x condition residuals do NOT have zero mean.
    # cc_ss_res is therefore an uncentred second moment: reporting it as a variance
    # overstates spread by exactly mean(R)^2. Both quantities are emitted below.
    cc_resid = codon_cond_k - codon_cond_fitted
    cc_resid_mean = float(cc_resid.mean())
    cc_ss_res_centred = float(np.sum((cc_resid - cc_resid_mean) ** 2))

    weighted_sum_beta = float(sum(n_obs[codon] * beta[codon] for codon in beta))
    summary = {
        "n_observations": acc["n_all"],
        "n_codons": len(k_seq),
        "n_conditions": len(condition_ids),
        "mu": mu,
        "r2_sequence_baseline": r2,
        "r2_codon_condition_means": r2_codon_condition,
        "var_k_obs": var_obs,
        "var_k_obs_minus_k_seq": var_residual,
        "var_k_obs_codon_condition": cc_ss_tot / len(codon_cond_k) if len(codon_cond_k) else None,
        "var_residual_codon_condition": cc_ss_res_centred / len(codon_cond_k) if len(codon_cond_k) else None,
        "mse_residual_codon_condition": cc_ss_res / len(codon_cond_k) if len(codon_cond_k) else None,
        "mean_residual_codon_condition": cc_resid_mean,
        "weighted_sum_n_times_beta": weighted_sum_beta,
        "identifiability": "sum_to_zero_weighted_by_n_obs",
        "model": "k_obs(i,c) = mu + beta_codon(i) + eps_ic",
        "k_seq": "k_seq(i) = mu + beta_codon(i)",
        "codon_condition_variance_note": (
            "var_residual_codon_condition is centred, Var(R); "
            "mse_residual_codon_condition is E[R^2] = Var(R) + mean(R)^2 and is the "
            "value previously emitted under the var_ name. "
            "r2_codon_condition_means keeps the standard definition "
            "1 - SS_res / SS_tot with uncentred SS_res, so it is unchanged."
        ),
    }
    return {
        "effects": effects,
        "matrix": matrix,
        "summary": summary,
        "k_seq": k_seq,
        "mu": mu,
        "beta": beta,
        "condition_ids": condition_ids,
    }
