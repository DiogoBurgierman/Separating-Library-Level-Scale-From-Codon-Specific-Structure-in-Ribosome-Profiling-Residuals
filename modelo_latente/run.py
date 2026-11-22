"""Orchestration of the latent condition-state stage.

Reproduce with:

    python -m modelo_latente.run

Every artefact is written under `modelo_latente/saidas/<comparator>/`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "modelo_latente"

import numpy as np
import pandas as pd

from . import config
from .analysis import condition_correlations, condition_dependence, low_dimensional_structure
from .data import load_comparator
from .models import count_active_ard, fit_pca, fit_ppca_em, fit_vae, fit_vbpca_ard
from .validation import (
    bi_cross_validation,
    bootstrap_stability,
    design_coherence,
    residual_correlation_reduction,
    variance_accounting,
)


GROUPINGS = ["dataset", "harvest", "drug", "protocol_family"]

# Groupings tested inside a single dataset, where the batch axis is held fixed.
WITHIN_DATASET_GROUPINGS = ["harvest", "drug", "lysis"]
MIN_CONDITIONS_FOR_WITHIN_DATASET = 5


def _nesting_check(design: pd.DataFrame, groupings: list[str]) -> list[dict]:
    """Flag groupings whose levels never cross datasets.

    GSE77617 reports neither harvesting method nor drug, so its four libraries
    form an 'unreported' level that coincides exactly with the dataset. A
    coherence result on such a grouping is partly a batch result, and saying so
    is required before the grouping is interpreted as protocol structure.
    """
    rows = []
    for grouping in groupings:
        table = design.groupby(grouping)["dataset"].nunique()
        single = table[table == 1].index.tolist()
        rows.append(
            {
                "grouping": grouping,
                "n_levels": int(table.size),
                "n_levels_confined_to_one_dataset": int((table == 1).sum()),
                "levels_confined_to_one_dataset": ";".join(map(str, single)),
                "fully_nested_in_dataset": bool((table == 1).all()),
            }
        )
    return rows


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _select_dimension(
    bcv: pd.DataFrame,
    structure: dict,
    ard_frame: pd.DataFrame,
    ppca_frame: pd.DataFrame,
) -> tuple[int, dict]:
    """Select d by the permutation test on the singular spectrum.

    Three candidate rules were computed and only one of them is admissible here.

    * Bi-cross-validation argmax: held-out skill saturates once the dominant
      direction is in the model, so the argmax lands on the largest d in the grid
      for a gain of a few thousandths. Picking the top of a plateau is not a
      dimensionality criterion.
    * One-standard-error rule on the same curve: with repeated random splits the
      standard error shrinks as 1/sqrt(n_splits) and the splits are not
      independent, so the rule degenerates back towards the argmax.
    * PPCA BIC: at 61 features and 14 conditions the free-parameter count reaches
      the same order as the number of cells and sigma^2 collapses towards zero, so
      the likelihood term outruns the penalty and BIC decreases monotonically in d.
      It is not usable in this p >> n regime.

    The parallel-analysis permutation test is the criterion that carries an
    explicit null. It is therefore primary,
    with the other three reported alongside as corroboration.
    """
    chosen = max(1, int(structure["n_components_above_permutation_null"]))

    detail = {
        "rule": "number of singular values above the within-codon permutation null",
        "selected_d": chosen,
        "n_components_above_permutation_null": int(
            structure["n_components_above_permutation_null"]
        ),
        "explained_variance_ratio_pc1": structure["explained_variance_ratio_pc1"],
    }

    if not bcv.empty:
        best_row = bcv.loc[bcv["mean_skill"].idxmax()]
        chosen_skill = float(bcv.loc[bcv["d"] == chosen, "mean_skill"].iloc[0])
        detail.update(
            {
                "bcv_argmax_d": int(best_row["d"]),
                "bcv_argmax_mean_skill": float(best_row["mean_skill"]),
                "bcv_mean_skill_at_selected_d": chosen_skill,
                "bcv_fraction_of_best_skill_at_selected_d": (
                    chosen_skill / float(best_row["mean_skill"])
                    if best_row["mean_skill"]
                    else None
                ),
                "bcv_rejected_rule": "argmax and one-standard-error both degenerate on a plateau",
            }
        )

    if not ard_frame.empty:
        top = ard_frame.loc[ard_frame["d_requested"] == max(ard_frame["d_requested"])]
        alphas = [float(v) for v in top["alpha_sorted"].iloc[0].split(";")]
        ratios = [alphas[i + 1] / alphas[i] for i in range(len(alphas) - 1)]
        detail.update(
            {
                "ard_n_active_by_threshold": int(top["n_active_components"].iloc[0]),
                "ard_alpha_sorted_at_max_d": alphas,
                "ard_largest_alpha_gap_after_component": int(np.argmax(ratios)) + 1,
                "ard_largest_alpha_gap_ratio": float(max(ratios)),
                "ard_threshold_caveat": (
                    "the active count depends on VB_ARD_ACTIVE_RATIO; the alpha spectrum "
                    "itself is reported so the dominance structure can be read directly"
                ),
            }
        )

    if not ppca_frame.empty:
        detail.update(
            {
                "ppca_bic_argmin_d": int(ppca_frame.loc[ppca_frame["bic"].idxmin(), "d"]),
                "ppca_sigma2_at_max_d": float(
                    ppca_frame.loc[ppca_frame["d"].idxmax(), "sigma2"]
                ),
                "ppca_bic_caveat": "monotone in d at p >> n; not used for selection",
            }
        )

    return chosen, detail


def run_comparator(name: str, spec: dict, outdir: Path) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    data = load_comparator(
        spec, design_fallback_path=config.COMPARATORS[config.PRIMARY_COMPARATOR]["conditions"]
    )
    matrix = data.matrix
    seed = config.RANDOM_SEED

    print(f"[{name}] R is {data.n_codons} codons x {data.n_conditions} conditions")
    data.design.to_csv(outdir / "condition_design.csv", index=False)

    # --- residual structure -----------------------------------------------
    dependence = condition_dependence(matrix, config.N_PERMUTATIONS, seed)
    structure, spectrum = low_dimensional_structure(
        matrix, config.N_PERMUTATIONS, seed + 1, config.MAX_COMPONENTS
    )
    correlations, corr_frame = condition_correlations(matrix, data.design)
    spectrum.to_csv(outdir / "spectrum_parallel_analysis.csv", index=False)
    corr_frame.to_csv(outdir / "condition_correlation_before.csv")
    print(
        f"[{name}] condition main effect = "
        f"{dependence['fraction_variance_condition_main_effect']:.4f} "
        f"(p={dependence['permutation_p_value']:.4g}); "
        f"{structure['n_components_above_permutation_null']} component(s) above null"
    )

    # --- latent estimators ------------------------------------------------
    fits: dict[tuple[str, int], object] = {}
    for d in config.D_GRID:
        fits[("pca", d)] = fit_pca(matrix, d)
        fits[("ppca_em", d)] = fit_ppca_em(matrix, d, config.EM_MAX_ITER, config.EM_TOL, seed)
        fits[("vbpca_ard", d)] = fit_vbpca_ard(matrix, d, config.VB_MAX_ITER, config.VB_TOL, seed)

    vae_rows = []
    for d in config.VAE_D_GRID:
        fit = fit_vae(
            matrix,
            d,
            config.VAE_HIDDEN,
            config.VAE_EPOCHS,
            config.VAE_LR,
            config.VAE_WEIGHT_DECAY,
            config.VAE_RESTARTS,
            seed,
        )
        if fit is None:
            break
        fits[("vae", d)] = fit
        vae_rows.append({"d": d, **{k: v for k, v in fit.extra.items() if k != "fitted_centred"}})
    if vae_rows:
        pd.DataFrame(vae_rows).to_csv(outdir / "vae_training_summary.csv", index=False)

    ard_rows = []
    for d in config.D_GRID:
        fit = fits[("vbpca_ard", d)]
        alpha = np.asarray(fit.extra["alpha"])
        ard_rows.append(
            {
                "d_requested": d,
                "n_active_components": count_active_ard(alpha, config.VB_ARD_ACTIVE_RATIO),
                "noise_variance": fit.extra["noise_variance"],
                "vb_iterations": fit.extra["vb_iterations"],
                "alpha_sorted": ";".join(f"{a:.6g}" for a in alpha),
            }
        )
    ard_frame = pd.DataFrame(ard_rows)
    ard_frame.to_csv(outdir / "vbpca_ard_components.csv", index=False)

    ppca_rows = [
        {
            "d": d,
            "sigma2": fits[("ppca_em", d)].extra["sigma2"],
            "loglik": fits[("ppca_em", d)].extra["loglik"],
            "bic": fits[("ppca_em", d)].extra["bic"],
            "n_free_params": fits[("ppca_em", d)].extra["n_free_params"],
            "em_iterations": fits[("ppca_em", d)].extra["em_iterations"],
        }
        for d in config.D_GRID
    ]
    ppca_frame = pd.DataFrame(ppca_rows)
    ppca_frame.to_csv(outdir / "ppca_em_model_selection.csv", index=False)

    # --- validation -------------------------------------------------------
    variance_rows = [variance_accounting(matrix, fit) for fit in fits.values()]
    variance_frame = pd.DataFrame(variance_rows).sort_values(["estimator", "d"])
    variance_frame.to_csv(outdir / "variance_accounting.csv", index=False)

    corr_rows = [residual_correlation_reduction(matrix, fit) for fit in fits.values()]
    corr_reduction = pd.DataFrame(corr_rows).sort_values(["estimator", "d"])
    corr_reduction.to_csv(outdir / "residual_correlation_reduction.csv", index=False)

    bcv = bi_cross_validation(
        matrix, config.D_GRID, config.N_SPLIT_HALF, seed + 2
    )
    bcv.to_csv(outdir / "bi_cross_validation.csv", index=False)
    d_star, selection = _select_dimension(bcv, structure, ard_frame, ppca_frame)
    print(
        f"[{name}] selected d = {d_star} by permutation test; "
        f"bcv argmax was d = {selection.get('bcv_argmax_d')}"
    )

    boot_summary, boot_conditions = bootstrap_stability(
        matrix, d_star, config.N_BOOTSTRAP, seed + 3
    )
    boot_conditions.insert(0, "condition_id", data.condition_ids)
    boot_conditions.to_csv(outdir / "bootstrap_condition_stability.csv", index=False)

    reference = fits[("pca", d_star)]
    coherence = design_coherence(
        reference.scores, data.design, GROUPINGS, config.N_PERMUTATIONS, seed + 4
    )
    nesting = pd.DataFrame(_nesting_check(data.design, GROUPINGS))
    coherence = coherence.merge(
        nesting[["grouping", "n_levels_confined_to_one_dataset", "fully_nested_in_dataset"]],
        on="grouping",
        how="left",
    )
    coherence.to_csv(outdir / "design_coherence.csv", index=False)

    # Protocol structure with the batch axis held fixed.
    within_rows = []
    for dataset, group in data.design.groupby("dataset"):
        idx = group.index.to_numpy()
        if idx.size < MIN_CONDITIONS_FOR_WITHIN_DATASET:
            continue
        sub_matrix = matrix[:, idx]
        sub_design = data.design.iloc[idx].reset_index(drop=True)
        sub_dependence = condition_dependence(sub_matrix, config.N_PERMUTATIONS, seed + 5)
        sub_structure, sub_spectrum = low_dimensional_structure(
            sub_matrix, config.N_PERMUTATIONS, seed + 6, min(config.MAX_COMPONENTS, idx.size - 1)
        )
        sub_d = max(1, int(sub_structure["n_components_above_permutation_null"]))
        sub_fit = fit_pca(sub_matrix, sub_d)
        sub_variance = variance_accounting(sub_matrix, sub_fit)
        sub_coherence = design_coherence(
            sub_fit.scores,
            sub_design,
            WITHIN_DATASET_GROUPINGS,
            config.N_PERMUTATIONS,
            seed + 7,
        )
        sub_boot, _ = bootstrap_stability(sub_matrix, sub_d, config.N_BOOTSTRAP, seed + 8)
        sub_spectrum.to_csv(outdir / f"within_{dataset}_spectrum.csv", index=False)
        sub_coherence.insert(0, "dataset", dataset)
        sub_coherence.to_csv(outdir / f"within_{dataset}_design_coherence.csv", index=False)
        sub_scores = pd.DataFrame(
            sub_fit.scores, columns=[f"Z{k + 1}" for k in range(sub_d)]
        )
        sub_scores.insert(0, "condition_id", sub_design["condition_id"])
        sub_scores = sub_scores.merge(sub_design, on="condition_id", how="left")
        sub_scores.to_csv(outdir / f"within_{dataset}_latent_scores_Zc.csv", index=False)
        within_rows.append(
            {
                "dataset": dataset,
                "n_conditions": int(idx.size),
                "selected_d": sub_d,
                "fraction_variance_condition_main_effect": sub_dependence[
                    "fraction_variance_condition_main_effect"
                ],
                "condition_permutation_p_value": sub_dependence["permutation_p_value"],
                "n_components_above_permutation_null": int(
                    sub_structure["n_components_above_permutation_null"]
                ),
                "explained_variance_ratio_pc1": sub_structure["explained_variance_ratio_pc1"],
                "latent_gain_vs_offset_only": sub_variance["latent_gain_vs_offset_only"],
                "mean_subspace_cosine_bootstrap": sub_boot["mean_subspace_cosine"],
                "coherence": sub_coherence.to_dict(orient="records"),
            }
        )

    # --- artefacts of the selected model ---------------------------------
    scores = pd.DataFrame(
        reference.scores, columns=[f"Z{k + 1}" for k in range(d_star)]
    )
    scores.insert(0, "condition_id", data.condition_ids)
    scores = scores.merge(data.design, on="condition_id", how="left")
    scores.to_csv(outdir / "latent_scores_Zc.csv", index=False)

    loadings = pd.DataFrame(
        reference.loadings, columns=[f"l{k + 1}" for k in range(d_star)]
    )
    loadings.insert(0, "codon", data.codons)
    loadings.insert(1, "amino_acid", data.amino_acids)
    loadings.insert(2, "codon_offset_m_i", reference.offset)
    loadings.to_csv(outdir / "latent_loadings.csv", index=False)

    xc = matrix.T - reference.offset
    after = xc - reference.latent_term()
    residual_after = pd.DataFrame(after.T, columns=[f"xi::{c}" for c in data.condition_ids])
    residual_after.insert(0, "codon", data.codons)
    residual_after.insert(1, "amino_acid", data.amino_acids)
    residual_after.to_csv(outdir / "residuals_after_latent.csv", index=False)

    selected_variance = variance_frame[
        (variance_frame["estimator"] == "pca") & (variance_frame["d"] == d_star)
    ].iloc[0].to_dict()
    selected_corr = corr_reduction[
        (corr_reduction["estimator"] == "pca") & (corr_reduction["d"] == d_star)
    ].iloc[0].to_dict()

    summary = {
        "comparator": name,
        "comparator_description": spec["description"],
        "residual_matrix_shape": [data.n_codons, data.n_conditions],
        "baseline_fit_summary_keys": {
            k: data.baseline_summary.get(k)
            for k in (
                "n_observations",
                "n_conditions",
                "mu",
                "var_residual_codon_condition",
                "mse_residual_codon_condition",
                "mean_residual_codon_condition",
                "r2_codon_condition_means",
            )
        },
        "step_1_8_6_condition_dependence": dependence,
        "step_1_8_6_low_dimensional_structure": structure,
        "step_1_8_6_condition_correlations": correlations,
        "step_1_8_7_dimension_selection": selection,
        "step_1_8_7_vbpca_active_components_at_d8": int(
            ard_frame.loc[ard_frame["d_requested"] == max(config.D_GRID), "n_active_components"].iloc[0]
        ),
        "step_1_8_7_ppca_bic_best_d": int(ppca_frame.loc[ppca_frame["bic"].idxmin(), "d"]),
        "step_1_8_8_a_variance_selected_model": selected_variance,
        "step_1_8_8_b_correlation_selected_model": selected_corr,
        "step_1_8_8_c_generalisation": bcv.to_dict(orient="records"),
        "step_1_8_8_d_bootstrap": boot_summary,
        "complementary_design_coherence": coherence.to_dict(orient="records"),
        "complementary_grouping_nesting_check": nesting.to_dict(orient="records"),
        "complementary_within_dataset_analysis": within_rows,
        "seed": seed,
        "n_permutations": config.N_PERMUTATIONS,
        "n_bootstrap": config.N_BOOTSTRAP,
        "n_bi_cross_validation_splits": config.N_SPLIT_HALF,
    }
    _write_json(outdir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit the latent condition representation Z_c on the frozen residual matrix."
    )
    parser.add_argument(
        "--comparators",
        nargs="+",
        choices=list(config.COMPARATORS),
        default=list(config.COMPARATORS),
    )
    args = parser.parse_args()

    all_summaries = {}
    for name in args.comparators:
        spec = config.COMPARATORS[name]
        summary = run_comparator(name, spec, config.OUTPUT_ROOT / name)
        all_summaries[name] = summary

    _write_json(config.OUTPUT_ROOT / "summary_all_comparators.json", all_summaries)
    print(json.dumps(all_summaries[config.PRIMARY_COMPARATOR], indent=2)[:2000])


if __name__ == "__main__":
    main()
