import json
import sys
from collections import Counter
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "modelo_baseline"

import numpy as np
import pandas as pd

from . import config
from .annotation import load_cds_records, load_genome_length
from .fit import add_gene_observations, empty_accumulators, fit_additive_codon_model
from .observables import gene_codon_occupancy, relative_occupancy_and_rate
from .run import DATASETS, _codons, _condition_id
from .tracks import (
    group_conditions,
    list_ribo_wigs,
    load_stranded_tracks,
    parse_soft_samples,
)

OUTPUT_DIR = config.ROOT / "modelo_baseline" / "saidas_robustez_freeze_mg"
POOLED_EFFECTS = config.ROOT / "modelo_baseline" / "saidas" / "codon_effects.csv"

FIT_LABELS = frozenset({"freeze_Mg", "freeze_Mg_ctrl"})

SER_CODONS = ("TCA", "TCC", "TCG", "TCT", "AGC", "AGT")
GLY_CODONS = ("GGA", "GGC", "GGG", "GGT")
PRO_CODONS = ("CCA", "CCC", "CCG", "CCT")
ILE_CODONS = ("ATA", "ATC", "ATT")
RARE_CODONS = ("AGG", "AGA", "ATA")


def _spearman(x, y) -> dict:
    frame = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(frame) < 5:
        return {"n": int(len(frame)), "rho": None, "note": "too_few_pairs"}
    rho = float(frame["x"].corr(frame["y"], method="spearman"))
    return {"n": int(len(frame)), "rho": rho}


def _mean_rho(matrix: pd.DataFrame, cond_id: str, codons: tuple[str, ...]) -> float:
    col = f"rho::{cond_id}"
    sub = matrix.loc[matrix["codon"].isin(codons), col]
    return float(sub.mean())


def _mean_k(matrix: pd.DataFrame, cond_id: str, codons: tuple[str, ...]) -> float:
    col = f"k_obs::{cond_id}"
    sub = matrix.loc[matrix["codon"].isin(codons), col]
    return float(sub.mean())


def _codon_usage(cds_records: list[dict]) -> pd.DataFrame:
    counts = Counter()
    for record in cds_records:
        for codon in _codons(record["seq"]):
            if codon in config.STOP_CODONS:
                continue
            if codon in config.STANDARD_GENETIC_CODE:
                counts[codon] += 1
    total = sum(counts.values())
    rows = []
    for codon, n in sorted(counts.items()):
        rows.append(
            {
                "codon": codon,
                "amino_acid": config.STANDARD_GENETIC_CODE[codon],
                "cds_count": n,
                "cds_frequency": n / total,
                "inv_frequency": total / n,
            }
        )
    return pd.DataFrame(rows)


def _apply_kseq(eval_acc, k_seq: dict) -> pd.DataFrame:
    condition_ids = sorted({cond for _, cond in eval_acc["n_kc"]})
    rows = []
    for codon in sorted(k_seq):
        row = {
            "codon": codon,
            "amino_acid": config.STANDARD_GENETIC_CODE[codon],
            "k_seq_freeze_mg": k_seq[codon],
        }
        for cond in condition_ids:
            n_kc = eval_acc["n_kc"].get((codon, cond), 0)
            if n_kc == 0:
                k_obs_c = np.nan
                rho_c = np.nan
            else:
                k_obs_c = eval_acc["sum_kc"][(codon, cond)] / n_kc
                n_rho = eval_acc["n_rho_kc"].get((codon, cond), 0)
                rho_c = eval_acc["sum_rho_kc"][(codon, cond)] / n_rho if n_rho else np.nan
            row[f"k_obs::{cond}"] = k_obs_c
            row[f"R::{cond}"] = k_obs_c - k_seq[codon]
            row[f"rho::{cond}"] = rho_c
            row[f"n::{cond}"] = n_kc
        rows.append(row)
    return pd.DataFrame(rows)


def run() -> dict:
    genome_len = load_genome_length()
    cds_records = load_cds_records()
    fit_acc = empty_accumulators()
    eval_acc = empty_accumulators()
    condition_rows = []

    for dataset, spec in DATASETS.items():
        members = list_ribo_wigs(spec["tar"], dataset)
        conditions = group_conditions(members)
        soft = parse_soft_samples(spec["soft"]) if spec["soft"].exists() else {}
        asite_shift = config.ASITE_SHIFT_NT[dataset]
        for condition in conditions:
            cond_id = _condition_id(dataset, condition["gsm"], condition["label"])
            used_in_fit = dataset == "GSE119104" and condition["label"] in FIT_LABELS
            print(f"loading {cond_id} fit={used_in_fit}", flush=True)
            tracks = load_stranded_tracks(spec["tar"], condition, genome_len)
            n_kept = 0
            n_obs = 0
            for record in cds_records:
                occupancy = gene_codon_occupancy(record, tracks, genome_len, asite_shift)
                if occupancy is None:
                    continue
                rho, k_obs, qc = relative_occupancy_and_rate(occupancy)
                if not qc["kept"]:
                    continue
                n_kept += 1
                n_obs += int(np.isfinite(k_obs).sum())
                add_gene_observations(eval_acc, _codons(record["seq"]), rho, k_obs, cond_id)
                if used_in_fit:
                    add_gene_observations(fit_acc, _codons(record["seq"]), rho, k_obs, cond_id)
            condition_rows.append(
                {
                    "condition_id": cond_id,
                    "dataset": dataset,
                    "gsm": condition["gsm"],
                    "label": condition["label"],
                    "used_in_kseq_fit": used_in_fit,
                    "title": soft.get(condition["gsm"], {}).get("title", ""),
                    "n_genes_kept": n_kept,
                    "n_rate_observations": n_obs,
                }
            )
            del tracks

    fit = fit_additive_codon_model(fit_acc)
    matrix = _apply_kseq(eval_acc, fit["k_seq"])
    usage = _codon_usage(cds_records)

    freeze_ids = [c for c in fit["condition_ids"]]
    rho_freeze = matrix[
        [col for col in matrix.columns if col.startswith("rho::") and any(x in col for x in freeze_ids)]
    ].mean(axis=1)
    effects = fit["effects"].copy()
    effects = effects.merge(usage, on=["codon", "amino_acid"], how="left")
    effects["pause_rho_freeze_mg"] = rho_freeze.to_numpy()
    effects["dwell_from_mean_rho"] = 1.0 / effects["pause_rho_freeze_mg"]

    pooled = pd.read_csv(POOLED_EFFECTS)
    pooled = pooled.rename(columns={"k_seq": "k_seq_pooled", "beta_codon": "beta_pooled"})
    effects = effects.merge(pooled[["codon", "k_seq_pooled", "beta_pooled"]], on="codon", how="left")

    cond_map = {row["label"]: row["condition_id"] for row in condition_rows if row["dataset"] == "GSE119104"}
    freeze_id = cond_map["freeze_Mg"]
    freeze_ctrl_id = cond_map["freeze_Mg_ctrl"]
    filter_id = cond_map["filter_Mg"]
    mpc_id = cond_map["freeze_Mg_MPC"]
    cm_id = cond_map["media_Cm"]

    alignment = {
        "spearman_kseq_freeze_vs_pooled": _spearman(effects["k_seq"], effects["k_seq_pooled"]),
        "spearman_pause_rho_vs_inv_frequency": _spearman(
            effects["pause_rho_freeze_mg"], effects["inv_frequency"]
        ),
        "spearman_kseq_vs_frequency": _spearman(effects["k_seq"], effects["cds_frequency"]),
        "spearman_kseq_vs_pause_rho": _spearman(effects["k_seq"], effects["pause_rho_freeze_mg"]),
        "proline_all_beta_negative": bool((effects.loc[effects["codon"].isin(PRO_CODONS), "beta_codon"] < 0).all()),
        "mean_kseq_proline": float(effects.loc[effects["codon"].isin(PRO_CODONS), "k_seq"].mean()),
        "mean_kseq_rare_agg_aga_ata": float(effects.loc[effects["codon"].isin(RARE_CODONS), "k_seq"].mean()),
        "mean_kseq_all": float(effects["k_seq"].mean()),
        "mean_pause_rho_proline": float(effects.loc[effects["codon"].isin(PRO_CODONS), "pause_rho_freeze_mg"].mean()),
        "mean_pause_rho_all": float(effects["pause_rho_freeze_mg"].mean()),
        "filter_minus_freeze_mean_rho_ser": _mean_rho(matrix, filter_id, SER_CODONS)
        - 0.5
        * (
            _mean_rho(matrix, freeze_id, SER_CODONS)
            + _mean_rho(matrix, freeze_ctrl_id, SER_CODONS)
        ),
        "filter_minus_freeze_mean_rho_gly": _mean_rho(matrix, filter_id, GLY_CODONS)
        - 0.5
        * (
            _mean_rho(matrix, freeze_id, GLY_CODONS)
            + _mean_rho(matrix, freeze_ctrl_id, GLY_CODONS)
        ),
        "mpc_minus_freeze_mean_rho_ile": _mean_rho(matrix, mpc_id, ILE_CODONS)
        - 0.5
        * (
            _mean_rho(matrix, freeze_id, ILE_CODONS)
            + _mean_rho(matrix, freeze_ctrl_id, ILE_CODONS)
        ),
        "media_cm_minus_freeze_mean_rho_ser": _mean_rho(matrix, cm_id, SER_CODONS)
        - 0.5
        * (
            _mean_rho(matrix, freeze_id, SER_CODONS)
            + _mean_rho(matrix, freeze_ctrl_id, SER_CODONS)
        ),
        "media_cm_minus_freeze_mean_rho_gly": _mean_rho(matrix, cm_id, GLY_CODONS)
        - 0.5
        * (
            _mean_rho(matrix, freeze_id, GLY_CODONS)
            + _mean_rho(matrix, freeze_ctrl_id, GLY_CODONS)
        ),
        "mean_k_ile_freeze": 0.5
        * (_mean_k(matrix, freeze_id, ILE_CODONS) + _mean_k(matrix, freeze_ctrl_id, ILE_CODONS)),
        "mean_k_ile_mpc": _mean_k(matrix, mpc_id, ILE_CODONS),
        "mean_k_ser_freeze": 0.5
        * (_mean_k(matrix, freeze_id, SER_CODONS) + _mean_k(matrix, freeze_ctrl_id, SER_CODONS)),
        "mean_k_ser_filter": _mean_k(matrix, filter_id, SER_CODONS),
        "mean_k_gly_freeze": 0.5
        * (_mean_k(matrix, freeze_id, GLY_CODONS) + _mean_k(matrix, freeze_ctrl_id, GLY_CODONS)),
        "mean_k_gly_filter": _mean_k(matrix, filter_id, GLY_CODONS),
        "expected_from_mohammad": {
            "pause_rho_vs_inv_usage_sign": "positive",
            "filter_ser_gly_pause_vs_freeze": "rho_filter > rho_freeze",
            "mupirocin_ile_pause_vs_freeze": "rho_mpc > rho_freeze",
            "proline_slower_than_mean": True,
            "rare_codons_slower_than_mean": True,
        },
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(condition_rows).to_csv(OUTPUT_DIR / "conditions.csv", index=False)
    effects.to_csv(OUTPUT_DIR / "codon_effects_freeze_mg.csv", index=False)
    usage.to_csv(OUTPUT_DIR / "codon_usage_cds.csv", index=False)
    residual_cols = ["codon", "amino_acid"] + [c for c in matrix.columns if c.startswith("R::")]
    rho_cols = ["codon", "amino_acid"] + [c for c in matrix.columns if c.startswith("rho::")]
    kobs_cols = ["codon", "amino_acid"] + [c for c in matrix.columns if c.startswith("k_obs::")]
    matrix[residual_cols].to_csv(OUTPUT_DIR / "residuals_using_freeze_kseq.csv", index=False)
    matrix[rho_cols].to_csv(OUTPUT_DIR / "relative_occupancy_codon_condition.csv", index=False)
    matrix[kobs_cols].to_csv(OUTPUT_DIR / "kobs_codon_condition.csv", index=False)
    matrix.to_csv(OUTPUT_DIR / "codon_condition_matrix.csv", index=False)

    summary = {
        **fit["summary"],
        "fit_libraries": sorted(FIT_LABELS),
        "fit_condition_ids": freeze_ids,
        "n_eval_conditions": len({c for _, c in eval_acc["n_kc"]}),
        "output_dir": OUTPUT_DIR.relative_to(config.ROOT).as_posix(),
        "alignment": alignment,
    }
    (OUTPUT_DIR / "fit_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    run()
