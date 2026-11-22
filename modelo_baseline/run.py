import argparse
import json
import sys
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
from .tracks import (
    group_conditions,
    list_ribo_wigs,
    load_stranded_tracks,
    parse_soft_samples,
)


DATASETS = {
    "GSE119104": {
        "tar": config.GSE119104_TAR,
        "soft": config.GSE119104_SOFT,
    },
    "GSE77617": {
        "tar": config.GSE77617_TAR,
        "soft": config.GSE77617_SOFT,
    },
}


def _condition_id(dataset: str, gsm: str, label: str) -> str:
    return f"{dataset}|{gsm}|{label}"


def _write_json(path: Path, payload: dict):
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def run(datasets: list[str] | None = None) -> dict:
    selected = datasets or list(DATASETS)
    genome_len = load_genome_length()
    cds_records = load_cds_records()
    acc = empty_accumulators()
    condition_rows = []
    qc_rows = []

    for dataset in selected:
        spec = DATASETS[dataset]
        if not spec["tar"].exists():
            raise FileNotFoundError(spec["tar"])
        members = list_ribo_wigs(spec["tar"], dataset)
        conditions = group_conditions(members)
        soft = parse_soft_samples(spec["soft"]) if spec["soft"].exists() else {}
        asite_shift = config.ASITE_SHIFT_NT[dataset]

        for condition in conditions:
            cond_id = _condition_id(dataset, condition["gsm"], condition["label"])
            meta = soft.get(condition["gsm"], {})
            print(f"loading {cond_id}", flush=True)
            tracks = load_stranded_tracks(spec["tar"], condition, genome_len)
            n_kept = 0
            n_low = 0
            n_other = 0
            n_obs = 0
            for record in cds_records:
                occupancy = gene_codon_occupancy(record, tracks, genome_len, asite_shift)
                if occupancy is None:
                    n_other += 1
                    continue
                rho, k_obs, qc = relative_occupancy_and_rate(occupancy)
                if not qc["kept"]:
                    if qc["reason"] == "low_coverage":
                        n_low += 1
                    else:
                        n_other += 1
                    continue
                n_kept += 1
                n_obs += int(np.isfinite(k_obs).sum())
                add_gene_observations(acc, _codons(record["seq"]), rho, k_obs, cond_id)
            condition_rows.append(
                {
                    "condition_id": cond_id,
                    "dataset": dataset,
                    "gsm": condition["gsm"],
                    "label": condition["label"],
                    "title": meta.get("title", ""),
                    "characteristics": " | ".join(meta.get("characteristics", [])),
                    "asite_shift_nt": asite_shift,
                    "n_genes_kept": n_kept,
                    "n_genes_low_coverage": n_low,
                    "n_genes_other_drop": n_other,
                    "n_rate_observations": n_obs,
                }
            )
            qc_rows.append(
                {
                    "condition_id": cond_id,
                    "dataset": dataset,
                    "n_cds_annotated": len(cds_records),
                    "n_genes_kept": n_kept,
                    "n_genes_low_coverage": n_low,
                    "n_genes_other_drop": n_other,
                    "n_rate_observations": n_obs,
                }
            )
            del tracks

    fit = fit_additive_codon_model(acc)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    conditions_df = pd.DataFrame(condition_rows)
    qc_df = pd.DataFrame(qc_rows)
    conditions_df.to_csv(config.OUTPUT_DIR / "conditions.csv", index=False)
    qc_df.to_csv(config.OUTPUT_DIR / "qc_per_condition.csv", index=False)
    fit["effects"].to_csv(config.OUTPUT_DIR / "codon_effects.csv", index=False)

    kobs_cols = ["codon", "amino_acid"] + [
        col for col in fit["matrix"].columns if col.startswith("k_obs::")
    ]
    residual_cols = ["codon", "amino_acid"] + [
        col for col in fit["matrix"].columns if col.startswith("R::")
    ]
    rho_cols = ["codon", "amino_acid"] + [
        col for col in fit["matrix"].columns if col.startswith("rho::")
    ]
    n_cols = ["codon", "amino_acid"] + [
        col for col in fit["matrix"].columns if col.startswith("n::")
    ]
    fit["matrix"][kobs_cols].to_csv(config.OUTPUT_DIR / "kobs_codon_condition.csv", index=False)
    fit["matrix"][residual_cols].to_csv(
        config.OUTPUT_DIR / "residuals_codon_condition.csv", index=False
    )
    fit["matrix"][rho_cols].to_csv(
        config.OUTPUT_DIR / "relative_occupancy_codon_condition.csv", index=False
    )
    fit["matrix"][n_cols].to_csv(config.OUTPUT_DIR / "n_obs_codon_condition.csv", index=False)
    fit["matrix"].to_csv(config.OUTPUT_DIR / "codon_condition_matrix.csv", index=False)

    summary = {
        **fit["summary"],
        "genome_accession": config.GENOME_ACCESSION,
        "genome_length": genome_len,
        "n_cds_used_annotation": len(cds_records),
        "exclude_start_nt": config.EXCLUDE_START_NT,
        "exclude_stop_nt": config.EXCLUDE_STOP_NT,
        "min_gene_mean_occupancy": config.MIN_GENE_MEAN_OCCUPANCY,
        "asite_shift_nt": config.ASITE_SHIFT_NT,
        "datasets": selected,
        "output_dir": config.OUTPUT_DIR.relative_to(config.ROOT).as_posix(),
    }
    _write_json(config.OUTPUT_DIR / "fit_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def _codons(seq: str) -> list[str]:
    return [seq[i : i + 3] for i in range(0, len(seq), 3)]


def main():
    parser = argparse.ArgumentParser(
        description="Fit the parsimonious codon baseline k_seq(i) = mu + beta_codon(i)."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=list(DATASETS),
        default=list(DATASETS),
    )
    args = parser.parse_args()
    run(args.datasets)


if __name__ == "__main__":
    main()
