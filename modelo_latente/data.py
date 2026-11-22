"""Loading of the frozen residual matrix and of the condition design labels.

The residual matrix is read exactly as written by `modelo_baseline`; no cell is
recomputed, rescaled, or filtered here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


RESIDUAL_PREFIX = "R::"


@dataclass
class ResidualData:
    """R_ic as codons x conditions, plus the experimental design of each column."""

    codons: list[str]
    amino_acids: list[str]
    condition_ids: list[str]
    matrix: np.ndarray  # shape (n_codons, n_conditions)
    design: pd.DataFrame  # one row per condition, aligned with condition_ids
    baseline_summary: dict

    @property
    def n_codons(self) -> int:
        return self.matrix.shape[0]

    @property
    def n_conditions(self) -> int:
        return self.matrix.shape[1]


def _parse_characteristics(text: str) -> dict[str, str]:
    """GEO characteristics are ' | '-separated 'key: value' fragments."""
    fields: dict[str, str] = {}
    if not isinstance(text, str):
        return fields
    for part in text.split("|"):
        part = part.strip()
        if not part:
            continue
        key, sep, value = part.partition(":")
        if not sep:
            continue
        key = key.strip().lower()
        value = value.strip()
        if not key:
            # GSE119104 writes the lysis buffer as ': lysis buffer containing: Cm'
            key, _, value = value.partition(":")
            key = key.strip().lower()
            value = value.strip()
        if key:
            fields[key] = value
    return fields


def _derive_design_labels(row: pd.Series) -> dict[str, str]:
    fields = _parse_characteristics(row.get("characteristics", ""))
    harvest = fields.get("harvesting method", "").lower()
    drug = fields.get("drug treatment in culture", "").lower()
    lysis = fields.get("lysis buffer containing", "").lower()

    if "freeze" in harvest:
        harvest_label = "direct_freeze"
    elif "filtration" in harvest:
        harvest_label = "filtration"
    else:
        harvest_label = "unreported"

    if "mupirocin" in drug:
        drug_label = "mupirocin"
    elif "chloramphenicol" in drug:
        drug_label = "chloramphenicol"
    elif drug in {"none", ""}:
        drug_label = "none" if drug == "none" else "unreported"
    else:
        drug_label = drug.replace(" ", "_")

    if "mgcl2" in lysis:
        lysis_label = "MgCl2_150mM"
    elif "nacl" in lysis:
        lysis_label = "NaCl_1M"
    elif lysis == "cm":
        lysis_label = "Cm"
    else:
        lysis_label = "unreported"

    return {
        "harvest": harvest_label,
        "drug": drug_label,
        "lysis": lysis_label,
    }


def load_residuals(
    residual_path: Path,
    conditions_path: Path,
    fit_summary_path: Path,
    design_fallback_path: Path | None = None,
) -> ResidualData:
    frame = pd.read_csv(residual_path)
    residual_cols = [c for c in frame.columns if c.startswith(RESIDUAL_PREFIX)]
    if not residual_cols:
        raise ValueError(f"no '{RESIDUAL_PREFIX}' columns in {residual_path}")

    condition_ids = [c[len(RESIDUAL_PREFIX) :] for c in residual_cols]
    matrix = frame[residual_cols].to_numpy(dtype=float)
    if not np.isfinite(matrix).all():
        raise ValueError(f"non-finite residuals in {residual_path}")

    conditions = pd.read_csv(conditions_path)
    if "characteristics" not in conditions.columns and design_fallback_path is not None:
        # The robustness refit writes a reduced conditions.csv without the GEO
        # characteristics field. The condition ids are the same libraries, so the
        # experimental design is taken from the full table rather than degenerating
        # every protocol label to 'unreported'.
        fallback = pd.read_csv(design_fallback_path)
        if "characteristics" in fallback.columns:
            conditions = conditions.merge(
                fallback[["condition_id", "characteristics"]],
                on="condition_id",
                how="left",
            )
    conditions = conditions.set_index("condition_id").reindex(condition_ids)
    if conditions["dataset"].isna().any():
        missing = [c for c, ok in zip(condition_ids, conditions["dataset"].notna()) if not ok]
        raise ValueError(f"conditions.csv is missing {missing}")

    design_rows = []
    for condition_id, row in conditions.iterrows():
        labels = _derive_design_labels(row)
        design_rows.append(
            {
                "condition_id": condition_id,
                "dataset": row["dataset"],
                "gsm": row["gsm"],
                "label": row["label"],
                **labels,
                "n_genes_kept": row.get("n_genes_kept"),
                "n_rate_observations": row.get("n_rate_observations"),
            }
        )
    design = pd.DataFrame(design_rows)
    design["protocol_family"] = (
        design["dataset"] + "|" + design["harvest"] + "|" + design["lysis"]
    )

    summary = json.loads(Path(fit_summary_path).read_text(encoding="utf-8"))

    return ResidualData(
        codons=frame["codon"].tolist(),
        amino_acids=frame["amino_acid"].tolist(),
        condition_ids=condition_ids,
        matrix=matrix,
        design=design,
        baseline_summary=summary,
    )


def load_comparator(spec: dict, design_fallback_path: Path | None = None) -> ResidualData:
    return load_residuals(
        residual_path=spec["residuals"],
        conditions_path=spec["conditions"],
        fit_summary_path=spec["fit_summary"],
        design_fallback_path=design_fallback_path,
    )
