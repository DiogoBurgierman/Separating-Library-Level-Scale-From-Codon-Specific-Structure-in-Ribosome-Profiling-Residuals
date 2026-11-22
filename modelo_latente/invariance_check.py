"""Invariance of Z_c to the choice of codon-only sequence baseline.

Two codon-only baselines (pooled over 14 libraries, and freeze-only) differ by a
function of codon identity alone, so their residual matrices differ, along each
codon row, by a constant that the offset m_i absorbs. This script measures that
numerically (it was previously verified by hand and not stored):

  - max over codons of the standard deviation, across conditions, of R_freeze - R_pooled
  - max absolute difference between the two m_i-centred matrices

Reproduce with:

    python -m modelo_latente.invariance_check

Writes modelo_latente/saidas/invariance_check.json.
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


def _load(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path).set_index("codon")
    cols = [c for c in frame.columns if c.startswith("R::")]
    out = frame[cols]
    out.columns = [c[3:] for c in cols]
    return out


def _display(path: Path) -> str:
    try:
        return path.resolve().relative_to(config.ROOT).as_posix()
    except ValueError:
        return str(path)


def check(pooled_path: Path, freeze_path: Path) -> dict:
    pooled = _load(pooled_path)
    freeze = _load(freeze_path).reindex(index=pooled.index, columns=pooled.columns)
    if freeze.isna().any().any():
        raise ValueError("codon or condition sets differ between comparators")
    diff = (freeze - pooled).to_numpy()
    row_sd = diff.std(axis=1)
    pc = pooled.to_numpy() - pooled.to_numpy().mean(axis=1, keepdims=True)
    fc = freeze.to_numpy() - freeze.to_numpy().mean(axis=1, keepdims=True)
    return {
        "max_row_sd_of_difference": float(row_sd.max()),
        "max_abs_difference_offset_centred": float(np.abs(fc - pc).max()),
        "max_abs_difference_raw": float(np.abs(diff).max()),
        "n_codons": int(diff.shape[0]),
        "n_conditions": int(diff.shape[1]),
        "pooled_residuals": _display(pooled_path),
        "freeze_residuals": _display(freeze_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pooled", type=Path, default=config.COMPARATORS["pooled"]["residuals"])
    parser.add_argument("--freeze", type=Path, default=config.COMPARATORS["freeze"]["residuals"])
    parser.add_argument("--out", type=Path, default=config.OUTPUT_ROOT / "invariance_check.json")
    args = parser.parse_args()
    result = check(args.pooled, args.freeze)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
