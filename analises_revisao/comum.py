"""Shared base for the supplementary analyses (R1 to R6).

Rules that every script in this package follows:

- the outputs of `modelo_baseline/saidas*` and `modelo_latente/saidas` are frozen inputs:
  nothing here rewrites them (the article's number verifier remains valid);
- each analysis writes only to `analises_revisao/saidas/<item>/`;
- every stochastic procedure uses `semente(item, k)`, derived from the same base seed as the
  latent stage (20260911);
- the same decomposition (`decomposicao_aditiva`) is used by every analysis, so that
  "gain over the additive model" has a single definition.

Matrix orientation: M is (codons x conditions) = (61 x 14), as in `modelo_latente`.
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = ROOT / "modelo_baseline" / "saidas"
FREEZE_DIR = ROOT / "modelo_baseline" / "saidas_robustez_freeze_mg"
LATENT_DIR = ROOT / "modelo_latente" / "saidas"
SAIDAS = ROOT / "analises_revisao" / "saidas"
CACHE = ROOT / "analises_revisao" / "_cache"  # outside git; rebuilt from the GEO .tar files

SEMENTE_BASE = 20260911  # equal to modelo_latente.config.RANDOM_SEED

SERINA = ("TCT", "TCC", "TCA", "TCG", "AGT", "AGC")
ISOLEUCINA = ("ATT", "ATC", "ATA")
PROLINA = ("CCT", "CCC", "CCA", "CCG")


def semente(item: int, k: int = 0) -> int:
    """Deterministic seed per analysis (1..6) and per procedure k."""
    return SEMENTE_BASE + 1000 * item + k


def dir_saida(item: str) -> Path:
    path = SAIDAS / item
    path.mkdir(parents=True, exist_ok=True)
    return path


def escrever_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def carregar_tabela(prefixo: str, caminho: Path | None = None):
    """Reads one quantity from the codon x condition table of the pooled baseline.

    prefixo: "R" (residual k_obs - k_seq), "k_obs", "rho" (mean of rho, zeros included)
    or "n" (number of positions with rho > 0).
    Returns (codons, amino acids, condition ids, 61 x 14 matrix).
    """
    frame = pd.read_csv(caminho or BASELINE_DIR / "codon_condition_matrix.csv")
    cols = [c for c in frame.columns if c.startswith(prefixo + "::")]
    if not cols:
        raise ValueError(f"sem colunas '{prefixo}::' em {caminho}")
    ids = [c.split("::", 1)[1] for c in cols]
    return (
        frame["codon"].tolist(),
        frame["amino_acid"].tolist(),
        ids,
        frame[cols].to_numpy(dtype=float),
    )


def carregar_desenho() -> pd.DataFrame:
    """Design labels per library, as used by the latent stage."""
    return pd.read_csv(LATENT_DIR / "pooled" / "condition_design.csv")


def decomposicao_aditiva(matriz: np.ndarray) -> dict:
    """Decomposes M_ic into a codon offset, a condition intercept, and an interaction.

        M_ic = g + m_i + a_c + D_ic ,  sum_c a_c = 0,  sum_i D_ic = sum_c D_ic = 0

    All fractions are given against the sum of squares centred on m_i (the denominator
    the article uses for the 93.0%), and the interaction also against itself.

    - `fracao_aditivo`: what the additive model m_i + a_c removes beyond m_i;
    - `fracao_posto1`: what rank one l_i z_c removes beyond m_i (the article's 93.0%);
    - `ganho_posto1_sobre_aditivo`: 1 - SS_res(rank one) / SS_res(additive);
    - `fracao_concorrencia`: Mandel row-regression model, D_ic ~ delta_i a_c,
      i.e. the same score a_c with a per-codon amplitude (1 + delta_i);
    - `interacao_pc1_*`: first component of the doubly centred matrix, which is what
      a latent factor would explain BEYOND the per-condition intercept.
    """
    m = np.asarray(matriz, dtype=float)
    n_cod, n_cond = m.shape
    offset = m.mean(axis=1)
    xc = m - offset[:, None]
    ss_offset_centred = float(np.sum(xc**2))

    a = xc.mean(axis=0)
    d = xc - a[None, :]
    ss_additive = float(n_cod * np.sum(a**2))
    ss_interaction = float(np.sum(d**2))

    u1, s1, vt1 = np.linalg.svd(xc.T, full_matrices=False)  # conditions as samples
    loadings1 = vt1[0]
    scores1 = u1[:, 0] * s1[0]
    if loadings1.sum() > 0:  # convention: same orientation as the article (negative loadings)
        loadings1, scores1 = -loadings1, -scores1
    ss_rank1 = float(s1[0] ** 2)

    ud, sd, vtd = np.linalg.svd(d.T, full_matrices=False)
    ss_int_pc1 = float(sd[0] ** 2)

    delta = d @ a / float(np.sum(a**2))
    ss_concurrence = float(np.sum(delta**2) * np.sum(a**2))

    res_additive = ss_offset_centred - ss_additive
    res_rank1 = ss_offset_centred - ss_rank1
    return {
        "n_codons": n_cod,
        "n_condicoes": n_cond,
        "offset_m_i": offset,
        "intercepto_a_c": a,
        "interacao_D": d,
        "cargas_posto1": loadings1,
        "escores_posto1": scores1,
        "cargas_interacao_pc1": vtd[0],
        "escores_interacao_pc1": ud[:, 0] * sd[0],
        "valores_singulares_interacao": sd,
        "delta_concorrencia": delta,
        "sq_centrada_em_m": ss_offset_centred,
        "sq_aditivo": ss_additive,
        "sq_interacao": ss_interaction,
        "sq_posto1": ss_rank1,
        "sq_interacao_pc1": ss_int_pc1,
        "sq_concorrencia": ss_concurrence,
        "fracao_aditivo": ss_additive / ss_offset_centred,
        "fracao_posto1": ss_rank1 / ss_offset_centred,
        "ganho_posto1_sobre_aditivo": 1.0 - res_rank1 / res_additive,
        "fracao_concorrencia_da_interacao": ss_concurrence / ss_interaction,
        "interacao_pc1_fracao_da_interacao": ss_int_pc1 / ss_interaction,
        "interacao_pc1_fracao_da_sq_centrada_em_m": ss_int_pc1 / ss_offset_centred,
        "qm_centrado_em_m": ss_offset_centred / m.size,
        "qm_apos_aditivo": res_additive / m.size,
        "qm_apos_posto1": res_rank1 / m.size,
        "qm_apos_aditivo_mais_pc1": (ss_interaction - ss_int_pc1) / m.size,
        "n_cargas_posto1_negativas": int(np.sum(loadings1 < 0)),
        "cos_cargas_posto1_com_vetor_constante": float(
            abs(loadings1.sum()) / np.sqrt(n_cod) / np.linalg.norm(loadings1)
        ),
    }


def escalares(decomp: dict) -> dict:
    """Serializable part (scalars only) of `decomposicao_aditiva`."""
    return {
        k: (v.item() if isinstance(v, np.generic) else v)
        for k, v in decomp.items()
        if not isinstance(v, np.ndarray)
    }


def pistas(dataset: str, condicao: dict, genome_len: int) -> dict:
    """Strands '+' and '-' of a library, cached as .npy (atomic write).

    `condicao` is an item of `modelo_baseline.tracks.group_conditions`. The cache only avoids
    re-reading the WIG; the content is that of `modelo_baseline.tracks.load_stranded_tracks`.
    """
    from modelo_baseline.run import DATASETS
    from modelo_baseline.tracks import load_stranded_tracks

    CACHE.mkdir(parents=True, exist_ok=True)
    out = {}
    paths = {s: CACHE / f"{dataset}_{condicao['gsm']}_{'mais' if s == '+' else 'menos'}.npy" for s in "+-"}
    if all(p.exists() for p in paths.values()):
        return {s: np.load(p) for s, p in paths.items()}
    out = load_stranded_tracks(DATASETS[dataset]["tar"], condicao, genome_len)
    for s, p in paths.items():
        tmp = p.with_name(f"{p.stem}.{uuid.uuid4().hex}.tmp.npy")
        np.save(tmp, out[s])
        os.replace(tmp, p)
    return out


def bibliotecas():
    """Iterates (dataset, condition, condition_id, asite_shift) in baseline order."""
    from modelo_baseline import config as bcfg
    from modelo_baseline.run import DATASETS, _condition_id
    from modelo_baseline.tracks import group_conditions, list_ribo_wigs

    for dataset, spec in DATASETS.items():
        for condicao in group_conditions(list_ribo_wigs(spec["tar"], dataset)):
            yield (
                dataset,
                condicao,
                _condition_id(dataset, condicao["gsm"], condicao["label"]),
                bcfg.ASITE_SHIFT_NT[dataset],
            )
