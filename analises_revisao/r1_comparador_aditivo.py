"""R1: additive comparator R_ic = m_i + a_c + xi_ic.

Question: is the rank-one latent factor (93.0%; skill 89.9%; cosine 0.99999) more than
a per-condition intercept? This script

1. reproduces the article's anchor with the SAME seeds as the latent stage (fails if they differ);
2. fits the additive model and the ladder of nested models
   m_i only  <  additive  <  rank one (= Mandel's concurrence by least squares)  <  additive + PC1 of the interaction,
   plus two "per-library scale factor" models with the same degrees of freedom as the additive one;
3. runs a parallel analysis on the doubly centred matrix, with three nulls;
4. bi-cross-validation on the same 500 partitions, against the additive model;
5. bootstrap cosine on the same 2000 resamples, separating the trivial part (constant vector);
6. shape of the interaction (scale, log, concurrence), residual correlation and positional share.

Usage (from the repository root):
    python -m analises_revisao.r1_comparador_aditivo

Outputs: analises_revisao/saidas/r1/ (CSV + summary.json). Nothing outside that folder is written.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from analises_revisao.comum import (
    BASELINE_DIR,
    FREEZE_DIR,
    ISOLEUCINA,
    LATENT_DIR,
    SERINA,
    carregar_desenho,
    carregar_tabela,
    decomposicao_aditiva,
    dir_saida,
    escalares,
    escrever_json,
    semente,
)
from modelo_latente import config as lcfg
from modelo_latente.analysis import low_dimensional_structure
from modelo_latente.models import fit_pca
from modelo_latente.validation import (
    bi_cross_validation,
    bootstrap_stability,
    residual_correlation_reduction,
    variance_accounting,
)

SEED = lcfg.RANDOM_SEED  # 20260911
SEED_BCV = SEED + 2  # modelo_latente.run: bi_cross_validation(..., seed + 2)
SEED_BOOT = SEED + 3  # modelo_latente.run: bootstrap_stability(..., seed + 3)
SEED_BOOT_WITHIN = SEED + 8  # modelo_latente.run: sub_boot with seed + 8
N_SPLIT = lcfg.N_SPLIT_HALF  # 500
N_BOOT = lcfg.N_BOOTSTRAP  # 2000
N_PERM = lcfg.N_PERMUTATIONS  # 2000
COL_HOLDOUT = 4

# new seeds for this analysis: comum.semente(1, k)
K_NULO = {"linhas": 0, "sinais": 1, "colunas": 2}
K_REF_RUIDO_BOOT = 3
K_NULO_LOG = 4
K_NULO_WITHIN = {"linhas": 5, "sinais": 6, "colunas": 7}
K_BCV_WITHIN = 8
K_PODER_NULO_ARTIGO = 9
N_REF_RUIDO = 200  # null matrices x bootstraps per matrix in the noise reference
N_PODER = 20  # purely additive matrices in the power test of the article's null
N_PERM_PODER = 200

OUT = dir_saida("r1")
TOL = 1e-10


# ---------------------------------------------------------------------------
# utilities
# ---------------------------------------------------------------------------


def dupla(x: np.ndarray) -> np.ndarray:
    return x - x.mean(1, keepdims=True) - x.mean(0, keepdims=True) + x.mean()


def cos_abs(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(abs(a @ b) / (na * nb)) if na > 0 and nb > 0 else float("nan")


def corr_residual(resid: np.ndarray) -> float:
    """Mean of |r| off the diagonal between conditions, computed over codons (as in the article)."""
    c = np.corrcoef(resid.T)
    iu = np.triu_indices(c.shape[0], 1)
    v = c[iu]
    return float(np.abs(v[np.isfinite(v)]).mean())


def json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [json_safe(v) for v in obj.tolist()]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


# ---------------------------------------------------------------------------
# 1. anchor
# ---------------------------------------------------------------------------

ancora_linhas: list[dict] = []


def checar(rotulo: str, impresso: str | None, fmt: str, reproduzido, armazenado=None, fonte: str = ""):
    ok = True
    if armazenado is not None:
        ok &= abs(float(reproduzido) - float(armazenado)) <= TOL * max(1.0, abs(float(armazenado)))
    texto = fmt.format(reproduzido) if not isinstance(reproduzido, str) else reproduzido
    if impresso is not None:
        ok &= texto == impresso
    ancora_linhas.append(
        {
            "quantidade": rotulo,
            "valor_no_artigo": impresso if impresso is not None else "",
            "reproduzido": float(reproduzido) if not isinstance(reproduzido, str) else reproduzido,
            "reproduzido_formatado": texto,
            "armazenado": armazenado if armazenado is not None else "",
            "fonte_armazenada": fonte,
            "ok": bool(ok),
        }
    )


def ancora(M: np.ndarray, M_within: np.ndarray, M_freeze: np.ndarray, n_ic, ss_pos) -> None:
    s = json.loads((LATENT_DIR / "pooled" / "summary.json").read_text(encoding="utf-8"))
    sf = json.loads((LATENT_DIR / "freeze" / "summary.json").read_text(encoding="utf-8"))
    bcv_csv = pd.read_csv(LATENT_DIR / "pooled" / "bi_cross_validation.csv").set_index("d")
    bcv_csv_f = pd.read_csv(LATENT_DIR / "freeze" / "bi_cross_validation.csv").set_index("d")
    va_s = s["step_1_8_8_a_variance_selected_model"]
    co_s = s["step_1_8_8_b_correlation_selected_model"]

    fit = fit_pca(M, 1)
    va = variance_accounting(M, fit)
    co = residual_correlation_reduction(M, fit)
    checar("fracao removida pelo posto 1 (pooled)", "0.930", "{:.3f}", va["latent_gain_vs_offset_only"], va_s["latent_gain_vs_offset_only"], "pooled/summary.json")
    checar("QM so m_i", "2.399", "{:.3f}", va["mse_after_offset_only"], va_s["mse_after_offset_only"], "pooled/summary.json")
    checar("QM apos posto 1", "0.167", "{:.3f}", va["mse_after_latent"], va_s["mse_after_latent"], "pooled/summary.json")
    checar("correlacao residual antes", "0.520", "{:.3f}", co["mean_absolute_offdiagonal_correlation_before"], co_s["mean_absolute_offdiagonal_correlation_before"], "pooled/summary.json")
    checar("correlacao residual depois", "0.363", "{:.3f}", co["mean_absolute_offdiagonal_correlation_after"], co_s["mean_absolute_offdiagonal_correlation_after"], "pooled/summary.json")

    st, spec = low_dimensional_structure(M, N_PERM, SEED + 1, lcfg.MAX_COMPONENTS)
    checar("componentes acima do nulo do artigo", "1", "{:.0f}", st["n_components_above_permutation_null"], s["step_1_8_6_low_dimensional_structure"]["n_components_above_permutation_null"], "pooled/summary.json")
    checar("PC1 / p95 do nulo do artigo", "2.24", "{:.2f}", spec.singular_value[0] / spec.permutation_null_p95_singular_value[0])

    bcv = bi_cross_validation(M, (1,), N_SPLIT, SEED_BCV).set_index("d")
    checar("skill posto 1 vs so m_i (modelo_latente)", "0.899", "{:.3f}", bcv.loc[1, "mean_skill"], bcv_csv.loc[1, "mean_skill"], "pooled/bi_cross_validation.csv")
    checar("fracao de particoes com skill > 0", "1.000", "{:.3f}", bcv.loc[1, "fraction_splits_with_positive_skill"], bcv_csv.loc[1, "fraction_splits_with_positive_skill"], "pooled/bi_cross_validation.csv")

    boot, _ = bootstrap_stability(M, 1, N_BOOT, SEED_BOOT)
    checar("cosseno de subespaco bootstrap", "0.99999", "{:.5f}", boot["mean_subspace_cosine"], s["step_1_8_8_d_bootstrap"]["mean_subspace_cosine"], "pooled/summary.json")

    # positional share (verificar_numeros_artigo.py)
    m = M.mean(1, keepdims=True)
    xi = (M - m) - fit.latent_term().T
    ss_off = float(np.sum(n_ic * (M - m) ** 2))
    ss_lat = float(np.sum(n_ic * xi**2))
    checar("posicional: removido por g(Z_c) alem de m_i (%)", "1.23", "{:.2f}", 100 * (ss_off - ss_lat) / ss_pos)
    xc = M - m
    checar("deslocamento igual para todos os codons (%)", "82.7", "{:.1f}", 100 * M.shape[0] * np.sum(xc.mean(0) ** 2) / np.sum(xc**2))

    # GSE119104 alone
    w = s["complementary_within_dataset_analysis"][0]
    fw = fit_pca(M_within, 1)
    checar("GSE119104: fracao posto 1", "83.0", "{:.1f}", 100 * variance_accounting(M_within, fw)["latent_gain_vs_offset_only"], 100 * w["latent_gain_vs_offset_only"], "pooled/summary.json (within)")
    bw, _ = bootstrap_stability(M_within, 1, N_BOOT, SEED_BOOT_WITHIN)
    checar("GSE119104: cosseno bootstrap", None, "{:.6f}", bw["mean_subspace_cosine"], w["mean_subspace_cosine_bootstrap"], "pooled/summary.json (within)")

    # freeze-only comparator
    ff = fit_pca(M_freeze, 1)
    checar("freeze: fracao posto 1", "0.930", "{:.3f}", variance_accounting(M_freeze, ff)["latent_gain_vs_offset_only"], sf["step_1_8_8_a_variance_selected_model"]["latent_gain_vs_offset_only"], "freeze/summary.json")
    bf = bi_cross_validation(M_freeze, (1,), N_SPLIT, SEED_BCV).set_index("d")
    checar("freeze: skill posto 1", "0.899", "{:.3f}", bf.loc[1, "mean_skill"], bcv_csv_f.loc[1, "mean_skill"], "freeze/bi_cross_validation.csv")

    frame = pd.DataFrame(ancora_linhas)
    frame.to_csv(OUT / "ancora.csv", index=False)
    falhas = frame[~frame.ok]
    if not falhas.empty:
        raise RuntimeError("ancora nao reproduzida:\n" + falhas.to_string())
    print(f"[r1] ancora: {len(frame)} quantidades reproduzidas")


# ---------------------------------------------------------------------------
# 2-3, 6. model ladder
# ---------------------------------------------------------------------------


def ajustes(M: np.ndarray, K: np.ndarray | None) -> dict:
    """Fitted values (I x J) and number of parameters of each model."""
    I, J = M.shape
    dec = decomposicao_aditiva(M)
    m, a = dec["offset_m_i"], dec["intercepto_a_c"]
    base = np.repeat(m[:, None], J, axis=1)
    out = {
        "so_m": (base, I),
        "aditivo": (base + a[None, :], I + J - 1),
    }
    if K is not None:
        kseq = (K - M).mean(1)
        kbar = K.mean(1)
        X = M - m[:, None]
        t = X.T @ kbar / float(kbar @ kbar)
        out["escala_kappa_fixo"] = (base + np.outer(kbar, t), I + J - 1)
        u, s, vt = np.linalg.svd(K, full_matrices=False)
        out["multiplicativo_puro"] = (s[0] * np.outer(u[:, 0], vt[0]) - kseq[:, None], I + J - 1)
    out["concorrencia_2_passos"] = (base + np.outer(1.0 + dec["delta_concorrencia"], a), I + (J - 1) + (I - 1))
    out["posto1"] = (fit_pca(M, 1).reconstruct().T, I + I + (J - 1) - 1)
    d1 = np.outer(dec["cargas_interacao_pc1"], dec["escores_interacao_pc1"])
    out["aditivo_mais_pc1_interacao"] = (base + a[None, :] + d1, I + (J - 1) + (I - 1) + (J - 1) - 1)
    return out


def tabela_modelos(nome: str, M: np.ndarray, K: np.ndarray | None, n_ic=None, ss_pos=None) -> pd.DataFrame:
    I, J = M.shape
    fits = ajustes(M, K)
    ss = {k: float(np.sum((M - f) ** 2)) for k, (f, _) in fits.items()}
    rows = []
    for k, (f, p) in fits.items():
        r = M - f
        row = {
            "matriz": nome,
            "modelo": k,
            "n_parametros": p,
            "gl_residual": I * J - p,
            "sq_residual": ss[k],
            "qm_por_celula": ss[k] / (I * J),
            "qm_por_gl": ss[k] / (I * J - p),
            "fracao_removida_vs_so_m": 1 - ss[k] / ss["so_m"],
            "ganho_vs_aditivo": 1 - ss[k] / ss["aditivo"],
            "correlacao_residual_media_abs": corr_residual(r),
        }
        if n_ic is not None:
            ssw = float(np.sum(n_ic * r**2))
            ss_off = float(np.sum(n_ic * (M - fits["so_m"][0]) ** 2))
            ss_add = float(np.sum(n_ic * (M - fits["aditivo"][0]) ** 2))
            row["posicional_removido_vs_so_m_pct"] = 100 * (ss_off - ssw) / ss_pos
            row["posicional_removido_vs_aditivo_pct"] = 100 * (ss_add - ssw) / ss_pos
        rows.append(row)
    frame = pd.DataFrame(rows)
    dec = decomposicao_aditiva(M)  # consistency with the single project-wide definition
    assert abs(frame.set_index("modelo").loc["aditivo", "fracao_removida_vs_so_m"] - dec["fracao_aditivo"]) < 1e-12
    assert abs(frame.set_index("modelo").loc["posto1", "fracao_removida_vs_so_m"] - dec["fracao_posto1"]) < 1e-12
    assert abs(frame.set_index("modelo").loc["posto1", "ganho_vs_aditivo"] - dec["ganho_posto1_sobre_aditivo"]) < 1e-12
    return frame


# ---------------------------------------------------------------------------
# 3. parallel analysis on the doubly centred matrix
# ---------------------------------------------------------------------------


def nulo_matriz(D: np.ndarray, tipo: str, rng: np.random.Generator) -> np.ndarray:
    if tipo == "linhas":  # the article's null applied to D: permutes cells within each codon
        out = np.empty_like(D)
        for i in range(D.shape[0]):
            out[i] = rng.permutation(D[i])
    elif tipo == "colunas":  # permutes codons within each condition
        out = np.empty_like(D)
        for j in range(D.shape[1]):
            out[:, j] = rng.permutation(D[:, j])
    elif tipo == "sinais":  # random cell-by-cell sign flip (preserves |D_ic|)
        out = D * rng.choice(np.array([-1.0, 1.0]), size=D.shape)
    else:
        raise ValueError(tipo)
    return dupla(out)


def paralela(nome: str, M: np.ndarray, tipo: str, seed: int) -> tuple[pd.DataFrame, dict]:
    dec = decomposicao_aditiva(M)
    D, m, a = dec["interacao_D"], dec["offset_m_i"], dec["intercepto_a_c"]
    k = min(D.shape) - 1
    obs = dec["valores_singulares_interacao"][:k]
    rng = np.random.default_rng(seed)
    null_sv = np.empty((N_PERM, k))
    null_ganho = np.empty(N_PERM)
    null_conc = np.empty(N_PERM)
    null_pc1 = np.empty(N_PERM)
    for b in range(N_PERM):
        Dp = nulo_matriz(D, tipo, rng)
        dp = decomposicao_aditiva(m[:, None] + a[None, :] + Dp)
        null_sv[b] = dp["valores_singulares_interacao"][:k]
        null_ganho[b] = dp["ganho_posto1_sobre_aditivo"]
        null_conc[b] = dp["fracao_concorrencia_da_interacao"]
        null_pc1[b] = dp["interacao_pc1_fracao_da_interacao"]
    p = (np.sum(null_sv >= obs, axis=0) + 1) / (N_PERM + 1)
    p95 = np.quantile(null_sv, 0.95, axis=0)
    n_sig, ainda = 0, True
    rows = []
    for c in range(k):
        sig = bool(p[c] < 0.05)
        if sig and ainda:
            n_sig += 1
        else:
            ainda = False
        rows.append(
            {
                "matriz": nome,
                "nulo": tipo,
                "componente_interacao": c + 1,
                "valor_singular": float(obs[c]),
                "fracao_da_interacao": float(obs[c] ** 2 / dec["sq_interacao"]),
                "nulo_media": float(null_sv[:, c].mean()),
                "nulo_p95": float(p95[c]),
                "razao_obs_p95": float(obs[c] / p95[c]),
                "p_valor": float(p[c]),
            }
        )

    def pv(x, o):
        return float((np.sum(x >= o) + 1) / (N_PERM + 1))

    resumo = {
        "n_componentes_da_interacao_acima_do_nulo": n_sig,
        "ganho_posto1_sobre_aditivo_obs": dec["ganho_posto1_sobre_aditivo"],
        "ganho_posto1_sobre_aditivo_nulo_p95": float(np.quantile(null_ganho, 0.95)),
        "ganho_posto1_sobre_aditivo_p": pv(null_ganho, dec["ganho_posto1_sobre_aditivo"]),
        "concorrencia_fracao_obs": dec["fracao_concorrencia_da_interacao"],
        "concorrencia_fracao_nulo_p95": float(np.quantile(null_conc, 0.95)),
        "concorrencia_p": pv(null_conc, dec["fracao_concorrencia_da_interacao"]),
        "pc1_interacao_fracao_obs": dec["interacao_pc1_fracao_da_interacao"],
        "pc1_interacao_fracao_nulo_p95": float(np.quantile(null_pc1, 0.95)),
        "pc1_interacao_p": pv(null_pc1, dec["interacao_pc1_fracao_da_interacao"]),
        "semente": seed,
        "n_permutacoes": N_PERM,
    }
    return pd.DataFrame(rows), resumo


def poder_nulo_artigo(M: np.ndarray) -> dict:
    """Does the article's null (permutation within rows of the m_i-centred matrix) declare d=1 for
    a matrix WITHOUT structured interaction? Synthetic matrices: m_i + a_c + row-permuted D."""
    dec = decomposicao_aditiva(M)
    rng = np.random.default_rng(semente(1, K_PODER_NULO_ARTIGO))
    n_comp, razao = [], []
    for j in range(N_PODER):
        Dp = nulo_matriz(dec["interacao_D"], "linhas", rng)
        Ms = dec["offset_m_i"][:, None] + dec["intercepto_a_c"][None, :] + Dp
        st, spec = low_dimensional_structure(Ms, N_PERM_PODER, int(rng.integers(0, 2**31 - 1)), lcfg.MAX_COMPONENTS)
        n_comp.append(int(st["n_components_above_permutation_null"]))
        razao.append(float(spec.singular_value[0] / spec.permutation_null_p95_singular_value[0]))
    return {
        "n_matrizes_aditivas_sinteticas": N_PODER,
        "n_permutacoes_cada": N_PERM_PODER,
        "componentes_acima_do_nulo_por_matriz": n_comp,
        "fracao_com_ao_menos_1_componente": float(np.mean(np.array(n_comp) >= 1)),
        "razao_pc1_p95_media": float(np.mean(razao)),
        "razao_pc1_p95_min": float(np.min(razao)),
        "semente": semente(1, K_PODER_NULO_ARTIGO),
    }


# ---------------------------------------------------------------------------
# 4. bi-cross-validation against the additive model
# ---------------------------------------------------------------------------


def bcv(M: np.ndarray, K: np.ndarray | None, seed: int, n_splits: int = N_SPLIT) -> pd.DataFrame:
    """Same partitions as modelo_latente.validation.bi_cross_validation (same draw order)."""
    I, J = M.shape
    rng = np.random.default_rng(seed)
    kseq = (K - M).mean(1) if K is not None else None
    rows = []
    for split in range(n_splits):
        r = rng.permutation(I)
        n_a = int(round(I * 0.5))
        A, B = r[:n_a], r[n_a:]
        c = rng.permutation(J)
        C2, C1 = c[:COL_HOLDOUT], c[COL_HOLDOUT:]

        off = M[:, C1].mean(axis=1, keepdims=True)
        E = M - off
        eA1, eA2 = E[np.ix_(A, C1)], E[np.ix_(A, C2)]
        eB1, eB2 = E[np.ix_(B, C1)], E[np.ix_(B, C2)]

        mse0 = float(np.mean(eB2**2))
        u, s, vt = np.linalg.svd(eA1, full_matrices=False)
        pred_r1 = eB1 @ (np.outer(vt[0], u[:, 0]) / s[0]) @ eA2
        a2 = eA2.mean(axis=0)  # a_c of C2 estimated on training rows A
        pred_add = np.broadcast_to(a2, eB2.shape)
        a1 = eA1.mean(axis=0)
        dA1, dA2, dB1 = eA1 - a1, eA2 - a2, eB1 - a1
        ud, sd, vtd = np.linalg.svd(dA1, full_matrices=False)
        pred_ai = a2 + dB1 @ (np.outer(vtd[0], ud[:, 0]) / sd[0]) @ dA2
        row = {
            "split": split,
            "mse_so_m": mse0,
            "mse_aditivo": float(np.mean((eB2 - pred_add) ** 2)),
            "mse_posto1": float(np.mean((eB2 - pred_r1) ** 2)),
            "mse_aditivo_mais_pc1_interacao": float(np.mean((eB2 - pred_ai) ** 2)),
        }
        if K is not None:
            # per-library scale with the same information as the additive model: kappa_i = mean of k_obs over C1,
            # s_c by least squares on rows A of columns C2
            kb = K[:, C1].mean(axis=1)
            sc = (K[np.ix_(A, C2)] * kb[A, None]).sum(0) / float(kb[A] @ kb[A])
            pred_k = np.outer(kb[B], sc)
            pred_e = pred_k - kseq[B, None] - off[B]
            row["mse_escala"] = float(np.mean((eB2 - pred_e) ** 2))
        rows.append(row)
    f = pd.DataFrame(rows)
    f["skill_posto1_vs_so_m"] = 1 - f.mse_posto1 / f.mse_so_m
    f["skill_aditivo_vs_so_m"] = 1 - f.mse_aditivo / f.mse_so_m
    f["skill_posto1_vs_aditivo"] = 1 - f.mse_posto1 / f.mse_aditivo
    f["skill_aditivo_mais_pc1_vs_aditivo"] = 1 - f.mse_aditivo_mais_pc1_interacao / f.mse_aditivo
    if K is not None:
        f["skill_escala_vs_so_m"] = 1 - f.mse_escala / f.mse_so_m
        f["skill_escala_vs_aditivo"] = 1 - f.mse_escala / f.mse_aditivo
        f["skill_posto1_vs_escala"] = 1 - f.mse_posto1 / f.mse_escala
    return f


def resumo_bcv(nome: str, f: pd.DataFrame, seed: int) -> pd.DataFrame:
    rows = []
    for col in [c for c in f.columns if c.startswith("skill_")]:
        v = f[col].to_numpy()
        rows.append(
            {
                "matriz": nome,
                "comparacao": col,
                "n_particoes": len(v),
                "skill_medio": float(v.mean()),
                "skill_mediano": float(np.median(v)),
                "p05": float(np.quantile(v, 0.05)),
                "p95": float(np.quantile(v, 0.95)),
                "fracao_particoes_positivas": float(np.mean(v > 0)),
                "n_particoes_positivas": int(np.sum(v > 0)),
                "semente": seed,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 5. codon bootstrap against the additive model
# ---------------------------------------------------------------------------


def bootstrap(M: np.ndarray, seed: int, n_boot: int = N_BOOT) -> pd.DataFrame:
    I = M.shape[0]
    ref = fit_pca(M, 1)
    l_ref, z_ref = ref.loadings[:, 0], ref.scores[:, 0]
    dref = decomposicao_aditiva(M)
    _, _, vt_ref = np.linalg.svd(dref["interacao_D"].T, full_matrices=False)  # codons in vt
    u_int, u_int2 = vt_ref[0], vt_ref[1]
    v_int = dref["escores_interacao_pc1"]
    one = np.ones(I)
    rng = np.random.default_rng(seed)
    rows = []
    for b in range(n_boot):
        idx = rng.integers(0, I, size=I)
        boot = M[idx, :]
        fb = fit_pca(boot, 1)
        lb, zb = fb.loadings[:, 0], fb.scores[:, 0]
        lr = l_ref[idx]
        db = decomposicao_aditiva(boot)
        uw, sw, vtw = np.linalg.svd(db["interacao_D"].T, full_matrices=False)
        ur, ur2 = u_int[idx], u_int2[idx]
        rows.append(
            {
                "b": b,
                "cos_cargas_posto1": cos_abs(lr, lb),  # = subspace cosine of the article (d=1)
                "cos_vetor_constante": cos_abs(one, one),  # additive loadings: 1 by construction
                "cos_cargas_ref_com_constante": cos_abs(lr, one),
                "cos_cargas_boot_com_constante": cos_abs(lb, one),
                "piso_trivial": cos_abs(lr, one) * cos_abs(lb, one),
                "cos_cargas_posto1_sem_media": cos_abs(lr - lr.mean(), lb - lb.mean()),
                "cos_cargas_pc1_interacao": cos_abs(ur - ur.mean(), vtw[0]),
                "cos_cargas_pc2_interacao": cos_abs(ur2 - ur2.mean(), vtw[1]),
                "cos_escores_posto1": cos_abs(z_ref, zb),
                "cos_escores_pc1_interacao": cos_abs(v_int, uw[:, 0] * sw[0]),
                "cos_escores_posto1_com_a_c": cos_abs(zb, db["intercepto_a_c"]),
            }
        )
    return pd.DataFrame(rows)


def referencia_ruido_boot(M: np.ndarray) -> dict:
    """Stability that the interaction PC1 (and the rank-one loadings without the mean) would have if
    the interaction were noise without alignment across codons (null 'linhas'), with the same additive part."""
    dec = decomposicao_aditiva(M)
    m, a, D = dec["offset_m_i"], dec["intercepto_a_c"], dec["interacao_D"]
    I = M.shape[0]
    rng = np.random.default_rng(semente(1, K_REF_RUIDO_BOOT))
    med_int, med_r1c = [], []
    for _ in range(N_REF_RUIDO):
        Mn = m[:, None] + a[None, :] + nulo_matriz(D, "linhas", rng)
        dn = decomposicao_aditiva(Mn)
        u_ref = dn["cargas_interacao_pc1"]
        l_ref = fit_pca(Mn, 1).loadings[:, 0]
        ci, cr = [], []
        for _ in range(N_REF_RUIDO):
            idx = rng.integers(0, I, size=I)
            bt = Mn[idx]
            _, _, vt = np.linalg.svd(dupla(bt).T, full_matrices=False)
            ur = u_ref[idx]
            ci.append(cos_abs(ur - ur.mean(), vt[0]))
            lb = fit_pca(bt, 1).loadings[:, 0]
            lr = l_ref[idx]
            cr.append(cos_abs(lr - lr.mean(), lb - lb.mean()))
        med_int.append(np.mean(ci))
        med_r1c.append(np.mean(cr))
    med_int, med_r1c = np.array(med_int), np.array(med_r1c)
    return {
        "n_matrizes_nulas": N_REF_RUIDO,
        "n_bootstraps_por_matriz": N_REF_RUIDO,
        "cos_pc1_interacao_nulo_media": float(med_int.mean()),
        "cos_pc1_interacao_nulo_p95": float(np.quantile(med_int, 0.95)),
        "cos_posto1_sem_media_nulo_media": float(med_r1c.mean()),
        "cos_posto1_sem_media_nulo_p95": float(np.quantile(med_r1c, 0.95)),
        "semente": semente(1, K_REF_RUIDO_BOOT),
    }


def resumo_boot(nome: str, f: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in [c for c in f.columns if c != "b"]:
        v = f[col].to_numpy()
        rows.append(
            {
                "matriz": nome,
                "medida": col,
                "media": float(v.mean()),
                "p05": float(np.quantile(v, 0.05)),
                "um_menos_media": float(1 - v.mean()),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 6. shape of the interaction
# ---------------------------------------------------------------------------


def forma(M: np.ndarray, K: np.ndarray, kseq_tab: np.ndarray) -> dict:
    dec = decomposicao_aditiva(M)
    D, a = dec["interacao_D"], dec["intercepto_a_c"]
    kbar = K.mean(1)
    w = kbar - kbar.mean()
    l1 = dec["cargas_posto1"]
    delta = dec["delta_concorrencia"]
    ss_d = dec["sq_interacao"]
    r = w @ D / float(w @ w)  # fixed codon direction, free scores
    pred0 = np.outer(w / kbar.mean(), a)  # pure scale, no free parameter in the interaction
    slope = float(np.polyfit(w / kbar.mean(), delta, 1)[0])
    v_int = dec["escores_interacao_pc1"]
    dl = decomposicao_aditiva(np.log(K))
    return {
        "corr_cargas_posto1_vs_media_kobs_codon": float(np.corrcoef(l1, kbar)[0, 1]),
        "corr_cargas_posto1_vs_k_seq": float(np.corrcoef(l1, kseq_tab)[0, 1]),
        "corr_delta_mandel_vs_media_kobs_codon": float(np.corrcoef(delta, kbar)[0, 1]),
        "inclinacao_delta_vs_kbar_relativo_esperado_1": slope,
        "corr_cargas_pc1_interacao_vs_media_kobs_codon": float(np.corrcoef(dec["cargas_interacao_pc1"], kbar)[0, 1]),
        "cos_escores_pc1_interacao_com_a_c": cos_abs(v_int, a),
        "cos_escores_posto1_com_a_c": cos_abs(dec["escores_posto1"], a),
        "cos_cargas_posto1_com_constante": dec["cos_cargas_posto1_com_vetor_constante"],
        "n_cargas_posto1_negativas": dec["n_cargas_posto1_negativas"],
        "interacao_fracao_escala_pura_sem_parametro": float(1 - np.sum((D - pred0) ** 2) / ss_d),
        "interacao_fracao_direcao_kbar_escores_livres": float(np.sum(r**2) * (w @ w) / ss_d),
        "interacao_fracao_concorrencia": dec["fracao_concorrencia_da_interacao"],
        "interacao_fracao_pc1": dec["interacao_pc1_fracao_da_interacao"],
        "interacao_pc1_fracao_da_sq_centrada_em_m": dec["interacao_pc1_fracao_da_sq_centrada_em_m"],
        "log_kobs": {
            k: dl[k]
            for k in (
                "fracao_aditivo",
                "fracao_posto1",
                "ganho_posto1_sobre_aditivo",
                "interacao_pc1_fracao_da_interacao",
                "interacao_pc1_fracao_da_sq_centrada_em_m",
                "fracao_concorrencia_da_interacao",
                "n_cargas_posto1_negativas",
            )
        },
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    codons, aas, ids, R = carregar_tabela("R")
    _, _, ids_k, K = carregar_tabela("k_obs")
    _, _, ids_n, n_ic = carregar_tabela("n")
    assert ids == ids_k == ids_n
    # the table is the same one the latent stage reads
    rl = pd.read_csv(BASELINE_DIR / "residuals_codon_condition.csv")
    assert np.array_equal(rl[[c for c in rl.columns if c.startswith("R::")]].to_numpy(float), R)
    fr = pd.read_csv(FREEZE_DIR / "residuals_using_freeze_kseq.csv")
    assert fr.codon.tolist() == codons and [c.split("::", 1)[1] for c in fr.columns if c.startswith("R::")] == ids
    RF = fr[[c for c in fr.columns if c.startswith("R::")]].to_numpy(float)
    kseq = (K - R).mean(1)
    assert np.max((K - R).std(1)) < 1e-12  # R = k_obs - k_seq exactly
    fs = json.loads((BASELINE_DIR / "fit_summary.json").read_text(encoding="utf-8"))
    ss_pos = fs["var_k_obs_minus_k_seq"] * fs["n_observations"]
    desenho = carregar_desenho().set_index("condition_id").loc[ids].reset_index()
    w = np.flatnonzero(desenho.dataset.to_numpy() == "GSE119104")
    assert w.size == 10

    # 1. anchor
    ancora(R, R[:, w], RF, n_ic, ss_pos)

    # 2-3, 6. models
    mod = pd.concat(
        [
            tabela_modelos("pooled_R", R, K, n_ic, ss_pos),
            tabela_modelos("freeze_R", RF, K, n_ic, ss_pos),
            tabela_modelos("GSE119104_R", R[:, w], K[:, w]),
            tabela_modelos("pooled_logkobs", np.log(K), None),
            tabela_modelos("GSE119104_logkobs", np.log(K[:, w]), None),
        ],
        ignore_index=True,
    )
    mod.to_csv(OUT / "modelos.csv", index=False)
    print("[r1] modelos ajustados")

    decs = {
        "pooled_R": decomposicao_aditiva(R),
        "freeze_R": decomposicao_aditiva(RF),
        "GSE119104_R": decomposicao_aditiva(R[:, w]),
        "pooled_logkobs": decomposicao_aditiva(np.log(K)),
        "GSE119104_logkobs": decomposicao_aditiva(np.log(K[:, w])),
    }

    # per library and per codon
    dp, dw = decs["pooled_R"], decs["GSE119104_R"]
    s_int = dp["escores_interacao_pc1"]
    sgn = 1.0 if s_int @ dp["intercepto_a_c"] >= 0 else -1.0
    cond = desenho[["condition_id", "dataset", "label"]].copy()
    cond["a_c_pooled"] = dp["intercepto_a_c"]
    cond["Z1_posto1_pooled"] = dp["escores_posto1"]
    cond["escore_pc1_interacao_pooled"] = sgn * s_int
    cond["media_kobs_biblioteca"] = K.mean(0)
    cond["escala_t_c_kappa_fixo"] = (R - R.mean(1, keepdims=True)).T @ K.mean(1) / float(K.mean(1) @ K.mean(1))
    cond["a_c_logkobs_pooled"] = decs["pooled_logkobs"]["intercepto_a_c"]
    cond["a_c_GSE119104"] = np.nan
    cond.loc[w, "a_c_GSE119104"] = dw["intercepto_a_c"]
    cond["a_c_freeze"] = decs["freeze_R"]["intercepto_a_c"]
    # PC1 of the interaction in log k_obs, oriented with filter_Mg_MPC negative (descriptive reading)
    dlog = decs["pooled_logkobs"]
    i_mpc = int(np.flatnonzero(desenho.label.to_numpy() == "filter_Mg_MPC")[0])
    sgn_log = -1.0 if dlog["escores_interacao_pc1"][i_mpc] > 0 else 1.0
    cond["escore_pc1_interacao_logkobs"] = sgn_log * dlog["escores_interacao_pc1"]
    cond.to_csv(OUT / "condicoes.csv", index=False)

    cod = pd.DataFrame({"codon": codons, "amino_acid": aas})
    cod["m_i"] = dp["offset_m_i"]
    cod["k_seq"] = kseq
    cod["media_kobs_codon"] = K.mean(1)
    cod["l_i_posto1"] = dp["cargas_posto1"]
    cod["l_i_posto1_sem_media"] = dp["cargas_posto1"] - dp["cargas_posto1"].mean()
    cod["delta_i_mandel"] = dp["delta_concorrencia"]
    cod["delta_i_esperado_escala"] = K.mean(1) / K.mean() - 1
    cod["carga_pc1_interacao"] = sgn * dp["cargas_interacao_pc1"]
    cod["carga_pc1_interacao_logkobs"] = sgn_log * dlog["cargas_interacao_pc1"]
    cod.to_csv(OUT / "codons.csv", index=False)
    carga_log = cod.set_index("codon")["carga_pc1_interacao_logkobs"]
    escore_log = cond.set_index("label")["escore_pc1_interacao_logkobs"]
    pc1_log = {
        "cargas_media_serina": float(carga_log.loc[list(SERINA)].mean()),
        "cargas_media_isoleucina": float(carga_log.loc[list(ISOLEUCINA)].mean()),
        "cargas_media_demais": float(carga_log.drop(list(SERINA) + list(ISOLEUCINA)).mean()),
        "seis_mais_negativas": carga_log.nsmallest(6).index.tolist(),
        "tres_mais_positivas": carga_log.nlargest(3).index.tolist(),
        "escores_mupirocina": escore_log.loc[["filter_Mg_MPC", "filter_Cm_MPC", "freeze_Mg_MPC"]].to_dict(),
        "escores_filtradas_sem_droga": escore_log.loc[["filter_Mg", "filter_NaCl", "filter_Mg_ctrl", "filter_Cm_ctrl"]].to_dict(),
        "escores_congeladas_sem_droga": escore_log.loc[["freeze_Mg", "freeze_Mg_ctrl"]].to_dict(),
        "escores_GSE77617": escore_log.loc[["rib_37_WT", "rib_CTRL", "rib_CUA", "rib_CUG"]].to_dict(),
        "escore_media_Cm": float(escore_log.loc["media_Cm"]),
    }

    # 3. parallel analysis on the interaction
    pa_frames, pa = [], {}
    for tipo, k in K_NULO.items():
        f, r = paralela("pooled_R", R, tipo, semente(1, k))
        pa_frames.append(f)
        pa[f"pooled_R|{tipo}"] = r
    for tipo, k in K_NULO_WITHIN.items():
        f, r = paralela("GSE119104_R", R[:, w], tipo, semente(1, k))
        pa_frames.append(f)
        pa[f"GSE119104_R|{tipo}"] = r
    f, r = paralela("pooled_logkobs", np.log(K), "linhas", semente(1, K_NULO_LOG))
    pa_frames.append(f)
    pa["pooled_logkobs|linhas"] = r
    pd.concat(pa_frames, ignore_index=True).to_csv(OUT / "paralela_interacao.csv", index=False)
    poder = poder_nulo_artigo(R)
    print("[r1] analise paralela concluida")

    # 4. bi-cross-validation
    b_pool = bcv(R, K, SEED_BCV)
    b_free = bcv(RF, K, SEED_BCV)
    b_log = bcv(np.log(K), None, SEED_BCV)
    b_with = bcv(R[:, w], K[:, w], semente(1, K_BCV_WITHIN))
    ref_csv = pd.read_csv(LATENT_DIR / "pooled" / "bi_cross_validation.csv").set_index("d").loc[1]
    assert abs(b_pool.skill_posto1_vs_so_m.mean() - ref_csv.mean_skill) < TOL  # same partitions
    b_pool.to_csv(OUT / "bcv_particoes_pooled.csv", index=False)
    bres = pd.concat(
        [
            resumo_bcv("pooled_R", b_pool, SEED_BCV),
            resumo_bcv("freeze_R", b_free, SEED_BCV),
            resumo_bcv("pooled_logkobs", b_log, SEED_BCV),
            resumo_bcv("GSE119104_R", b_with, semente(1, K_BCV_WITHIN)),
        ],
        ignore_index=True,
    )
    bres.to_csv(OUT / "bcv_resumo.csv", index=False)
    print("[r1] bi-validacao concluida")

    # 5. bootstrap
    bt_pool = bootstrap(R, SEED_BOOT)
    bt_with = bootstrap(R[:, w], SEED_BOOT_WITHIN)
    ref_boot = json.loads((LATENT_DIR / "pooled" / "summary.json").read_text(encoding="utf-8"))
    assert abs(bt_pool.cos_cargas_posto1.mean() - ref_boot["step_1_8_8_d_bootstrap"]["mean_subspace_cosine"]) < TOL
    bt_pool.to_csv(OUT / "bootstrap_replicas_pooled.csv", index=False)
    pd.concat([resumo_boot("pooled_R", bt_pool), resumo_boot("GSE119104_R", bt_with)], ignore_index=True).to_csv(
        OUT / "bootstrap_resumo.csv", index=False
    )
    ruido = referencia_ruido_boot(R)
    print("[r1] bootstrap concluido")

    # 6. shape
    fm = {"pooled_R": forma(R, K, kseq), "GSE119104_R": forma(R[:, w], K[:, w], kseq)}

    # freeze == pooled
    iguais = {
        "max_dif_interacao_D": float(np.abs(decs["freeze_R"]["interacao_D"] - decs["pooled_R"]["interacao_D"]).max()),
        "max_dif_a_c": float(np.abs(decs["freeze_R"]["intercepto_a_c"] - decs["pooled_R"]["intercepto_a_c"]).max()),
        "max_dif_skill_por_particao": float(
            np.abs(b_free.filter(like="skill_").to_numpy() - b_pool.filter(like="skill_").to_numpy()).max()
        ),
        "max_dif_tabela_modelos_fracoes": float(
            np.abs(
                mod[mod.matriz == "freeze_R"].fracao_removida_vs_so_m.to_numpy()
                - mod[mod.matriz == "pooled_R"].fracao_removida_vs_so_m.to_numpy()
            ).max()
        ),
    }

    def bsum(nome):
        return bres[bres.matriz == nome].set_index("comparacao")[["skill_medio", "n_particoes_positivas", "n_particoes"]].to_dict(orient="index")

    summary = {
        "item": "R1 comparador aditivo",
        "ancora": pd.DataFrame(ancora_linhas).to_dict(orient="records"),
        "decomposicao_aditiva": {k: escalares(v) for k, v in decs.items()},
        "modelos": mod.to_dict(orient="records"),
        "analise_paralela_interacao": pa,
        "poder_do_nulo_do_artigo_contra_matriz_aditiva": poder,
        "bcv": {k: bsum(k) for k in ["pooled_R", "freeze_R", "pooled_logkobs", "GSE119104_R"]},
        "bootstrap": {
            "pooled_R": resumo_boot("pooled_R", bt_pool).set_index("medida")[["media", "p05"]].to_dict(orient="index"),
            "GSE119104_R": resumo_boot("GSE119104_R", bt_with).set_index("medida")[["media", "p05"]].to_dict(orient="index"),
            "referencia_ruido_pooled": ruido,
        },
        "forma_da_interacao": fm,
        "pc1_interacao_logkobs_descritivo": pc1_log,
        "freeze_igual_pooled": iguais,
        "sementes": {
            "bcv_pooled_freeze_log": SEED_BCV,
            "bootstrap_pooled": SEED_BOOT,
            "bootstrap_GSE119104": SEED_BOOT_WITHIN,
            "paralela": {k: semente(1, v) for k, v in K_NULO.items()},
            "paralela_GSE119104": {k: semente(1, v) for k, v in K_NULO_WITHIN.items()},
            "paralela_log": semente(1, K_NULO_LOG),
            "bcv_GSE119104": semente(1, K_BCV_WITHIN),
        },
    }
    escrever_json(OUT / "summary.json", json_safe(summary))
    print(f"[r1] saidas em {OUT}")


if __name__ == "__main__":
    main()
