"""R5: within-study analysis (GSE119104) without the media_Cm library, with correction for
multiple label tests, scale of the "12.6 units", leave-one-library-out, and an additive version.

Reproduce (from the repository root):

    python -m analises_revisao.r5_intra_estudo

Frozen inputs (read only): the R:: table of modelo_baseline/saidas and the
`within_GSE119104_*` outputs of modelo_latente. Outputs in analises_revisao/saidas/r5/.
Only numpy/pandas; new seeds via comum.semente(5, k).
"""

from __future__ import annotations

import hashlib
import itertools
import sys

import numpy as np
import pandas as pd

from modelo_latente import config as lcfg
from modelo_latente.analysis import condition_dependence, low_dimensional_structure
from modelo_latente.models import fit_pca
from modelo_latente.validation import design_coherence

from . import comum

ITEM = 5
GRUPOS = ["harvest", "drug", "lysis"]  # order of WITHIN_DATASET_GROUPINGS in modelo_latente.run
N_PERM = lcfg.N_PERMUTATIONS  # 2000
TOL = 1e-9
DATASET = "GSE119104"
CM = "media_Cm"


# ----------------------------------------------------------------------------
# utilities
# ----------------------------------------------------------------------------

def curto(cid: str) -> str:
    return cid.split("|")[-1]


def rotulacoes_distintas(rotulos) -> np.ndarray:
    """All distinct assignments of labels (multiset) to the n libraries."""
    rotulos = np.asarray(rotulos)
    niveis, codigos = np.unique(rotulos, return_inverse=True)
    tamanhos = [int(np.sum(codigos == k)) for k in range(niveis.size)]
    n = rotulos.size
    saida = []

    def rec(livres, nivel, atual):
        if nivel == len(tamanhos) - 1:
            a = atual.copy()
            a[livres] = nivel
            saida.append(a)
            return
        for comb in itertools.combinations(livres, tamanhos[nivel]):
            a = atual.copy()
            a[list(comb)] = nivel
            resto = [x for x in livres if x not in comb]
            rec(resto, nivel + 1, a)

    rec(list(range(n)), 0, np.full(n, -1, dtype=np.int8))
    return np.array(saida, dtype=np.int8)


def razoes(pair_dist, iu, codigos):
    """Within/between distance ratio for each labelling (rows of `codigos`)."""
    same = codigos[:, iu[0]] == codigos[:, iu[1]]
    ns, nd = same.sum(1), (~same).sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = ((same * pair_dist).sum(1) / ns) / (((~same) * pair_dist).sum(1) / nd)
    r[(ns == 0) | (nd == 0)] = np.nan
    return r


def p_exato(scores, rotulos) -> dict:
    """Exact p by enumeration of all distinct labellings (same statistic as
    `design_coherence`: mean within/between ratio of distances in the scores)."""
    n = scores.shape[0]
    dist = np.sqrt(np.sum((scores[:, None, :] - scores[None, :, :]) ** 2, axis=2))
    iu = np.triu_indices(n, k=1)
    pd_ = dist[iu]
    cod = rotulacoes_distintas(rotulos)
    r = razoes(pd_, iu, cod)
    obs_codigos = np.unique(np.asarray(rotulos), return_inverse=True)[1].astype(np.int8)[None, :]
    obs = float(razoes(pd_, iu, obs_codigos)[0])
    ok = np.isfinite(r)
    menor = r[ok] <= obs * (1 + TOL)
    n_lab = int(ok.sum())
    k = int(menor.sum())
    rmin = float(r[ok].min())
    n_emp_min = int(np.sum(r[ok] <= rmin * (1 + TOL)))
    return {
        "razao_observada": obs,
        "n_rotulacoes_distintas": n_lab,
        "n_rotulacoes_com_razao_menor_ou_igual": k,
        "p_exato": k / n_lab,
        "p_minimo_atingivel": n_emp_min / n_lab,
    }


def holm(ps):
    ps = np.asarray(ps, float)
    ordem = np.argsort(ps)
    m = ps.size
    aj = np.empty(m)
    corrente = 0.0
    for rank, idx in enumerate(ordem):
        corrente = max(corrente, (m - rank) * ps[idx])
        aj[idx] = min(1.0, corrente)
    return aj


def orientar(fit):
    """Article convention: PC1 loadings with negative sum (untreated scores > 0)."""
    s = fit.scores.copy()
    l = fit.loadings.copy()
    if l[:, 0].sum() > 0:
        s[:, 0] *= -1
        l[:, 0] *= -1
    return s, l


def analisar(sub, desenho, seed_est, seed_coer, n_perm=N_PERM, max_comp=None):
    """Replicates the 'within' routine of modelo_latente.run for a submatrix (61 x n)."""
    n = sub.shape[1]
    estr, espectro = low_dimensional_structure(
        sub, n_perm, seed_est, min(lcfg.MAX_COMPONENTS, n - 1) if max_comp is None else max_comp
    )
    d = max(1, int(estr["n_components_above_permutation_null"]))
    fit = fit_pca(sub, d)
    coer = design_coherence(fit.scores, desenho, GRUPOS, n_perm, seed_coer)
    return estr, espectro, d, fit, coer


def linhas_coerencia(rotulo_caso, scores, desenho, coer, extra=None):
    linhas = []
    for g in GRUPOS:
        row = coer[coer["grouping"] == g].iloc[0]
        ex = p_exato(scores, desenho[g].to_numpy())
        linha = {
            "caso": rotulo_caso,
            "grouping": g,
            "n_groups": int(row["n_groups"]),
            "tamanhos": "/".join(
                str(int(c)) for c in desenho[g].value_counts().sort_index().to_numpy()
            ),
            "mean_within": float(row["mean_within_group_distance"]),
            "mean_between": float(row["mean_between_group_distance"]),
            "razao_intra_inter": float(row["within_over_between_ratio"]),
            "p_monte_carlo": float(row["permutation_p_value"]),
            "p_mc_minimo": 1.0 / (N_PERM + 1),
            **{k: v for k, v in ex.items() if k != "razao_observada"},
        }
        if extra:
            linha.update(extra)
        linhas.append(linha)
    return linhas


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def main() -> None:
    saida = comum.dir_saida("r5")
    cod, aa, ids, R = comum.carregar_tabela("R")
    _, _, ids_k, K = comum.carregar_tabela("k_obs")
    assert ids == ids_k
    desenho_full = comum.carregar_desenho().set_index("condition_id").loc[ids].reset_index()
    idx10 = np.where(desenho_full["dataset"].to_numpy() == DATASET)[0]
    R10 = R[:, idx10]
    K10 = K[:, idx10]
    des10 = desenho_full.iloc[idx10].reset_index(drop=True)
    nomes10 = [curto(c) for c in des10["condition_id"]]
    pos = {n: i for i, n in enumerate(nomes10)}
    sem = lcfg.RANDOM_SEED
    resultados = {}

    # ---------------- 1. anchor ----------------
    estr, esp, d, fit, coer = analisar(R10, des10, sem + 6, sem + 7)
    dep = condition_dependence(R10, N_PERM, sem + 5)
    congelado_coer = pd.read_csv(comum.LATENT_DIR / "pooled" / "within_GSE119104_design_coherence.csv")
    congelado_sc = pd.read_csv(comum.LATENT_DIR / "pooled" / "within_GSE119104_latent_scores_Zc.csv")
    congelado_esp = pd.read_csv(comum.LATENT_DIR / "pooled" / "within_GSE119104_spectrum.csv")
    assert d == 1, d
    assert np.allclose(fit.scores[:, 0], congelado_sc["Z1"].to_numpy(), atol=1e-9), "escores nao batem"
    assert np.allclose(esp["singular_value"], congelado_esp["singular_value"], atol=1e-9), "espectro"
    for g in GRUPOS:
        p_novo = float(coer.loc[coer["grouping"] == g, "permutation_p_value"].iloc[0])
        p_old = float(congelado_coer.loc[congelado_coer["grouping"] == g, "permutation_p_value"].iloc[0])
        assert abs(p_novo - p_old) < 1e-12, (g, p_novo, p_old)
        r_novo = float(coer.loc[coer["grouping"] == g, "within_over_between_ratio"].iloc[0])
        r_old = float(congelado_coer.loc[congelado_coer["grouping"] == g, "within_over_between_ratio"].iloc[0])
        assert abs(r_novo - r_old) < 1e-9, g
    assert round(estr["explained_variance_ratio_pc1"] * 100, 1) == 83.0
    sc10, ld10 = orientar(fit)
    z10 = sc10[:, 0]
    nn_dist = {n: abs(z10[pos[CM]] - z10[pos[n]]) for n in nomes10 if n != CM}
    vizinho = min(nn_dist, key=nn_dist.get)
    assert round(nn_dist[vizinho], 1) == 12.6, nn_dist[vizinho]
    ancora = {
        "fracao_pc1": estr["explained_variance_ratio_pc1"],
        "n_componentes_acima_do_nulo": estr["n_components_above_permutation_null"],
        "p_condicao_efeito_principal": dep["permutation_p_value"],
        "p_droga": float(coer.loc[coer.grouping == "drug", "permutation_p_value"].iloc[0]),
        "p_colheita": float(coer.loc[coer.grouping == "harvest", "permutation_p_value"].iloc[0]),
        "p_lise": float(coer.loc[coer.grouping == "lysis", "permutation_p_value"].iloc[0]),
        "distancia_media_Cm_ao_vizinho": nn_dist[vizinho],
        "vizinho_mais_proximo": vizinho,
        "razao_sv1_p95_nulo": float(esp["singular_value"].iloc[0] / esp["permutation_null_p95_singular_value"].iloc[0]),
        "verificacao": "escores, espectro e p-valores reproduzem os arquivos congelados (atol 1e-9)",
    }
    resultados["ancora_10_bibliotecas"] = ancora
    print("ancora OK:", ancora)

    esp10 = esp.copy()
    esp10.insert(0, "caso", "10_bibliotecas")
    pd.DataFrame(
        {"biblioteca": nomes10, "Z1_orientado": z10, "harvest": des10["harvest"], "drug": des10["drug"], "lysis": des10["lysis"]}
    ).to_csv(saida / "escores_10_bibliotecas.csv", index=False)

    # ---------------- 2. without media_Cm ----------------
    keep9 = [i for i, n in enumerate(nomes10) if n != CM]
    R9 = R10[:, keep9]
    des9 = des10.iloc[keep9].reset_index(drop=True)
    nomes9 = [nomes10[i] for i in keep9]
    estr9, esp9, d9, fit9, coer9 = analisar(R9, des9, comum.semente(ITEM, 0), comum.semente(ITEM, 1))
    sc9, ld9 = orientar(fit9)
    esp9c = esp9.copy()
    esp9c.insert(0, "caso", "9_sem_media_Cm")
    pd.concat([esp10, esp9c]).to_csv(saida / "espectro_paralelo_10_e_9.csv", index=False)
    pd.DataFrame(
        {"biblioteca": nomes9, **{f"Z{k+1}": sc9[:, k] for k in range(d9)},
         "harvest": des9["harvest"], "drug": des9["drug"], "lysis": des9["lysis"]}
    ).to_csv(saida / "escores_9_sem_media_Cm.csv", index=False)

    # overlap mupirocin x untreated (PC1 axis of the 9-library case)
    z9 = sc9[:, 0]
    mpc = des9["drug"].to_numpy() == "mupirocin"
    none = des9["drug"].to_numpy() == "none"
    lo, hi = z9[mpc].min(), z9[mpc].max()
    dentro = [nomes9[i] for i in np.where(none)[0] if lo <= z9[i] <= hi]
    # exact U statistic (Mann-Whitney) without a package: P(none > mpc) by counting
    u = sum(float(a > b) + 0.5 * float(a == b) for a in z9[none] for b in z9[mpc])
    sobre = {
        "faixa_mupirocina": [float(lo), float(hi)],
        "faixa_nao_tratadas": [float(z9[none].min()), float(z9[none].max())],
        "nao_tratadas_dentro_da_faixa_mupirocina": dentro,
        "pares_none_maior_que_mpc": u,
        "pares_total": int(none.sum() * mpc.sum()),
        "separacao_perfeita_no_eixo": bool(z9[none].min() > hi),
        "folga_entre_min_none_e_max_mpc": float(z9[none].min() - hi),
        "mpc_escores": {nomes9[i]: float(z9[i]) for i in np.where(mpc)[0]},
        "none_escores": {nomes9[i]: float(z9[i]) for i in np.where(none)[0]},
    }
    resultados["sobreposicao_mupirocina_9"] = sobre
    resultados["analise_paralela_9"] = {
        "d_selecionado": d9,
        "n_componentes_acima_do_nulo": estr9["n_components_above_permutation_null"],
        "fracao_pc1": estr9["explained_variance_ratio_pc1"],
        "sv1": float(esp9["singular_value"].iloc[0]),
        "p95_nulo_pc1": float(esp9["permutation_null_p95_singular_value"].iloc[0]),
        "razao_sv1_p95": float(esp9["singular_value"].iloc[0] / esp9["permutation_null_p95_singular_value"].iloc[0]),
        "p_pc1": float(esp9["permutation_p_value"].iloc[0]),
        "p_pc2": float(esp9["permutation_p_value"].iloc[1]),
        "razao_sv2_p95": float(esp9["singular_value"].iloc[1] / esp9["permutation_null_p95_singular_value"].iloc[1]),
    }
    resultados["analise_paralela_10"] = {
        "razao_sv2_p95": float(esp["singular_value"].iloc[1] / esp["permutation_null_p95_singular_value"].iloc[1]),
        "p_pc2": float(esp["permutation_p_value"].iloc[1]),
    }

    linhas = []
    linhas += linhas_coerencia("10_bibliotecas", fit.scores, des10, coer)
    linhas += linhas_coerencia("9_sem_media_Cm", fit9.scores, des9, coer9)
    # sensitivity: 9 libraries evaluated on PC1 only (d = 1, as in the 10-library case)
    sc9_pc1 = fit9.scores[:, :1]
    co9_pc1 = design_coherence(sc9_pc1, des9, GRUPOS, N_PERM, comum.semente(ITEM, 5))
    linhas += linhas_coerencia("9_sem_media_Cm_somente_PC1", sc9_pc1, des9, co9_pc1)
    tab_coer = pd.DataFrame(linhas)
    tab_coer.to_csv(saida / "coerencia_10_e_9.csv", index=False)

    # ---------------- 3. correction for 3 tests ----------------
    corr_rows = []
    for caso in ["10_bibliotecas", "9_sem_media_Cm", "9_sem_media_Cm_somente_PC1"]:
        sub = tab_coer[tab_coer.caso == caso].set_index("grouping").loc[GRUPOS]
        for tipo, col in [("monte_carlo", "p_monte_carlo"), ("exato", "p_exato")]:
            p = sub[col].to_numpy()
            bonf = np.minimum(1.0, 3 * p)
            hol = holm(p)
            for g, a, b, c in zip(GRUPOS, p, bonf, hol):
                corr_rows.append(
                    {"caso": caso, "tipo_p": tipo, "grouping": g, "p_bruto": a,
                     "p_bonferroni": b, "p_holm": c,
                     "significativo_0.05_bonferroni": bool(b < 0.05), "significativo_0.05_holm": bool(c < 0.05)}
                )
    tab_corr = pd.DataFrame(corr_rows)
    tab_corr.to_csv(saida / "correcao_multiplos_testes.csv", index=False)

    # ---------------- 4. scale of the 12.6 units ----------------
    a, b = pos[CM], pos[vizinho]
    dk = K10[:, a] - K10[:, b]  # = Delta R (common k_seq)
    dR = R10[:, a] - R10[:, b]
    assert np.allclose(dk, dR, atol=1e-12)
    l1 = ld10[:, 0]
    dz = float(z10[a] - z10[b])
    dz_check = float(l1 @ dR)
    assert abs(abs(dz) - abs(dz_check)) < 1e-9
    soma_l = float(l1.sum())
    n_cod = R10.shape[0]
    parte_uniforme = soma_l * float(dk.mean())
    parte_nao_unif = float(np.sum((l1 - l1.mean()) * (dk - dk.mean())))
    outros = np.array([z10[i] for i in range(len(nomes10)) if i != a])
    sd_outros = float(outros.std(ddof=1))
    d_mg = abs(z10[pos["filter_Mg"]] - z10[pos["filter_Mg_ctrl"]])
    d_fz = abs(z10[pos["freeze_Mg"]] - z10[pos["freeze_Mg_ctrl"]])
    dk_mg = float((K10[:, pos["filter_Mg"]] - K10[:, pos["filter_Mg_ctrl"]]).mean())
    dk_fz = float((K10[:, pos["freeze_Mg"]] - K10[:, pos["freeze_Mg_ctrl"]]).mean())
    kmed = K10.mean(axis=0)
    escala = {
        "unidade_do_escore": "z_c = sum_i l_i (R_ic - m_i), R = k_obs - k_seq (k_seq comum), sum l_i^2 = 1; unidade = k_obs",
        "n_codons": n_cod,
        "biblioteca_extrema": CM,
        "vizinho": vizinho,
        "dz_12_6": abs(dz),
        "soma_cargas": soma_l,
        "abs_soma_cargas": abs(soma_l),
        "sqrt_61": float(np.sqrt(n_cod)),
        "cargas_min": float(l1.min()), "cargas_max": float(l1.max()),
        "n_cargas_negativas": int(np.sum(l1 < 0)),
        "cos_cargas_vetor_constante": float(abs(soma_l) / np.sqrt(n_cod)),
        "diferenca_media_k_obs_por_codon": float(dk.mean()),
        "conversao_dz_sobre_sqrt61": abs(dz) / np.sqrt(n_cod),
        "conversao_dz_sobre_abs_soma_cargas": abs(dz) / abs(soma_l),
        "parte_uniforme_do_dz": float(parte_uniforme),
        "parte_nao_uniforme_do_dz": float(parte_nao_unif),
        "rms_diferenca_por_codon": float(np.sqrt(np.mean(dk**2))),
        "norma_euclidiana_diferenca": float(np.linalg.norm(dk)),
        "cos_diferenca_cargas": float(abs(l1 @ dk) / np.linalg.norm(dk)),
        "k_obs_medio_media_Cm": float(kmed[a]),
        "k_obs_medio_vizinho": float(kmed[b]),
        "razao_k_obs_medio": float(kmed[a] / kmed[b]),
        "diferenca_relativa_k_obs_medio": float((kmed[a] - kmed[b]) / kmed[b]),
        "k_obs_medio_das_10": float(kmed.mean()),
        "sd_escores_outras_9": sd_outros,
        "dz_em_sd_das_outras_9": abs(dz) / sd_outros,
        "dist_filter_Mg_vs_ctrl": float(d_mg),
        "dist_freeze_Mg_vs_ctrl": float(d_fz),
        "dz_em_multiplos_filter_Mg_pair": abs(dz) / float(d_mg),
        "dz_em_multiplos_freeze_Mg_pair": abs(dz) / float(d_fz),
        "dif_media_k_obs_filter_Mg_pair": dk_mg,
        "dif_media_k_obs_freeze_Mg_pair": dk_fz,
        "dif_media_k_obs_filter_Mg_pair_via_sqrt61": float(d_mg / np.sqrt(n_cod)),
        "dif_media_k_obs_freeze_Mg_pair_via_sqrt61": float(d_fz / np.sqrt(n_cod)),
        "amplitude_escores_outras_9": float(outros.max() - outros.min()),
        "dz_sobre_amplitude_outras_9": abs(dz) / float(outros.max() - outros.min()),
        "dist_media_Cm_ao_centroide_das_outras_9": float(abs(z10[a] - outros.mean())),
    }
    resultados["escala_12_6"] = escala
    # distances between all pairs (on the axis), for reference
    dm = pd.DataFrame(
        np.abs(z10[:, None] - z10[None, :]), index=nomes10, columns=nomes10
    )
    dm.to_csv(saida / "distancias_pares_10_bibliotecas.csv")

    # ---------------- 5. leave one library out ----------------
    loo_rows = []
    for j, nome in enumerate(nomes10):
        keep = [i for i in range(len(nomes10)) if i != j]
        subR = R10[:, keep]
        subD = des10.iloc[keep].reset_index(drop=True)
        e, s_, dd, ft, co = analisar(subR, subD, comum.semente(ITEM, 10 + j), comum.semente(ITEM, 30 + j))
        row = {
            "retirada": nome, "n": len(keep), "d": dd,
            "fracao_pc1": e["explained_variance_ratio_pc1"],
            "n_componentes_acima_do_nulo": e["n_components_above_permutation_null"],
            "razao_sv1_p95": float(s_["singular_value"].iloc[0] / s_["permutation_null_p95_singular_value"].iloc[0]),
        }
        for g in GRUPOS:
            r_ = co[co.grouping == g].iloc[0]
            ex = p_exato(ft.scores, subD[g].to_numpy())
            row[f"p_mc_{g}"] = float(r_["permutation_p_value"])
            row[f"p_exato_{g}"] = ex["p_exato"]
            row[f"pmin_{g}"] = ex["p_minimo_atingivel"]
            row[f"razao_{g}"] = float(r_["within_over_between_ratio"])
        loo_rows.append(row)
    loo = pd.DataFrame(loo_rows)
    loo.to_csv(saida / "retirar_uma_biblioteca.csv", index=False)

    # ---------------- 6. additive version ----------------
    ad_rows = []
    ad_info = {}
    for caso, sub, des, kseed in [("10_bibliotecas", R10, des10, 2), ("9_sem_media_Cm", R9, des9, 4)]:
        dec = comum.decomposicao_aditiva(sub)
        a_c = dec["intercepto_a_c"]
        fz = fit_pca(sub, 1)
        z1 = orientar(fz)[0][:, 0]
        # a_c and z_c: sign of the correlation
        corr = float(np.corrcoef(a_c, z1)[0, 1])
        scores_a = a_c[:, None]
        co = design_coherence(scores_a, des, GRUPOS, N_PERM, comum.semente(ITEM, kseed))
        ad_rows += linhas_coerencia(caso + "_aditivo", scores_a, des, co)
        ad_info[caso] = {
            "fracao_aditivo": dec["fracao_aditivo"],
            "fracao_posto1": dec["fracao_posto1"],
            "ganho_posto1_sobre_aditivo": dec["ganho_posto1_sobre_aditivo"],
            "fracao_concorrencia_da_interacao": dec["fracao_concorrencia_da_interacao"],
            "interacao_pc1_fracao_da_sq_centrada_em_m": dec["interacao_pc1_fracao_da_sq_centrada_em_m"],
            "cos_cargas_posto1_com_vetor_constante": dec["cos_cargas_posto1_com_vetor_constante"],
            "n_cargas_posto1_negativas": dec["n_cargas_posto1_negativas"],
            "correlacao_a_c_z1": corr,
            "a_c": {curto(c): float(v) for c, v in zip(des["condition_id"], a_c)},
        }
    tab_ad = pd.DataFrame(ad_rows)
    tab_ad.to_csv(saida / "coerencia_aditiva.csv", index=False)
    resultados["aditiva"] = ad_info

    # correction also for the additive version
    corr_ad = []
    for caso in tab_ad["caso"].unique():
        sub = tab_ad[tab_ad.caso == caso].set_index("grouping").loc[GRUPOS]
        for tipo, col in [("monte_carlo", "p_monte_carlo"), ("exato", "p_exato")]:
            p = sub[col].to_numpy()
            for g, a_, b_, c_ in zip(GRUPOS, p, np.minimum(1, 3 * p), holm(p)):
                corr_ad.append({"caso": caso, "tipo_p": tipo, "grouping": g, "p_bruto": a_, "p_bonferroni": b_, "p_holm": c_})
    pd.DataFrame(corr_ad).to_csv(saida / "correcao_multiplos_testes_aditiva.csv", index=False)

    comum.escrever_json(saida / "resumo_r5.json", resultados)

    # ---------------- printing ----------------
    pd.set_option("display.width", 250, "display.max_columns", 40)
    print("\n== coerencia 10 e 9 ==");  print(tab_coer.drop(columns=["mean_within", "mean_between"]).to_string(index=False))
    print("\n== correcao ==");  print(tab_corr.to_string(index=False))
    print("\n== retirar uma ==");  print(loo.to_string(index=False))
    print("\n== aditiva ==");  print(tab_ad.drop(columns=["mean_within", "mean_between"]).to_string(index=False))
    for caso, v in ad_info.items():
        print(caso, {k: x for k, x in v.items() if k != "a_c"})
    print("\n== analise paralela 9 =="); print(resultados["analise_paralela_9"])
    print("\n== sobreposicao =="); print(sobre)
    print("\n== escala =="); print(escala)

    # hashes of the outputs
    print("\n== sha256 ==")
    for p in sorted(saida.glob("*")):
        print(hashlib.sha256(p.read_bytes()).hexdigest()[:16], p.name)


if __name__ == "__main__":
    sys.exit(main())
