"""R3: serine prediction in the post-latent residual.

Prespecified test:
    T_c = mean, over the six serine codons, of xi_ic (rank one fitted on the 10
    GSE119104 libraries); Delta = mean(filtered) - mean(frozen);
    predicted direction Delta < 0 (rate); exact one-sided p over the C(10,3) = 120
    label assignments.

Everything is deterministic (exact enumeration, no sampling). Frozen inputs; writes
only to analises_revisao/saidas/r3/.
"""

from __future__ import annotations

import itertools
from math import comb

import numpy as np
import pandas as pd

from modelo_latente.models import fit_pca

from . import comum
from .comum import ISOLEUCINA, LATENT_DIR, SERINA

# ---------------------------------------------------------------------------
# Prespecification (not to be changed after the first run)
# ---------------------------------------------------------------------------
DATASET_PRIMARIO = "GSE119104"
PRIMARIO = {
    "residuo": "xi_intra",  # rank one on GSE119104 (10 libraries)
    "codons": SERINA,
    "grupo_alvo": "filtration",  # group in which the pause is predicted
    "grupo_referencia": "direct_freeze",
    "alfa_unicaudal": 0.05,
}
DIRECAO_PREVISTA = -1  # in rate: pause => lower k_obs => lower residual
GLICINA = ("GGT", "GGC", "GGA", "GGG")
SEM_DROGA_FILTRADAS = ("filter_Mg", "filter_NaCl", "filter_Mg_ctrl", "filter_Cm_ctrl")
SEM_DROGA_CONGELADAS = ("freeze_Mg_ctrl", "freeze_Mg")
EPS = 1e-12

SAIDA = comum.dir_saida("r3")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def duplo_centro(m: np.ndarray) -> np.ndarray:
    return m - m.mean(axis=1, keepdims=True) - m.mean(axis=0, keepdims=True) + m.mean()


def residuo_posto1(m: np.ndarray):
    fit = fit_pca(m, 1)
    xi = (m.T - fit.offset - fit.latent_term()).T
    return xi, fit


def estatistica(res: np.ndarray, linhas: np.ndarray) -> np.ndarray:
    """T_c = mean of the selected rows, per column."""
    return res[linhas].mean(axis=0)


def atribuicoes(n: int, k: int) -> np.ndarray:
    """Boolean matrix (C(n,k) x n): all choices of k positions of the target group."""
    combos = list(itertools.combinations(range(n), k))
    out = np.zeros((len(combos), n), dtype=bool)
    for r, c in enumerate(combos):
        out[r, list(c)] = True
    return out


def contrastes(t: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Delta = mean(target) - mean(rest) for each assignment (rows of masks)."""
    k = masks.sum(axis=1)
    soma_alvo = masks.astype(float) @ t
    return soma_alvo / k - (t.sum() - soma_alvo) / (t.size - k)


def teste_exato(t: np.ndarray, alvo: np.ndarray, direcao: int) -> dict:
    """Exact one-sided p in the predicted direction; AUC and perfect separation."""
    n, k = t.size, int(alvo.sum())
    masks = atribuicoes(n, k)
    nulos = contrastes(t, masks) * direcao
    obs = float(contrastes(t, alvo[None, :])[0])
    obs_d = obs * direcao
    p = float(np.mean(nulos >= obs_d - EPS))  # extreme = further in the predicted direction
    p_oposto = float(np.mean(nulos <= obs_d + EPS))
    ta, tr = t[alvo] * direcao, t[~alvo] * direcao
    # AUC: P(target further in the predicted direction than reference); ties = 1/2
    pares = (ta[:, None] > tr[None, :]).astype(float) + 0.5 * (ta[:, None] == tr[None, :])
    auc = float(pares.mean())
    return {
        "n_alvo": k,
        "n_referencia": n - k,
        "n_atribuicoes": int(masks.shape[0]),
        "p_minimo_possivel": 1.0 / masks.shape[0],
        "delta": obs,
        "direcao_prevista": "negativa" if direcao < 0 else "positiva",
        "na_direcao_prevista": bool(obs_d > 0),
        "p_unicaudal_exato": p,
        "p_unicaudal_direcao_oposta": p_oposto,
        "posto_da_atribuicao_observada": int(np.sum(nulos > obs_d + EPS)) + 1,
        "auc": auc,
        "separacao_perfeita": bool(auc == 1.0),
    }


def p_ajustado_minp(res: np.ndarray, grupos: dict, alvo: np.ndarray, direcao: int) -> pd.DataFrame:
    """Exact p per row group and min-p adjusted p (exact Westfall-Young)."""
    n, k = alvo.size, int(alvo.sum())
    masks = atribuicoes(n, k)
    obs_idx = int(np.flatnonzero((masks == alvo[None, :]).all(axis=1))[0])
    nomes = list(grupos)
    nul = np.stack([contrastes(estatistica(res, grupos[g]), masks) * direcao for g in nomes])
    # p of each assignment pi, for each group: fraction of assignments at least as extreme
    p_perm = (nul[:, None, :] >= nul[:, :, None] - EPS).mean(axis=2)  # (G, N)
    min_p = p_perm.min(axis=0)
    p_obs = p_perm[:, obs_idx]
    linhas = []
    for gi, g in enumerate(nomes):
        delta = nul[gi, obs_idx] * direcao
        linhas.append(
            {
                "grupo": g,
                "n_codons": int(len(grupos[g])),
                "delta": float(delta),
                "p_unicaudal_exato": float(p_obs[gi]),
                "p_ajustado_minp": float(np.mean(min_p <= p_obs[gi] + EPS)),
            }
        )
    df = pd.DataFrame(linhas)
    df["posto_por_delta"] = (df["delta"] * direcao).rank(ascending=False, method="min").astype(int)
    return df.sort_values(["posto_por_delta", "grupo"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def main() -> None:
    codons, aas, ids, R = comum.carregar_tabela("R")
    _, _, ids_k, K = comum.carregar_tabela("k_obs")
    _, _, ids_r, RHO = comum.carregar_tabela("rho")
    assert ids == ids_k == ids_r
    codons = np.array(codons)
    aas = np.array(aas)
    desenho = comum.carregar_desenho().set_index("condition_id").loc[ids].reset_index()
    rotulo = desenho["label"].to_numpy()

    idx = np.flatnonzero(desenho["dataset"].to_numpy() == DATASET_PRIMARIO)
    sub = desenho.iloc[idx].reset_index(drop=True)
    lab = sub["label"].to_numpy()
    filtr = (sub["harvest"] == PRIMARIO["grupo_alvo"]).to_numpy()
    cong = (sub["harvest"] == PRIMARIO["grupo_referencia"]).to_numpy()
    assert filtr.sum() == 7 and cong.sum() == 3 and (filtr | cong).all()
    mpc = (sub["drug"] == "mupirocin").to_numpy()
    assert mpc.sum() == 3

    ser = np.flatnonzero(np.isin(codons, SERINA))
    ile = np.flatnonzero(np.isin(codons, ISOLEUCINA))
    gly = np.flatnonzero(np.isin(codons, GLICINA))
    assert ser.size == 6 and ile.size == 3 and gly.size == 4
    assert set(aas[ser]) == {"S"} and set(aas[ile]) == {"I"}

    R_sub = R[:, idx]

    # --- anchors -----------------------------------------------------------
    xi_intra, fit_intra = residuo_posto1(R_sub)
    z_ref = pd.read_csv(LATENT_DIR / "pooled" / f"within_{DATASET_PRIMARIO}_latent_scores_Zc.csv")
    z_ref = z_ref.set_index("condition_id").loc[sub["condition_id"], "Z1"].to_numpy()
    sinal = float(np.sign(np.dot(z_ref, fit_intra.scores[:, 0])))
    ancora_z = float(np.max(np.abs(sinal * fit_intra.scores[:, 0] - z_ref)))

    xi_pool_file = pd.read_csv(LATENT_DIR / "pooled" / "residuals_after_latent.csv")
    assert xi_pool_file["codon"].tolist() == codons.tolist()
    xi_pool = xi_pool_file[[f"xi::{c}" for c in ids]].to_numpy()
    xi_pool_rec, _ = residuo_posto1(R)
    ancora_xi_pool = float(np.max(np.abs(xi_pool_rec - xi_pool)))
    assert ancora_z < 1e-8 and ancora_xi_pool < 1e-8, (ancora_z, ancora_xi_pool)

    # --- residual definitions on the 10 libraries ---------------------------
    centrado = R_sub - R_sub.mean(axis=1, keepdims=True)  # R - m_i (before the latent term)
    decomp = comum.decomposicao_aditiva(R_sub)
    D = decomp["interacao_D"]
    logk = duplo_centro(np.log(K[:, idx]))
    logrho = duplo_centro(np.log(RHO[:, idx]))
    residuos = {
        "xi_intra": (xi_intra, DIRECAO_PREVISTA, "primario"),
        "xi_agregado": (xi_pool[:, idx], DIRECAO_PREVISTA, "secundario (a)"),
        "D_aditivo_intra": (D, DIRECAO_PREVISTA, "secundario (b)"),
        "log_kobs_duplo_centro": (logk, DIRECAO_PREVISTA, "secundario (c)"),
        "log_rho_duplo_centro": (logrho, -DIRECAO_PREVISTA, "secundario (c)"),
        "R_menos_m_antes_do_latente": (centrado, DIRECAO_PREVISTA, "referencia (absorcao)"),
    }

    # --- primary and secondary tests (serine; isoleucine; glycine) ----------
    linhas = []
    tabela_lib = {"condition_id": sub["condition_id"], "label": lab,
                  "harvest": sub["harvest"], "drug": sub["drug"], "lysis": sub["lysis"]}
    for nome, (res, direc, papel) in residuos.items():
        for familia, linhas_cod, alvo, rot in (
            ("serina", ser, filtr, "filtradas vs congeladas"),
            ("glicina", gly, filtr, "filtradas vs congeladas"),
            ("isoleucina", ile, mpc, "mupirocina vs demais"),
        ):
            t = estatistica(res, linhas_cod)
            r = teste_exato(t, alvo, direc)
            r.update({"residuo": nome, "papel": papel, "familia": familia, "contraste": rot,
                      "subconjunto": "10 bibliotecas GSE119104"})
            linhas.append(r)
            tabela_lib[f"T_{familia}::{nome}"] = t

    # (d) untreated libraries only
    sel_d = np.isin(lab, SEM_DROGA_FILTRADAS + SEM_DROGA_CONGELADAS)
    assert sel_d.sum() == 6
    for nome in ("xi_intra", "D_aditivo_intra", "log_rho_duplo_centro"):
        res, direc, _ = residuos[nome]
        t = estatistica(res, ser)[sel_d]
        r = teste_exato(t, np.isin(lab[sel_d], SEM_DROGA_FILTRADAS), direc)
        r.update({"residuo": nome, "papel": "secundario (d)", "familia": "serina",
                  "contraste": "filtradas sem droga (4) vs congeladas sem droga (2)",
                  "subconjunto": "6 bibliotecas sem droga"})
        linhas.append(r)

    # (e) excluding media_Cm: restricted and refitted
    sel_e = lab != "media_Cm"
    xi_9, _ = residuo_posto1(R_sub[:, sel_e])
    for nome, res_e in (("xi_intra (restrito a 9)", xi_intra[:, sel_e]),
                        ("xi_reajustado_9", xi_9),
                        ("D_aditivo_9", comum.decomposicao_aditiva(R_sub[:, sel_e])["interacao_D"])):
        t = estatistica(res_e, ser)
        r = teste_exato(t, filtr[sel_e], DIRECAO_PREVISTA)
        r.update({"residuo": nome, "papel": "secundario (e)", "familia": "serina",
                  "contraste": "filtradas sem media_Cm (6) vs congeladas (3)",
                  "subconjunto": "9 bibliotecas (sem media_Cm)"})
        linhas.append(r)

    testes = pd.DataFrame(linhas)
    primeiras = ["papel", "residuo", "familia", "contraste", "subconjunto"]
    testes = testes[primeiras + [c for c in testes.columns if c not in primeiras]]
    testes.to_csv(SAIDA / "testes_exatos.csv", index=False)

    tab_lib = pd.DataFrame(tabela_lib)
    tab_lib.insert(5, "Z1_intra", sinal * fit_intra.scores[:, 0])
    tab_lib.to_csv(SAIDA / "estatistica_por_biblioteca.csv", index=False)

    # --- specificity -----------------------------------------------------
    aa_grupos = {a: np.flatnonzero(aas == a) for a in sorted(set(aas))}
    cod_grupos = {c: np.array([i]) for i, c in enumerate(codons)}
    espec = []
    for nome in ("xi_intra", "D_aditivo_intra", "log_rho_duplo_centro"):
        res, direc, _ = residuos[nome]
        for nivel, grupos in (("aminoacido", aa_grupos), ("codon", cod_grupos)):
            df = p_ajustado_minp(res, grupos, filtr, direc)
            df.insert(0, "nivel", nivel)
            df.insert(0, "residuo", nome)
            df.insert(0, "contraste", "filtradas vs congeladas")
            espec.append(df)
        df = p_ajustado_minp(res, aa_grupos, mpc, direc)
        df.insert(0, "nivel", "aminoacido")
        df.insert(0, "residuo", nome)
        df.insert(0, "contraste", "mupirocina vs demais")
        espec.append(df)
    espec = pd.concat(espec, ignore_index=True)
    if "codon" in espec.columns:
        espec = espec.drop(columns="codon")
    aa_de = dict(zip(codons, aas))
    espec["aminoacido_do_codon"] = [aa_de.get(g, "") if n == "codon" else ""
                                    for g, n in zip(espec["grupo"], espec["nivel"])]
    espec.to_csv(SAIDA / "especificidade.csv", index=False)

    # --- absorption by the latent/additive term ---------------------------
    absorcao = []
    for familia, linhas_cod, alvo in (("serina", ser, filtr), ("isoleucina", ile, mpc)):
        d_antes = float(contrastes(estatistica(centrado, linhas_cod), alvo[None, :])[0])
        for nome in ("xi_intra", "D_aditivo_intra", "xi_agregado"):
            d_dep = float(contrastes(estatistica(residuos[nome][0], linhas_cod), alvo[None, :])[0])
            absorcao.append({"familia": familia, "residuo_depois": nome,
                             "delta_antes_R_menos_m": d_antes, "delta_depois": d_dep,
                             "fracao_do_delta_absorvida": 1.0 - d_dep / d_antes})
        # what the latent term removed from the serine mean, per group
    termo_lat = centrado - xi_intra
    absorcao_df = pd.DataFrame(absorcao)
    absorcao_df.to_csv(SAIDA / "absorcao.csv", index=False)
    contraste_intercepto = float(contrastes(decomp["intercepto_a_c"], filtr[None, :])[0])
    contraste_termo_lat_ser = float(contrastes(estatistica(termo_lat, ser), filtr[None, :])[0])

    # --- GSE77617: CTA in the CUA mini-ORF ---------------------------------
    idx2 = np.flatnonzero(desenho["dataset"].to_numpy() == "GSE77617")
    lab2 = rotulo[idx2]
    R2 = R[:, idx2]
    defs2 = {
        "xi_agregado": (xi_pool[:, idx2], -1),
        "D_aditivo_intra": (comum.decomposicao_aditiva(R2)["interacao_D"], -1),
        "log_kobs_duplo_centro": (duplo_centro(np.log(K[:, idx2])), -1),
        "log_rho_duplo_centro": (duplo_centro(np.log(RHO[:, idx2])), +1),
    }
    gse2 = []
    for alvo_lab, codon_alvo, papel in (("rib_CUA", "CTA", "controle positivo"),
                                        ("rib_CUG", "CTG", "controle negativo")):
        j = int(np.flatnonzero(lab2 == alvo_lab)[0])
        outros = np.flatnonzero(lab2 != alvo_lab)
        i = int(np.flatnonzero(codons == codon_alvo)[0])
        for nome, (res, direc) in defs2.items():
            dif = res[:, j] - res[:, outros].mean(axis=1)
            posto = int(np.sum(dif * direc > dif[i] * direc + EPS)) + 1
            gse2.append({"papel": papel, "biblioteca": alvo_lab, "codon": codon_alvo,
                         "residuo": nome,
                         "direcao_prevista": "negativa" if direc < 0 else "positiva",
                         "diferenca_vs_media_das_outras_3": float(dif[i]),
                         "posto_entre_61_na_direcao_prevista": posto})
    jc = {l: int(np.flatnonzero(lab2 == l)[0]) for l in lab2}
    i_cta = int(np.flatnonzero(codons == "CTA")[0])
    rho2 = RHO[:, idx2]
    razao_cta = {l: float(rho2[i_cta, jc[l]] / rho2[i_cta, jc["rib_CTRL"]]) for l in ("rib_CUA", "rib_CUG")}
    gse2_df = pd.DataFrame(gse2)
    gse2_df.to_csv(SAIDA / "gse77617_cta.csv", index=False)

    # --- summary -------------------------------------------------------
    prim = testes[(testes.papel == "primario") & (testes.familia == "serina")].iloc[0].to_dict()
    resumo = {
        "pre_especificacao": {**{k: (list(v) if isinstance(v, tuple) else v) for k, v in PRIMARIO.items()},
                              "direcao_prevista": "negativa (taxa)",
                              "n_atribuicoes": comb(10, 3), "p_minimo": 1 / comb(10, 3)},
        "ancoras": {"max_abs_dif_Z1_intra_vs_arquivo": ancora_z,
                    "sinal_aplicado_ao_Z1_recalculado": sinal,
                    "max_abs_dif_xi_agregado_recalculado_vs_arquivo": ancora_xi_pool,
                    "razao_rho_CTA_vs_rib_CTRL": razao_cta},
        "primario": prim,
        "contraste_do_intercepto_a_c_filtradas_menos_congeladas": contraste_intercepto,
        "contraste_serina_do_termo_latente_intra": contraste_termo_lat_ser,
        "fracao_variancia_pc1_intra": fit_intra.extra["explained_variance_ratio"][0],
    }
    comum.escrever_json(SAIDA / "resumo.json", resumo)

    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(testes[["papel", "residuo", "familia", "subconjunto", "delta", "p_unicaudal_exato",
                  "p_unicaudal_direcao_oposta", "auc", "separacao_perfeita"]].to_string())
    print(tab_lib.drop(columns=["condition_id"]).round(4).to_string())
    print(espec[(espec.nivel == "aminoacido")].head(80).round(4).to_string())
    print(absorcao_df.round(4).to_string())
    print(gse2_df.to_string())
    print(resumo["ancoras"], contraste_intercepto, contraste_termo_lat_ser)


if __name__ == "__main__":
    main()
