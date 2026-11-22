#!/usr/bin/env python3
"""Recomputes, from the pipeline outputs, every number the article reports and compares it with
the value printed in the text (rounded to the same precision).

Usage (repository root):
    python reprodutibilidade/verificar_numeros_artigo.py
    python reprodutibilidade/verificar_numeros_artigo.py --baseline DIR --freeze DIR --latent DIR --revisao DIR

Output: table (article number | recomputed value | source file | OK/DIVERGE) and exit code 1 if
any number diverges.

The numbers cover the baseline and freeze-only stages, the latent stage, and the supplementary
analyses (`analises_revisao/saidas/`): library-level comparators, sequencing depth and rate
estimator, A-site assignment, codon-specific signatures, and the within-study analysis without
the chloramphenicol library.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

rows: list[tuple[str, str, str, str, bool]] = []


def check(label: str, printed: str, computed, fmt: str, source: str) -> None:
    text = fmt.format(computed) if not isinstance(computed, str) else computed
    rows.append((label, printed, text, source, text == printed))


def one(frame: pd.DataFrame, **filters):
    sel = frame
    for col, val in filters.items():
        sel = sel[sel[col] == val]
    if len(sel) != 1:
        raise ValueError(f"esperava 1 linha para {filters}, achei {len(sel)}")
    return sel.iloc[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", type=Path, default=ROOT / "modelo_baseline" / "saidas")
    ap.add_argument("--freeze", type=Path, default=ROOT / "modelo_baseline" / "saidas_robustez_freeze_mg")
    ap.add_argument("--latent", type=Path, default=ROOT / "modelo_latente" / "saidas")
    ap.add_argument("--revisao", type=Path, default=ROOT / "analises_revisao" / "saidas")
    a = ap.parse_args()
    B, F, L = a.baseline, a.freeze, a.latent / "pooled"

    # ---------------- baseline (baseline table and text)
    fs = json.loads((B / "fit_summary.json").read_text(encoding="utf-8"))
    cond = pd.read_csv(B / "conditions.csv")
    eff = pd.read_csv(B / "codon_effects.csv").sort_values("k_seq").reset_index(drop=True)
    check("CDS usados", "4141", fs["n_cds_used_annotation"], "{:d}", "fit_summary.json")
    check("condicoes", "14", fs["n_conditions"], "{:d}", "fit_summary.json")
    check("bibliotecas GSE119104", "10", int((cond.dataset == "GSE119104").sum()), "{:d}", "conditions.csv")
    check("bibliotecas GSE77617", "4", int((cond.dataset == "GSE77617").sum()), "{:d}", "conditions.csv")
    check("codons", "61", fs["n_codons"], "{:d}", "fit_summary.json")
    check("observacoes", "6.572e+06", float(fs["n_observations"]), "{:.3e}", "fit_summary.json")
    check("mu", "3.052", fs["mu"], "{:.3f}", "fit_summary.json")
    check("R2 posicional", "0.00141", fs["r2_sequence_baseline"], "{:.5f}", "fit_summary.json")
    check("R2 posicional (%)", "0.14", 100 * fs["r2_sequence_baseline"], "{:.2f}", "fit_summary.json")
    check("R2 codon x condicao", "0.117", fs["r2_codon_condition_means"], "{:.3f}", "fit_summary.json")
    check("R2 codon x condicao (%)", "11.7", 100 * fs["r2_codon_condition_means"], "{:.1f}", "fit_summary.json")
    check("Var(k_obs)", "238.79", fs["var_k_obs"], "{:.2f}", "fit_summary.json")
    check("Var(k_obs-k_seq)", "238.46", fs["var_k_obs_minus_k_seq"], "{:.2f}", "fit_summary.json")
    check("media quadratica residual", "2.455", fs["mse_residual_codon_condition"], "{:.3f}", "fit_summary.json")
    check("ordem de sum n_j beta_j", "1e-06", 10 ** np.floor(np.log10(abs(fs["weighted_sum_n_times_beta"]))), "{:.0e}", "fit_summary.json")
    check("genes retidos (min-max)", "3906-4079", f"{cond.n_genes_kept.min()}-{cond.n_genes_kept.max()}", "{}", "conditions.csv")
    check("deslocamento sitio A", "11/0", f"{fs['asite_shift_nt']['GSE119104']}/{fs['asite_shift_nt']['GSE77617']}", "{}", "fit_summary.json")
    check("janela 5'/3' (nt)", "27/12", f"{fs['exclude_start_nt']}/{fs['exclude_stop_nt']}", "{}", "fit_summary.json")
    check("sete codons mais lentos", "AGG,ATA,TCA,GGA,TCG,AGT,AGA", ",".join(eff.codon[:7]), "{}", "codon_effects.csv")
    check("quatro Pro com beta<0 (pooled)", "True", str(bool((eff[eff.amino_acid == "P"].beta_codon < 0).all())), "{}", "codon_effects.csv")

    # ---------------- freeze-only comparator
    ff = json.loads((F / "fit_summary.json").read_text(encoding="utf-8"))
    fe = pd.read_csv(F / "codon_effects_freeze_mg.csv")
    check("freeze: observacoes", "1.067e+06", float(ff["n_observations"]), "{:.3e}", "freeze fit_summary.json")
    check("freeze: mu", "1.406", ff["mu"], "{:.3f}", "freeze fit_summary.json")
    check("freeze: R2 entre bibliotecas de ajuste", "0.933", ff["r2_codon_condition_means"], "{:.3f}", "freeze fit_summary.json")
    check("freeze x pooled Spearman", "0.55", ff["alignment"]["spearman_kseq_freeze_vs_pooled"]["rho"], "{:.2f}", "freeze fit_summary.json")
    check("quatro Pro com beta<0 (freeze)", "True", str(bool((fe[fe.amino_acid == "P"].beta_codon < 0).all())), "{}", "codon_effects_freeze_mg.csv")
    res = pd.read_csv(F / "residuals_using_freeze_kseq.csv").set_index("codon").drop(columns="amino_acid")
    res.columns = [c.split("|")[-1] for c in res.columns]
    m = res.mean()
    fit_libs = ["freeze_Mg", "freeze_Mg_ctrl"]
    gse119 = [c for c in m.index if not c.startswith("rib_")]
    others = [c for c in gse119 if c not in fit_libs + ["media_Cm"]]
    check("freeze: residuo medio das bibliotecas de ajuste (|max|)", "0.05", m[fit_libs].abs().max(), "{:.2f}", "residuals_using_freeze_kseq.csv")
    check("freeze: residuo medio demais GSE119104", "-0.2..0.7", f"{m[others].min():.1f}..{m[others].max():.1f}", "{}", "residuals_using_freeze_kseq.csv")
    check("freeze: residuo medio media_Cm", "2.2", m["media_Cm"], "{:.1f}", "residuals_using_freeze_kseq.csv")
    rib = [c for c in m.index if c.startswith("rib_")]
    check("freeze: residuo medio GSE77617", "2.8..3.9", f"{m[rib].min():.1f}..{m[rib].max():.1f}", "{}", "residuals_using_freeze_kseq.csv")
    rho = pd.read_csv(F / "relative_occupancy_codon_condition.csv").set_index("codon")
    aa = rho.pop("amino_acid")
    rho.columns = [c.split("|")[-1] for c in rho.columns]
    ref = rho[fit_libs].mean(axis=1)

    def contrast(lib, a_):
        idx = aa[aa == a_].index
        return float((rho.loc[idx, lib] - ref[idx]).mean())

    filt = ["filter_Mg", "filter_NaCl", "filter_Mg_ctrl", "filter_Cm_ctrl"]
    ser = [contrast(x, "S") for x in filt]
    check("Ser nas filtradas sem droga", "0.54..2.80", f"{min(ser):.2f}..{max(ser):.2f}", "{}", "relative_occupancy_codon_condition.csv")
    check("Ser filter_Mg", "2.80", contrast("filter_Mg", "S"), "{:.2f}", "relative_occupancy_codon_condition.csv")
    check("Ser filter_Mg_ctrl", "0.60", contrast("filter_Mg_ctrl", "S"), "{:.2f}", "relative_occupancy_codon_condition.csv")
    check("Ser media_Cm", "-0.07", contrast("media_Cm", "S"), "{:.2f}", "relative_occupancy_codon_condition.csv")
    mpc = ["filter_Mg_MPC", "filter_Cm_MPC", "freeze_Mg_MPC"]
    ile = [contrast(x, "I") for x in mpc]
    check("Ile nas tres mupirocina", "1.55..2.95", f"{min(ile):.2f}..{max(ile):.2f}", "{}", "relative_occupancy_codon_condition.csv")
    check("Ile max nas demais", "0.11", max(contrast(x, "I") for x in rho.columns if x not in mpc), "{:.2f}", "relative_occupancy_codon_condition.csv")

    # ---------------- latent stage (text and comparator table)
    s = json.loads((L / "summary.json").read_text(encoding="utf-8"))
    va = s["step_1_8_8_a_variance_selected_model"]
    co = s["step_1_8_8_b_correlation_selected_model"]
    bo = s["step_1_8_8_d_bootstrap"]
    cr = s["step_1_8_6_condition_correlations"]
    spec = pd.read_csv(L / "spectrum_parallel_analysis.csv")
    bcv = pd.read_csv(L / "bi_cross_validation.csv")
    ppca = pd.read_csv(L / "ppca_em_model_selection.csv")
    load = pd.read_csv(L / "latent_loadings.csv")
    sys.path.insert(0, str(ROOT))
    from modelo_latente import config as lcfg  # noqa: E402
    from modelo_latente import run as lrun  # noqa: E402

    check("credito retido por m_i (%)", "2.31", 100 * (1 - va["mse_after_offset_only"] / va["mse_R_raw"]), "{:.2f}", "summary.json")
    check("componentes acima do nulo", "1", s["step_1_8_6_low_dimensional_structure"]["n_components_above_permutation_null"], "{:d}", "summary.json")
    check("PC1 / p95 do nulo", "2.24", spec.singular_value[0] / spec.permutation_null_p95_singular_value[0], "{:.2f}", "spectrum_parallel_analysis.csv")
    check("PC2 / p95 do nulo", "0.43", spec.singular_value[1] / spec.permutation_null_p95_singular_value[1], "{:.2f}", "spectrum_parallel_analysis.csv")
    check("parametros livres PPCA d=8", "522", int(ppca.loc[ppca.d == 8, "n_free_params"].iloc[0]), "{:d}", "ppca_em_model_selection.csv")
    check("EM no limite de iteracoes em todo d", "True", str(bool((ppca.em_iterations == lcfg.EM_MAX_ITER).all())), "{}", "ppca_em_model_selection.csv")
    check("celulas", "854", s["residual_matrix_shape"][0] * s["residual_matrix_shape"][1], "{:d}", "summary.json")
    check("PC1 variancia explicada (%)", "93.0", 100 * va["latent_gain_vs_offset_only"], "{:.1f}", "summary.json")
    check("PC1 variancia explicada (tabela)", "0.930", va["latent_gain_vs_offset_only"], "{:.3f}", "summary.json")
    check("media quadratica antes", "2.399", va["mse_after_offset_only"], "{:.3f}", "summary.json")
    check("media quadratica depois", "0.167", va["mse_after_latent"], "{:.3f}", "summary.json")
    b1 = bcv[bcv.d == 1].iloc[0]
    check("habilidade fora da amostra (%)", "89.9", 100 * b1.mean_skill, "{:.1f}", "bi_cross_validation.csv")
    check("habilidade fora da amostra (tabela)", "0.899", b1.mean_skill, "{:.3f}", "bi_cross_validation.csv")
    check("particoes com habilidade positiva", "500/500", f"{int(round(b1.fraction_splits_with_positive_skill * b1.n_splits))}/{int(b1.n_splits)}", "{}", "bi_cross_validation.csv")
    check("cosseno de subespaco bootstrap", "0.99999", bo["mean_subspace_cosine"], "{:.5f}", "summary.json")
    check("correlacao residual antes", "0.520", co["mean_absolute_offdiagonal_correlation_before"], "{:.3f}", "summary.json")
    check("correlacao residual depois", "0.363", co["mean_absolute_offdiagonal_correlation_after"], "{:.3f}", "summary.json")
    check("correlacao dentro do dataset", "+0.42", cr["mean_correlation_within_dataset"], "{:+.2f}", "summary.json")
    check("correlacao entre datasets", "-0.57", cr["mean_correlation_between_datasets"], "{:+.2f}", "summary.json")
    check("61 cargas com o mesmo sinal", "True", str(bool((np.sign(load.l1) == np.sign(load.l1.iloc[0])).all() and len(load) == 61)), "{}", "latent_loadings.csv")
    inv = L.parent / "invariance_check.json"
    if inv.exists():
        iv = json.loads(inv.read_text(encoding="utf-8"))
        check("invariancia: dp max por linha", "<=8.6e-16", "<=8.6e-16" if iv["max_row_sd_of_difference"] <= 8.65e-16 else f"{iv['max_row_sd_of_difference']:.1e}", "{}", "invariance_check.json")
        check("invariancia: matrizes centradas", "<=1.3e-15", "<=1.3e-15" if iv["max_abs_difference_offset_centred"] <= 1.35e-15 else f"{iv['max_abs_difference_offset_centred']:.1e}", "{}", "invariance_check.json")
    else:
        rows.append(("invariancia", "8.6e-16 / 1.3e-15", "invariance_check.json ausente", "-", False))
    w = s["complementary_within_dataset_analysis"][0]
    check("dentro do estudo: bibliotecas", "10", w["n_conditions"], "{:d}", "summary.json")
    check("dentro do estudo: componentes acima do nulo", "1", w["n_components_above_permutation_null"], "{:d}", "summary.json")
    check("dentro do estudo: PC1 (%)", "83.0", 100 * w["explained_variance_ratio_pc1"], "{:.1f}", "summary.json")
    z = pd.read_csv(L / "within_GSE119104_latent_scores_Zc.csv").set_index("label")["Z1"]
    zm = z[["filter_Cm_MPC", "filter_Mg_MPC", "freeze_Mg_MPC"]]
    check("filter_Mg dentro da faixa da mupirocina", "True", str(bool(zm.min() <= z["filter_Mg"] <= zm.max())), "{}", "within_GSE119104_latent_scores_Zc.csv")
    rest = z.drop(["media_Cm", "filter_Mg", "filter_NaCl", "filter_Cm_MPC", "filter_Mg_MPC", "freeze_Mg_MPC"])
    check("filter_NaCl entre mupirocina e demais controles", "True", str(bool(zm.max() < z["filter_NaCl"] < rest.min())), "{}", "within_GSE119104_latent_scores_Zc.csv")
    zs = z.sort_values()
    check("media_Cm: distancia ao vizinho", "12.6", zs.iloc[1] - zs.iloc[0] if zs.index[0] == "media_Cm" else float("nan"), "{:.1f}", "within_GSE119104_latent_scores_Zc.csv")

    # ---------------- reach and content of the axis (Results, "Three derived quantities" and GSE77617)
    def _mat(path):
        d = pd.read_csv(path).set_index("codon")
        d = d[[c for c in d.columns if "::" in c]]
        d.columns = [c.split("|")[-1] for c in d.columns]
        return d

    n_ic = _mat(B / "n_obs_codon_condition.csv").astype(float)
    r_ic = _mat(B / "residuals_codon_condition.csv")
    k_ic = _mat(B / "kobs_codon_condition.csv")
    xi_ic = _mat(L / "residuals_after_latent.csv")
    m_i = load.set_index("codon")["codon_offset_m_i"]
    zp = pd.read_csv(L / "latent_scores_Zc.csv").set_index("label")
    check("contagens por celula somam N", "True", str(bool(n_ic.values.sum() == fs["n_observations"])), "{}", "n_obs_codon_condition.csv")
    ss_pos = fs["var_k_obs_minus_k_seq"] * fs["n_observations"]
    ss_cell = float((n_ic * r_ic**2).values.sum())
    ss_offset = float((n_ic * r_ic.sub(m_i, axis=0) ** 2).values.sum())
    ss_latent = float((n_ic * xi_ic**2).values.sum())
    check("posicional: parcela entre celulas (%)", "1.29", 100 * ss_cell / ss_pos, "{:.2f}", "n_obs + residuals_codon_condition.csv")
    check("posicional: removido por g(Z_c) alem de m_i (%)", "1.23", 100 * (ss_offset - ss_latent) / ss_pos, "{:.2f}", "residuals_after_latent.csv")
    check("posicional: parcela dentro das celulas (%)", "98.7", 100 * (1 - ss_cell / ss_pos), "{:.1f}", "n_obs + residuals_codon_condition.csv")
    rc = r_ic.sub(m_i, axis=0)
    lib_mean = k_ic.mean(axis=0)
    check("r(Z1, media de k_obs da biblioteca)", "-0.9995", float(np.corrcoef(zp.loc[lib_mean.index, "Z1"], lib_mean)[0, 1]), "{:.4f}", "latent_scores_Zc.csv + kobs_codon_condition.csv")
    check("media de k_obs por biblioteca: minimo", "1.20 filter_Cm_ctrl", f"{lib_mean.min():.2f} {lib_mean.idxmin()}", "{}", "kobs_codon_condition.csv")
    check("media de k_obs por biblioteca: maximo", "5.21 rib_CUA", f"{lib_mean.max():.2f} {lib_mean.idxmax()}", "{}", "kobs_codon_condition.csv")
    check("intercepto por biblioteca: fracao removida (%)", "82.7", 100 * float(len(rc) * (rc.mean(axis=0) ** 2).sum() / (rc.values**2).sum()), "{:.1f}", "residuals_codon_condition.csv")
    pos = zp["n_rate_observations"]
    mpc_libs = ["filter_Mg_MPC", "filter_Cm_MPC", "freeze_Mg_MPC"]
    other119 = [x for x in zp.index[zp.dataset == "GSE119104"] if x not in mpc_libs]
    check("posicoes com densidade positiva, mupirocina", "2.3e+05..2.4e+05", f"{pos[mpc_libs].min():.1e}..{pos[mpc_libs].max():.1e}", "{}", "latent_scores_Zc.csv")
    check("posicoes com densidade positiva, demais GSE119104", "3.5e+05..5.6e+05", f"{pos[other119].min():.1e}..{pos[other119].max():.1e}", "{}", "latent_scores_Zc.csv")
    ribs = list(zp.index[zp.dataset == "GSE77617"])
    check("Z1 agregado: filter_Cm_ctrl e o maximo", "11.5", zp["Z1"].max() if zp["Z1"].idxmax() == "filter_Cm_ctrl" else float("nan"), "{:.1f}", "latent_scores_Zc.csv")
    check("Z1 agregado: GSE77617 sao os quatro menores", "-12.3..-21.6", f"{zp.loc[ribs, 'Z1'].max():.1f}..{zp.loc[ribs, 'Z1'].min():.1f}" if set(zp["Z1"].nsmallest(4).index) == set(ribs) else "nao", "{}", "latent_scores_Zc.csv")
    check("Z1: rib_CUA e o menor da serie", "True", str(bool(zp.loc[ribs, "Z1"].idxmin() == "rib_CUA")), "{}", "latent_scores_Zc.csv")
    check("Z1: |rib_CUA - rib_CTRL|", "3.9", abs(zp.loc["rib_CUA", "Z1"] - zp.loc["rib_CTRL", "Z1"]), "{:.1f}", "latent_scores_Zc.csv")
    check("Z1: |rib_CTRL - rib_37_WT|", "5.4", abs(zp.loc["rib_CTRL", "Z1"] - zp.loc["rib_37_WT", "Z1"]), "{:.1f}", "latent_scores_Zc.csv")
    rho_b = _mat(B / "relative_occupancy_codon_condition.csv")
    check("rho CTA: rib_CUA / rib_CTRL", "1.17", rho_b.loc["CTA", "rib_CUA"] / rho_b.loc["CTA", "rib_CTRL"], "{:.2f}", "relative_occupancy_codon_condition.csv")
    check("rho CTA: rib_CUG / rib_CTRL", "0.94", rho_b.loc["CTA", "rib_CUG"] / rho_b.loc["CTA", "rib_CTRL"], "{:.2f}", "relative_occupancy_codon_condition.csv")

    # ---------------- procedure parameters cited in Methods
    check("semente", "20260911", s["seed"], "{:d}", "summary.json")
    check("permutacoes", "2000", s["n_permutations"], "{:d}", "summary.json")
    check("reamostras bootstrap", "2000", s["n_bootstrap"], "{:d}", "summary.json")
    check("particoes de bi-validacao", "500", s["n_bi_cross_validation_splits"], "{:d}", "summary.json")
    n_cod = s["residual_matrix_shape"][0]
    check("codons retidos na bi-validacao", "31 of 61", f"{n_cod - int(round(n_cod * 0.5))} of {n_cod}", "{}", "validation.bi_cross_validation")
    check("condicoes retidas na bi-validacao", "4 of 14", f"4 of {s['residual_matrix_shape'][1]}", "{}", "validation.bi_cross_validation (col_holdout=4)")
    check("minimo de bibliotecas no estudo", "5", lrun.MIN_CONDITIONS_FOR_WITHIN_DATASET, "{:d}", "modelo_latente/run.py")
    gsm = cond.groupby("dataset")["gsm"].apply(lambda v: ",".join(sorted(v)))
    check("GSMs GSE119104", "GSM3358133..GSM3358142", f"{min(cond[cond.dataset=='GSE119104'].gsm)}..{max(cond[cond.dataset=='GSE119104'].gsm)}", "{}", "conditions.csv")
    check("GSMs GSE77617", "GSM2055244,GSM2462933,GSM2462934,GSM2462935", gsm["GSE77617"], "{}", "conditions.csv")

    revisao(a.revisao)

    w0 = max(len(r[0]) for r in rows)
    bad = 0
    for label, printed, comp, src, ok in rows:
        bad += not ok
        print(f"{'OK     ' if ok else 'DIVERGE'}  {label:<{w0}}  artigo={printed:<26} recalculado={comp:<26} [{src}]")
    print(f"\n{len(rows)} numeros conferidos; {bad} divergencia(s).")
    return 1 if bad else 0


def revisao(R: Path) -> None:
    """Numbers of the supplementary analyses (analises_revisao/saidas/)."""
    # ---- comparators (Methods; Results, "Condition-Level Structure of the Residual"; comparator table)
    mod = pd.read_csv(R / "r1" / "modelos.csv")
    bcv = pd.read_csv(R / "r1" / "bcv_resumo.csv")
    boot = pd.read_csv(R / "r1" / "bootstrap_resumo.csv")
    par = pd.read_csv(R / "r1" / "paralela_interacao.csv")
    s1 = json.loads((R / "r1" / "summary.json").read_text(encoding="utf-8"))
    f = "r1/modelos.csv"

    def md(matriz, modelo):
        return one(mod, matriz=matriz, modelo=modelo)

    def bc(matriz, comp):
        return one(bcv, matriz=matriz, comparacao=comp)

    for modelo, n in (("so_m", "61"), ("aditivo", "74"), ("multiplicativo_puro", "74"), ("posto1", "134"), ("aditivo_mais_pc1_interacao", "146")):
        check(f"parametros: {modelo}", n, int(md("pooled_R", modelo).n_parametros), "{:d}", f)
    check("parametros a mais: aditivo - so m_i", "13", int(md("pooled_R", "aditivo").n_parametros - md("pooled_R", "so_m").n_parametros), "{:d}", f)
    check("parametros a mais: posto 1 - aditivo", "60", int(md("pooled_R", "posto1").n_parametros - md("pooled_R", "aditivo").n_parametros), "{:d}", f)
    for modelo, pct, tab in (("aditivo", "82.7", "0.827"), ("multiplicativo_puro", "90.5", "0.905"), ("posto1", "93.0", "0.930"), ("aditivo_mais_pc1_interacao", None, "0.938")):
        v = md("pooled_R", modelo).fracao_removida_vs_so_m
        if pct:
            check(f"fracao removida (%): {modelo}", pct, 100 * v, "{:.1f}", f)
        check(f"fracao removida (tabela): {modelo}", tab, v, "{:.3f}", f)
    for modelo, val in (("so_m", "0.520"), ("aditivo", "0.520"), ("multiplicativo_puro", "0.395"), ("posto1", "0.363"), ("aditivo_mais_pc1_interacao", "0.367")):
        check(f"correlacao residual: {modelo}", val, md("pooled_R", modelo).correlacao_residual_media_abs, "{:.3f}", f)
    check("ganho do posto 1 sobre o aditivo (%)", "59.7", 100 * md("pooled_R", "posto1").ganho_vs_aditivo, "{:.1f}", f)
    check("ganho do posto 1 sobre o aditivo (tabela)", "0.597", md("pooled_R", "posto1").ganho_vs_aditivo, "{:.3f}", f)
    check("ganho da escala sobre o aditivo (%)", "45.3", 100 * md("pooled_R", "multiplicativo_puro").ganho_vs_aditivo, "{:.1f}", f)
    check("ganho da escala sobre o aditivo (tabela)", "0.453", md("pooled_R", "multiplicativo_puro").ganho_vs_aditivo, "{:.3f}", f)
    ratio_in = md("pooled_R", "multiplicativo_puro").ganho_vs_aditivo / md("pooled_R", "posto1").ganho_vs_aditivo
    ratio_out = bc("pooled_R", "skill_escala_vs_aditivo").skill_medio / bc("pooled_R", "skill_posto1_vs_aditivo").skill_medio
    check("escala reproduz ~3/4 do ganho (dentro e fora)", "True", str(bool(0.70 <= ratio_in <= 0.80 and 0.70 <= ratio_out <= 0.80)), "{}", f + " + r1/bcv_resumo.csv")
    check("log k_obs: aditivo (%)", "82.6", 100 * md("pooled_logkobs", "aditivo").fracao_removida_vs_so_m, "{:.1f}", f)
    check("log k_obs: ganho do posto 1 sobre o aditivo (%)", "14.6", 100 * md("pooled_logkobs", "posto1").ganho_vs_aditivo, "{:.1f}", f)
    check("GSE119104: aditivo (%)", "75.3", 100 * md("GSE119104_R", "aditivo").fracao_removida_vs_so_m, "{:.1f}", f)
    check("posicional: intercepto (pontos %)", "1.12", md("pooled_R", "aditivo").posicional_removido_vs_so_m_pct, "{:.2f}", f)
    check("posicional: posto 1 alem do intercepto (pontos %)", "0.11", md("pooled_R", "posto1").posicional_removido_vs_aditivo_pct, "{:.2f}", f)

    f = "r1/bcv_resumo.csv"
    for comp, pct, tab in (("skill_aditivo_vs_so_m", "81.1", "0.811"), ("skill_escala_vs_so_m", "87.8", "0.878"), ("skill_posto1_vs_so_m", "89.9", "0.899")):
        check(f"habilidade (%): {comp}", pct, 100 * bc("pooled_R", comp).skill_medio, "{:.1f}", f)
        check(f"habilidade (tabela): {comp}", tab, bc("pooled_R", comp).skill_medio, "{:.3f}", f)
        check(f"particoes positivas: {comp}", "500", int(bc("pooled_R", comp).n_particoes_positivas), "{:d}", f)
    check("habilidade posto 1 vs aditivo (%)", "48.8", 100 * bc("pooled_R", "skill_posto1_vs_aditivo").skill_medio, "{:.1f}", f)
    check("habilidade posto 1 vs aditivo (tabela)", "0.488", bc("pooled_R", "skill_posto1_vs_aditivo").skill_medio, "{:.3f}", f)
    check("particoes positivas: posto 1 vs aditivo", "492", int(bc("pooled_R", "skill_posto1_vs_aditivo").n_particoes_positivas), "{:d}", f)
    check("habilidade escala vs aditivo (%)", "37.7", 100 * bc("pooled_R", "skill_escala_vs_aditivo").skill_medio, "{:.1f}", f)
    check("habilidade escala vs aditivo (tabela)", "0.377", bc("pooled_R", "skill_escala_vs_aditivo").skill_medio, "{:.3f}", f)
    check("habilidade posto 1 vs escala (%)", "18.0", 100 * bc("pooled_R", "skill_posto1_vs_escala").skill_medio, "{:.1f}", f)
    check("particoes positivas: posto 1 vs escala", "439", int(bc("pooled_R", "skill_posto1_vs_escala").n_particoes_positivas), "{:d}", f)
    check("log k_obs: habilidade posto 1 vs aditivo (%)", "1.1", 100 * bc("pooled_logkobs", "skill_posto1_vs_aditivo").skill_medio, "{:.1f}", f)
    check("log k_obs: particoes positivas", "337", int(bc("pooled_logkobs", "skill_posto1_vs_aditivo").n_particoes_positivas), "{:d}", f)
    g = bc("GSE119104_R", "skill_posto1_vs_aditivo")
    check("GSE119104: habilidade media posto 1 vs aditivo", "-0.61", g.skill_medio, "{:.2f}", f)
    check("GSE119104: habilidade mediana", "-0.04", g.skill_mediano, "{:.2f}", f)
    check("GSE119104: particoes positivas", "192 of 500", f"{int(g.n_particoes_positivas)} of {int(g.n_particoes)}", "{}", f)

    f = "r1/summary.json"
    fi = s1["forma_da_interacao"]["pooled_R"]
    check("cos(escores do PC1 da interacao, a_c)", "0.953", fi["cos_escores_pc1_interacao_com_a_c"], "{:.3f}", f)
    check("norma das cargas na direcao constante (%)", "88.8", 100 * fi["cos_cargas_posto1_com_constante"] ** 2, "{:.1f}", f)
    cal = s1["poder_do_nulo_do_artigo_contra_matriz_aditiva"]
    check("matrizes aditivas simuladas", "20", cal["n_matrizes_aditivas_sinteticas"], "{:d}", f)
    check("permutacoes por matriz aditiva", "200", cal["n_permutacoes_cada"], "{:d}", f)
    check("matrizes aditivas com 1 componente", "20", sum(1 for v in cal["componentes_acima_do_nulo_por_matriz"] if v == 1), "{:d}", f)
    check("razao PC1/p95 media nas aditivas", "2.27", cal["razao_pc1_p95_media"], "{:.2f}", f)
    ap = s1["analise_paralela_interacao"]
    check("interacao: componentes no agregado (3 nulos)", "1,1,1", ",".join(str(ap[f"pooled_R|{n}"]["n_componentes_da_interacao_acima_do_nulo"]) for n in ("linhas", "sinais", "colunas")), "{}", f)
    check("interacao: componentes no GSE119104", "3", ap["GSE119104_R|linhas"]["n_componentes_da_interacao_acima_do_nulo"], "{:d}", f)
    check("interacao: componentes em log", "2", ap["pooled_logkobs|linhas"]["n_componentes_da_interacao_acima_do_nulo"], "{:d}", f)
    razoes = par[(par.matriz == "pooled_R") & (par.componente_interacao == 1)].razao_obs_p95
    check("interacao: razao PC1/p95 (faixa)", "1.52-1.70", f"{razoes.min():.2f}-{razoes.max():.2f}", "{}", "r1/paralela_interacao.csv")
    rz = s1["bootstrap"]["referencia_ruido_pooled"]
    check("referencia de ruido: matrizes x reamostras", "200x200", f"{rz['n_matrizes_nulas']}x{rz['n_bootstraps_por_matriz']}", "{}", f)
    check("cosseno do PC1 da interacao em ruido", "0.76", rz["cos_pc1_interacao_nulo_media"], "{:.2f}", f)
    check("cosseno com vetor constante (aditivo)", "1.000", one(boot, matriz="pooled_R", medida="cos_vetor_constante").media, "{:.3f}", "r1/bootstrap_resumo.csv")
    check("cosseno do PC1 da interacao", "0.9989", one(boot, matriz="pooled_R", medida="cos_cargas_pc1_interacao").media, "{:.4f}", "r1/bootstrap_resumo.csv")

    # ---- within-study (Results, "Holding Study Fixed"; Discussion, Sample)
    d5 = R / "r5"
    coe = pd.read_csv(d5 / "coerencia_10_e_9.csv")
    cor = pd.read_csv(d5 / "correcao_multiplos_testes.csv")
    adi = pd.read_csv(d5 / "coerencia_aditiva.csv")
    esp = pd.read_csv(d5 / "espectro_paralelo_10_e_9.csv")
    loo = pd.read_csv(d5 / "retirar_uma_biblioteca.csv")
    s5 = json.loads((d5 / "resumo_r5.json").read_text(encoding="utf-8"))
    e = s5["escala_12_6"]
    casos = list(esp.caso.unique())
    c10, c9 = casos[0], casos[1]
    f = "r5/espectro_paralelo_10_e_9.csv"
    check("intra: componentes sem media_Cm", "2", int(esp[esp.caso == c9]["significant_at_0.05"].cumprod().sum()), "{:d}", f)
    check("intra: PC1 sem media_Cm (%)", "60.1", 100 * one(esp, caso=c9, component=1).explained_variance_ratio, "{:.1f}", f)
    check("intra: PC2 sem media_Cm (%)", "34.3", 100 * one(esp, caso=c9, component=2).explained_variance_ratio, "{:.1f}", f)
    check("intra: componentes, 10 bibliotecas (r5)", "1", int(esp[esp.caso == c10]["significant_at_0.05"].cumprod().sum()), "{:d}", f)
    f = "r5/resumo_r5.json"
    check("12,6 unidades: distancia", "12.6", e["dz_12_6"], "{:.1f}", f)
    check("12,6 unidades: diferenca media de k_obs por codon", "1.53", e["diferenca_media_k_obs_por_codon"], "{:.2f}", f)
    check("12,6 unidades: k_obs medio media_Cm", "3.57", e["k_obs_medio_media_Cm"], "{:.2f}", f)
    check("12,6 unidades: k_obs medio do vizinho", "2.04", e["k_obs_medio_vizinho"], "{:.2f}", f)
    check("12,6 unidades: em DP das outras 9", "4.9", e["dz_em_sd_das_outras_9"], "{:.1f}", f)
    check("12,6 unidades: multiplo do par filter_Mg", "2.7", e["dz_em_multiplos_filter_Mg_pair"], "{:.1f}", f)
    check("12,6 unidades: multiplo do par freeze_Mg", "16", e["dz_em_multiplos_freeze_Mg_pair"], "{:.0f}", f)
    outras = loo[~loo.retirada.astype(str).str.contains("media_Cm")]
    f = "r5/retirar_uma_biblioteca.csv"
    check("retirar outra biblioteca: componentes", "1", ",".join(sorted({str(int(v)) for v in outras.n_componentes_acima_do_nulo})), "{}", f)
    check("retirar outra biblioteca: PC1 (%)", "82-88", f"{100 * outras.fracao_pc1.min():.0f}-{100 * outras.fracao_pc1.max():.0f}", "{}", f)
    f = "r5/coerencia_10_e_9.csv"
    d10 = one(coe, caso="10_bibliotecas", grouping="drug")
    d9 = one(coe, caso="9_sem_media_Cm", grouping="drug")
    check("droga: razao intra/inter, 10", "0.29", d10.razao_intra_inter, "{:.2f}", f)
    check("droga: p exato, 10", "0.0060", d10.p_exato, "{:.4f}", f)
    check("droga: razao intra/inter, 9", "0.58", d9.razao_intra_inter, "{:.2f}", f)
    check("droga: p exato, 9", "0.0119", d9.p_exato, "{:.4f}", f)
    check("droga: p minimo atingivel, 9", "0.0119", d9.p_minimo_atingivel, "{:.4f}", f)
    check("droga: rotulacoes possiveis, 9", "84", int(d9.n_rotulacoes_distintas), "{:d}", f)
    check("droga: observada e a mais extrema, 9", "1", int(d9.n_rotulacoes_com_razao_menor_ou_igual), "{:d}", f)
    check("droga so no PC1, 9", "0.048", one(coe, caso="9_sem_media_Cm_somente_PC1", grouping="drug").p_exato, "{:.3f}", f)
    for caso, grp, val in (("10_bibliotecas", "harvest", "0.65"), ("9_sem_media_Cm", "harvest", "0.56"), ("10_bibliotecas", "lysis", "0.23"), ("9_sem_media_Cm", "lysis", "0.61")):
        check(f"{grp}: p exato ({caso})", val, one(coe, caso=caso, grouping=grp).p_exato, "{:.2f}", f)
    f = "r5/correcao_multiplos_testes.csv"
    for caso, val in (("10_bibliotecas", "0.018"), ("9_sem_media_Cm", "0.036")):
        lin = one(cor, caso=caso, tipo_p="exato", grouping="drug")
        check(f"droga Bonferroni ({caso})", val, lin.p_bonferroni, "{:.3f}", f)
        check(f"droga Holm = Bonferroni ({caso})", "True", str(bool(abs(lin.p_holm - lin.p_bonferroni) < 1e-12)), "{}", f)
    check("droga com intercepto, 10", "0.0083", one(adi, caso="10_bibliotecas_aditivo", grouping="drug").p_exato, "{:.4f}", "r5/coerencia_aditiva.csv")
    check("droga com intercepto, 9", "0.083", one(adi, caso="9_sem_media_Cm_aditivo", grouping="drug").p_exato, "{:.3f}", "r5/coerencia_aditiva.csv")

    # ---- depth and estimator (Methods; Results, "Sequencing Depth and the Rate Estimator"; tables)
    d2 = R / "r2"
    corr = pd.read_csv(d2 / "correlacoes_zc_covariaveis.csv")
    cve = pd.read_csv(d2 / "covariaveis_vs_estudo.csv")
    rar = pd.read_csv(d2 / "rarefacao_resumo.csv")
    rep = pd.read_csv(d2 / "rarefacao_replicas.csv")
    est = pd.read_csv(d2 / "estimadores_resumo.csv")
    cov = pd.read_csv(d2 / "covariaveis_por_biblioteca.csv").set_index("label")
    s2 = json.loads((d2 / "resumo.json").read_text(encoding="utf-8"))

    def cz(recorte, x):
        return one(corr, alvo="z_c_agregado", recorte=recorte, covariavel=x, metodo="pearson")

    f = "r2/correlacoes_zc_covariaveis.csv"
    for recorte, x, rv, pv, rfmt in (
        ("N14", "log10_reads_ENA", "-0.76", "0.002", "{:.2f}"),
        ("N14", "fracao_posicoes_rho_pos", "-0.54", "0.047", "{:.2f}"),
        ("N14_parcial_estudo", "log10_reads_ENA", "-0.30", "0.41", "{:.2f}"),
        ("N14_parcial_estudo", "fracao_posicoes_rho_pos", "+0.08", "0.83", "{:+.2f}"),
        ("N10_GSE119104", "log10_reads_ENA", "-0.30", "0.47", "{:.2f}"),
        ("N10_GSE119104", "fracao_posicoes_rho_pos", "+0.02", "0.96", "{:+.2f}"),
    ):
        lin = cz(recorte, x)
        check(f"r {x} ({recorte})", rv, lin.r, rfmt, f)
        check(f"p {x} ({recorte})", pv, lin.p_perm, "{:.3f}" if len(pv) == 5 else "{:.2f}", f)
    check("r genes retidos (N14)", "-0.02", cz("N14", "n_genes_retidos").r, "{:.2f}", f)
    check("eta2 do estudo no escore", "0.84", one(cve, covariavel="z_c_agregado").eta2_estudo, "{:.2f}", "r2/covariaveis_vs_estudo.csv")
    check("eta2 do estudo em log reads", "0.54", one(cve, covariavel="log10_reads_ENA").eta2_estudo, "{:.2f}", "r2/covariaveis_vs_estudo.csv")
    check("permutacoes de Monte Carlo (N=14)", "100000", s2["n_perm_monte_carlo"], "{:d}", "r2/resumo.json")
    mupi = ["filter_Mg_MPC", "filter_Cm_MPC", "freeze_Mg_MPC"]
    check("mupirocina sao as tres mais rasas do GSE119104", "True", str(bool(set(cov[cov.dataset == "GSE119104"].ena_read_count.nsmallest(3).index) == set(mupi))), "{}", "r2/covariaveis_por_biblioteca.csv")
    # per-library covariate table
    for lab, zt, kt, rt, ft, gt in (
        ("media_Cm", "-6.9", "3.57", "36.6", "0.38", "4057"), ("filter_Mg", "5.5", "1.94", "28.3", "0.42", "4018"),
        ("filter_NaCl", "7.3", "1.73", "26.4", "0.45", "4025"), ("filter_Mg_ctrl", "10.6", "1.30", "14.4", "0.35", "4028"),
        ("filter_Mg_MPC", "6.4", "1.89", "8.2", "0.19", "4009"), ("filter_Cm_ctrl", "11.5", "1.20", "9.6", "0.28", "3944"),
        ("filter_Cm_MPC", "5.3", "2.04", "7.9", "0.20", "3976"), ("freeze_Mg_ctrl", "10.9", "1.31", "25.2", "0.41", "4046"),
        ("freeze_Mg_MPC", "6.9", "1.83", "6.4", "0.19", "3906"), ("freeze_Mg", "10.2", "1.40", "28.6", "0.45", "4071"),
        ("rib_37_WT", "-12.3", "4.13", "61.3", "0.60", "4079"), ("rib_CTRL", "-17.7", "4.76", "61.7", "0.48", "3986"),
        ("rib_CUA", "-21.6", "5.21", "68.7", "0.50", "4014"), ("rib_CUG", "-16.2", "4.53", "43.1", "0.40", "3949"),
    ):
        r = cov.loc[lab]
        got = f"{r.z_c:.1f} {r.media_k_obs:.2f} {r.ena_read_count / 1e6:.1f} {r.fracao_posicoes_rho_pos:.2f} {int(r.n_genes_retidos)}"
        check(f"tabela de profundidade: {lab}", f"{zt} {kt} {rt} {ft} {gt}", got, "{}", "r2/covariaveis_por_biblioteca.csv")
    # rarefaction
    f = "r2/rarefacao_resumo.csv"
    grade = rar[rar.alvo != "original"]
    check("rarefacao: bibliotecas", "filter_NaCl,freeze_Mg,freeze_Mg_ctrl", ",".join(sorted(rep.biblioteca.unique())), "{}", "r2/rarefacao_replicas.csv")
    check("rarefacao: fracoes da grade", "0.1,0.2,0.4,0.6,0.8", ",".join(f"{v:g}" for v in sorted(rep[rep.alvo == "grade"].fracao.round(3).unique())), "{}", "r2/rarefacao_replicas.csv")
    nrep = rep.groupby(["biblioteca", "alvo", "fracao"]).size().reset_index(name="n")
    check("rarefacao: replicas na grade / nos alvos", "5/10", f"{'/'.join(str(v) for v in sorted(set(nrep[nrep.alvo == 'grade'].n)))}/{'/'.join(str(v) for v in sorted(set(nrep[nrep.alvo.isin(['n_pos_mupirocina', 'reads_mupirocina'])].n)))}", "{}", "r2/rarefacao_replicas.csv")
    check("rarefacao: 61 codons no mesmo sentido", "1.00", grade["delta_k::fracao_codons_mesmo_sinal"].min(), "{:.2f}", f)
    check("rarefacao: SQ no termo uniforme (%)", "82-97", f"{100 * grade['delta_k::fracao_sq_no_deslocamento_uniforme'].min():.0f}-{100 * grade['delta_k::fracao_sq_no_deslocamento_uniforme'].max():.0f}", "{}", f)
    check("rarefacao: cosseno com as cargas", "0.86-0.92", f"{grade['delta_k::cos_com_cargas_l'].min():.2f}-{grade['delta_k::cos_com_cargas_l'].max():.2f}", "{}", f)
    check("rarefacao: freeze_Mg original", "1.40", one(rar, biblioteca="freeze_Mg", alvo="original")["media_k_obs::media"], "{:.2f}", f)
    check("rarefacao: freeze_Mg a 10%", "0.77", one(rar[rar.fracao.round(3) == 0.1], biblioteca="freeze_Mg", alvo="grade")["media_k_obs::media"], "{:.2f}", f)
    alvo = rar[rar.alvo == "n_pos_mupirocina"]["z_c_projetado::delta"]
    check("rarefacao ate n_pos da mupirocina: delta z", "+3.3 to +5.5", f"{alvo.min():+.1f} to {alvo.max():+.1f}", "{}", f)
    check("rarefacao: maior desvio, k_obs", "5.5", grade["z_c_projetado::delta"].abs().max(), "{:.1f}", f)
    check("rarefacao: maior desvio, (i)", "5.0", grade["escore::i_media_log_k::N14::delta"].abs().max(), "{:.1f}", f)
    check("rarefacao: maior desvio, (ii) < 0.14", "True", str(bool(grade["escore::ii_razao_1_sobre_media_rho::N14::delta"].abs().max() < 0.14)), "{}", f)
    check("rarefacao: maior desvio, (iii) < 0.09", "True", str(bool(grade["escore::iii_log_razao::N14::delta"].abs().max() < 0.09)), "{}", f)
    curva = s2["curva_de_rarefacao"]["escore::k_obs_original::N14"]
    check("lacuna nao tratadas - mupirocina", "3.13", curva["gap_nao_tratadas_menos_mupirocina_observado"], "{:.2f}", "r2/resumo.json")
    check("diferenca entre estudos no escore", "23.7", curva["diferenca_entre_estudos_observada"], "{:.1f}", "r2/resumo.json")
    check("parte coberta pela curva de rarefacao", "2.7", curva["diferenca_entre_estudos_observada"] - curva["diferenca_entre_estudos_ajustada_extrapolada"], "{:.1f}", "r2/resumo.json")
    # estimators (table)
    f = "r2/estimadores_resumo.csv"
    for nome, ad, p1, sinal, comp, eta, sobrep, postos in (
        ("k_obs_original", "0.827", "0.930", "61", "1", "0.84", "False", "2, 4, 5"),
        ("i_media_log_k", "0.617", "0.698", "61", "2", "0.65", "False", "7, 8, 10"),
        ("ii_razao_1_sobre_media_rho", "0.185", "0.440", "51", "2", "0.29", "True", "8, 9, 10"),
        ("iii_log_razao", "0.087", "0.522", "41", "2", "0.09", "True", "8, 9, 10"),
    ):
        lin = one(est, estimador=nome, subconjunto="N14")
        got = f"{lin.fracao_aditivo:.3f} {lin.fracao_posto1:.3f} {int(lin.n_cargas_mesmo_sinal_max)} {int(lin.pa_matriz_n_componentes)} {lin.eta2_estudo_no_escore:.2f} {lin.escore_faixas_se_sobrepoem} {str(lin.postos_mupirocina_de_10).replace(';', ', ')}"
        check(f"estimador {nome}", f"{ad} {p1} {sinal} {comp} {eta} {sobrep} {postos}", got, "{}", f)
    check("estimadores imunes: aditivo (%)", "18.5/8.7", f"{100 * one(est, estimador='ii_razao_1_sobre_media_rho', subconjunto='N14').fracao_aditivo:.1f}/{100 * one(est, estimador='iii_log_razao', subconjunto='N14').fracao_aditivo:.1f}", "{}", f)

    def ser_ile(nome):
        top = str(one(est, estimador=nome, subconjunto="N14").pc1_interacao_codons_top).split(";")
        cods = [t.split("(")[0] for t in top]
        sinais = {t.split("(")[0]: float(t.split("(")[1].rstrip(")")) for t in top}
        sers = {"TCT", "TCC", "TCA", "TCG", "AGT", "AGC"}
        iles = {"ATT", "ATC", "ATA"}
        n = sum(c in sers | iles for c in cods)
        opostos = all(sinais[c] < 0 for c in cods if c in sers) and all(sinais[c] > 0 for c in cods if c in iles)
        return n, len(cods), opostos

    n2, t2, o2 = ser_ile("ii_razao_1_sobre_media_rho")
    n3, t3, o3 = ser_ile("iii_log_razao")
    check("PC1 da interacao: Ser/Ile entre os 6 maiores, (ii)", "5 of 6", f"{n2} of {t2}", "{}", f)
    check("PC1 da interacao: Ser/Ile entre os 6 maiores, (iii)", "6 of 6", f"{n3} of {t3}", "{}", f)
    check("PC1 da interacao: Ser e Ile em sinais opostos", "True", str(bool(o2 and o3)), "{}", f)

    # ---- A site (Methods; Results, "A-Site Assignment"; Discussion, Confounding)
    d4 = R / "r4"
    var = pd.read_csv(d4 / "variantes.csv")
    sweep = pd.read_csv(d4 / "variantes_varredura_deslocamento.csv")
    picos = pd.read_csv(d4 / "picos.csv")
    ce = json.loads((d4 / "comparacao_estudos.json").read_text(encoding="utf-8"))
    dif = "dif_media_kobs_77617_menos_119104"
    ref = var[var.variante.str.startswith("ref_")].iloc[0]
    ile_sem_cm = picos[(picos.sinal == "Ile_contraste") & ~picos.biblioteca.str.contains("Cm")].s_pico
    ser_sem_cm = picos[(picos.sinal == "Ser_contraste") & ~picos.biblioteca.str.contains("Cm")].s_pico
    check("picos Ile e Ser sem Cm na lise (nt)", "10-11", f"{min(ile_sem_cm.min(), ser_sem_cm.min())}-{max(ile_sem_cm.max(), ser_sem_cm.max())}", "{}", "r4/picos.csv")
    lams = [ce["filter_Cm_ctrl|xcorr_lambda"]["argmax_nt"], ce["media_GSE119104_sem_MPC|xcorr_lambda"]["argmax_nt"],
            ce["sinais_ancorados_na_extremidade_3"]["Pro"]["defasagem_A_menos_B"], ce["sinais_ancorados_na_extremidade_3"]["Ile"]["defasagem_A_menos_B"]]
    check("deslocamento 0 do GSE77617 equivale a (nt)", "12-14", f"{min(lams)}-{max(lams)}", "{}", "r4/comparacao_estudos.json")
    check("diferenca entre estudos na media de k_obs", "2.83", ref[dif], "{:.2f}", "r4/variantes.csv")
    check("variantes de deslocamento", "122", len(sweep), "{:d}", "r4/variantes_varredura_deslocamento.csv")
    check("varredura: faixa da diferenca", "2.61-3.18", f"{sweep[dif].min():.2f}-{sweep[dif].max():.2f}", "{}", "r4/variantes_varredura_deslocamento.csv")
    check("varredura: intercepto separa em todas", "True", str(bool(sweep.separacao_perfeita_no_intercepto.all())), "{}", "r4/variantes_varredura_deslocamento.csv")
    check("varredura: corr. dentro", "+0.13 to +0.77", f"{sweep.corr_dentro_estudo.min():+.2f} to {sweep.corr_dentro_estudo.max():+.2f}", "{}", "r4/variantes_varredura_deslocamento.csv")
    check("varredura: corr. entre", "-0.30 to -0.84", f"{sweep.corr_entre_estudos.max():+.2f} to {sweep.corr_entre_estudos.min():+.2f}", "{}", "r4/variantes_varredura_deslocamento.csv")
    check("GSE77617 no equivalente ao sitio A", "2.66", var[var.variante.str.startswith("A11_B-1")].iloc[0][dif], "{:.2f}", "r4/variantes.csv")
    jan = var[var.variante.str.startswith("A_janela[12")]
    wins = jan.variante.str.extract(r"w=(\d+)")[0].astype(int)
    check("suavizacao: janelas (nt)", "2-16", f"{wins.min()}-{wins.max()}", "{}", "r4/variantes.csv")
    check("suavizacao: faixa da diferenca", "2.38-2.55", f"{jan[dif].min():.2f}-{jan[dif].max():.2f}", "{}", "r4/variantes.csv")
    check("suavizacao: reducao maxima (%)", "16", 100 * (1 - jan[dif].min() / ref[dif]), "{:.0f}", "r4/variantes.csv")

    # ---- codon-specific signatures (Methods; Results, "Codon-Specific Signatures")
    d3 = R / "r3"
    t = pd.read_csv(d3 / "testes_exatos.csv")
    espc = pd.read_csv(d3 / "especificidade.csv")
    ab = pd.read_csv(d3 / "absorcao.csv")
    cta = pd.read_csv(d3 / "gse77617_cta.csv")
    lib = pd.read_csv(d3 / "estatistica_por_biblioteca.csv")
    s3 = json.loads((d3 / "resumo.json").read_text(encoding="utf-8"))
    f = "r3/testes_exatos.csv"
    ser10 = t[(t.familia == "serina") & (t.contraste == "filtradas vs congeladas") & (t.subconjunto == "10 bibliotecas GSE119104")]
    prim = one(ser10, residuo="xi_intra")
    check("serina: contraste primario", "-0.674", prim.delta, "{:.3f}", f)
    check("serina: p exato", "0.0083", prim.p_unicaudal_exato, "{:.4f}", f)
    check("serina: atribuicoes", "120", int(prim.n_atribuicoes), "{:d}", f)
    check("serina: separacao perfeita", "True", str(bool(prim.separacao_perfeita)), "{}", f)
    col = [c for c in lib.columns if "serina" in c.lower() and "xi_intra" in c][0]
    fz = ["freeze_Mg_ctrl", "freeze_Mg_MPC", "freeze_Mg"]
    lab = lib.set_index(lib.columns[0]) if "label" not in lib.columns else lib.set_index("label")
    check("serina: margem entre os grupos", "0.07", lab.loc[fz, col].min() - lab.drop(index=fz)[col].max(), "{:.2f}", "r3/estatistica_por_biblioteca.csv")
    for residuo, dv, pv in (("D_aditivo_intra", "-0.656", "0.0083"), ("log_kobs_duplo_centro", None, "0.0083"), ("xi_agregado", None, "0.117"), ("log_rho_duplo_centro", None, "0.058"), ("R_menos_m_antes_do_latente", "-0.216", "0.33")):
        lin = one(ser10, residuo=residuo)
        if dv:
            check(f"serina: contraste ({residuo})", dv, lin.delta, "{:.3f}", f)
        check(f"serina: p ({residuo})", pv, lin.p_unicaudal_exato, "{:.4f}" if pv == "0.0083" else ("{:.3f}" if len(pv) == 5 else "{:.2f}"), f)
    sem9 = one(t[t.subconjunto == "9 bibliotecas (sem media_Cm)"], residuo="xi_intra (restrito a 9)")
    check("serina: residuo restrito a 9 (p; separacao)", "0.012 True", f"{sem9.p_unicaudal_exato:.3f} {bool(sem9.separacao_perfeita)}", "{}", f)
    sem = one(t[t.subconjunto == "9 bibliotecas (sem media_Cm)"], residuo="xi_reajustado_9")
    check("serina: reajuste sem media_Cm (p; separacao)", "0.036 False", f"{sem.p_unicaudal_exato:.3f} {bool(sem.separacao_perfeita)}", "{}", f)
    ile = t[(t.familia == "isoleucina") & (t.subconjunto == "10 bibliotecas GSE119104")]
    check("isoleucina: p em todas as definicoes", "0.0083", f"{ile.p_unicaudal_exato.max():.4f}" if np.allclose(ile.p_unicaudal_exato, 1 / 120) else "nao", "{}", f)
    gli = one(t[(t.familia == "glicina") & (t.subconjunto == "10 bibliotecas GSE119104")], residuo="xi_intra")
    check("glicina: p", "0.042", gli.p_unicaudal_exato, "{:.3f}", f)
    f = "r3/especificidade.csv"
    aa_ = espc[(espc.contraste == "filtradas vs congeladas") & (espc.residuo == "xi_intra") & (espc.nivel == "aminoacido")]
    check("serina: posto entre 20 aminoacidos", "1", int(one(aa_, grupo="S").posto_por_delta), "{:d}", f)
    check("glicina: posto entre 20 aminoacidos", "2", int(one(aa_, grupo="G").posto_por_delta), "{:d}", f)
    check("aminoacidos avaliados", "20", len(aa_), "{:d}", f)
    check("serina: p ajustado (20)", "0.10", one(aa_, grupo="S").p_ajustado_minp, "{:.2f}", f)
    cod = espc[(espc.contraste == "filtradas vs congeladas") & (espc.residuo == "xi_intra") & (espc.nivel == "codon")]
    check("codons avaliados", "61", len(cod), "{:d}", f)
    check("postos dos seis codons de serina", "1,2,3,4,5,6", ",".join(str(int(v)) for v in sorted(cod[cod.aminoacido_do_codon == "S"].posto_por_delta)), "{}", f)
    check("serina antes de retirar o nivel (absorcao.csv)", "-0.216", one(ab, familia="serina", residuo_depois="xi_intra").delta_antes_R_menos_m, "{:.3f}", "r3/absorcao.csv")
    check("contraste do intercepto, filtradas - congeladas", "+0.44", s3["contraste_do_intercepto_a_c_filtradas_menos_congeladas"], "{:+.2f}", "r3/resumo.json")
    check("CTA: posto em taxa (residuo agregado)", "1", int(one(cta, biblioteca="rib_CUA", residuo="xi_agregado").posto_entre_61_na_direcao_prevista), "{:d}", "r3/gse77617_cta.csv")
    ctg = cta[(cta.biblioteca == "rib_CUG") & (cta.direcao_prevista == "negativa")].posto_entre_61_na_direcao_prevista
    check("CTG: postos na biblioteca CUG", "31-33", f"{int(ctg.min())}-{int(ctg.max())}", "{}", "r3/gse77617_cta.csv")

    # ---- references (Methods)
    d6 = R / "r6"
    dr = json.loads((d6 / "diferencas_referencias_resumo.json").read_text(encoding="utf-8"))
    f = "r6/diferencas_referencias_resumo.json"
    check("eventos entre NC_000913.2 e .3", "7", dr["n_eventos"], "{:d}", f)
    check("diferenca liquida (nt)", "1977", dr["soma_delta_comprimento"], "{:d}", f)
    check("CDS sobrepostos a um evento", "3 of 4141", f"{dr['n_cds_sobrepostos']} of {dr['n_cds_artigo']}", "{}", f)


if __name__ == "__main__":
    sys.exit(main())
