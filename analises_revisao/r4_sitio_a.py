"""R4: A-site assignment in the two studies (GSE119104 x GSE77617).

Question: is the A-site assignment the same in both studies? If not, does that explain the
study-of-origin axis?

Shift convention (identical to `modelo_baseline.observables._shifted_lookup`): the value
assigned to nucleotide i of a codon is the track value at position i + s along the 5'->3'
direction of the mRNA (strand '+': genomic coordinate + s; strand '-': genomic coordinate - s).
Thus s > 0 reads the track s nt DOWNSTREAM (3') of the codon, i.e. it moves the density s nt
towards 5'. With density aligned by the 3' end (GSE119104), s = 11 / 14 / 17 places the codon in
the A / P / E sites (Mohammad et al. 2019, p. 37). The baseline uses s = 11 for GSE119104 and
s = 0 for GSE77617.

Usage: python -m analises_revisao.r4_sitio_a
Outputs: analises_revisao/saidas/r4/
"""

from __future__ import annotations

import hashlib
import json
import time

import numpy as np
import pandas as pd

from analises_revisao import comum
from modelo_baseline import config as bcfg
from modelo_baseline.annotation import codon_nt_indices, load_cds_records, load_genome_length

DESLOCS = np.arange(-30, 31)  # nt
I0 = 30  # index of s = 0 in DESLOCS
SENTIDO = sorted(c for c, a in bcfg.STANDARD_GENETIC_CODE.items() if a != "*")
AA = np.array([bcfg.STANDARD_GENETIC_CODE[c] for c in SENTIDO])
N_BOOT = 200
LAGS_ACF = np.arange(0, 41)

# GEO labels used in the positive controls
PARES_MUPIROCINA = [
    ("filter_Mg_MPC", "filter_Mg_ctrl"),
    ("filter_Cm_MPC", "filter_Cm_ctrl"),
    ("freeze_Mg_MPC", "freeze_Mg_ctrl"),
]
FILTRADAS_SEM_DROGA = ["filter_Mg", "filter_NaCl", "filter_Mg_ctrl", "filter_Cm_ctrl"]
CONGELADAS_REF = ["freeze_Mg", "freeze_Mg_ctrl"]
REF_PROTOCOLO = "filter_Cm_ctrl"  # GSE119104 with the protocol nominally closest to GSE77617


# --------------------------------------------------------------------------------------------
# CDS geometry (concatenated in the order of load_cds_records, as in the baseline)
# --------------------------------------------------------------------------------------------


def geometria(records, genome_len):
    idx, sinal, gene_cod, pos_cod, cid = [], [], [], [], []
    ncod = np.zeros(len(records), dtype=np.int64)
    mapa = {c: k for k, c in enumerate(SENTIDO)}
    for g, rec in enumerate(records):
        ind = np.asarray(codon_nt_indices(rec["start"], rec["end"], rec["strand"], genome_len), dtype=np.int64)
        assert len(ind) == len(rec["seq"])
        n = len(ind) // 3
        ncod[g] = n
        idx.append(ind)
        sinal.append(np.full(len(ind), 1 if rec["strand"] == "+" else -1, dtype=np.int8))
        gene_cod.append(np.full(n, g, dtype=np.int64))
        pos_cod.append(np.arange(n, dtype=np.int64))
        seq = rec["seq"]
        cid.append(np.array([mapa.get(seq[3 * k : 3 * k + 3], -1) for k in range(n)], dtype=np.int64))
    idx = np.concatenate(idx)
    sinal = np.concatenate(sinal)
    gene_cod = np.concatenate(gene_cod)
    pos_cod = np.concatenate(pos_cod)
    cid = np.concatenate(cid)
    drop0 = bcfg.EXCLUDE_START_NT // 3
    drop1 = bcfg.EXCLUDE_STOP_NT // 3
    interior = (pos_cod >= drop0) & (pos_cod < ncod[gene_cod] - drop1)
    usar = interior & (cid >= 0)
    return {
        "idx_mais": idx[sinal == 1],
        "pos_mais": np.flatnonzero(sinal == 1),
        "idx_menos": idx[sinal == -1],
        "pos_menos": np.flatnonzero(sinal == -1),
        "n_nt": len(idx),
        "gene_cod": gene_cod,
        "gene_nt": np.repeat(gene_cod, 3),
        "ncod": ncod,
        "interior": interior,
        "usar": usar,
        "cid": cid,
        "n_genes": len(records),
        "n_interior": np.bincount(gene_cod[interior], minlength=len(records)),
        "N_gene_codon": np.bincount(
            gene_cod[usar] * 61 + cid[usar], minlength=len(records) * 61
        ).reshape(len(records), 61),
    }


def valores_nt(trilhas, geo, s):
    """Track read on the CDS with shift s (same rule as _shifted_lookup)."""
    v = np.zeros(geo["n_nt"], dtype=np.float64)
    for fita, idx, pos, d in (("+", geo["idx_mais"], geo["pos_mais"], s), ("-", geo["idx_menos"], geo["pos_menos"], -s)):
        tr = trilhas[fita]
        lk = idx + d
        ok = (lk >= 0) & (lk < len(tr))
        v[pos[ok]] = tr[lk[ok]]
    return v


def extrair(trilhas, geo, deslocs, por_gene=False):
    """Per-codon statistics for the occupancy read as the mean of the tracks over `deslocs`.

    With a single shift it reproduces gene_codon_occupancy + relative_occupancy_and_rate +
    add_gene_observations. With several, the per-nt occupancy is the mean of the shifted
    reads (rectangular window = smoothing of the 3' density into coverage).
    """
    deslocs = list(deslocs)
    v = valores_nt(trilhas, geo, deslocs[0])
    for s in deslocs[1:]:
        v = v + valores_nt(trilhas, geo, s)
    if len(deslocs) > 1:
        v = v / len(deslocs)
    occ = v.reshape(-1, 3).mean(axis=1)
    G = geo["n_genes"]
    gc = geo["gene_cod"]
    gmean = np.bincount(gc, occ, minlength=G) / geo["ncod"]
    it = geo["interior"]
    isum = np.bincount(gc[it], occ[it], minlength=G)
    npos = np.bincount(gc[it], (occ[it] > 0).astype(np.float64), minlength=G)
    kept = (gmean > bcfg.MIN_GENE_MEAN_OCCUPANCY) & (isum / np.maximum(geo["n_interior"], 1) > 0) & (npos > 0)
    sel = geo["usar"] & kept[gc]
    rho = occ[sel] / gmean[gc[sel]]
    cid = geo["cid"][sel]
    pos = rho > 0
    out = {
        "sum_rho": np.bincount(cid, rho, minlength=61),
        "n_rho": np.bincount(cid, minlength=61).astype(np.float64),
        "sum_k": np.bincount(cid[pos], 1.0 / rho[pos], minlength=61),
        "n_k": np.bincount(cid[pos], minlength=61).astype(np.float64),
        "n_genes": int(kept.sum()),
    }
    if por_gene:
        out["S"] = np.bincount(gc[sel] * 61 + cid, rho, minlength=G * 61).reshape(G, 61)
        out["kept"] = kept
    return out


def acf_trilha(trilhas, geo, lags):
    """Autocorrelation (Pearson) of the track along the CDS, per nt, in pairs within the same gene.

    Each nt is normalized by the mean of its gene (genes with mean > 0)."""
    v = valores_nt(trilhas, geo, 0)
    gn = geo["gene_nt"]
    m = np.bincount(gn, v, minlength=geo["n_genes"]) / (3 * geo["ncod"])
    ok = m[gn] > 0
    z = np.where(ok, v / np.where(m[gn] > 0, m[gn], 1.0), np.nan)
    res = []
    for lag in lags:
        a, b = z[: len(z) - lag], z[lag:]
        same = (gn[: len(z) - lag] == gn[lag:]) & np.isfinite(a) & np.isfinite(b)
        res.append(float(np.corrcoef(a[same], b[same])[0, 1]))
    return np.array(res)


# --------------------------------------------------------------------------------------------
# profiles
# --------------------------------------------------------------------------------------------


def enriquecimento(T, D):
    """log2 of the mean rho per codon divided by the global mean. T, D: (..., n_s, 61)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        E = T / D
        Gm = T.sum(-1) / D.sum(-1)
        return np.log2(E / Gm[..., None])


def enriq_grupo(T, D, mask):
    with np.errstate(divide="ignore", invalid="ignore"):
        e = T[..., mask].sum(-1) / D[..., mask].sum(-1)
        Gm = T.sum(-1) / D.sum(-1)
        return np.log2(e / Gm)


def var_global(T, D):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.var(np.log(T / D), axis=-1)


def pico(perfil):
    """argmax and full width at half maximum (FWHM, nt) over a baseline = median of the profile."""
    f = np.asarray(perfil, dtype=float)
    k = int(np.nanargmax(f))
    base = float(np.nanmedian(f))
    meia = base + (f[k] - base) / 2.0
    lo = k
    while lo - 1 >= 0 and f[lo - 1] >= meia:
        lo -= 1
    hi = k
    while hi + 1 < len(f) and f[hi + 1] >= meia:
        hi += 1
    return {
        "s_pico": int(DESLOCS[k]),
        "altura": float(f[k]),
        "linha_base": base,
        "fwhm_nt": int(hi - lo + 1),
        "fwhm_de": int(DESLOCS[lo]),
        "fwhm_ate": int(DESLOCS[hi]),
    }


def corr(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def ic(x):
    x = np.asarray(x, dtype=float)
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


# --------------------------------------------------------------------------------------------
# variants of the k_obs table and decomposition
# --------------------------------------------------------------------------------------------


def kobs_de(st):
    with np.errstate(divide="ignore", invalid="ignore"):
        return st["sum_k"] / st["n_k"]


def metricas_variante(M, datasets, rotulos):
    dec = comum.decomposicao_aditiva(M)
    a = dec["intercepto_a_c"]
    z = dec["escores_posto1"]
    media_k = M.mean(axis=0)
    isA = np.array([d == "GSE119104" for d in datasets])
    xc = M - M.mean(axis=1, keepdims=True)
    C = np.corrcoef(xc.T)
    iu = np.triu_indices(M.shape[1], 1)
    same = isA[iu[0]] == isA[iu[1]]

    def eta2(v):
        tot = np.sum((v - v.mean()) ** 2)
        ent = isA.sum() * (v[isA].mean() - v.mean()) ** 2 + (~isA).sum() * (v[~isA].mean() - v.mean()) ** 2
        return float(ent / tot)

    return {
        "fracao_aditivo": dec["fracao_aditivo"],
        "fracao_posto1": dec["fracao_posto1"],
        "media_kobs_GSE119104": float(media_k[isA].mean()),
        "media_kobs_GSE77617": float(media_k[~isA].mean()),
        "dif_media_kobs_77617_menos_119104": float(media_k[~isA].mean() - media_k[isA].mean()),
        "corr_dentro_estudo": float(C[iu][same].mean()),
        "corr_entre_estudos": float(C[iu][~same].mean()),
        "eta2_estudo_no_intercepto": eta2(a),
        "eta2_estudo_no_escore_posto1": eta2(z),
        "max_a_c_GSE119104": float(a[isA].max()),
        "min_a_c_GSE119104": float(a[isA].min()),
        "max_a_c_GSE77617": float(a[~isA].max()),
        "min_a_c_GSE77617": float(a[~isA].min()),
        "separacao_perfeita_no_intercepto": bool(a[~isA].min() > a[isA].max() or a[~isA].max() < a[isA].min()),
        "corr_escore_posto1_media_kobs": float(np.corrcoef(z, media_k)[0, 1]),
    }, a, z, media_k


# --------------------------------------------------------------------------------------------


def main():
    t0 = time.time()
    out = comum.dir_saida("r4")
    genome_len = load_genome_length()
    records = load_cds_records()
    geo = geometria(records, genome_len)
    libs = list(comum.bibliotecas())
    ids = [lib[2] for lib in libs]
    datasets = [lib[0] for lib in libs]
    rotulos = [lib[1]["label"] for lib in libs]
    nativo = [lib[3] for lib in libs]
    L, nS = len(libs), len(DESLOCS)
    print(f"geometria: {geo['n_genes']} genes, {geo['n_nt']} nt ({time.time() - t0:.0f}s)", flush=True)

    rng = np.random.default_rng(comum.semente(4, 0))
    W = rng.multinomial(geo["n_genes"], np.full(geo["n_genes"], 1.0 / geo["n_genes"]), size=N_BOOT).astype(np.float64)

    sum_rho = np.zeros((L, nS, 61))
    n_rho = np.zeros((L, nS, 61))
    sum_k = np.zeros((L, nS, 61))
    n_k = np.zeros((L, nS, 61))
    n_genes = np.zeros((L, nS), dtype=int)
    Tb = np.zeros((L, N_BOOT, nS, 61), dtype=np.float32)
    Db = np.zeros((L, N_BOOT, nS, 61), dtype=np.float32)
    acf = np.zeros((L, len(LAGS_ACF)))
    trilhas_cache = {}
    N = geo["N_gene_codon"].astype(np.float64)
    for li, (dataset, cond, cid_, _) in enumerate(libs):
        tr = comum.pistas(dataset, cond, genome_len)
        for si, s in enumerate(DESLOCS):
            st = extrair(tr, geo, [int(s)], por_gene=True)
            sum_rho[li, si], n_rho[li, si] = st["sum_rho"], st["n_rho"]
            sum_k[li, si], n_k[li, si] = st["sum_k"], st["n_k"]
            n_genes[li, si] = st["n_genes"]
            Tb[li, :, si] = W @ st["S"]
            Db[li, :, si] = W[:, st["kept"]] @ N[st["kept"]]
        acf[li] = acf_trilha(tr, geo, LAGS_ACF)
        if dataset == "GSE119104":
            trilhas_cache[li] = tr
        print(f"varredura {cid_} ({time.time() - t0:.0f}s)", flush=True)

    # ---------------- 1. reproduction anchor ----------------
    ref = pd.read_csv(comum.BASELINE_DIR / "codon_condition_matrix.csv")
    assert ref["codon"].tolist() == SENTIDO
    dif = {"k_obs": 0.0, "rho": 0.0}
    nan_mismatch = 0
    for li, cid_ in enumerate(ids):
        si = int(np.flatnonzero(DESLOCS == nativo[li])[0])
        with np.errstate(divide="ignore", invalid="ignore"):
            k = sum_k[li, si] / n_k[li, si]
            r = np.where(n_k[li, si] > 0, sum_rho[li, si] / n_rho[li, si], np.nan)
        for nome, x in (("k_obs", k), ("rho", r)):
            y = ref[f"{nome}::{cid_}"].to_numpy(float)
            nan_mismatch += int(np.sum(np.isnan(x) != np.isnan(y)))
            ok = np.isfinite(x) & np.isfinite(y)
            dif[nome] = max(dif[nome], float(np.max(np.abs(x[ok] - y[ok]))))
    ancora = {
        "max_abs_dif_k_obs": dif["k_obs"],
        "max_abs_dif_rho": dif["rho"],
        "celulas_nan_divergentes": nan_mismatch,
        "tolerancia": 1e-9,
        "reproduz": bool(max(dif.values()) <= 1e-9 and nan_mismatch == 0),
    }
    comum.escrever_json(out / "ancora.json", ancora)
    print("ancora", ancora, flush=True)
    if not ancora["reproduz"]:
        raise SystemExit("ancora de reproducao falhou")

    # ---------------- 2. sweep ----------------
    T0 = sum_rho
    D0 = n_rho
    P0 = enriquecimento(T0, D0)  # (L, nS, 61)
    Pb = enriquecimento(Tb.astype(np.float64), Db.astype(np.float64))  # (L, B, nS, 61)
    linhas = []
    for li in range(L):
        for si, s in enumerate(DESLOCS):
            with np.errstate(divide="ignore", invalid="ignore"):
                kk = sum_k[li, si] / n_k[li, si]
                rr = sum_rho[li, si] / n_rho[li, si]
            for j, c in enumerate(SENTIDO):
                linhas.append((ids[li], int(s), c, AA[j], rr[j], int(n_rho[li, si, j]), kk[j], int(n_k[li, si, j]), P0[li, si, j]))
    pd.DataFrame(linhas, columns=["condition_id", "s_nt", "codon", "amino_acid", "rho_medio", "n_rho", "k_obs", "n_k", "log2_enriquecimento"]).to_csv(out / "varredura_codon.csv", index=False)

    mIle = np.isin(SENTIDO, comum.ISOLEUCINA)
    mSer = np.isin(SENTIDO, comum.SERINA)
    mPro = np.isin(SENTIDO, comum.PROLINA)
    mCTA = np.array([c == "CTA" for c in SENTIDO])
    lab = {r: i for i, r in enumerate(rotulos)}

    def perfis(T, D):
        return {
            "Ile": enriq_grupo(T, D, mIle),
            "Ser": enriq_grupo(T, D, mSer),
            "Pro": enriq_grupo(T, D, mPro),
            "CTA": enriq_grupo(T, D, mCTA),
            "var_global": var_global(T, D),
        }

    pf0 = perfis(T0, D0)  # each (L, nS)
    pfb = perfis(Tb.astype(np.float64), Db.astype(np.float64))  # each (L, B, nS)

    sinais = {}  # name -> (point profile (nS), bootstrap profiles (B, nS))
    for t, ct in PARES_MUPIROCINA:
        sinais[f"Ile_contraste|{t}-{ct}"] = (pf0["Ile"][lab[t]] - pf0["Ile"][lab[ct]], pfb["Ile"][lab[t]] - pfb["Ile"][lab[ct]])
    ref_cong0 = np.mean([pf0["Ser"][lab[r]] for r in CONGELADAS_REF], axis=0)
    ref_congb = np.mean([pfb["Ser"][lab[r]] for r in CONGELADAS_REF], axis=0)
    for f in FILTRADAS_SEM_DROGA:
        sinais[f"Ser_contraste|{f}-freeze"] = (pf0["Ser"][lab[f]] - ref_cong0, pfb["Ser"][lab[f]] - ref_congb)
    sinais["CTA_contraste|rib_CUA-rib_CTRL"] = (pf0["CTA"][lab["rib_CUA"]] - pf0["CTA"][lab["rib_CTRL"]], pfb["CTA"][lab["rib_CUA"]] - pfb["CTA"][lab["rib_CTRL"]])
    sinais["CTA_contraste|rib_CUG-rib_CTRL"] = (pf0["CTA"][lab["rib_CUG"]] - pf0["CTA"][lab["rib_CTRL"]], pfb["CTA"][lab["rib_CUG"]] - pfb["CTA"][lab["rib_CTRL"]])
    for li in range(L):
        for nome in ("Pro", "Ser", "Ile", "var_global"):
            sinais[f"{nome}|{rotulos[li]}"] = (pf0[nome][li], pfb[nome][li])

    picos = []
    for nome, (p0, pb) in sinais.items():
        pk = pico(p0)
        boot_s = np.array([DESLOCS[int(np.nanargmax(x))] for x in pb])
        boot_w = np.array([pico(x)["fwhm_nt"] for x in pb])
        picos.append({
            "sinal": nome.split("|")[0],
            "biblioteca": nome.split("|")[1],
            **pk,
            "boot_s_pico_ic95": ic(boot_s),
            "boot_frac_no_pico": float(np.mean(boot_s == pk["s_pico"])),
            "boot_fwhm_ic95": ic(boot_w),
            "perfil_em_0": float(p0[I0]),
            "perfil_em_11": float(p0[I0 + 11]),
            "perfil_em_14": float(p0[I0 + 14]),
            "perfil_em_17": float(p0[I0 + 17]),
        })
    picos_df = pd.DataFrame(picos)
    picos_df["boot_s_pico_ic95"] = picos_df["boot_s_pico_ic95"].map(lambda v: f"[{v[0]:g}, {v[1]:g}]")
    picos_df["boot_fwhm_ic95"] = picos_df["boot_fwhm_ic95"].map(lambda v: f"[{v[0]:g}, {v[1]:g}]")
    picos_df.to_csv(out / "picos.csv", index=False)

    perf_rows = []
    for nome, (p0, _) in sinais.items():
        for si, s in enumerate(DESLOCS):
            perf_rows.append((nome.split("|")[0], nome.split("|")[1], int(s), float(p0[si])))
    pd.DataFrame(perf_rows, columns=["sinal", "biblioteca", "s_nt", "valor"]).to_csv(out / "perfis.csv", index=False)

    pd.DataFrame(acf, index=ids, columns=[f"lag_{l}" for l in LAGS_ACF]).rename_axis("condition_id").to_csv(out / "autocorrelacao_trilhas.csv")

    # ---------------- 3. comparison between studies ----------------
    iA = [i for i, d in enumerate(datasets) if d == "GSE119104"]
    iB = [i for i, d in enumerate(datasets) if d == "GSE77617"]
    iA_semMPC = [i for i in iA if "MPC" not in rotulos[i]]
    iref = lab[REF_PROTOCOLO]

    def perfil_medio(P, grupo):  # P (..., L, nS, 61) with L on axis -3
        return np.nanmean(np.take(P, grupo, axis=-3), axis=-3)

    def curvas(P):
        """P: (L, nS, 61). Returns the per-codon (61) correlation curves between studies."""
        B0 = perfil_medio(P, iB)  # (nS, 61)
        refs = {"filter_Cm_ctrl": P[iref], "media_GSE119104_sem_MPC": perfil_medio(P, iA_semMPC)}
        res = {}
        for nr, R in refs.items():
            # which GSE119104 shift resembles GSE77617 at s = 0
            res[f"{nr}|corr_B0_vs_A(t)"] = np.array([corr(B0[I0], R[si]) for si in range(nS)])
            # which GSE77617 shift resembles GSE119104 at the A site (s = 11)
            res[f"{nr}|corr_B(s)_vs_A11"] = np.array([corr(B0[si], R[I0 + 11]) for si in range(nS)])
            # cross-correlation of the whole profile (s x codon) by lag lam: B(s) ~ A(s + lam)
            cc = []
            for lam in DESLOCS:
                sa = np.arange(nS) + lam
                ok = (sa >= 0) & (sa < nS)
                cc.append(corr(B0[np.arange(nS)[ok]].ravel(), R[sa[ok]].ravel()))
            res[f"{nr}|xcorr_lambda"] = np.array(cc)
        return res, B0, refs

    c0, B0p, refs0 = curvas(P0)
    cb = [curvas(Pb[:, b])[0] for b in range(N_BOOT)]
    comp = {}
    linhas_c = []
    for nome, v in c0.items():
        k = int(np.nanargmax(v))
        bs = np.array([DESLOCS[int(np.nanargmax(x[nome]))] for x in cb])
        comp[nome] = {
            "argmax_nt": int(DESLOCS[k]),
            "max": float(v[k]),
            "boot_argmax_ic95": ic(bs),
            "boot_frac_no_argmax": float(np.mean(bs == DESLOCS[k])),
            "valor_em_0": float(v[I0]),
            "valor_em_11": float(v[I0 + 11]),
            "valor_em_14": float(v[I0 + 14]),
            "valor_em_17": float(v[I0 + 17]),
        }
        for si, s in enumerate(DESLOCS):
            linhas_c.append((nome, int(s), float(v[si])))
    pd.DataFrame(linhas_c, columns=["curva", "s_ou_lambda_nt", "correlacao"]).to_csv(out / "comparacao_estudos_curvas.csv", index=False)

    # A/P/E mixture: B(0) ~ a*A(11) + p*A(14) + e*A(17) + c, per codon
    def mistura(B0, R):
        X = np.column_stack([R[I0 + 11], R[I0 + 14], R[I0 + 17], np.ones(61)])
        ok = np.all(np.isfinite(X), axis=1) & np.isfinite(B0[I0])
        coef, *_ = np.linalg.lstsq(X[ok], B0[I0][ok], rcond=None)
        pred = X[ok] @ coef
        r2 = 1 - np.sum((B0[I0][ok] - pred) ** 2) / np.sum((B0[I0][ok] - B0[I0][ok].mean()) ** 2)
        return coef[:3], float(r2)

    mist = {}
    for nr in refs0:
        coef, r2 = mistura(B0p, refs0[nr])
        bcoef = []
        for b in range(N_BOOT):
            Bb = perfil_medio(Pb[:, b], iB)
            Rb = Pb[iref, b] if nr == "filter_Cm_ctrl" else perfil_medio(Pb[:, b], iA_semMPC)
            bcoef.append(mistura(Bb, Rb)[0])
        bcoef = np.array(bcoef)
        mist[nr] = {
            "coef_A11": float(coef[0]), "coef_P14": float(coef[1]), "coef_E17": float(coef[2]), "r2": r2,
            "ic95_A11": ic(bcoef[:, 0]), "ic95_P14": ic(bcoef[:, 1]), "ic95_E17": ic(bcoef[:, 2]),
        }
    comp["mistura_B0_sobre_A11_P14_E17"] = mist

    # effective track width from the autocorrelation: w = 1 + 2 * sum_{lag>=1} r(lag)
    w_ef = {ids[li]: float(1 + 2 * acf[li, 1:31].sum()) for li in range(L)}
    comp["largura_efetiva_acf_nt"] = w_ef
    comp["largura_efetiva_media_GSE119104"] = float(np.mean([w_ef[ids[i]] for i in iA]))
    comp["largura_efetiva_media_GSE77617"] = float(np.mean([w_ef[ids[i]] for i in iB]))

    # signals anchored at the 3' end of the read (GSE119104 peak near s = 0, far from the
    # ribosome sites): where do they reappear in GSE77617? Under the central-residue rule
    # (>= 12 nt from each end), the prediction is s_B ~ s_A - 12 - (width - 1) / 2.
    anc = {}
    janela_B = (DESLOCS >= -25) & (DESLOCS <= -5)
    for nome in ("Pro", "Ile"):
        pa = np.mean([pf0[nome][i] for i in iA_semMPC], axis=0)
        pb_ = np.mean([pf0[nome][i] for i in iB], axis=0)
        kA = int(np.argmax(pa))
        kB = int(np.flatnonzero(janela_B)[np.argmax(pb_[janela_B])])
        bA = np.array([DESLOCS[int(np.argmax(np.mean([pfb[nome][i][b] for i in iA_semMPC], axis=0)))] for b in range(N_BOOT)])
        bB = np.array([DESLOCS[np.flatnonzero(janela_B)[int(np.argmax(np.mean([pfb[nome][i][b] for i in iB], axis=0)[janela_B]))]] for b in range(N_BOOT)])
        anc[nome] = {
            "s_pico_GSE119104_media_sem_MPC": int(DESLOCS[kA]), "valor_A": float(pa[kA]),
            "s_pico_local_GSE77617_media_em_[-25,-5]": int(DESLOCS[kB]), "valor_B": float(pb_[kB]),
            "defasagem_A_menos_B": int(DESLOCS[kA] - DESLOCS[kB]),
            "boot_defasagem_ic95": ic(bA - bB),
        }
    comp["sinais_ancorados_na_extremidade_3"] = anc
    comum.escrever_json(out / "comparacao_estudos.json", comp)

    # ---------------- 4. variants of the k_obs table ----------------
    # primary criterion: lag of the whole-profile cross-correlation against filter_Cm_ctrl
    # (narrowest bootstrap); GSE77617 at s is equivalent to GSE119104 at s + lam_star
    lam_star = comp["filter_Cm_ctrl|xcorr_lambda"]["argmax_nt"]
    sB_star = 11 - lam_star  # GSE77617 shift equivalent to the GSE119104 A site
    w_mimic = int(round(comp["largura_efetiva_media_GSE77617"] - comp["largura_efetiva_media_GSE119104"] + 1))
    w_mimic = max(w_mimic, 1)
    K = sum_k / n_k  # (L, nS, 61)
    FP = n_k.sum(-1) / n_rho.sum(-1)  # fraction of positions with rho > 0, (L, nS)

    variantes, frac_pos = {}, {}

    def base(sA, sB):
        sel_ = [I0 + (sA if datasets[li] == "GSE119104" else sB) for li in range(L)]
        return [K[li, si] for li, si in enumerate(sel_)], [float(FP[li, si]) for li, si in enumerate(sel_)]

    def add(nome, par):
        variantes[nome], frac_pos[nome] = par

    add("ref_A11_B0", base(11, 0))
    for d in sorted({sB_star - 6, sB_star - 3, sB_star, sB_star + 3, sB_star + 6}):
        if -30 <= d <= 30 and d != 0:
            add(f"A11_B{d:+d}" + (" (equivalente empirico ao sitio A)" if d == sB_star else ""), base(11, d))
    add("A0_B0 (GSE119104 sem deslocamento de sitio A)", base(0, 0))
    add(f"A{lam_star}_B0 (GSE119104 no deslocamento equivalente ao B0)", base(lam_star, 0))
    # smoothed GSE119104: mean of the shifted reads over a window of w nt
    janelas = {}
    for w in sorted({2, w_mimic, 4, 6, 9, 12, 16}):
        janelas[f"A_janela[12..{12 + w - 1}]_B0 (w={w}; cobertura de residuos centrais, geometria Burkhardt)"] = list(range(12, 12 + w))
    janelas[f"A_janela[{11 - w_mimic // 2}..{11 - w_mimic // 2 + w_mimic - 1}]_B0 (w={w_mimic}; centrada no sitio A)"] = list(range(11 - w_mimic // 2, 11 - w_mimic // 2 + w_mimic))
    for nome, jan in janelas.items():
        cols, fp = [], []
        for li in range(L):
            if datasets[li] == "GSE119104":
                st = extrair(trilhas_cache[li], geo, jan)
                cols.append(kobs_de(st))
                fp.append(float(st["n_k"].sum() / st["n_rho"].sum()))
            else:
                cols.append(K[li, I0])
                fp.append(float(FP[li, I0]))
        add(nome, (cols, fp))

    isA_ = np.array([d == "GSE119104" for d in datasets])
    reg, por_lib = [], []
    for nome, cols in variantes.items():
        M = np.column_stack(cols)
        met, a, z, mk = metricas_variante(M, datasets, rotulos)
        fp = np.array(frac_pos[nome])
        reg.append({"variante": nome, **met,
                    "frac_rho_pos_media_GSE119104": float(fp[isA_].mean()),
                    "frac_rho_pos_media_GSE77617": float(fp[~isA_].mean())})
        for li in range(L):
            por_lib.append({
                "variante": nome, "condition_id": ids[li], "intercepto_a_c": float(a[li]),
                "escore_posto1": float(z[li]), "media_kobs": float(mk[li]),
                "frac_posicoes_rho_pos": float(fp[li]),
            })
    # full sweep of the GSE77617 shift (GSE119104 fixed at 11) and of the GSE119104 shift (GSE77617 fixed at 0)
    curva_var = []
    for s in DESLOCS:
        for qual, (cols, _) in (("varia_GSE77617", base(11, int(s))), ("varia_GSE119104", base(int(s), 0))):
            met, *_ = metricas_variante(np.column_stack(cols), datasets, rotulos)
            curva_var.append({"varia": qual, "s_nt": int(s), **met})
    pd.DataFrame(reg).to_csv(out / "variantes.csv", index=False)
    pd.DataFrame(por_lib).to_csv(out / "variantes_por_biblioteca.csv", index=False)
    pd.DataFrame(curva_var).to_csv(out / "variantes_varredura_deslocamento.csv", index=False)

    frac_nativa = {ids[li]: float(n_k[li, I0 + nativo[li]].sum() / n_rho[li, I0 + nativo[li]].sum()) for li in range(L)}
    resumo = {
        "convencao": "s>0: o codon recebe o valor da trilha s nt a jusante (3') no mRNA; GSE119104 s=11/14/17 = sitio A/P/E",
        "deslocamentos_nt": [int(DESLOCS[0]), int(DESLOCS[-1])],
        "n_bootstrap_genes": N_BOOT,
        "semente": comum.semente(4, 0),
        "n_genes_anotados": geo["n_genes"],
        "ancora": ancora,
        "lambda_estrela_B0_equivale_a_A": lam_star,
        "sB_estrela_sitio_A_no_GSE77617": sB_star,
        "largura_janela_imitacao_nt": w_mimic,
        "frac_posicoes_rho_positivo_nativo": frac_nativa,
        "n_genes_retidos_nativo": {ids[li]: int(n_genes[li, I0 + nativo[li]]) for li in range(L)},
    }
    comum.escrever_json(out / "resumo.json", resumo)

    # hash manifest (without the run time, which varies)
    hashes = {}
    for p in sorted(out.glob("*")):
        if p.name in {"manifesto_sha256.json", "tempo_execucao.json"}:
            continue
        hashes[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    comum.escrever_json(out / "manifesto_sha256.json", hashes)
    comum.escrever_json(out / "tempo_execucao.json", {"segundos": round(time.time() - t0, 1)})
    print(f"fim ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
