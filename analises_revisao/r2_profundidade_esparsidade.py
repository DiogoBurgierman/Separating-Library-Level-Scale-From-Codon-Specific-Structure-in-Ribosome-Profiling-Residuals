"""R2: sequencing depth and track sparsity versus the Z_c axis.

Questions: does Z_c correlate with total reads, with the fraction of positions with
rho > 0, and with the number of retained genes? Does the result hold on a log scale, or
with the codon mean of rho inverted afterwards? Does k_obs = 1/rho per position, with
rho = 0 excluded, produce a uniform per-library shift on its own?

Steps (all deterministic):

1. anchor: re-extracts from the tracks the codon x library table of k_obs, mean rho and n,
   gene by gene with `modelo_baseline.observables`, aggregated with bincount, and requires
   agreement <= 1e-9 with `codon_condition_matrix.csv` and `conditions.csv`;
   a fully vectorized extractor (used in the rarefaction) is checked against the per-gene one;
2. depth/sparsity covariates per library (frozen ENA + track);
3. correlations of z_c (pooled and within GSE119104) and of the additive intercept a_c with
   them, with permutation p values (exact for N <= 10, Monte Carlo for N = 14) and partial
   correlations by study;
4. alternative estimators at the same granularity: mean of log k_obs, 1/mean(rho), and
   log of 1/mean(rho);
5. binomial rarefaction of the deepest GSE119104 libraries (exact counts: the tracks are
   integer multiples of 1e6/N, see covariates).

Usage: python -m analises_revisao.r2_profundidade_esparsidade [--atualizar-ena]
Outputs: analises_revisao/saidas/r2/. Nothing outside that folder is rewritten, except the
frozen ENA CSV when --atualizar-ena is passed.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import gzip
import io
import itertools
import math
import re
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from analises_revisao import comum
from modelo_baseline import config as bcfg
from modelo_baseline.annotation import codon_nt_indices, load_cds_records, load_genome_length
from modelo_baseline.observables import gene_codon_occupancy, relative_occupancy_and_rate
from modelo_latente.analysis import low_dimensional_structure

ITEM = 2
OUT = comum.SAIDAS / "r2"
ENA_CSV = comum.ROOT / "analises_revisao" / "dados" / "profundidade_ena.csv"
ENA_URL = (
    "https://www.ebi.ac.uk/ena/portal/api/filereport?accession={acc}&result=read_run"
    "&fields=run_accession,experiment_accession,sample_alias,experiment_title,"
    "read_count,base_count&format=tsv"
)
BIOPROJETOS = {"GSE119104": "PRJNA488109", "GSE77617": "PRJNA311041"}
USER_AGENT = "python-urllib/3"  # generic, no name or e-mail

TOL_ANCORA = 1e-9
N_PERM_PA = 2000
N_PERM_MC = 100_000
RAREF_LIBS = ("freeze_Mg", "freeze_Mg_ctrl", "filter_NaCl")
RAREF_FRACOES = (0.8, 0.6, 0.4, 0.2, 0.1)
RAREF_REPLICAS = 5
RAREF_REPLICAS_ALVO = 10

# indices k of comum.semente(2, k) per procedure
K_MC14 = 0
K_PARCIAL = 1
K_PA = 10  # + 2*j (matrix) and + 2*j + 1 (doubly centred)
K_RAREF = 1000  # + 100*lib + 10*fraction + replicate
K_ALVO = 5000  # + 100*lib + 10*target + replicate

CODONS = sorted(c for c, aa in bcfg.STANDARD_GENETIC_CODE.items() if aa != "*")
CODON_ID = {c: i for i, c in enumerate(CODONS)}
DROP_START = bcfg.EXCLUDE_START_NT // 3
DROP_END = bcfg.EXCLUDE_STOP_NT // 3


# ---------------------------------------------------------------------------
# Genes and extraction
# ---------------------------------------------------------------------------


class Genes:
    """Genomic indices and codon identity of all CDS, concatenated."""

    def __init__(self, records: list[dict], genome_len: int):
        self.records = records
        self.genome_len = genome_len
        idx_list, plus_list, ncod, cid_list, per_record_cid = [], [], [], [], []
        self.n_skip = 0
        self.usados = []
        for r in records:
            idx = np.asarray(
                codon_nt_indices(r["start"], r["end"], r["strand"], genome_len), dtype=np.int64
            )
            if len(idx) != len(r["seq"]):
                self.n_skip += 1
                per_record_cid.append(None)
                continue
            seq = r["seq"]
            cid = np.array(
                [CODON_ID.get(seq[i : i + 3], -1) for i in range(0, len(seq), 3)], dtype=np.int64
            )
            idx_list.append(idx)
            plus_list.append(np.full(idx.size, r["strand"] == "+"))
            ncod.append(cid.size)
            cid_list.append(cid)
            per_record_cid.append(cid)
            self.usados.append(r)
        self.per_record_cid = per_record_cid
        self.nt_idx = np.concatenate(idx_list)
        self.nt_plus = np.concatenate(plus_list)
        self.ncod = np.asarray(ncod, dtype=np.int64)
        self.cod_start = np.concatenate([[0], np.cumsum(self.ncod)[:-1]])
        self.cid = np.concatenate(cid_list)
        self.gene_of = np.repeat(np.arange(self.ncod.size), self.ncod)
        pos_in_gene = np.arange(self.cid.size) - self.cod_start[self.gene_of]
        self.interior = (pos_in_gene >= DROP_START) & (pos_in_gene < self.ncod[self.gene_of] - DROP_END)
        self._lookup = {}
        # CDS nucleotide mask per strand (no shift), for the sum over CDS
        self.cds_mask = {"+": np.zeros(genome_len, bool), "-": np.zeros(genome_len, bool)}
        for r in records:
            self.cds_mask[r["strand"]][r["start"] - 1 : r["end"]] = True

    def lookup(self, shift: int):
        if shift not in self._lookup:
            lk = self.nt_idx + np.where(self.nt_plus, shift, -shift)
            valid = (lk >= 0) & (lk < self.genome_len)
            self._lookup[shift] = (np.where(valid, lk, 0), valid)
        return self._lookup[shift]


def agregar(cid, rho, kobs, gene_index) -> dict:
    """Aggregates interior positions of retained genes into the 61-codon table."""
    sense = cid >= 0
    pos = rho > 0
    sp = sense & pos
    n = np.bincount(cid[sp], minlength=61)
    sum_k = np.bincount(cid[sp], weights=kobs[sp], minlength=61)
    sum_logk = np.bincount(cid[sp], weights=-np.log(rho[sp]), minlength=61)
    sum_rho = np.bincount(cid[sense], weights=rho[sense], minlength=61)
    cnt_rho = np.bincount(cid[sense], minlength=61)
    _, gi = np.unique(gene_index, return_inverse=True)
    zeros_gene = np.bincount(gi, weights=(rho == 0).astype(float)) / np.bincount(gi)
    with np.errstate(divide="ignore", invalid="ignore"):
        return {
            "k_obs": sum_k / n,
            "log_k": sum_logk / n,
            "rho": sum_rho / cnt_rho,
            "n": n.astype(float),
            "n_genes_kept": int(np.unique(gene_index).size),
            "n_rate_observations": int(np.isfinite(kobs).sum()),
            "n_interior": int(rho.size),
            "n_interior_sentido": int(sense.sum()),
            "n_pos_sentido": int(sp.sum()),
            "mediana_frac_zeros_gene": float(np.median(zeros_gene)),
        }


def extrair_por_gene(G: Genes, tracks: dict, shift: int) -> dict:
    """Path faithful to the baseline: the same `observables` functions, gene by gene."""
    cids, rhos, ks, gidx = [], [], [], []
    for g, r in enumerate(G.records):
        cid_r = G.per_record_cid[g]
        occ = gene_codon_occupancy(r, tracks, G.genome_len, shift)
        if occ is None or cid_r is None:
            continue
        rho, k, qc = relative_occupancy_and_rate(occ)
        if not qc["kept"]:
            continue
        cids.append(cid_r[DROP_START : DROP_START + rho.size])
        rhos.append(rho)
        ks.append(k)
        gidx.append(np.full(rho.size, g))
    return agregar(np.concatenate(cids), np.concatenate(rhos), np.concatenate(ks), np.concatenate(gidx))


def extrair_rapido(G: Genes, tracks: dict, shift: int) -> dict:
    """Same quantity, without a per-gene loop (checked against `extrair_por_gene`)."""
    lk, valid = G.lookup(shift)
    vals = np.where(G.nt_plus, tracks["+"][lk], tracks["-"][lk])
    vals[~valid] = 0.0
    occ = vals.reshape(-1, 3).mean(axis=1)
    gene_mean = np.add.reduceat(occ, G.cod_start) / G.ncod
    inter_sum = np.bincount(G.gene_of[G.interior], weights=occ[G.interior], minlength=G.ncod.size)
    kept = (gene_mean > bcfg.MIN_GENE_MEAN_OCCUPANCY) & (inter_sum > 0)
    sel = G.interior & kept[G.gene_of]
    rho = occ[sel] / gene_mean[G.gene_of[sel]]
    with np.errstate(divide="ignore"):
        k = np.where(rho > 0, 1.0 / np.where(rho > 0, rho, 1.0), np.nan)
    return agregar(G.cid[sel], rho, k, G.gene_of[sel])


# ---------------------------------------------------------------------------
# ENA
# ---------------------------------------------------------------------------


def srx_do_soft(soft_path: Path) -> dict:
    out, cur = {}, None
    with gzip.open(soft_path, "rt", encoding="utf-8", errors="replace") as h:
        for line in h:
            if line.startswith("^SAMPLE = "):
                cur = line.split("=", 1)[1].strip()
            elif cur and line.startswith("!Sample_relation = SRA:"):
                m = re.search(r"(SRX\d+)", line)
                if m:
                    out[cur] = m.group(1)
    return out


def atualizar_ena(gsms_artigo: set) -> None:
    rows = []
    hoje = _dt.date.today().isoformat()
    for dataset, proj in BIOPROJETOS.items():
        url = ENA_URL.format(acc=proj)
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=120) as resp:
            text = resp.read().decode("utf-8")
        srx = srx_do_soft(comum.ROOT / "dados" / "alta_confiabilidade" / dataset / f"{dataset}_family.soft.gz")
        for rec in csv.DictReader(io.StringIO(text), delimiter="\t"):
            gsm = rec["sample_alias"].strip()
            rows.append(
                {
                    "dataset": dataset,
                    "bioproject": proj,
                    "gsm": gsm,
                    "run_accession": rec["run_accession"],
                    "experiment_accession": rec["experiment_accession"],
                    "srx_no_soft": srx.get(gsm, ""),
                    "srx_confere": srx.get(gsm, "") == rec["experiment_accession"],
                    "biblioteca_do_artigo": gsm in gsms_artigo,
                    "experiment_title": rec["experiment_title"],
                    "read_count": int(rec["read_count"]),
                    "base_count": int(rec["base_count"]),
                    "data_consulta": hoje,
                    "url_consulta": url,
                }
            )
    frame = pd.DataFrame(rows).sort_values(["dataset", "gsm", "run_accession"])
    ENA_CSV.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(ENA_CSV, index=False)
    print(f"ENA congelado em {ENA_CSV} ({len(frame)} runs)")


def carregar_ena() -> pd.DataFrame | None:
    if not ENA_CSV.exists():
        print(f"AVISO: {ENA_CSV} ausente; seguindo so com a profundidade derivada da trilha")
        return None
    f = pd.read_csv(ENA_CSV)
    f = f[f["biblioteca_do_artigo"]]
    if not f["srx_confere"].all():
        raise RuntimeError("run do ENA com SRX diferente do registrado no SOFT do GEO")
    return f.groupby("gsm").agg(
        ena_n_runs=("run_accession", "count"),
        ena_read_count=("read_count", "sum"),
        ena_base_count=("base_count", "sum"),
        ena_runs=("run_accession", lambda s: ";".join(sorted(s))),
    )


# ---------------------------------------------------------------------------
# Track scale
# ---------------------------------------------------------------------------


def escala_da_trilha(dataset: str, tracks: dict, G: Genes) -> dict:
    allv = np.concatenate([tracks["+"], tracks["-"]])
    nz = allv[allv > 0]
    vmin = float(nz.min())
    ratio = nz / vmin
    dev = np.abs(ratio - np.round(ratio))
    frac_int = float(np.mean(dev < 1e-6 * np.maximum(1.0, ratio)))
    soma = float(allv.sum())
    soma_cds = float(tracks["+"][G.cds_mask["+"]].sum() + tracks["-"][G.cds_mask["-"]].sum())
    if frac_int == 1.0:
        contagens = soma / vmin
        contagens_cds = soma_cds / vmin
        if abs(vmin - 1.0) < 1e-12:
            tipo = "contagens inteiras (min = 1)"
        elif abs(soma - 1e6) < 1e-3:
            tipo = "RPM: multiplos inteiros de 1e6/N, soma = 1e6"
        else:
            tipo = "multiplos inteiros do minimo"
    else:
        # read fractions spread over central residues: each read adds 1 to the track
        contagens = soma
        contagens_cds = soma_cds
        tipo = "fracoes de read (cada read soma 1; soma = numero de reads)"
    return {
        "trilha_tipo": tipo,
        "trilha_min_positivo": vmin,
        "trilha_frac_valores_multiplos_inteiros_do_min": frac_int,
        "trilha_max_desvio_de_inteiro": float(dev.max()),
        "trilha_soma_genoma_unidade_nativa": soma,
        "trilha_soma_cds_unidade_nativa": soma_cds,
        "trilha_soma_genoma_quase_inteira": bool(abs(soma - round(soma)) < 1e-3),
        "reads_mapeados_inferidos": contagens,
        "reads_cds_inferidos": contagens_cds,
        "fracao_sinal_em_cds": soma_cds / soma,
        "n_nt_com_sinal": int(nz.size),
    }


def contagens_inteiras(tracks: dict) -> tuple[dict, float]:
    vmin = min(float(tracks[s][tracks[s] > 0].min()) for s in "+-")
    out = {}
    for s in "+-":
        c = tracks[s] / vmin
        r = np.round(c)
        if np.max(np.abs(c - r)) > 1e-6:
            raise RuntimeError("trilha nao e multiplo inteiro do minimo")
        out[s] = r.astype(np.int64)
    return out, vmin


# ---------------------------------------------------------------------------
# Correlations with permutation p values
# ---------------------------------------------------------------------------

_PERMS: dict[int, np.ndarray] = {}


def todas_permutacoes(n: int) -> np.ndarray:
    if n not in _PERMS:
        total = math.factorial(n)
        flat = np.fromiter(
            itertools.chain.from_iterable(itertools.permutations(range(n))),
            dtype=np.int8,
            count=total * n,
        )
        _PERMS[n] = flat.reshape(total, n)
    return _PERMS[n]


def _padronizar(v):
    v = np.asarray(v, float)
    v = v - v.mean()
    s = np.sqrt(np.sum(v**2))
    return v / s if s > 0 else np.full_like(v, np.nan)


def _ranks(v):
    return pd.Series(np.asarray(v, float)).rank(method="average").to_numpy()


def _residuo_estrato(v, estrato):
    v = np.asarray(v, float).copy()
    for g in np.unique(estrato):
        m = estrato == g
        v[m] -= v[m].mean()
    return v


def correlacoes_perm(y, X, *, estrato=None, seed=None, n_mc=N_PERM_MC, chunk=400_000):
    """Observed r and two-sided permutation p value by permuting y.

    estrato=None: free permutation (exact if n <= 10, Monte Carlo otherwise).
    estrato given: y and X already residualized by stratum; y is permuted within the stratum.
    """
    ys = _padronizar(y)
    Xs = np.column_stack([_padronizar(X[:, j]) for j in range(X.shape[1])])
    r_obs = Xs.T @ ys
    n = ys.size
    ok = np.isfinite(r_obs)
    Xs_ok = np.where(np.isfinite(Xs), Xs, 0.0)
    count = np.zeros(X.shape[1])
    if estrato is None and n <= 10:
        P = todas_permutacoes(n)
        for a in range(0, P.shape[0], chunk):
            R = ys[P[a : a + chunk]] @ Xs_ok
            count += np.sum(np.abs(R) >= np.abs(r_obs) - 1e-12, axis=0)
        p = count / P.shape[0]
        tipo, nperm = "exato", P.shape[0]
    else:
        rng = np.random.default_rng(seed)
        done = 0
        while done < n_mc:
            m = min(chunk, n_mc - done)
            if estrato is None:
                P = rng.permuted(np.tile(np.arange(n), (m, 1)), axis=1)
            else:
                P = np.tile(np.arange(n), (m, 1))
                for g in np.unique(estrato):
                    cols = np.flatnonzero(estrato == g)
                    P[:, cols] = rng.permuted(np.tile(cols, (m, 1)), axis=1)
            R = ys[P] @ Xs_ok
            count += np.sum(np.abs(R) >= np.abs(r_obs) - 1e-12, axis=0)
            done += m
        p = (count + 1) / (n_mc + 1)
        tipo, nperm = ("monte_carlo" if estrato is None else "monte_carlo_estratificado"), n_mc
    r_obs = np.where(ok, r_obs, np.nan)
    p = np.where(ok, p, np.nan)
    return r_obs, p, tipo, nperm


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--atualizar-ena", action="store_true", help="consulta o ENA e regrava o CSV congelado")
    args = ap.parse_args()
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)

    libs = list(comum.bibliotecas())
    ids = [cid for _, _, cid, _ in libs]
    gsms = [c["gsm"] for _, c, _, _ in libs]
    labels = [c["label"] for _, c, _, _ in libs]
    datasets = np.array([d for d, _, _, _ in libs])
    if args.atualizar_ena:
        try:
            atualizar_ena(set(gsms))
        except Exception as exc:  # network unavailable: log it and continue with (b)
            print(f"AVISO: consulta ao ENA falhou ({exc!r}); segue com o CSV congelado, se houver")

    genome_len = load_genome_length()
    records = load_cds_records()
    G = Genes(records, genome_len)
    print(f"{len(records)} CDS, {G.n_skip} descartados por comprimento; {time.time()-t0:.1f}s")

    # ---------------- 1. anchor --------------------------------------------
    codons_csv, _, ids_csv, K_csv = comum.carregar_tabela("k_obs")
    _, _, _, RHO_csv = comum.carregar_tabela("rho")
    _, _, _, N_csv = comum.carregar_tabela("n")
    assert codons_csv == CODONS and ids_csv == ids, "ordem de codons/bibliotecas diverge do baseline"
    cond = pd.read_csv(comum.BASELINE_DIR / "conditions.csv").set_index("condition_id").loc[ids]

    lento, rapido, escalas, pist = {}, {}, {}, {}
    for dataset, condicao, cid, shift in libs:
        tr = comum.pistas(dataset, condicao, genome_len)
        if min(tr["+"].min(), tr["-"].min()) < 0:
            raise RuntimeError(f"trilha negativa em {cid}")
        lento[cid] = extrair_por_gene(G, tr, shift)
        rapido[cid] = extrair_rapido(G, tr, shift)
        escalas[cid] = escala_da_trilha(dataset, tr, G)
        if dataset == "GSE119104" and condicao["label"] in RAREF_LIBS:
            pist[condicao["label"]] = (tr, shift)
        print(f"  {cid}: extraido ({time.time()-t0:.1f}s)", flush=True)

    def tabela(src, key):
        return np.column_stack([src[c][key] for c in ids])

    K = tabela(lento, "k_obs")
    RHO = tabela(lento, "rho")
    N = tabela(lento, "n")
    LOGK = tabela(lento, "log_k")
    ancora = {
        "max_abs_dif_k_obs": float(np.max(np.abs(K - K_csv))),
        "max_abs_dif_rho": float(np.max(np.abs(RHO - RHO_csv))),
        "max_abs_dif_n": float(np.max(np.abs(N - N_csv))),
        "max_abs_dif_n_genes_kept": int(
            max(abs(lento[c]["n_genes_kept"] - int(cond.loc[c, "n_genes_kept"])) for c in ids)
        ),
        "max_abs_dif_n_rate_observations": int(
            max(abs(lento[c]["n_rate_observations"] - int(cond.loc[c, "n_rate_observations"])) for c in ids)
        ),
        "tolerancia": TOL_ANCORA,
    }
    rel = lambda a, b: float(np.nanmax(np.abs(a - b) / np.maximum(np.abs(b), 1e-300)))
    ancora["rapido_vs_por_gene_max_dif_relativa"] = {
        key: rel(tabela(rapido, key), tabela(lento, key)) for key in ("k_obs", "log_k", "rho", "n")
    }
    ancora["rapido_vs_por_gene_contadores_iguais"] = all(
        rapido[c][k] == lento[c][k]
        for c in ids
        for k in ("n_genes_kept", "n_rate_observations", "n_interior", "n_pos_sentido")
    )
    ok = (
        ancora["max_abs_dif_k_obs"] <= TOL_ANCORA
        and ancora["max_abs_dif_rho"] <= TOL_ANCORA
        and ancora["max_abs_dif_n"] == 0
        and ancora["max_abs_dif_n_genes_kept"] == 0
        and ancora["max_abs_dif_n_rate_observations"] == 0
        and max(ancora["rapido_vs_por_gene_max_dif_relativa"].values()) <= 1e-12
        and ancora["rapido_vs_por_gene_contadores_iguais"]
    )
    ancora["aprovada"] = bool(ok)
    comum.escrever_json(OUT / "ancora_reproducao.json", ancora)
    if not ok:
        raise SystemExit(f"ancora reprovada: {ancora}")
    print("ancora aprovada", flush=True)

    # frozen article scores and projection
    eff = pd.read_csv(comum.BASELINE_DIR / "codon_effects.csv").set_index("codon").loc[CODONS]
    kseq = eff["k_seq"].to_numpy()
    lo = pd.read_csv(comum.LATENT_DIR / "pooled" / "latent_loadings.csv").set_index("codon").loc[CODONS]
    m_lat, l_lat = lo["codon_offset_m_i"].to_numpy(), lo["l1"].to_numpy()
    zf = pd.read_csv(comum.LATENT_DIR / "pooled" / "latent_scores_Zc.csv").set_index("condition_id").loc[ids]
    z_pooled = zf["Z1"].to_numpy()

    def z_proj(kcol):
        return (kcol - kseq - m_lat) @ l_lat

    z_rep = np.array([z_proj(K[:, j]) for j in range(len(ids))])
    is119 = datasets == "GSE119104"
    idx119 = np.flatnonzero(is119)
    _, _, _, R_csv = comum.carregar_tabela("R")
    xw = R_csv[:, idx119].T
    m_w = xw.mean(axis=0)
    _, _, vt = np.linalg.svd(xw - m_w, full_matrices=False)
    l_w = vt[0]
    zw_file = (
        pd.read_csv(comum.LATENT_DIR / "pooled" / "within_GSE119104_latent_scores_Zc.csv")
        .set_index("condition_id")
        .loc[[ids[j] for j in idx119], "Z1"]
        .to_numpy()
    )
    if np.dot((xw - m_w) @ l_w, zw_file) < 0:
        l_w = -l_w
    z_within = (xw - m_w) @ l_w

    def zw_proj(kcol):
        return (kcol - kseq - m_w) @ l_w

    ancora_z = {
        "max_abs_dif_z_pooled_projecao_vs_arquivo": float(np.max(np.abs(z_rep - z_pooled))),
        "max_abs_dif_z_intra_GSE119104_vs_arquivo": float(np.max(np.abs(z_within - zw_file))),
        "max_abs_dif_z_intra_projecao_de_k_reextraido": float(
            np.max(np.abs(np.array([zw_proj(K[:, j]) for j in idx119]) - zw_file))
        ),
    }
    if max(ancora_z.values()) > TOL_ANCORA:
        raise SystemExit(f"projecao nas cargas congeladas nao reproduz Z_c: {ancora_z}")
    ancora.update(ancora_z)
    comum.escrever_json(OUT / "ancora_reproducao.json", ancora)

    # ---------------- 2. covariates ----------------------------------------
    ena = carregar_ena()
    media_k = np.nanmean(K, axis=0)
    cov_rows = []
    for j, c in enumerate(ids):
        e = lento[c]
        row = {
            "condition_id": c,
            "dataset": datasets[j],
            "gsm": gsms[j],
            "label": labels[j],
            "z_c": z_pooled[j],
            "media_k_obs": media_k[j],
            **escalas[c],
            "n_genes_retidos": e["n_genes_kept"],
            "n_posicoes_rho_pos": e["n_rate_observations"],
            "n_posicoes_interiores": e["n_interior"],
            "fracao_posicoes_rho_pos": e["n_rate_observations"] / e["n_interior"],
            "mediana_frac_zeros_por_gene": e["mediana_frac_zeros_gene"],
        }
        if ena is not None and gsms[j] in ena.index:
            row.update(ena.loc[gsms[j]].to_dict())
        cov_rows.append(row)
    cov = pd.DataFrame(cov_rows)
    cov.to_csv(OUT / "covariaveis_por_biblioteca.csv", index=False)

    COVS = {
        "log10_reads_ENA": np.log10(cov["ena_read_count"].to_numpy(float)) if ena is not None else None,
        "log10_reads_mapeados_trilha": np.log10(cov["reads_mapeados_inferidos"].to_numpy(float)),
        "log10_reads_cds_trilha": np.log10(cov["reads_cds_inferidos"].to_numpy(float)),
        "fracao_posicoes_rho_pos": cov["fracao_posicoes_rho_pos"].to_numpy(float),
        "n_posicoes_rho_pos": cov["n_posicoes_rho_pos"].to_numpy(float),
        "n_genes_retidos": cov["n_genes_retidos"].to_numpy(float),
        "mediana_frac_zeros_por_gene": cov["mediana_frac_zeros_por_gene"].to_numpy(float),
    }
    COVS = {k: v for k, v in COVS.items() if v is not None}
    cov_names = list(COVS)
    Xcov = np.column_stack([COVS[k] for k in cov_names])
    estudo = (datasets == "GSE77617").astype(int)

    # ---------------- 3. correlations --------------------------------------
    dec14 = comum.decomposicao_aditiva(K)
    dec10 = comum.decomposicao_aditiva(K[:, idx119])
    a14 = dec14["intercepto_a_c"]
    a10 = dec10["intercepto_a_c"]
    idx77 = np.flatnonzero(~is119)
    alvos = [
        ("z_c_agregado", "N14", np.arange(14), z_pooled),
        ("a_c_aditivo_14", "N14", np.arange(14), a14),
        ("z_c_agregado", "N10_GSE119104", idx119, z_pooled[idx119]),
        ("z_c_intra_GSE119104", "N10_GSE119104", idx119, z_within),
        ("a_c_aditivo_10", "N10_GSE119104", idx119, a10),
        ("z_c_agregado", "N4_GSE77617", idx77, z_pooled[idx77]),
        ("a_c_aditivo_14", "N4_GSE77617", idx77, a14[idx77]),
    ]
    corr_rows = []
    kmc = 0
    for alvo, recorte, idx, y in alvos:
        X = Xcov[idx]
        for metodo in ("pearson", "spearman"):
            yy = y if metodo == "pearson" else _ranks(y)
            XX = X if metodo == "pearson" else np.column_stack([_ranks(X[:, k]) for k in range(X.shape[1])])
            seed = comum.semente(ITEM, K_MC14 + 100 * kmc)
            r, p, tipo, nperm = correlacoes_perm(yy, XX, seed=seed)
            kmc += 1
            for k, nm in enumerate(cov_names):
                corr_rows.append(
                    dict(alvo=alvo, recorte=recorte, n=len(idx), covariavel=nm, metodo=metodo,
                         r=r[k], p_perm=p[k], tipo_p=tipo, n_perm=nperm)
                )
        if recorte == "N14":  # partial correlation removing study
            for metodo in ("pearson", "spearman"):
                yy = y if metodo == "pearson" else _ranks(y)
                XX = X if metodo == "pearson" else np.column_stack([_ranks(X[:, k]) for k in range(X.shape[1])])
                yr = _residuo_estrato(yy, estudo)
                Xr = np.column_stack([_residuo_estrato(XX[:, k], estudo) for k in range(XX.shape[1])])
                seed = comum.semente(ITEM, K_PARCIAL + 100 * kmc)
                r, p, tipo, nperm = correlacoes_perm(yr, Xr, estrato=estudo, seed=seed)
                kmc += 1
                for k, nm in enumerate(cov_names):
                    corr_rows.append(
                        dict(alvo=alvo, recorte="N14_parcial_estudo", n=14, covariavel=nm, metodo=metodo,
                             r=r[k], p_perm=p[k], tipo_p=tipo, n_perm=nperm)
                    )
    corr = pd.DataFrame(corr_rows)
    corr.to_csv(OUT / "correlacoes_zc_covariaveis.csv", index=False)

    # covariates against study and against mupirocin (what N = 14 can separate)
    mpc = np.array(["MPC" in lb for lb in labels])
    conf_rows = []
    for k, nm in enumerate(cov_names):
        v = Xcov[:, k]
        conf_rows.append(
            dict(
                covariavel=nm,
                r_com_indicador_GSE77617=float(np.corrcoef(v, estudo)[0, 1]),
                eta2_estudo=float(1 - np.sum(_residuo_estrato(v, estudo) ** 2) / np.sum((v - v.mean()) ** 2)),
                r_com_mupirocina_no_GSE119104=float(np.corrcoef(v[idx119], mpc[idx119])[0, 1]),
                min_GSE119104=float(v[idx119].min()), max_GSE119104=float(v[idx119].max()),
                min_GSE77617=float(v[idx77].min()), max_GSE77617=float(v[idx77].max()),
            )
        )
    conf_rows.append(
        dict(covariavel="z_c_agregado",
             r_com_indicador_GSE77617=float(np.corrcoef(z_pooled, estudo)[0, 1]),
             eta2_estudo=float(1 - np.sum(_residuo_estrato(z_pooled, estudo) ** 2) / np.sum((z_pooled - z_pooled.mean()) ** 2)),
             r_com_mupirocina_no_GSE119104=float(np.corrcoef(z_pooled[idx119], mpc[idx119])[0, 1]),
             min_GSE119104=float(z_pooled[idx119].min()), max_GSE119104=float(z_pooled[idx119].max()),
             min_GSE77617=float(z_pooled[idx77].min()), max_GSE77617=float(z_pooled[idx77].max()))
    )
    pd.DataFrame(conf_rows).to_csv(OUT / "covariaveis_vs_estudo.csv", index=False)

    # ---------------- 4. alternative estimators ----------------------------
    Q = 1.0 / RHO
    ESTIM = {
        "k_obs_original": K,
        "i_media_log_k": LOGK,
        "ii_razao_1_sobre_media_rho": Q,
        "iii_log_razao": np.log(Q),
    }
    ancora["ii_vs_1_sobre_coluna_rho_csv_max_dif_relativa"] = rel(Q, 1.0 / RHO_csv)
    comum.escrever_json(OUT / "ancora_reproducao.json", ancora)
    mat_rows = []
    for nm, M in ESTIM.items():
        for i, cdn in enumerate(CODONS):
            mat_rows.append({"estimador": nm, "codon": cdn, **{ids[j]: M[i, j] for j in range(14)}})
    pd.DataFrame(mat_rows).to_csv(OUT / "estimadores_matrizes.csv", index=False)

    untreated = np.array(["MPC" not in lb and lb != "media_Cm" for lb in labels]) & is119
    resumo_rows, esp_rows, esc_rows, ecorr_rows = [], [], [], []
    frozen = {}  # loadings and offsets per (estimator, subset), for the rarefaction
    jpa = 0
    for nm, M in ESTIM.items():
        for sub, idx in (("N14", np.arange(14)), ("N10_GSE119104", idx119)):
            Ms = M[:, idx]
            d = comum.decomposicao_aditiva(Ms)
            score = d["escores_posto1"]
            # orient the score like the article's z_c (positive correlation with the reference z)
            zref = z_pooled[idx] if sub == "N14" else z_within
            sgn = 1.0 if np.corrcoef(score, zref)[0, 1] >= 0 else -1.0
            score_o = sgn * score
            frozen[(nm, sub)] = (d["offset_m_i"], sgn * d["cargas_posto1"])
            pc1_int = d["escores_interacao_pc1"]
            nmax = min(13, len(idx) - 1)
            pa_m, sp_m = low_dimensional_structure(Ms, N_PERM_PA, comum.semente(ITEM, K_PA + 2 * jpa), nmax)
            pa_d, sp_d = low_dimensional_structure(
                d["interacao_D"], N_PERM_PA, comum.semente(ITEM, K_PA + 2 * jpa + 1), nmax
            )
            jpa += 1
            for tipo_m, sp in (("matriz", sp_m), ("duplamente_centrada", sp_d)):
                sp = sp.copy()
                sp.insert(0, "matriz_analisada", tipo_m)
                sp.insert(0, "subconjunto", sub)
                sp.insert(0, "estimador", nm)
                esp_rows.append(sp)
            row = {"estimador": nm, "subconjunto": sub, **comum.escalares(d)}
            row["pa_matriz_n_componentes"] = pa_m["n_components_above_permutation_null"]
            row["pa_matriz_razao_sv1_p95"] = float(sp_m.loc[0, "singular_value"] / sp_m.loc[0, "permutation_null_p95_singular_value"])
            row["pa_matriz_razao_sv2_p95"] = float(sp_m.loc[1, "singular_value"] / sp_m.loc[1, "permutation_null_p95_singular_value"])
            row["pa_dupla_n_componentes"] = pa_d["n_components_above_permutation_null"]
            row["pa_dupla_razao_sv1_p95"] = float(sp_d.loc[0, "singular_value"] / sp_d.loc[0, "permutation_null_p95_singular_value"])
            row["pa_dupla_p_sv1"] = float(sp_d.loc[0, "permutation_p_value"])
            row["r_escore_lider_com_z_ref"] = float(np.corrcoef(score_o, zref)[0, 1])
            row["r_escore_lider_com_media_k_obs"] = float(np.corrcoef(score_o, media_k[idx])[0, 1])
            row["r_escore_lider_com_a_c"] = float(np.corrcoef(score_o, d["intercepto_a_c"])[0, 1])
            row["n_cargas_mesmo_sinal_max"] = int(max(np.sum(d["cargas_posto1"] < 0), np.sum(d["cargas_posto1"] > 0)))
            if sub == "N14":
                g1, g2 = score_o[is119], score_o[~is119]
                row["escore_media_GSE119104"] = float(g1.mean())
                row["escore_media_GSE77617"] = float(g2.mean())
                row["escore_faixas_se_sobrepoem"] = bool(min(g1.max(), g2.max()) >= max(g1.min(), g2.min()))
                row["eta2_estudo_no_escore"] = float(
                    1 - np.sum(_residuo_estrato(score_o, estudo) ** 2) / np.sum((score_o - score_o.mean()) ** 2)
                )
                row["eta2_estudo_no_pc1_interacao"] = float(
                    1 - np.sum(_residuo_estrato(pc1_int, estudo) ** 2) / np.sum((pc1_int - pc1_int.mean()) ** 2)
                )
            # ordering within GSE119104 (from lowest to highest oriented score)
            s119 = score_o if sub == "N10_GSE119104" else score_o[is119]
            lab119 = [labels[j] for j in idx119]
            ordem = [lab119[t] for t in np.argsort(s119)]
            row["ordem_GSE119104_crescente"] = ";".join(ordem)
            pos_mpc = sorted(ordem.index(lb) + 1 for lb in lab119 if "MPC" in lb)
            row["postos_mupirocina_de_10"] = ";".join(map(str, pos_mpc))
            row["posto_media_Cm_de_10"] = ordem.index("media_Cm") + 1
            row["mupirocina_e_Cm_nas_4_menores"] = bool(set(ordem[:4]) == {"media_Cm", *[lb for lb in lab119 if "MPC" in lb]})
            unt = untreated[idx119]
            row["gap_nao_tratadas_menos_mupirocina"] = float(
                s119[unt].mean() - s119[np.array(["MPC" in lb for lb in lab119])].mean()
            )
            top = np.argsort(-np.abs(d["cargas_interacao_pc1"]))[:6]
            row["pc1_interacao_codons_top"] = ";".join(f"{CODONS[t]}({d['cargas_interacao_pc1'][t]:+.3f})" for t in top)
            resumo_rows.append(row)
            for t, j in enumerate(idx):
                esc_rows.append(
                    dict(estimador=nm, subconjunto=sub, condition_id=ids[j], label=labels[j],
                         escore_lider_orientado=score_o[t], intercepto_a_c=d["intercepto_a_c"][t],
                         escore_pc1_interacao=pc1_int[t])
                )
            # correlations of the leading score and of the interaction PC1 with the covariates
            for alvo_nm, yv in (("escore_lider", score_o), ("pc1_interacao", pc1_int)):
                X = Xcov[idx]
                for metodo in ("pearson", "spearman"):
                    yy = yv if metodo == "pearson" else _ranks(yv)
                    XX = X if metodo == "pearson" else np.column_stack([_ranks(X[:, k]) for k in range(X.shape[1])])
                    r, p, tipo, nperm = correlacoes_perm(yy, XX, seed=comum.semente(ITEM, 200 + 100 * kmc))
                    kmc += 1
                    for k, cn in enumerate(cov_names):
                        ecorr_rows.append(dict(estimador=nm, subconjunto=sub, alvo=alvo_nm, covariavel=cn,
                                               metodo=metodo, r=r[k], p_perm=p[k], tipo_p=tipo))
                    if sub == "N14":
                        yr = _residuo_estrato(yy, estudo)
                        Xr = np.column_stack([_residuo_estrato(XX[:, k], estudo) for k in range(XX.shape[1])])
                        r, p, tipo, nperm = correlacoes_perm(yr, Xr, estrato=estudo, seed=comum.semente(ITEM, 300 + 100 * kmc))
                        kmc += 1
                        for k, cn in enumerate(cov_names):
                            ecorr_rows.append(dict(estimador=nm, subconjunto="N14_parcial_estudo", alvo=alvo_nm,
                                                   covariavel=cn, metodo=metodo, r=r[k], p_perm=p[k], tipo_p=tipo))
        print(f"  estimador {nm} ({time.time()-t0:.1f}s)", flush=True)
    pd.DataFrame(resumo_rows).to_csv(OUT / "estimadores_resumo.csv", index=False)
    pd.concat(esp_rows).to_csv(OUT / "estimadores_espectro_paralelo.csv", index=False)
    pd.DataFrame(esc_rows).to_csv(OUT / "estimadores_escores.csv", index=False)
    pd.DataFrame(ecorr_rows).to_csv(OUT / "estimadores_correlacoes.csv", index=False)

    # ---------------- 5. rarefaction ---------------------------------------
    lab2j = {labels[j]: j for j in range(14)}
    est_fun = {
        "k_obs_original": lambda e: e["k_obs"],
        "i_media_log_k": lambda e: e["log_k"],
        "ii_razao_1_sobre_media_rho": lambda e: 1.0 / e["rho"],
        "iii_log_razao": lambda e: np.log(1.0 / e["rho"]),
    }

    def medir(e):
        out = {
            "n_genes_retidos": e["n_genes_kept"],
            "n_posicoes_rho_pos": e["n_rate_observations"],
            "fracao_posicoes_rho_pos": e["n_rate_observations"] / e["n_interior"],
            "media_k_obs": float(np.mean(e["k_obs"])),
            "z_c_projetado": float(z_proj(e["k_obs"])),
            "z_intra_projetado": float(zw_proj(e["k_obs"])),
        }
        for nm, f in est_fun.items():
            x = f(e)
            for sub in ("N14", "N10_GSE119104"):
                off, lo_ = frozen[(nm, sub)]
                out[f"escore::{nm}::{sub}"] = float((x - off) @ lo_)
            out[f"a_c::{nm}"] = float(np.mean(x - frozen[(nm, "N14")][0]))
        return out

    rare_rows, delta_vec = [], {}
    base_meas = {}
    for li, lb in enumerate(RAREF_LIBS):
        tr, shift = pist[lb]
        cnt, vmin = contagens_inteiras(tr)
        total = int(cnt["+"].sum() + cnt["-"].sum())
        e0 = extrair_rapido(G, {s: cnt[s].astype(float) for s in "+-"}, shift)
        m0 = medir(e0)
        base_meas[lb] = (m0, e0)
        rare_rows.append(dict(biblioteca=lb, fracao=1.0, replica=0, alvo="original", reads_retidos=total, **m0))
        for fi, f in enumerate(RAREF_FRACOES):
            dv = []
            for rep in range(RAREF_REPLICAS):
                rng = np.random.default_rng(comum.semente(ITEM, K_RAREF + 100 * li + 10 * fi + rep))
                th = {s: rng.binomial(cnt[s], f).astype(float) for s in "+-"}
                e = extrair_rapido(G, th, shift)
                rare_rows.append(dict(biblioteca=lb, fracao=f, replica=rep, alvo="grade",
                                      reads_retidos=int(th["+"].sum() + th["-"].sum()), **medir(e)))
                dv.append(e["k_obs"] - e0["k_obs"])
            delta_vec[(lb, f)] = np.mean(dv, axis=0)
        print(f"  rarefacao {lb} ({time.time()-t0:.1f}s)", flush=True)
    rare = pd.DataFrame(rare_rows)

    # targets: (a) number of positions with rho > 0 of the mupirocin libraries; (b) their reads
    mpc_labels = [lb for lb in labels if "MPC" in lb]
    alvo_npos = float(np.mean([lento[ids[lab2j[lb]]]["n_rate_observations"] for lb in mpc_labels]))
    alvo_reads = float(np.mean([cov.loc[lab2j[lb], "reads_mapeados_inferidos"] for lb in mpc_labels]))
    alvo_rows = []
    for li, lb in enumerate(RAREF_LIBS):
        tr, shift = pist[lb]
        cnt, _ = contagens_inteiras(tr)
        total = int(cnt["+"].sum() + cnt["-"].sum())
        g = rare[(rare.biblioteca == lb)].groupby("fracao")["n_posicoes_rho_pos"].mean()
        fr = g.index.to_numpy(float)
        f_npos = float(np.exp(np.interp(np.log(alvo_npos), np.log(g.to_numpy()), np.log(fr))))
        f_reads = min(1.0, alvo_reads / total)
        for ai, (anm, f) in enumerate((("n_pos_mupirocina", f_npos), ("reads_mupirocina", f_reads))):
            for rep in range(RAREF_REPLICAS_ALVO):
                rng = np.random.default_rng(comum.semente(ITEM, K_ALVO + 100 * li + 10 * ai + rep))
                th = {s: rng.binomial(cnt[s], f).astype(float) for s in "+-"}
                e = extrair_rapido(G, th, shift)
                alvo_rows.append(dict(biblioteca=lb, fracao=f, replica=rep, alvo=anm,
                                      reads_retidos=int(th["+"].sum() + th["-"].sum()), **medir(e)))
                if rep == 0:
                    delta_vec[(lb, anm)] = []
                delta_vec[(lb, anm)].append(e["k_obs"] - base_meas[lb][1]["k_obs"])
            delta_vec[(lb, anm)] = np.mean(delta_vec[(lb, anm)], axis=0)
    rare = pd.concat([rare, pd.DataFrame(alvo_rows)], ignore_index=True)
    rare.to_csv(OUT / "rarefacao_replicas.csv", index=False)

    # rarefaction summary: shift relative to the original library
    medidas = ["n_posicoes_rho_pos", "fracao_posicoes_rho_pos", "n_genes_retidos", "media_k_obs",
               "z_c_projetado", "z_intra_projetado"] + [c for c in rare.columns if c.startswith(("escore::", "a_c::"))]
    res_rows = []
    for (lb, alvo, f), grp in rare.groupby(["biblioteca", "alvo", "fracao"], sort=True):
        m0 = base_meas[lb][0]
        row = dict(biblioteca=lb, alvo=alvo, fracao=f, n_replicas=len(grp), reads_retidos_media=grp["reads_retidos"].mean())
        for c in medidas:
            row[f"{c}::media"] = grp[c].mean()
            row[f"{c}::dp"] = grp[c].std(ddof=1) if len(grp) > 1 else 0.0
            row[f"{c}::delta"] = grp[c].mean() - m0[c]
        key = (lb, f) if alvo == "grade" else (lb, alvo)
        if key in delta_vec:
            dv = delta_vec[key]
            row["delta_k::fracao_codons_mesmo_sinal"] = float(max(np.mean(dv > 0), np.mean(dv < 0)))
            row["delta_k::fracao_sq_no_deslocamento_uniforme"] = float(61 * dv.mean() ** 2 / np.sum(dv**2))
            row["delta_k::cos_com_cargas_l"] = float(abs(dv @ l_lat) / np.linalg.norm(dv) / np.linalg.norm(l_lat))
        res_rows.append(row)
    rres = pd.DataFrame(res_rows)
    rres.to_csv(OUT / "rarefacao_resumo.csv", index=False)

    # observed references to compare the size of the shifts
    def faixa(v):
        v119 = v[is119]
        return {
            "faixa_GSE119104_max_menos_min": float(v119.max() - v119.min()),
            "faixa_GSE119104_sem_media_Cm": float(np.ptp(np.delete(v119, labels.index("media_Cm")))),
            "gap_nao_tratadas_menos_mupirocina": float(v[untreated].mean() - v[mpc].mean()),
            "diferenca_medias_entre_estudos_119_menos_77": float(v[is119].mean() - v[~is119].mean()),
            "mupirocina_min": float(v[mpc].min()),
            "mupirocina_max": float(v[mpc].max()),
        }
    referencias = {
        "z_c_agregado": faixa(z_pooled),
        "media_k_obs": faixa(media_k),
        "a_c_k_obs": faixa(a14),
    }
    for nm in ESTIM:
        off, lo_ = frozen[(nm, "N14")]
        sc = np.array([(ESTIM[nm][:, j] - off) @ lo_ for j in range(14)])
        referencias[f"escore::{nm}::N14"] = faixa(sc)
        referencias[f"a_c::{nm}"] = faixa(np.mean(ESTIM[nm] - off[:, None], axis=0))

    # profile of the observed mupirocin - control contrast, compared with that of the rarefaction
    pares = [("freeze_Mg_MPC", "freeze_Mg"), ("freeze_Mg_MPC", "freeze_Mg_ctrl"),
             ("filter_Mg_MPC", "filter_Mg_ctrl"), ("filter_Cm_MPC", "filter_Cm_ctrl")]
    perfil_rows = []
    for a_lb, b_lb in pares:
        dobs = K[:, lab2j[a_lb]] - K[:, lab2j[b_lb]]
        row = dict(contraste=f"{a_lb} - {b_lb}",
                   fracao_codons_mesmo_sinal=float(max(np.mean(dobs > 0), np.mean(dobs < 0))),
                   fracao_sq_no_deslocamento_uniforme=float(61 * dobs.mean() ** 2 / np.sum(dobs**2)),
                   media_delta_k=float(dobs.mean()),
                   delta_z=float(z_pooled[lab2j[a_lb]] - z_pooled[lab2j[b_lb]]))
        for lb in RAREF_LIBS:
            dr = delta_vec[(lb, "n_pos_mupirocina")]
            row[f"r_perfil_com_rarefacao_{lb}"] = float(np.corrcoef(dobs, dr)[0, 1])
        perfil_rows.append(row)
    pd.DataFrame(perfil_rows).to_csv(OUT / "rarefacao_perfil_vs_contraste_observado.csv", index=False)

    # descriptive adjustment by the rarefaction curve: y_adj = y - f(ln n_pos - ln n_ref),
    # f quadratic without intercept, fitted on the replicates (centred on the original library)
    alvos_aj = {"z_c_agregado": ("z_c_projetado", z_pooled), "media_k_obs": ("media_k_obs", media_k)}
    for nm in ESTIM:
        off, lo_ = frozen[(nm, "N14")]
        alvos_aj[f"escore::{nm}::N14"] = (
            f"escore::{nm}::N14", np.array([(ESTIM[nm][:, j] - off) @ lo_ for j in range(14)])
        )
    npos_lib = np.array([lento[c]["n_rate_observations"] for c in ids], float)
    n_ref = float(npos_lib[untreated].max())
    dx_rows, dy_rows = [], {k: [] for k in alvos_aj}
    for lb in RAREF_LIBS:
        g = rare[rare.biblioteca == lb]
        m0 = base_meas[lb][0]
        dx_rows.append(np.log(g["n_posicoes_rho_pos"].to_numpy(float)) - np.log(m0["n_posicoes_rho_pos"]))
        for k, (col, _) in alvos_aj.items():
            dy_rows[k].append(g[col].to_numpy(float) - m0[col])
    dx = np.concatenate(dx_rows)
    A = np.column_stack([dx, dx**2])
    aj_rows, curva = [], {}
    dxl = np.log(npos_lib) - np.log(n_ref)
    for k, (col, yobs) in alvos_aj.items():
        coef, *_ = np.linalg.lstsq(A, np.concatenate(dy_rows[k]), rcond=None)
        pred = coef[0] * dxl + coef[1] * dxl**2
        curva[k] = {"b1": float(coef[0]), "b2": float(coef[1]),
                    "faixa_dx_ajustada": [float(dx.min()), float(dx.max())]}
        yadj = yobs - pred
        for j in range(14):
            aj_rows.append(dict(medida=k, condition_id=ids[j], label=labels[j], dataset=datasets[j],
                                n_posicoes_rho_pos=npos_lib[j], dx_ln_npos_vs_ref=dxl[j],
                                extrapolado=bool(dxl[j] > 1e-12 or dxl[j] < dx.min() - 1e-12),
                                observado=yobs[j], efeito_profundidade_previsto=pred[j], ajustado=yadj[j]))
        a119 = yadj[is119]
        ordem = [labels[j] for j in idx119[np.argsort(a119)]]
        curva[k].update({
            "ordem_GSE119104_ajustada_crescente": ";".join(ordem),
            "gap_nao_tratadas_menos_mupirocina_observado": float(yobs[untreated].mean() - yobs[mpc].mean()),
            "gap_nao_tratadas_menos_mupirocina_ajustado": float(yadj[untreated].mean() - yadj[mpc].mean()),
            "diferenca_entre_estudos_observada": float(yobs[is119].mean() - yobs[~is119].mean()),
            "diferenca_entre_estudos_ajustada_extrapolada": float(yadj[is119].mean() - yadj[~is119].mean()),
            "dp_nao_tratadas_observado": float(yobs[untreated].std(ddof=1)),
            "dp_nao_tratadas_ajustado": float(yadj[untreated].std(ddof=1)),
        })
    pd.DataFrame(aj_rows).to_csv(OUT / "ajuste_pela_curva_de_rarefacao.csv", index=False)

    resumo = {
        "ancora": ancora,
        "n_ref_ajuste": n_ref,
        "curva_de_rarefacao": curva,
        "alvo_n_posicoes_rho_pos_mupirocina": alvo_npos,
        "alvo_reads_mapeados_mupirocina": alvo_reads,
        "referencias_observadas": referencias,
        "covariaveis": cov_names,
        "n_perm_monte_carlo": N_PERM_MC,
        "n_perm_analise_paralela": N_PERM_PA,
        "sementes": {
            "monte_carlo_correlacoes": "comum.semente(2, 0 + 100*k), k = ordem da chamada",
            "parcial_estrato": "comum.semente(2, 1 + 100*k)",
            "analise_paralela": f"comum.semente(2, {K_PA} + 2*j) e +1 para a duplamente centrada",
            "rarefacao_grade": f"comum.semente(2, {K_RAREF} + 100*lib + 10*fracao + replica)",
            "rarefacao_alvos": f"comum.semente(2, {K_ALVO} + 100*lib + 10*alvo + replica)",
        },
        "r_z_c_com_media_k_obs": float(np.corrcoef(z_pooled, media_k)[0, 1]),
    }
    comum.escrever_json(OUT / "resumo.json", resumo)
    print(f"concluido em {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
