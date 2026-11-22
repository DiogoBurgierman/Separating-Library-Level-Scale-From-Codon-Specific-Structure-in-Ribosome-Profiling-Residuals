"""R6: remapping of the full FASTQ files with removal of the 3' adapter, and direct comparison
of the two E. coli reference sequences (NC_000913.2 and NC_000913.3).

Usage (from the repository root):

    python -m analises_revisao.r6_remapeamento_adaptadores                 # everything
    python -m analises_revisao.r6_remapeamento_adaptadores --reusar-cache  # reuse the FASTQ pass cache

Steps (outputs in analises_revisao/saidas/r6/):
  1. MD5 of the two FASTQ files against dados/remapeamento_bruto/resultados/runs_selecionados.csv.
  2. Differences NC_000913.2 x NC_000913.3 measured directly (unique k-mer anchors,
     collinear chain, recursion with decreasing k and Needleman-Wunsch on small gaps;
     the differences reconstruct NC_000913.3 exactly from NC_000913.2).
  3. 3' adapter by evidence: GEO protocol fields + reads (most frequent suffixes and the
     sequence that follows the end of the genomic match of a 25-nt prefix).
  4. Remapping by exact match of the whole read after trimming (both strands, explicit
     unique x multiple count), with a 64-bit polynomial hash per genome window.
  5. Reads discordant between references and per-allele support (20-nt k-mers covering the
     divergent sites), which is what actually distinguishes the references.

Determinism: there is no stochastic procedure (semente(6, 0) is only recorded). FASTQ files are
read in file order; hashes use fixed constants.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import platform
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from analises_revisao.comum import CACHE, ROOT, dir_saida, escrever_json, semente

# --------------------------------------------------------------------------- paths
BASE = ROOT / "dados" / "remapeamento_bruto"
FASTQ_DIR = BASE / "fastq"
REF_DIR = BASE / "referencias_candidatas"
MANIFESTO = BASE / "resultados" / "runs_selecionados.csv"
GFF_REF3 = ROOT / "dados" / "referencias" / "Ecoli_MG1655_ASM584v2" / "genomic.gff.gz"
GEO_DIR = ROOT / "dados" / "alta_confiabilidade"
OUT = dir_saida("r6")
CACHE_R6 = CACHE / "r6"

REFS = ("NC_000913.2", "NC_000913.3")
RUNS = {"SRR7759813": "GSE119104", "SRR5186138": "GSE77617"}

# --------------------------------------------------------------------------- parameters
ADAPTADOR = "CTGTAGGCACCATCAAT"  # confirmed in step 3 (see saidas/r6/adaptador_*.json)
SOBREPOSICAO_MIN = 5  # minimum partial overlap of the adapter at the end of the read
COMPRIMENTO_MIN = 20  # main threshold after trimming
LIMIARES = (15, 20, 25)  # sensitivity
LMAX = 51  # longest observed read length
K_PROBE = 20  # allele-support k-mers
CHUNK_READS = 200_000
HASH_M = np.uint64(0x9E3779B97F4A7C15)  # fixed (odd) multiplier

LUT = np.zeros(256, dtype=np.uint8)
for _c, _v in zip("ACGT", (1, 2, 3, 4)):
    LUT[ord(_c)] = _v
    LUT[ord(_c.lower())] = _v
COMP = np.array([0, 4, 3, 2, 1], dtype=np.uint8)  # A<->T, C<->G, 0 (N) stays 0

_t0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - _t0:7.1f}s] {msg}", flush=True)


# =========================================================================== utilities
def ler_fasta(path: Path) -> np.ndarray:
    """Codes 1..4 (A,C,G,T); 0 for any other symbol."""
    partes = []
    with open(path, "rb") as fh:
        for linha in fh:
            if linha.startswith(b">"):
                continue
            partes.append(linha.strip())
    return LUT[np.frombuffer(b"".join(partes), dtype=np.uint8)]


def codigos_para_str(c: np.ndarray) -> str:
    return "".join("NACGT"[int(x)] for x in c)


def str_para_codigos(s: str) -> np.ndarray:
    return LUT[np.frombuffer(s.encode(), dtype=np.uint8)]


def revcomp_codigos(c: np.ndarray) -> np.ndarray:
    return COMP[c[::-1]]


def md5_arquivo(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for bloco in iter(lambda: fh.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def sha256_arquivo(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# =========================================================================== 1. MD5
def verificar_md5() -> pd.DataFrame:
    man = pd.read_csv(MANIFESTO, dtype={"fastq_md5": str})
    linhas = []
    for run in RUNS:
        row = man[man["run_accession"] == run].iloc[0]
        esperado = str(row["fastq_md5"]).split(";")[0].strip().lower()
        caminho = FASTQ_DIR / f"{run}.fastq.gz"
        obtido = md5_arquivo(caminho)
        if obtido != esperado:
            raise RuntimeError(f"MD5 divergente em {caminho}: {obtido} != {esperado}")
        log(f"[md5] {run} ok ({esperado})")
        linhas.append(
            {"run": run, "dataset": RUNS[run], "md5_manifesto": esperado, "md5_obtido": obtido,
             "bytes": caminho.stat().st_size, "confere": True}
        )
    return pd.DataFrame(linhas)


# =========================================================================== 2. differences
def kmers_2bit(c: np.ndarray, k: int) -> np.ndarray:
    n = len(c)
    v = np.zeros(n - k + 1, dtype=np.uint64)
    q = (c.astype(np.uint64) - np.uint64(1))
    for j in range(k):
        v = (v << np.uint64(2)) | q[j : n - k + 1 + j]
    return v


def ancoras_unicas(a: np.ndarray, b: np.ndarray, k: int):
    """Pairs (posA, posB) of k-mers that occur exactly once in a and once in b (forward strand)."""
    out = []
    for c in (a, b):
        v = kmers_2bit(c, k)
        # windows with 0 (N) do not exist in the genomes; in a local excerpt they may -> discard
        cs = np.concatenate([[0], np.cumsum(c == 0, dtype=np.int64)])
        bad = (cs[k:] - cs[:-k]) > 0
        ordem = np.argsort(v, kind="stable")
        vs = v[ordem]
        uniq = np.ones(len(vs), dtype=bool)
        uniq[1:] &= vs[1:] != vs[:-1]
        uniq[:-1] &= vs[:-1] != vs[1:]
        keep = uniq & ~bad[ordem]
        out.append((vs[keep], ordem[keep]))
    (ka, pa), (kb, pb) = out
    _, ia, ib = np.intersect1d(ka, kb, assume_unique=True, return_indices=True)
    pos_a, pos_b = pa[ia], pb[ib]
    o = np.argsort(pos_a, kind="stable")
    return pos_a[o].astype(np.int64), pos_b[o].astype(np.int64)


def _lis_indices(vals: np.ndarray) -> np.ndarray:
    """Longest strictly increasing subsequence (indices), O(n log n)."""
    import bisect

    caudas: list[int] = []
    idx_cauda: list[int] = []
    pred = np.full(len(vals), -1, dtype=np.int64)
    for i, v in enumerate(vals.tolist()):
        j = bisect.bisect_left(caudas, v)
        if j == len(caudas):
            caudas.append(v)
            idx_cauda.append(i)
        else:
            caudas[j] = v
            idx_cauda[j] = i
        pred[i] = idx_cauda[j - 1] if j > 0 else -1
    saida = []
    i = idx_cauda[-1] if idx_cauda else -1
    while i >= 0:
        saida.append(i)
        i = int(pred[i])
    return np.array(saida[::-1], dtype=np.int64)


def _nw_eventos(a: np.ndarray, b: np.ndarray):
    """Edit distance with traceback; returns operations ('=','X','I','D') in a->b."""
    n, m = len(a), len(b)
    d = np.zeros((n + 1, m + 1), dtype=np.int32)
    d[:, 0] = np.arange(n + 1)
    d[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        ai = a[i - 1]
        for j in range(1, m + 1):
            d[i, j] = min(d[i - 1, j - 1] + (ai != b[j - 1]), d[i - 1, j] + 1, d[i, j - 1] + 1)
    ops = []
    i, j = n, m
    while i > 0 or j > 0:
        # preference: diagonal, then deletion (left alignment of indels comes from the earlier trim)
        if i > 0 and j > 0 and d[i, j] == d[i - 1, j - 1] + (a[i - 1] != b[j - 1]):
            ops.append("=" if a[i - 1] == b[j - 1] else "X")
            i -= 1
            j -= 1
        elif i > 0 and d[i, j] == d[i - 1, j] + 1:
            ops.append("D")  # base present in a (.2) and absent in b (.3)
            i -= 1
        else:
            ops.append("I")  # base present in b (.3) and absent in a (.2)
            j -= 1
    return ops[::-1]


def _eventos_de_ops(ops, a0: int, b0: int):
    """Groups consecutive non-'=' operations into events [a_start,a_end) x [b_start,b_end) (0-based)."""
    ev = []
    ia, ib = a0, b0
    i = 0
    while i < len(ops):
        if ops[i] == "=":
            ia += 1
            ib += 1
            i += 1
            continue
        sa, sb = ia, ib
        while i < len(ops) and ops[i] != "=":
            if ops[i] in "XD":
                ia += 1
            if ops[i] in "XI":
                ib += 1
            i += 1
        ev.append((sa, ia, sb, ib))
    return ev


def alinhar_regiao(A, B, a1, a2, b1, b2, ks=(32, 20, 14, 10), nw_max=300):
    """Events between A[a1:a2] and B[b1:b2] (0-based, half-open). Each event: (a_start,a_end,b_start,b_end)."""
    # trim common suffix and then prefix (aligns indels to the left)
    while a2 > a1 and b2 > b1 and A[a2 - 1] == B[b2 - 1]:
        a2 -= 1
        b2 -= 1
    while a2 > a1 and b2 > b1 and A[a1] == B[b1]:
        a1 += 1
        b1 += 1
    if a1 == a2 and b1 == b2:
        return []
    if a1 == a2 or b1 == b2:
        return [(a1, a2, b1, b2)]
    la, lb = a2 - a1, b2 - b1
    if la <= nw_max and lb <= nw_max and not ks:
        return _eventos_de_ops(_nw_eventos(A[a1:a2], B[b1:b2]), a1, b1)
    if not ks:
        return [(a1, a2, b1, b2)]  # large complex region: reported as a block
    k = ks[0]
    if min(la, lb) < k:
        return alinhar_regiao(A, B, a1, a2, b1, b2, ks[1:], nw_max)
    sa, sb = A[a1:a2], B[b1:b2]
    pa, pb = ancoras_unicas(sa, sb, k)
    if len(pa) == 0:
        return alinhar_regiao(A, B, a1, a2, b1, b2, ks[1:], nw_max)
    # runs on the same diagonal
    diag = pb - pa
    quebra = np.flatnonzero(np.diff(diag) != 0) + 1
    ini = np.concatenate([[0], quebra])
    fim = np.concatenate([quebra, [len(pa)]])
    corridas = np.stack(
        [pa[ini], pa[fim - 1] + k, diag[ini]], axis=1
    )  # a_start, a_end, d  (local coordinates)
    # collinear chain by LIS on b_start
    b_ini = corridas[:, 0] + corridas[:, 2]
    sel = _lis_indices(b_ini)
    corridas = corridas[sel]
    eventos = []
    # trim overlaps
    recortadas = []
    ult_a, ult_b = 0, 0
    for a_i, a_f, d in corridas.tolist():
        b_i, b_f = a_i + d, a_f + d
        corte = max(ult_a - a_i, ult_b - b_i, 0)
        a_i, b_i = a_i + corte, b_i + corte
        if a_i >= a_f:
            continue
        recortadas.append((a_i, a_f, d))
        ult_a, ult_b = a_f, b_f
    # differences inside runs (same diagonal): base-by-base comparison
    for a_i, a_f, d in recortadas:
        x = np.flatnonzero(sa[a_i:a_f] != sb[a_i + d : a_f + d]) + a_i
        if len(x):
            # group adjacent mismatches
            grupos = np.split(x, np.flatnonzero(np.diff(x) > 1) + 1)
            for g in grupos:
                eventos.append((a1 + int(g[0]), a1 + int(g[-1]) + 1, b1 + int(g[0]) + d, b1 + int(g[-1]) + 1 + d))
    # gaps between runs (and borders)
    prev_a, prev_b = 0, 0
    lacunas = []
    for a_i, a_f, d in recortadas:
        lacunas.append((prev_a, a_i, prev_b, a_i + d))
        prev_a, prev_b = a_f, a_f + d
    lacunas.append((prev_a, la, prev_b, lb))
    for ga1, ga2, gb1, gb2 in lacunas:
        if ga1 == ga2 and gb1 == gb2:
            continue
        eventos += alinhar_regiao(A, B, a1 + ga1, a1 + ga2, b1 + gb1, b1 + gb2, ks[1:], nw_max)
    eventos.sort()
    return eventos


def diferencas_referencias(g2: np.ndarray, g3: np.ndarray) -> pd.DataFrame:
    ev = alinhar_regiao(g2, g3, 0, len(g2), 0, len(g3))
    linhas = []
    for n, (a1, a2, b1, b2) in enumerate(sorted(ev), 1):
        la, lb = a2 - a1, b2 - b1
        if la == lb and la > 0 and la <= 50:
            tipo = "substituicao" if la == 1 else f"substituicao_multibase"
        elif la == 0 and lb > 0:
            tipo = "insercao_em_NC_000913.3"
        elif lb == 0 and la > 0:
            tipo = "delecao_em_NC_000913.3"
        else:
            tipo = "complexo"
        linhas.append(
            {
                "id": n,
                "tipo": tipo,
                "ncbi2_inicio_1based": a1 + 1,
                "ncbi2_fim_1based": a2,
                "ncbi2_comprimento": la,
                "ncbi3_inicio_1based": b1 + 1,
                "ncbi3_fim_1based": b2,
                "ncbi3_comprimento": lb,
                "seq_ncbi2": codigos_para_str(g2[a1:a2]) if la <= 60 else codigos_para_str(g2[a1:a1 + 30]) + f"...({la} nt)",
                "seq_ncbi3": codigos_para_str(g3[b1:b2]) if lb <= 60 else codigos_para_str(g3[b1:b1 + 30]) + f"...({lb} nt)",
                "flanco_esq": codigos_para_str(g2[max(a1 - 10, 0) : a1]),
                "flanco_dir": codigos_para_str(g2[a2 : a2 + 10]),
                "x_a1": a1, "x_a2": a2, "x_b1": b1, "x_b2": b2,
            }
        )
    return pd.DataFrame(linhas)


def reconstruir_ncbi3(g2: np.ndarray, df: pd.DataFrame, g3: np.ndarray) -> bool:
    """Applies the events to NC_000913.2 and compares with NC_000913.3 (consistency check)."""
    partes, pos = [], 0
    for r in df.itertuples():
        partes.append(g2[pos : r.x_a1])
        partes.append(g3[r.x_b1 : r.x_b2])
        pos = r.x_a2
    partes.append(g2[pos:])
    rec = np.concatenate(partes)
    return bool(len(rec) == len(g3) and np.array_equal(rec, g3))


def sobreposicao_cds(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    from modelo_baseline.annotation import load_cds_records

    cds = load_cds_records()
    ini = np.array([c["start"] for c in cds])
    fim = np.array([c["end"] for c in cds])
    linhas = []
    cds_tocados = set()
    for r in df.itertuples():
        a1, a2 = r.ncbi2_inicio_1based, r.ncbi2_fim_1based  # 1-based inclusive; empty when length is 0
        if r.ncbi2_comprimento == 0:
            # junction between a1-1 and a1 (1-based: between positions `a2` and `a2+1`): inside a CDS if both bases are covered by it
            esq, dir_ = a2, a2 + 1
            hit = np.flatnonzero((ini <= esq) & (fim >= dir_))
        else:
            hit = np.flatnonzero((ini <= a2) & (fim >= a1))
        nomes = [f"{cds[i]['gene'] or cds[i]['locus_tag']}({cds[i]['locus_tag']},{cds[i]['strand']})" for i in hit]
        cds_tocados.update(hit.tolist())
        linhas.append({"id": r.id, "n_cds": len(hit), "cds": ";".join(nomes)})
    out = pd.DataFrame(linhas)
    resumo = {
        "n_cds_artigo": len(cds),
        "n_eventos_sobrepostos_a_cds": int((out["n_cds"] > 0).sum()),
        "n_cds_sobrepostos": len(cds_tocados),
        "cds_sobrepostos": sorted(
            f"{cds[i]['gene'] or cds[i]['locus_tag']} ({cds[i]['locus_tag']})" for i in cds_tocados
        ),
    }
    return out, resumo


# =========================================================================== FASTQ reading
def blocos_fastq(path: Path, max_reads: int | None = None, chunk: int = CHUNK_READS):
    """Yields (codes[n, LMAX] uint8, lengths[n]) per block, in file order.

    Reads the gzip in byte blocks and locates the lines by '\n' (record = 4 lines)."""
    resto = b""
    total = 0
    with gzip.open(path, "rb") as fh:
        while True:
            dados = fh.read(1 << 25)
            buf = resto + dados if resto else dados
            if not buf:
                break
            arr = np.frombuffer(buf, dtype=np.uint8)
            nl = np.flatnonzero(arr == 10)
            nrec = len(nl) // 4
            if not dados:  # end of file
                if len(nl) % 4:
                    raise RuntimeError(f"FASTQ truncado: {path}")
            if nrec == 0:
                resto = buf
                if not dados:
                    break
                continue
            nl = nl[: 4 * nrec]
            ini = nl[0::4] + 1  # start of the sequence
            fimseq = nl[1::4]
            comp = (fimseq - ini).astype(np.int32)
            resto = buf[nl[-1] + 1 :]
            for s in range(0, nrec, chunk):
                e = min(s + chunk, nrec)
                if max_reads is not None and total + (e - s) > max_reads:
                    e = s + (max_reads - total)
                    if e <= s:
                        return
                ln = comp[s:e]
                if ln.max() > LMAX:
                    raise RuntimeError(f"read com {ln.max()} nt > LMAX")
                idx = ini[s:e, None] + np.arange(LMAX, dtype=np.int64)[None, :]
                np.clip(idx, 0, len(arr) - 1, out=idx)
                m = LUT[arr[idx]]
                m[np.arange(LMAX)[None, :] >= ln[:, None]] = 0
                total += e - s
                yield m, ln
                if max_reads is not None and total >= max_reads:
                    return
            if not dados:
                break


# =========================================================================== adapter
def corte_adaptador(m: np.ndarray, ln: np.ndarray, ad: np.ndarray, sob_min: int = SOBREPOSICAO_MIN):
    """Cut position (adapter start) per read: exact match of the whole adapter at
    any position, or an adapter prefix (>= sob_min nt) ending exactly at the end of the read.
    Returns (cut, type) with type 0 = no adapter, 1 = whole adapter, 2 = partial."""
    n = len(ln)
    la = len(ad)
    corte = ln.copy()
    tipo = np.zeros(n, dtype=np.uint8)
    for s in range(0, LMAX - la + 1):
        eq = np.ones(n, dtype=bool)
        for j in range(la):
            eq &= m[:, s + j] == ad[j]
        eq &= (s + la) <= ln
        novo = eq & (s < corte)
        corte[novo] = s
        tipo[novo] = 1
    ar = np.arange(n)
    for ov in range(la - 1, sob_min - 1, -1):
        eq = ln >= ov
        for j in range(ov):
            col = np.clip(ln - ov + j, 0, LMAX - 1)
            eq &= m[ar, col] == ad[j]
        novo = eq & ((ln - ov) < corte)
        corte[novo] = (ln - ov)[novo]
        tipo[novo] = 2
    return corte, tipo


def hash_linhas(m: np.ndarray, L: int, reverso: bool = False) -> np.ndarray:
    """Polynomial hash (mod 2^64) of the first L bases of each row (or of its reverse complement)."""
    h = np.zeros(len(m), dtype=np.uint64)
    sub = COMP[m[:, L - 1 :: -1]] if reverso else m
    for j in range(L):
        h = h * HASH_M + sub[:, j].astype(np.uint64)
    return h


def hashes_genoma(g: np.ndarray, Lmin: int, Lmax: int):
    """Yields (L, hash of all windows of length L) by the recurrence h_L = h_{L-1}*M + b."""
    b = g.astype(np.uint64)
    n = len(g)
    h = np.zeros(n, dtype=np.uint64)
    for L in range(1, Lmax + 1):
        h = h[: n - L + 1] * HASH_M + b[L - 1 : n]
        if L >= Lmin:
            yield L, h




# ---- step 3: adapter (empirical evidence) -----------------------------------------------
def inferir_adaptador_empirico(run: str, g3: np.ndarray, n_reads: int = 1_000_000) -> dict:
    """First `n_reads` reads: (i) most frequent 15-nt suffixes; (ii) for reads whose 25-nt prefix
    matches the genome exactly ONCE (both strands), the sequence that follows the first base
    that stops matching; (iii) fraction of reads with the adapter (corte_adaptador)."""
    n_pref = 25
    n = len(g3)
    rc_g = revcomp_codigos(g3)
    h_f = next(h for L, h in hashes_genoma(g3, n_pref, n_pref))
    h_r = next(h for L, h in hashes_genoma(rc_g, n_pref, n_pref))
    todos = np.concatenate([h_f, h_r])
    ordem = np.argsort(todos, kind="stable")
    hs = todos[ordem]
    base = np.concatenate([g3, rc_g, np.zeros(LMAX + 1, dtype=np.uint8)])
    n_f = len(h_f)
    ad = str_para_codigos(ADAPTADOR)
    la = len(ad)

    suf15, seguintes, fim_casa = Counter(), Counter(), Counter()
    total = com_ad = n_unico = 0
    for m, ln in blocos_fastq(FASTQ_DIR / f"{run}.fastq.gz", max_reads=n_reads):
        total += len(ln)
        _, tipo = corte_adaptador(m, ln, ad)
        com_ad += int((tipo > 0).sum())
        ar = np.arange(len(ln))
        # (i) final 15 nt
        idx = np.clip(ln[:, None] - 15 + np.arange(15)[None, :], 0, LMAX - 1)
        suf = m[ar[:, None], idx]
        ok = (suf > 0).all(axis=1) & (ln >= 15)
        for linha, c in zip(*np.unique(suf[ok], axis=0, return_counts=True)):
            suf15["".join("NACGT"[x] for x in linha)] += int(c)
        # (ii) 25-nt prefix with a unique match
        validos = (m[:, :n_pref] > 0).all(axis=1) & (ln >= n_pref)
        h = hash_linhas(m, n_pref)
        esq = np.searchsorted(hs, h, side="left")
        dire = np.searchsorted(hs, h, side="right")
        sel = np.flatnonzero(validos & (dire - esq == 1))
        n_unico += len(sel)
        if not len(sel):
            continue
        t = ordem[esq[sel]]
        gi = np.where(t < n_f, t, n + (t - n_f))
        gen = base[gi[:, None] + np.arange(LMAX)[None, :]]
        cols = np.arange(LMAX)[None, :]
        mism = (m[sel] != gen) & (cols < ln[sel, None])
        fim = np.where(mism.any(axis=1), mism.argmax(axis=1), ln[sel])
        for v, c in zip(*np.unique(fim, return_counts=True)):
            fim_casa[int(v)] += int(c)
        cabe = fim + la <= ln[sel]
        if cabe.any():
            msel = m[sel][cabe]
            sub = msel[np.arange(len(msel))[:, None], fim[cabe][:, None] + np.arange(la)[None, :]]
            for linha, c in zip(*np.unique(sub, axis=0, return_counts=True)):
                seguintes["".join("NACGT"[x] for x in linha)] += int(c)
    return {
        "reads_analisados": total,
        "reads_com_adaptador_(ADAPTADOR_testado)": com_ad,
        "fracao_com_adaptador_nos_primeiros_reads": com_ad / total,
        "sufixos_15nt_mais_frequentes": [
            {"seq": s, "reads": c, "fracao": c / total} for s, c in suf15.most_common(6)
        ],
        "reads_com_prefixo25_casando_uma_vez_no_genoma": n_unico,
        "posicao_da_primeira_base_nao_genomica_(mais_comuns)": [
            {"posicao": int(p), "reads": c, "fracao": c / max(n_unico, 1)} for p, c in fim_casa.most_common(5)
        ],
        "17nt_apos_o_fim_do_casamento_mais_frequentes": [
            {"seq": s, "reads": c, "fracao": c / max(n_unico, 1)} for s, c in seguintes.most_common(6)
        ],
    }


def campos_geo() -> dict:
    out = {}
    for gse, gsm in {"GSE119104": "GSM3358141", "GSE77617": "GSM2462935"}.items():
        txt = gzip.open(GEO_DIR / gse / f"{gse}_family.soft.gz", "rt", errors="replace").read()
        i = txt.index(f"^SAMPLE = {gsm}")
        j = txt.find("^SAMPLE", i + 5)
        sel = [
            l[:900] for l in txt[i:j].splitlines()
            if l.startswith(("!Sample_extract_protocol_ch1", "!Sample_data_processing", "!Sample_instrument_model"))
        ]
        out[gse] = {"amostra": gsm, "linhas": sel}
    return out


# ---- allele probes -----------------------------------------------------------------------
def _kmers_ambas_fitas(g: np.ndarray, k: int) -> np.ndarray:
    return np.sort(np.concatenate([kmers_2bit(g, k), kmers_2bit(revcomp_codigos(g), k)]))


def _contagem(ordenado: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.searchsorted(ordenado, v, side="right") - np.searchsorted(ordenado, v, side="left")


def _revcomp_kmer(v: np.ndarray, k: int) -> np.ndarray:
    out = np.zeros_like(v)
    x = v.copy()
    for _ in range(k):
        out = (out << np.uint64(2)) | (np.uint64(3) - (x & np.uint64(3)))
        x = x >> np.uint64(2)
    return out


def construir_sondas(g2: np.ndarray, g3: np.ndarray, df: pd.DataFrame) -> dict:
    """Allele-specific K_PROBE-mers per divergent site.

    class 0 = allele NC_000913.2 ; 1 = allele NC_000913.3 ; 2/3 = flank controls (window that
    ends at the base before / starts at the base after the site, identical in both references).
    An allele probe is kept if it occurs exactly once (both strands) in its source reference and
    zero times in the other; a control, if it occurs once in each. Probes are valid in both orientations."""
    K = K_PROBE
    km = {"NC_000913.2": _kmers_ambas_fitas(g2, K), "NC_000913.3": _kmers_ambas_fitas(g3, K)}
    chaves, ids, tab = [], [], []
    for sitio, r in enumerate(df.itertuples()):
        for classe, (ref, g, x1, x2) in enumerate(
            [("NC_000913.2", g2, r.x_a1, r.x_a2), ("NC_000913.3", g3, r.x_b1, r.x_b2)]
        ):
            outro = "NC_000913.3" if ref == "NC_000913.2" else "NC_000913.2"
            n = len(g)
            ss = np.arange(max(x1 - K, 0), min(x2 + 1, n - K) + 1)
            ok = np.zeros(len(ss), dtype=bool)
            if 0 < x2 - x1 <= K:
                ok |= (ss < x2) & (ss + K > x1)
            ok |= (ss <= x1 - 1) & (ss + K - 1 >= x1)
            ok |= (ss <= x2 - 1) & (ss + K - 1 >= x2)
            ss = ss[ok]
            v = np.array([int(kmers_2bit(g[s : s + K], K)[0]) for s in ss], dtype=np.uint64)
            manter = (_contagem(km[ref], v) == 1) & (_contagem(km[outro], v) == 0)
            v = v[manter]
            chaves.append(np.concatenate([v, _revcomp_kmer(v, K)]))
            ids.append(np.full(2 * len(v), 4 * sitio + classe, dtype=np.int64))
            tab.append({"sitio": r.id, "classe": ["alelo_NC_000913.2", "alelo_NC_000913.3"][classe],
                        "janelas_candidatas": int(len(ss)), "sondas_especificas": int(len(v))})
        # flank controls (NC_000913.2 coordinates)
        for classe, s, nome in ((2, r.x_a1 - K, "controle_flanco_esquerdo"), (3, r.x_a2, "controle_flanco_direito")):
            if s < 0 or s + K > len(g2):
                tab.append({"sitio": r.id, "classe": nome, "janelas_candidatas": 0, "sondas_especificas": 0})
                continue
            v = kmers_2bit(g2[s : s + K], K)
            manter = (_contagem(km["NC_000913.2"], v) == 1) & (_contagem(km["NC_000913.3"], v) == 1)
            v = v[manter]
            chaves.append(np.concatenate([v, _revcomp_kmer(v, K)]))
            ids.append(np.full(2 * len(v), 4 * sitio + classe, dtype=np.int64))
            tab.append({"sitio": r.id, "classe": nome, "janelas_candidatas": 1, "sondas_especificas": int(len(v))})
    chaves = np.concatenate(chaves)
    ids = np.concatenate(ids)
    o = np.argsort(chaves, kind="stable")
    chaves, ids = chaves[o], ids[o]
    dup = np.zeros(len(chaves), dtype=bool)  # k-mer in two entries (e.g. palindrome): discard
    dup[1:] |= chaves[1:] == chaves[:-1]
    dup[:-1] |= chaves[:-1] == chaves[1:]
    return {
        "chaves": chaves[~dup],
        "id": ids[~dup],
        "n_sitio_alelo": 4 * len(df),
        "tabela": pd.DataFrame(tab),
    }


# ---- pass over the FASTQ --------------------------------------------------------------------
def _tabela_unica(m, tl, elegiveis, ad_flag):
    """Unique reads (by L and direct hash) with total count and adapter count, ordered by L."""
    partes = []
    for L in np.unique(tl[elegiveis]):
        sel = np.flatnonzero(elegiveis & (tl == L))
        hf = hash_linhas(m[sel], int(L))
        hr = hash_linhas(m[sel], int(L), reverso=True)
        u, ix, inv = np.unique(hf, return_index=True, return_inverse=True)
        ct = np.bincount(inv, minlength=len(u)).astype(np.int32)
        ca = np.bincount(inv, weights=ad_flag[sel], minlength=len(u)).astype(np.int32)
        partes.append(np.rec.fromarrays([np.full(len(u), L, np.uint8), u, hr[ix], ct, ca], names="L,hf,hr,ct,ca"))
    return np.concatenate(partes) if partes else None


def passar_fastq(run: str, sondas: dict, lmin_tab: int, dominante: int, max_reads: int | None = None) -> dict:
    ad = str_para_codigos(ADAPTADOR)
    pasta = CACHE_R6 / run
    pasta.mkdir(parents=True, exist_ok=True)
    for f in pasta.glob("*.npy"):
        f.unlink()
    st = Counter()
    h_corte, h_pos, h_bruto, apos = Counter(), Counter(), Counter(), Counter()
    ns = sondas["n_sitio_alelo"]
    chaves, sid = sondas["chaves"], sondas["id"]
    sup, sup_ad = np.zeros(ns, dtype=np.int64), np.zeros(ns, dtype=np.int64)
    nb = 0
    K = K_PROBE
    mask_k = np.uint64((1 << (2 * K)) - 1)
    for m, ln in blocos_fastq(FASTQ_DIR / f"{run}.fastq.gz", max_reads=max_reads):
        n = len(ln)
        cols = np.arange(LMAX)[None, :]
        st["reads_totais"] += n
        for L, c in zip(*np.unique(ln, return_counts=True)):
            h_bruto[int(L)] += int(c)
        corte, tipo = corte_adaptador(m, ln, ad)
        com = tipo > 0
        st["reads_com_adaptador"] += int(com.sum())
        st["adaptador_inteiro"] += int((tipo == 1).sum())
        st["adaptador_parcial_no_fim"] += int((tipo == 2).sum())
        for v, c in zip(*np.unique(corte[com], return_counts=True)):
            h_corte[int(v)] += int(c)
        for v, c in zip(*np.unique(corte, return_counts=True)):
            h_pos[int(v)] += int(c)
        if nb < 5:  # bases following the whole adapter (sample of the first 5 blocks)
            i1 = np.flatnonzero((tipo == 1) & (corte + len(ad) + 8 <= ln))
            if len(i1):
                p = corte[i1] + len(ad)
                sub = m[i1[:, None], p[:, None] + np.arange(8)[None, :]]
                for linha, c in zip(*np.unique(sub, axis=0, return_counts=True)):
                    apos["".join("NACGT"[x] for x in linha)] += int(c)
        invalida = (m == 0) & (cols < ln[:, None])
        st["reads_com_N_no_read_inteiro"] += int(invalida.any(axis=1).sum())
        # (a) no trimming, dominant length only: reproduces the article's criterion
        eleg0 = (ln == dominante) & ~invalida.any(axis=1)
        t0 = _tabela_unica(m, ln, eleg0, np.zeros(n))
        if t0 is not None:
            np.save(pasta / f"s{nb:05d}.npy", t0)
        # (b) with trimming
        invalida_pos = ((m == 0) & (cols < corte[:, None])).any(axis=1)
        eleg = (corte >= lmin_tab) & ~invalida_pos
        st["reads_pos_corte_>=15"] += int((corte >= 15).sum())
        st["reads_pos_corte_>=15_com_N"] += int(((corte >= 15) & invalida_pos).sum())
        t1 = _tabela_unica(m, corte, eleg, com.astype(np.float64))
        if t1 is not None:
            np.save(pasta / f"c{nb:05d}.npy", t1)
        # (c) allele probes
        run_ok = np.zeros(n, dtype=np.int16)
        val = np.zeros(n, dtype=np.uint64)
        pl, pi_ = [], []
        for s in range(LMAX):
            col = m[:, s]
            run_ok = np.where(col > 0, run_ok + 1, 0)
            val = ((val << np.uint64(2)) | ((col.astype(np.uint64) - np.uint64(1)) & np.uint64(3))) & mask_k
            if s < K - 1:
                continue
            idx = np.flatnonzero((run_ok >= K) & (s < corte))
            if not len(idx):
                continue
            q = val[idx]
            pos = np.searchsorted(chaves, q)
            pos[pos == len(chaves)] = 0
            acerto = chaves[pos] == q
            if acerto.any():
                pl.append(idx[acerto])
                pi_.append(sid[pos[acerto]])
        if pl:
            comb = np.unique(np.concatenate(pl).astype(np.int64) * ns + np.concatenate(pi_))
            rd, sa = comb // ns, comb % ns
            np.add.at(sup, sa, 1)
            np.add.at(sup_ad, sa, com[rd].astype(np.int64))
        nb += 1
        if nb % 25 == 0:
            log(f"  {run}: {st['reads_totais']:,} reads")
    return {
        "estatisticas": dict(st),
        "hist_inicio_do_adaptador_(reads_com_adaptador)": dict(sorted(h_corte.items())),
        "hist_comprimento_pos_corte_(todos_os_reads)": dict(sorted(h_pos.items())),
        "hist_comprimento_bruto": dict(sorted(h_bruto.items())),
        "8nt_apos_adaptador_inteiro_(amostra)": apos.most_common(5),
        "suporte_leituras": sup,
        "suporte_leituras_com_adaptador": sup_ad,
        "n_blocos": nb,
    }


# ---- step 4: exact matching -------------------------------------------------------------
def _juntar_por_L(run: str, prefixo: str, L: int):
    """Unique reads of length L from all blocks (prefix 'c' = trimmed, 's' = untrimmed)."""
    partes = []
    for f in sorted((CACHE_R6 / run).glob(f"{prefixo}*.npy")):
        t = np.load(f, mmap_mode="r")
        lo, hi = np.searchsorted(t["L"], L, "left"), np.searchsorted(t["L"], L, "right")
        if hi > lo:
            partes.append(np.array(t[lo:hi]))
    if not partes:
        return None
    t = np.concatenate(partes)
    u, ix, inv = np.unique(t["hf"], return_index=True, return_inverse=True)
    return {
        "hf": u,
        "hr": t["hr"][ix],
        "ct": np.bincount(inv, weights=t["ct"], minlength=len(u)).astype(np.int64),
        "ca": np.bincount(inv, weights=t["ca"], minlength=len(u)).astype(np.int64),
    }


def consultar(hs, ordem, hf, hr):
    """Exact occurrences of the read (forward strand + reverse complement) and bounds to recover positions."""
    lf, rf = np.searchsorted(hs, hf, "left"), np.searchsorted(hs, hf, "right")
    lr, rr = np.searchsorted(hs, hr, "left"), np.searchsorted(hs, hr, "right")
    cnt = (rf - lf) + (rr - lr)
    primeira = np.where(rf > lf, ordem[np.minimum(lf, len(ordem) - 1)], ordem[np.minimum(lr, len(ordem) - 1)])
    return cnt.astype(np.int32), primeira.astype(np.int64), (lf, rf, lr, rr)


def categorias_gff(n: int) -> np.ndarray:
    """0 outside annotated features, 1 CDS, 2 ncRNA, 3 tRNA, 4 rRNA (NC_000913.3 coordinates)."""
    cat = np.zeros(n, dtype=np.int8)
    tipos = {"CDS": 1, "ncRNA": 2, "SRP_RNA": 2, "RNase_P_RNA": 2, "tmRNA": 2, "antisense_RNA": 2,
             "tRNA": 3, "rRNA": 4}
    feats = []
    with gzip.open(GFF_REF3, "rt") as fh:
        for l in fh:
            if l.startswith("#"):
                continue
            c = l.rstrip("\n").split("\t")
            if len(c) > 7 and c[2] in tipos:
                feats.append((tipos[c[2]], int(c[3]), int(c[4])))
    for tipo, a, b in sorted(feats):
        cat[a - 1 : b] = tipo
    return cat


NOMES_CAT = {0: "fora_de_CDS_e_RNAs_anotados", 1: "CDS", 2: "ncRNA_outros", 3: "tRNA", 4: "rRNA"}


def mapear_runs(g: dict, df_ev: pd.DataFrame, lmin_tab: int, dominante: dict) -> dict:
    """Maps the unique-read tables onto the two references, L by L."""
    gens = {r: hashes_genoma(g[r], lmin_tab, LMAX) for r in REFS}
    cat3 = categorias_gff(len(g["NC_000913.3"]))
    evs = {
        "NC_000913.2": [(r.id, r.x_a1, r.x_a2) for r in df_ev.itertuples()],
        "NC_000913.3": [(r.id, r.x_b1, r.x_b2) for r in df_ev.itertuples()],
    }

    def perto(ref, posicoes, L):
        achados = []
        for sid_, x1, x2 in evs[ref]:
            if any(p < x2 + 1 and p + L > x1 - 1 for p, _f in posicoes):
                achados.append(sid_)
        return achados

    def posicoes(ordem, lim, i):
        lf, rf, lr, rr = (a[i] for a in lim)
        return [(int(p), "+") for p in ordem[lf:rf]] + [(int(p), "-") for p in ordem[lr:rr]]

    por_L, sem_corte, cats, discord = [], [], [], []
    for L in range(lmin_tab, LMAX + 1):
        hh = {r: next(gens[r])[1] for r in REFS}
        proc = {}
        for run in RUNS:
            proc[(run, "c")] = _juntar_por_L(run, "c", L)
            proc[(run, "s")] = _juntar_por_L(run, "s", L) if L == dominante[run] else None
        if all(v is None for v in proc.values()):
            continue
        idx = {}
        for r in REFS:
            o = np.argsort(hh[r], kind="stable")
            idx[r] = (hh[r][o], o)
        for (run, modo), t in proc.items():
            if t is None:
                continue
            res = {r: consultar(idx[r][0], idx[r][1], t["hf"], t["hr"]) for r in REFS}
            for r in REFS:
                cnt = res[r][0]
                for nome, msk in (("unico", cnt == 1), ("multiplo", cnt >= 2)):
                    (por_L if modo == "c" else sem_corte).append(
                        {"run": run, "referencia": r, "L": L, "classe": nome,
                         "reads": int(t["ct"][msk].sum()), "reads_com_adaptador": int(t["ca"][msk].sum()),
                         "sequencias_distintas": int(msk.sum())}
                    )
            if modo != "c":
                continue
            # feature categories (NC_000913.3, midpoint of the first match)
            cnt3, prim3, _ = res["NC_000913.3"]
            mp = cnt3 >= 1
            c3 = cat3[np.minimum(prim3[mp] + L // 2, len(cat3) - 1)]
            for nome, msk in (("unico", cnt3[mp] == 1), ("multiplo", cnt3[mp] >= 2)):
                for k in range(5):
                    w = int(t["ct"][mp][msk & (c3 == k)].sum())
                    if w:
                        cats.append({"run": run, "L": L, "classe": nome, "categoria": NOMES_CAT[k], "reads": w})
            # discordant
            cl2, cl3 = np.minimum(res["NC_000913.2"][0], 2), np.minimum(res["NC_000913.3"][0], 2)
            for i in np.flatnonzero(cl2 != cl3):
                p2 = posicoes(idx["NC_000913.2"][1], res["NC_000913.2"][2], i)
                p3 = posicoes(idx["NC_000913.3"][1], res["NC_000913.3"][2], i)
                ref_seq, (p0, f0) = ("NC_000913.2", p2[0]) if p2 else ("NC_000913.3", p3[0])
                seq = g[ref_seq][p0 : p0 + L]
                seq = seq if f0 == "+" else revcomp_codigos(seq)
                nomes_cl = ["nao_mapeia", "unico", "multiplo"]
                discord.append(
                    {"run": run, "L": L, "reads": int(t["ct"][i]), "reads_com_adaptador": int(t["ca"][i]),
                     "classe_NC_000913.2": nomes_cl[int(cl2[i])], "classe_NC_000913.3": nomes_cl[int(cl3[i])],
                     "posicoes_NC_000913.2_(1-based,fita)": ";".join(f"{p + 1}{f}" for p, f in p2[:6]),
                     "posicoes_NC_000913.3_(1-based,fita)": ";".join(f"{p + 1}{f}" for p, f in p3[:6]),
                     "sitios_tocados_NC_000913.2": ",".join(map(str, perto("NC_000913.2", p2, L))),
                     "sitios_tocados_NC_000913.3": ",".join(map(str, perto("NC_000913.3", p3, L))),
                     "sequencia": codigos_para_str(seq)}
                )
        log(f"  L={L} ok")
    return {
        "por_L": pd.DataFrame(por_L),
        "sem_corte": pd.DataFrame(sem_corte),
        "categorias": pd.DataFrame(cats),
        "discordantes": pd.DataFrame(discord),
    }


# ---- independent validation of the hash ----------------------------------------------------------
def _para_str(g: np.ndarray) -> str:
    return bytes(np.frombuffer(b"NACGT", dtype=np.uint8)[g]).decode()


def _contar_str(gs: str, rc: str, s: str) -> int:
    total = 0
    for alvo in (gs, rc):
        i = alvo.find(s)
        while i >= 0:
            total += 1
            i = alvo.find(s, i + 1)
    return total


def validar_hash(g: dict, n_reads: int = 300_000, passo: int = 11, teto: int = 2000) -> dict:
    """Compares the hash count with string search (str.find, both strands) on already trimmed reads."""
    ad = str_para_codigos(ADAPTADOR)
    amostra = {}
    for run in RUNS:
        for m, ln in blocos_fastq(FASTQ_DIR / f"{run}.fastq.gz", max_reads=n_reads):
            corte, _ = corte_adaptador(m, ln, ad)
            sem_n = ~(((m == 0) & (np.arange(LMAX)[None, :] < corte[:, None])).any(axis=1))
            ok = np.flatnonzero((corte >= 20) & sem_n)
            for i in ok[::passo]:
                amostra.setdefault(run, []).append(m[i, : corte[i]].copy())
        amostra[run] = amostra[run][:teto]
    gs = {r: _para_str(g[r]) for r in REFS}
    rcs = {r: _para_str(revcomp_codigos(g[r])) for r in REFS}
    por_L = {}
    for run, lista in amostra.items():
        for k, seq in enumerate(lista):
            por_L.setdefault(len(seq), []).append((run, k))
    res = {r: 0 for r in REFS}
    div = {r: 0 for r in REFS}
    n_map = {r: 0 for r in REFS}
    gens = {r: hashes_genoma(g[r], min(por_L), max(por_L)) for r in REFS}
    for L in range(min(por_L), max(por_L) + 1):
        hh = {r: next(gens[r])[1] for r in REFS}
        if L not in por_L:
            continue
        seqs = np.array([amostra[run][k] for run, k in por_L[L]], dtype=np.uint8)
        hf, hr = hash_linhas(seqs, L), hash_linhas(seqs, L, reverso=True)
        for r in REFS:
            o = np.argsort(hh[r], kind="stable")
            cnt, _, _ = consultar(hh[r][o], o, hf, hr)
            for j, s in enumerate(seqs):
                ref_cnt = _contar_str(gs[r], rcs[r], _para_str(s))
                res[r] += 1
                n_map[r] += int(ref_cnt > 0)
                div[r] += int(ref_cnt != cnt[j])
    return {"reads_verificados_por_referencia": res, "reads_com_casamento_por_busca_de_strings": n_map,
            "divergencias_hash_x_strings": div}


# ---- aggregation ----------------------------------------------------------------------------
def resumir(por_L: pd.DataFrame, passagens: dict) -> pd.DataFrame:
    linhas = []
    for run in RUNS:
        st = passagens[run]["estatisticas"]
        hpos = {int(k): v for k, v in passagens[run]["hist_comprimento_pos_corte_(todos_os_reads)"].items()}
        for ref in REFS:
            sub = por_L[(por_L["run"] == run) & (por_L["referencia"] == ref)]
            for T in LIMIARES:
                s = sub[sub["L"] >= T]
                un = int(s[s["classe"] == "unico"]["reads"].sum())
                mu = int(s[s["classe"] == "multiplo"]["reads"].sum())
                un_ad = int(s[s["classe"] == "unico"]["reads_com_adaptador"].sum())
                mu_ad = int(s[s["classe"] == "multiplo"]["reads_com_adaptador"].sum())
                ge = sum(v for k, v in hpos.items() if k >= T)
                linhas.append(
                    {"run": run, "dataset": RUNS[run], "referencia": ref, "comprimento_minimo": T,
                     "reads_totais": st["reads_totais"], "reads_com_adaptador": st["reads_com_adaptador"],
                     "reads_com_comprimento_pos_corte_>=T": ge,
                     "mapeados": un + mu, "unicos": un, "multiplos": mu,
                     "mapeados_com_adaptador": un_ad + mu_ad,
                     "taxa_mapeados_sobre_totais": (un + mu) / st["reads_totais"],
                     "taxa_mapeados_sobre_>=T": (un + mu) / ge if ge else float("nan"),
                     "taxa_unicos_sobre_totais": un / st["reads_totais"],
                     "taxa_multiplos_sobre_totais": mu / st["reads_totais"]}
                )
    return pd.DataFrame(linhas)


def tabela_sitios(df_ev, passagens):
    linhas = []
    for run in RUNS:
        sup = passagens[run]["suporte_leituras"]
        for s, r in enumerate(df_ev.itertuples()):
            linhas.append(
                {"run": run, "sitio": r.id, "tipo": r.tipo, "ncbi2_inicio": r.ncbi2_inicio_1based,
                 "reads_com_sonda_alelo_NC_000913.2": int(sup[4 * s]),
                 "reads_com_sonda_alelo_NC_000913.3": int(sup[4 * s + 1]),
                 "reads_controle_flanco_esq": int(sup[4 * s + 2]),
                 "reads_controle_flanco_dir": int(sup[4 * s + 3])}
            )
    return pd.DataFrame(linhas)


def main() -> None:
    import json

    ap = argparse.ArgumentParser()
    ap.add_argument("--reusar-cache", action="store_true", help="pula a passagem nos FASTQ se o cache existir")
    ap.add_argument("--max-reads", type=int, default=None, help="subamostra dos N primeiros reads por execucao")
    ap.add_argument("--saida", default=None, help="subpasta de saida (padrao: r6)")
    args = ap.parse_args()
    global OUT, CACHE_R6
    if args.saida:
        OUT = OUT / args.saida
        OUT.mkdir(parents=True, exist_ok=True)
        CACHE_R6 = CACHE_R6 / args.saida
    t_ini = time.time()
    tempos = {}

    def marca(nome):
        tempos[nome] = round(time.time() - t_ini, 1)
        log(f"== {nome} ({tempos[nome]} s acumulados)")

    # 1. MD5
    md5 = verificar_md5()
    md5.to_csv(OUT / "md5_fastq.csv", index=False)
    marca("md5")

    # 2. differences between references
    g = {r: ler_fasta(REF_DIR / f"{r}.fasta") for r in REFS}
    df = diferencas_referencias(g["NC_000913.2"], g["NC_000913.3"])
    reconstroi = reconstruir_ncbi3(g["NC_000913.2"], df, g["NC_000913.3"])
    if not reconstroi:
        raise RuntimeError("diferencas nao reconstroem NC_000913.3")
    cds_tab, cds_res = sobreposicao_cds(df)
    publico = df.drop(columns=[c for c in df.columns if c.startswith("x_")]).merge(cds_tab, on="id")
    publico.to_csv(OUT / "diferencas_referencias.csv", index=False)
    escrever_json(OUT / "diferencas_referencias_resumo.json", {
        "comprimento_NC_000913.2": len(g["NC_000913.2"]), "comprimento_NC_000913.3": len(g["NC_000913.3"]),
        "n_eventos": len(df), "eventos_por_tipo": df["tipo"].value_counts().to_dict(),
        "soma_delta_comprimento": int((df["ncbi3_comprimento"] - df["ncbi2_comprimento"]).sum()),
        "reconstrucao_exata_de_NC_000913.3_a_partir_de_NC_000913.2": reconstroi,
        "bases_substituidas": int(df.loc[df.tipo.str.startswith("substituicao"), "ncbi2_comprimento"].sum()),
        "sha256_fasta": {r: sha256_arquivo(REF_DIR / f"{r}.fasta") for r in REFS},
        **cds_res,
    })
    marca("diferencas")

    # 3. adapter
    geo = campos_geo()
    emp = {run: inferir_adaptador_empirico(run, g["NC_000913.3"]) for run in RUNS}
    escrever_json(OUT / "adaptador_evidencias.json", {
        "adaptador_usado_5p3p": ADAPTADOR, "sobreposicao_parcial_minima_nt": SOBREPOSICAO_MIN,
        "geo_protocolo": geo, "empirico_primeiros_1e6_reads": emp,
    })
    marca("adaptador")

    # 4/5. pass over the FASTQ
    sondas = construir_sondas(g["NC_000913.2"], g["NC_000913.3"], df)
    sondas["tabela"].to_csv(OUT / "sondas_de_alelo.csv", index=False)
    dominante = {"SRR7759813": 50, "SRR5186138": 51}
    passagens = {}
    for run in RUNS:
        cache_json = CACHE_R6 / run / "passagem.json"
        if args.reusar_cache and cache_json.exists():
            p = json.loads(cache_json.read_text(encoding="utf-8"))
            p["suporte_leituras"] = np.array(p["suporte_leituras"])
            p["suporte_leituras_com_adaptador"] = np.array(p["suporte_leituras_com_adaptador"])
            passagens[run] = p
            log(f"cache reutilizado: {run}")
        else:
            passagens[run] = passar_fastq(run, sondas, 15, dominante[run], args.max_reads)
            ser = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in passagens[run].items()}
            cache_json.write_text(json.dumps(ser), encoding="utf-8")
        marca(f"passagem_{run}")
    mp = mapear_runs(g, df, 15, dominante)
    marca("mapeamento")
    mp["por_L"].to_csv(OUT / "mapeamento_por_comprimento.csv", index=False)
    resumo = resumir(mp["por_L"], passagens)
    resumo.to_csv(OUT / "resumo_remapeamento_com_corte.csv", index=False)
    mp["categorias"].groupby(["run", "classe", "categoria"], as_index=False)["reads"].sum().to_csv(
        OUT / "categorias_de_feature_NC_000913.3.csv", index=False)
    mp["categorias"][mp["categorias"]["L"] >= COMPRIMENTO_MIN].groupby(
        ["run", "classe", "categoria"], as_index=False)["reads"].sum().to_csv(
        OUT / "categorias_de_feature_NC_000913.3_L20.csv", index=False)
    mp["discordantes"].sort_values(["run", "L", "sequencia"]).to_csv(OUT / "reads_discordantes.csv", index=False)
    sitios = tabela_sitios(df, passagens)
    sitios.to_csv(OUT / "suporte_por_alelo_nos_sitios.csv", index=False)

    # replicates the article's criterion (no trimming, dominant length)
    sc = mp["sem_corte"]
    rep = []
    for run in RUNS:
        for ref in REFS:
            s = sc[(sc["run"] == run) & (sc["referencia"] == ref)]
            un, mu = int(s[s.classe == "unico"].reads.sum()), int(s[s.classe == "multiplo"].reads.sum())
            rep.append({"run": run, "referencia": ref, "reads_totais": passagens[run]["estatisticas"]["reads_totais"],
                        "mapeados": un + mu, "unicos": un, "multiplos": mu})
    rep = pd.DataFrame(rep)
    orig = pd.read_csv(BASE / "resultados" / "resumo_fullrun_remapeamento.csv")
    rep = rep.merge(
        orig[["run", "reference", "mapped_reads", "unique_best_reads", "multi_best_reads", "total_reads_fullrun"]]
        .rename(columns={"reference": "referencia", "mapped_reads": "artigo_mapeados", "unique_best_reads": "artigo_unicos",
                         "multi_best_reads": "artigo_multiplos", "total_reads_fullrun": "artigo_reads_totais"}),
        on=["run", "referencia"])
    rep.to_csv(OUT / "replica_criterio_do_artigo.csv", index=False)
    marca("agregacao")

    val = validar_hash(g)
    escrever_json(OUT / "validacao_hash.json", val)
    marca("validacao_hash")

    for run in RUNS:
        p = passagens[run]
        escrever_json(OUT / f"adaptador_{run}_distribuicoes.json", {
            "estatisticas": p["estatisticas"],
            "hist_inicio_do_adaptador_(reads_com_adaptador)": p["hist_inicio_do_adaptador_(reads_com_adaptador)"],
            "hist_comprimento_pos_corte_(todos_os_reads)": p["hist_comprimento_pos_corte_(todos_os_reads)"],
            "hist_comprimento_bruto": p["hist_comprimento_bruto"],
            "8nt_apos_adaptador_inteiro_(amostra)": p["8nt_apos_adaptador_inteiro_(amostra)"],
        })
    resumo_hash = {}
    for nome in ("diferencas_referencias.csv", "resumo_remapeamento_com_corte.csv", "reads_discordantes.csv",
                 "suporte_por_alelo_nos_sitios.csv", "replica_criterio_do_artigo.csv", "mapeamento_por_comprimento.csv"):
        resumo_hash[nome] = sha256_arquivo(OUT / nome)
    escrever_json(OUT / "execucao.json", {
        "comando": "python -m analises_revisao.r6_remapeamento_adaptadores"
        + (f" --max-reads {args.max_reads}" if args.max_reads else ""),
        "max_reads": args.max_reads, "semente": semente(6, 0), "aleatoriedade": "nenhuma",
        "parametros": {"adaptador": ADAPTADOR, "sobreposicao_min": SOBREPOSICAO_MIN,
                       "comprimento_min_principal": COMPRIMENTO_MIN, "limiares": list(LIMIARES), "k_sonda": K_PROBE},
        "tempo_total_s": round(time.time() - t_ini, 1), "tempos_acumulados_s": tempos,
        "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
        "sha256_saidas_resumo": resumo_hash,
    })
    log(f"concluido em {time.time() - t_ini:.1f} s")


if __name__ == "__main__":
    main()
