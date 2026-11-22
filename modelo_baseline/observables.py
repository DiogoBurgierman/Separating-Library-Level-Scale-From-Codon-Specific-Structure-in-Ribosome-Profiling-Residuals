import numpy as np

from . import config
from .annotation import codon_nt_indices


def _shifted_lookup(track: np.ndarray, indices: np.ndarray, shift: int, strand: str) -> np.ndarray:
    delta = shift if strand == "+" else -shift
    lookup = indices + delta
    values = np.zeros(len(indices), dtype=np.float64)
    valid = (lookup >= 0) & (lookup < len(track))
    values[valid] = track[lookup[valid]]
    return values


def gene_codon_occupancy(record: dict, tracks: dict, genome_len: int, asite_shift: int) -> np.ndarray | None:
    indices = np.array(
        codon_nt_indices(record["start"], record["end"], record["strand"], genome_len),
        dtype=np.int64,
    )
    if len(indices) != len(record["seq"]):
        return None
    values = _shifted_lookup(tracks[record["strand"]], indices, asite_shift, record["strand"])
    return values.reshape(-1, 3).mean(axis=1)


def elongation_codon_slice(n_codons: int) -> slice:
    drop_start = config.EXCLUDE_START_NT // 3
    drop_end = config.EXCLUDE_STOP_NT // 3
    return slice(drop_start, n_codons - drop_end)


def relative_occupancy_and_rate(occupancy: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    keep = elongation_codon_slice(len(occupancy))
    interior = occupancy[keep]
    gene_mean = float(np.mean(occupancy)) if occupancy.size else 0.0
    interior_mean = float(np.mean(interior)) if interior.size else 0.0
    qc = {
        "gene_mean": gene_mean,
        "interior_mean": interior_mean,
        "n_codons": int(len(occupancy)),
        "n_interior": int(interior.size),
        "kept": False,
        "reason": "",
    }
    if occupancy.size == 0 or gene_mean <= config.MIN_GENE_MEAN_OCCUPANCY:
        qc["reason"] = "low_coverage"
        return np.array([]), np.array([]), qc
    if interior.size == 0 or interior_mean <= 0:
        qc["reason"] = "empty_interior"
        return np.array([]), np.array([]), qc
    rho = interior / gene_mean
    positive = rho > 0
    if not np.any(positive):
        qc["reason"] = "zero_relative_occupancy"
        return np.array([]), np.array([]), qc
    k_obs = np.full(rho.shape, np.nan, dtype=np.float64)
    k_obs[positive] = 1.0 / rho[positive]
    qc["kept"] = True
    return rho, k_obs, qc
