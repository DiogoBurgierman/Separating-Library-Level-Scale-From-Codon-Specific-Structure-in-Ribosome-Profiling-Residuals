from pathlib import Path
from urllib.request import Request, urlopen

from . import config


def ensure_cds_fasta() -> Path:
    path = config.CDS_FASTA
    if path.exists() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    request = Request(
        config.CDS_FASTA_URL,
        headers={"User-Agent": "codon-baseline/1.0"},
    )
    with urlopen(request, timeout=180) as response:
        data = response.read()
    if not data.startswith(b">"):
        raise RuntimeError("CDS download did not return FASTA")
    path.write_bytes(data)
    return path


def load_genome_length(fasta_path: Path | None = None) -> int:
    path = fasta_path or config.GENOME_FASTA
    length = 0
    with path.open("rt", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith(">"):
                continue
            length += len(line.strip())
    return length


def _parse_location(location: str):
    if "join(" in location:
        return None
    if location.startswith("complement(") and location.endswith(")"):
        inner = location[len("complement(") : -1]
        if ".." not in inner or "," in inner:
            return None
        start_s, end_s = inner.split("..")
        return int(start_s), int(end_s), "-"
    if ".." not in location or "," in location:
        return None
    start_s, end_s = location.split("..")
    return int(start_s), int(end_s), "+"


def _header_field(header: str, key: str) -> str:
    token = f"[{key}="
    start = header.find(token)
    if start < 0:
        return ""
    start += len(token)
    end = header.find("]", start)
    if end < 0:
        return ""
    return header[start:end]


def load_cds_records(fasta_path: Path | None = None) -> list[dict]:
    path = fasta_path or ensure_cds_fasta()
    records = []
    header = None
    chunks: list[str] = []

    def flush():
        if header is None:
            return
        if "pseudo=true" in header.lower() or "[transl_except=" in header:
            return
        location = _header_field(header, "location")
        parsed = _parse_location(location)
        if parsed is None:
            return
        start, end, strand = parsed
        seq = "".join(chunks).upper().replace("U", "T")
        if end < start:
            return
        coord_len = end - start + 1
        if coord_len != len(seq) or coord_len % 3 != 0 or len(seq) < (
            config.EXCLUDE_START_NT + config.EXCLUDE_STOP_NT + 3
        ):
            return
        records.append(
            {
                "locus_tag": _header_field(header, "locus_tag"),
                "gene": _header_field(header, "gene"),
                "protein_id": _header_field(header, "protein_id"),
                "start": start,
                "end": end,
                "strand": strand,
                "seq": seq,
            }
        )

    with path.open("rt", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith(">"):
                flush()
                header = line.rstrip("\n")
                chunks = []
            else:
                chunks.append(line.strip())
        flush()
    return records


def codon_nt_indices(start: int, end: int, strand: str, genome_len: int) -> list[int]:
    """0-based genomic indices in 5'->3' codon order."""
    if strand == "+":
        return list(range(start - 1, end))
    return list(range(end - 1, start - 2, -1))
