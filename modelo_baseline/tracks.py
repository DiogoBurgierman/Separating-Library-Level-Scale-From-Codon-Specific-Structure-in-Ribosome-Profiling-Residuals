import gzip
import re
import tarfile
from pathlib import Path

import numpy as np

from . import config


_STRAND_RE = re.compile(
    r"^(GSM\d+)_(.+)_(plus|minus|for|rev)\.wig\.gz$",
    re.IGNORECASE,
)


def _normalize_strand(token: str) -> str:
    token = token.lower()
    if token in {"plus", "for"}:
        return "+"
    if token in {"minus", "rev"}:
        return "-"
    raise ValueError(f"unrecognized strand token: {token}")


def list_ribo_wigs(tar_path: Path, dataset: str) -> list[str]:
    with tarfile.open(tar_path, "r") as tar:
        names = [member.name for member in tar.getmembers() if member.isfile()]
    if dataset == "GSE119104":
        return [name for name in names if name.endswith(".wig.gz")]
    if dataset == "GSE77617":
        return [name for name in names if "_rib_" in name and name.endswith(".wig.gz")]
    raise ValueError(dataset)


def parse_track_name(member_name: str) -> dict:
    base = Path(member_name).name
    match = _STRAND_RE.match(base)
    if not match:
        raise ValueError(f"cannot parse WIG name: {base}")
    gsm, label, strand_token = match.groups()
    return {
        "gsm": gsm,
        "label": label,
        "strand": _normalize_strand(strand_token),
        "member": member_name,
    }


def group_conditions(members: list[str]) -> list[dict]:
    grouped: dict[str, dict] = {}
    for name in members:
        parsed = parse_track_name(name)
        key = parsed["gsm"]
        item = grouped.setdefault(
            key,
            {"gsm": parsed["gsm"], "label": parsed["label"], "tracks": {}},
        )
        item["tracks"][parsed["strand"]] = parsed["member"]
    conditions = []
    for gsm, item in grouped.items():
        if "+" not in item["tracks"] or "-" not in item["tracks"]:
            continue
        conditions.append(item)
    conditions.sort(key=lambda row: row["gsm"])
    return conditions


def _fill_fixed_step(lines, genome_len: int) -> np.ndarray:
    values = np.zeros(genome_len, dtype=np.float64)
    pos = 0
    started = False
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("track"):
            continue
        if line.startswith("fixedStep"):
            started = True
            pos = 0
            start_match = re.search(r"start=(\d+)", line)
            if start_match:
                pos = int(start_match.group(1)) - 1
            continue
        if not started:
            continue
        if pos >= genome_len:
            break
        values[pos] = float(line.split()[-1])
        pos += 1
    return values


def _fill_variable_step(lines, genome_len: int) -> np.ndarray:
    values = np.zeros(genome_len, dtype=np.float64)
    for raw in lines:
        line = raw.strip()
        if (
            not line
            or line.startswith("#")
            or line.startswith("track")
            or line.startswith("variableStep")
            or line.startswith("fixedStep")
        ):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        pos = int(float(parts[0])) - 1
        if 0 <= pos < genome_len:
            values[pos] = float(parts[1])
    return values


def load_wig_array(tar_path: Path, member_name: str, genome_len: int) -> np.ndarray:
    with tarfile.open(tar_path, "r") as tar:
        extracted = tar.extractfile(member_name)
        if extracted is None:
            raise FileNotFoundError(member_name)
        payload = extracted.read()
    if member_name.endswith(".gz"):
        text = gzip.decompress(payload).decode("utf-8", errors="replace")
    else:
        text = payload.decode("utf-8", errors="replace")
    lines = text.splitlines()
    body = "\n".join(lines[:30])
    if "variableStep" in body:
        return _fill_variable_step(lines, genome_len)
    return _fill_fixed_step(lines, genome_len)


def load_stranded_tracks(tar_path: Path, condition: dict, genome_len: int) -> dict:
    return {
        "+": load_wig_array(tar_path, condition["tracks"]["+"], genome_len),
        "-": load_wig_array(tar_path, condition["tracks"]["-"], genome_len),
    }


def parse_soft_samples(soft_path: Path) -> dict[str, dict]:
    samples: dict[str, dict] = {}
    current_id = None
    with gzip.open(soft_path, "rt", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            if line.startswith("^SAMPLE = "):
                current_id = line.split("=", 1)[1].strip()
                samples[current_id] = {
                    "title": "",
                    "characteristics": [],
                }
                continue
            if current_id is None:
                continue
            if line.startswith("!Sample_title = "):
                samples[current_id]["title"] = line.split("=", 1)[1].strip()
            elif line.startswith("!Sample_characteristics_ch1 = "):
                samples[current_id]["characteristics"].append(
                    line.split("=", 1)[1].strip()
                )
    return samples
