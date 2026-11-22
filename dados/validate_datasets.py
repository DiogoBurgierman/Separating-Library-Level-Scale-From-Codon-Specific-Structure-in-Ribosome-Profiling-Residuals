import gzip
import json
import re
import tarfile
from pathlib import Path


BASE = Path("dados")
DATASET_LOCATIONS = {
    "GSE52119": BASE / "baixa_confiabilidade" / "GSE52119",
    "GSE75897": BASE / "baixa_confiabilidade" / "GSE75897",
    "GSE119104": BASE / "alta_confiabilidade" / "GSE119104",
    "GSE77617": BASE / "alta_confiabilidade" / "GSE77617",
}


def read_soft(path: Path):
    series = {}
    samples = []
    current = None

    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if line.startswith("^SAMPLE = "):
                if current:
                    samples.append(current)
                current = {
                    "id": line.split("=", 1)[1].strip(),
                    "title": "",
                    "organism": "",
                    "characteristics": [],
                }
                continue

            if line.startswith("!Series_title = "):
                series["title"] = line.split("=", 1)[1].strip()
            elif line.startswith("!Series_overall_design = "):
                series["overall_design"] = line.split("=", 1)[1].strip()
            elif line.startswith("!Series_relation = "):
                series.setdefault("relations", []).append(line.split("=", 1)[1].strip())
            elif current is not None:
                if line.startswith("!Sample_title = "):
                    current["title"] = line.split("=", 1)[1].strip()
                elif line.startswith("!Sample_organism_ch1 = "):
                    current["organism"] = line.split("=", 1)[1].strip()
                elif line.startswith("!Sample_characteristics_ch1 = "):
                    current["characteristics"].append(line.split("=", 1)[1].strip())

    if current:
        samples.append(current)
    return series, samples


def normalize_chromosome_name(raw_name: str) -> str:
    name = raw_name.strip().strip('"')
    nc_match = re.search(r"(NC_\d+\.\d+)", name)
    if nc_match:
        return nc_match.group(1)
    return name.replace("_", ".")


def build_reference_aliases(fasta_path: Path, gff_path: Path, kind: str):
    aliases = set()

    with gzip.open(fasta_path, "rt", encoding="utf-8", errors="replace") as fasta:
        for line in fasta:
            if line.startswith(">"):
                accession = line[1:].split()[0]
                aliases.add(normalize_chromosome_name(accession))

    with gzip.open(gff_path, "rt", encoding="utf-8", errors="replace") as gff:
        for line in gff:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9 or fields[2] != "region":
                continue

            seqid = fields[0]
            attrs = fields[8]
            aliases.add(normalize_chromosome_name(seqid))

            for key in ("chromosome", "Name"):
                match = re.search(rf"(?:^|;){key}=([^;]+)", attrs)
                if match:
                    value = match.group(1)
                    aliases.add(value)
                    aliases.add(f"chr{value}")

            if "mitochond" in attrs.lower():
                aliases.update({"chrM", "chrMT", "MT", "M"})

    if kind == "ecoli":
        aliases.update({"MG1655", "NC_000913.2", "NC_000913_2", "NC.000913.2"})

    return aliases


def list_ribo_members(gse_name: str, all_members):
    if gse_name == "GSE52119":
        return [name for name in all_members if "_RPF." in name]
    if gse_name == "GSE119104":
        return [name for name in all_members if name.endswith(".wig.gz")]
    if gse_name == "GSE77617":
        return [name for name in all_members if "_rib_" in name]
    if gse_name == "GSE75897":
        return [
            name
            for name in all_members
            if ("rpf" in name.lower() or "ribosome" in name.lower()) and "rpkm" not in name.lower()
        ]
    return []


def extract_chromosomes_from_members(tar_path: Path, members):
    chromosomes = set()
    with tarfile.open(tar_path, "r") as tar:
        for member_name in members:
            extracted = tar.extractfile(member_name)
            if extracted is None:
                continue
            data = extracted.read()
            lines = (
                gzip.decompress(data).decode("utf-8", errors="replace").splitlines()
                if member_name.endswith(".gz")
                else data.decode("utf-8", errors="replace").splitlines()
            )

            for line in lines:
                if not line or line.startswith("#") or line.startswith("track"):
                    continue
                if line.startswith("fixedStep") or line.startswith("variableStep"):
                    match = re.search(r"chrom=([^\s]+)", line)
                    if match:
                        chromosomes.add(match.group(1))
                else:
                    columns = line.split("\t")
                    if len(columns) >= 3:
                        chromosomes.add(columns[0])
    return {normalize_chromosome_name(chrom) for chrom in chromosomes}


def main():
    references = {
        "scer": build_reference_aliases(
            BASE / "referencias" / "S_cerevisiae_R64" / "genomic.fna.gz",
            BASE / "referencias" / "S_cerevisiae_R64" / "genomic.gff.gz",
            "scer",
        ),
        "spar": build_reference_aliases(
            BASE / "referencias" / "S_paradoxus_ASM16683v2" / "genomic.fna.gz",
            BASE / "referencias" / "S_paradoxus_ASM16683v2" / "genomic.gff.gz",
            "spar",
        ),
        "ecoli": build_reference_aliases(
            BASE / "referencias" / "Ecoli_MG1655_ASM584v2" / "genomic.fna.gz",
            BASE / "referencias" / "Ecoli_MG1655_ASM584v2" / "genomic.gff.gz",
            "ecoli",
        ),
    }

    report = {}
    for gse_name, gse_dir in DATASET_LOCATIONS.items():
        soft_path = gse_dir / f"{gse_name}_family.soft.gz"
        tar_path = gse_dir / f"{gse_name}_supplementary.tar"
        series, samples = read_soft(soft_path)

        with tarfile.open(tar_path, "r") as tar:
            all_members = [member.name for member in tar.getmembers() if member.isfile()]

        ribo_members = list_ribo_members(gse_name, all_members)
        chromosomes = extract_chromosomes_from_members(tar_path, ribo_members) if ribo_members else set()

        if gse_name in {"GSE119104", "GSE77617"}:
            matched = sorted(chromosomes & references["ecoli"])
            unmatched = sorted(chromosomes - references["ecoli"])
            contig_check = {
                "matched": len(matched),
                "total": len(chromosomes),
                "matched_values": matched,
                "unmatched_values": unmatched,
            }
        elif gse_name == "GSE52119":
            matched_scer = sorted(chromosomes & references["scer"])
            matched_spar = sorted(chromosomes & references["spar"])
            unmatched_scer = sorted(chromosomes - references["scer"])
            unmatched_spar = sorted(chromosomes - references["spar"])
            contig_check = {
                "matched_scer": len(matched_scer),
                "matched_spar": len(matched_spar),
                "total": len(chromosomes),
                "matched_scer_values": matched_scer,
                "matched_spar_values": matched_spar,
                "unmatched_scer_values": unmatched_scer,
                "unmatched_spar_values": unmatched_spar,
            }
        else:
            contig_check = {"matched": 0, "total": len(chromosomes), "matched_values": []}

        report[gse_name] = {
            "title": series.get("title", ""),
            "overall_design": series.get("overall_design", ""),
            "relations": series.get("relations", []),
            "organisms": sorted({sample["organism"] for sample in samples if sample["organism"]}),
            "all_members_count": len(all_members),
            "ribo_members_count": len(ribo_members),
            "chromosomes": sorted(chromosomes),
            "contig_check": contig_check,
        }

    output_path = BASE / "validacao_resumo.json"
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
