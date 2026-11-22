from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

GENOME_ACCESSION = "NC_000913.2"
GENOME_FASTA = ROOT / "dados" / "remapeamento_bruto" / "referencias_candidatas" / "NC_000913.2.fasta"
CDS_FASTA = (
    ROOT
    / "dados"
    / "remapeamento_bruto"
    / "referencias_candidatas"
    / "NC_000913.2_cds_from_genomic.fna"
)
CDS_FASTA_URL = (
    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    "?db=nuccore&id=NC_000913.2&rettype=fasta_cds_na&retmode=text"
)

GSE119104_TAR = (
    ROOT / "dados" / "alta_confiabilidade" / "GSE119104" / "GSE119104_supplementary.tar"
)
GSE119104_SOFT = (
    ROOT / "dados" / "alta_confiabilidade" / "GSE119104" / "GSE119104_family.soft.gz"
)
GSE77617_TAR = (
    ROOT / "dados" / "alta_confiabilidade" / "GSE77617" / "GSE77617_supplementary.tar"
)
GSE77617_SOFT = (
    ROOT / "dados" / "alta_confiabilidade" / "GSE77617" / "GSE77617_family.soft.gz"
)

OUTPUT_DIR = ROOT / "modelo_baseline" / "saidas"

# Mohammad et al., eLife 2019: A-site pause scores use a -11 nt shift from 3'-aligned density.
ASITE_SHIFT_NT = {
    "GSE119104": 11,
    "GSE77617": 0,
}

# Mohammad et al., eLife 2019: drop the first 27 nt after start and 12 nt before stop.
EXCLUDE_START_NT = 27
EXCLUDE_STOP_NT = 12

# Mohammad et al. used 0.5 raw reads/nt. GEO WIG values are already scaled, so that
# cutoff is not applied in native track units. Genes with non-positive mean occupancy
# are dropped because relative occupancy is otherwise undefined.
MIN_GENE_MEAN_OCCUPANCY = 0.0

STOP_CODONS = frozenset({"TAA", "TAG", "TGA"})

STANDARD_GENETIC_CODE = {
    "TTT": "F",
    "TTC": "F",
    "TTA": "L",
    "TTG": "L",
    "TCT": "S",
    "TCC": "S",
    "TCA": "S",
    "TCG": "S",
    "TAT": "Y",
    "TAC": "Y",
    "TAA": "*",
    "TAG": "*",
    "TGT": "C",
    "TGC": "C",
    "TGA": "*",
    "TGG": "W",
    "CTT": "L",
    "CTC": "L",
    "CTA": "L",
    "CTG": "L",
    "CCT": "P",
    "CCC": "P",
    "CCA": "P",
    "CCG": "P",
    "CAT": "H",
    "CAC": "H",
    "CAA": "Q",
    "CAG": "Q",
    "CGT": "R",
    "CGC": "R",
    "CGA": "R",
    "CGG": "R",
    "ATT": "I",
    "ATC": "I",
    "ATA": "I",
    "ATG": "M",
    "ACT": "T",
    "ACC": "T",
    "ACA": "T",
    "ACG": "T",
    "AAT": "N",
    "AAC": "N",
    "AAA": "K",
    "AAG": "K",
    "AGT": "S",
    "AGC": "S",
    "AGA": "R",
    "AGG": "R",
    "GTT": "V",
    "GTC": "V",
    "GTA": "V",
    "GTG": "V",
    "GCT": "A",
    "GCC": "A",
    "GCA": "A",
    "GCG": "A",
    "GAT": "D",
    "GAC": "D",
    "GAA": "E",
    "GAG": "E",
    "GGT": "G",
    "GGC": "G",
    "GGA": "G",
    "GGG": "G",
}
