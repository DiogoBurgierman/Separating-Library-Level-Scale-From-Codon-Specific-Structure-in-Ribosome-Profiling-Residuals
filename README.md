# Separating Library-Level Scale From Codon-Specific Structure in Ribosome-Profiling Residuals of a Sequence-Only Baseline

Code and stored outputs for the article of the same title. The article audits what a
condition-level latent factor represents when it is fitted to the discrepancy between a
sequence-only codon baseline and an occupancy-derived rate proxy, using fourteen
*Escherichia coli* ribosome-profiling libraries from two public studies (GSE119104 and GSE77617).

## Contents

| Path | Purpose |
|---|---|
| `dados/validate_datasets.py` | dataset selection and contig-consistency workflow; writes `dados/validacao_resumo.json` |
| `modelo_baseline/` | sequence-only codon baseline (`run.py`) and the freeze-only comparator (`robustez_freeze_mg.py`) |
| `modelo_latente/` | latent-stage code (`run.py`) and the baseline-invariance check (`invariance_check.py`) |
| `analises_revisao/` | one script per supplementary analysis: additive and scale comparators (`r1`), sequencing depth and rate estimator (`r2`), codon-specific signatures (`r3`), A-site assignment (`r4`), within-study analysis (`r5`), comparison of the two reference sequences (`r6`); `dados/profundidade_ena.csv` holds the frozen ENA read counts |
| `reprodutibilidade/` | execution order (`README.md`) and a script that recomputes every number reported in the article from the stored outputs (`verificar_numeros_artigo.py`) |
| `requirements-analise.txt` | pinned analysis environment |
| `*/saidas*/` | machine-readable output tables of every stage |

Directory and file names are in Portuguese because the article cites them by path.

## Environment

Python 3.12 and `pip install -r requirements-analise.txt` (numpy 2.4.4, pandas 3.0.3; torch only
for the exploratory variational autoencoder branch of the latent stage, which is not part of the
article's results).

## Inputs

All inputs are public. Place them under `dados/` as follows; the scripts read these paths.

| Path | Content | Source |
|---|---|---|
| `dados/alta_confiabilidade/GSE119104/`, `dados/alta_confiabilidade/GSE77617/` | `<GSE>_supplementary.tar`, `<GSE>_family.soft.gz`, `<GSE>_series_matrix.txt.gz` | GEO: `https://www.ncbi.nlm.nih.gov/geo/download/?acc=<GSE>&format=file` and the GEO FTP (`https://ftp.ncbi.nlm.nih.gov/geo/series/<prefix>/<GSE>/soft/` and `.../matrix/`) |
| `dados/baixa_confiabilidade/GSE52119/`, `dados/baixa_confiabilidade/GSE75897/` | the same three files per series (dataset-selection step only) | GEO, as above |
| `dados/referencias/` | `genomic.fna.gz`, `genomic.gff.gz`, `cds_from_genomic.fna.gz` in `Ecoli_MG1655_ASM584v2/` (GCF_000005845.2), `S_cerevisiae_R64/` (GCF_000146045.2), `S_paradoxus_ASM16683v2/` (GCF_002079055.1) | NCBI RefSeq |
| `dados/remapeamento_bruto/referencias_candidatas/` | `NC_000913.2.fasta`, `NC_000913.3.fasta`, and `NC_000913.2_cds_from_genomic.fna` (the last one is downloaded automatically by `modelo_baseline/annotation.py` if absent) | NCBI E-utilities (`efetch`, `db=nuccore`, `rettype=fasta` / `fasta_cds_na`) |

The read-level analysis `r6` additionally reads the FASTQ files of ENA runs SRR7759813 (GSE119104)
and SRR5186138 (GSE77617) in `dados/remapeamento_bruto/fastq/` (`<run>.fastq.gz`, from the ENA
addresses listed in the manifest below). Two small frozen tables are included in this repository
under `dados/remapeamento_bruto/resultados/`:

- `runs_selecionados.csv`: the two selected runs with their ENA metadata and the expected MD5 of
  each FASTQ file (`r6` aborts if a downloaded file does not match);
- `resumo_fullrun_remapeamento.csv`: read counts of an earlier exact-match remapping, which `r6`
  reproduces and compares against.

Both contain only public ENA accession metadata and counts computed by the authors.

## Reproduction

The execution order, expected outputs, and the final number check are in
[`reprodutibilidade/README.md`](reprodutibilidade/README.md).
