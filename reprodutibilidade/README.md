# Reproduction order

All commands run from the repository root, after the inputs described in the top-level
[README](../README.md) are in place and the environment of `requirements-analise.txt` is installed.

| Step | Command | Produces | Used in the article |
|---|---|---|---|
| 1 | `python dados/validate_datasets.py` | `dados/validacao_resumo.json` | dataset-selection table and eligibility criteria |
| 2 | `python -m modelo_baseline.run` | `modelo_baseline/saidas/` | sequence-only baseline |
| 3 | `python -m modelo_baseline.robustez_freeze_mg` | `modelo_baseline/saidas_robustez_freeze_mg/` | freeze-only comparator |
| 4 | `python -m modelo_latente.run` | `modelo_latente/saidas/{pooled,freeze}/` | rank-one structure, dimension selection, within-study analysis |
| 5 | `python -m modelo_latente.invariance_check` | `modelo_latente/saidas/invariance_check.json` | invariance with respect to the baseline |
| 6 | `python -m analises_revisao.r1_comparador_aditivo` | `analises_revisao/saidas/r1/` | additive and per-library scale comparators, dimension-test calibration, stability |
| 7 | `python -m analises_revisao.r5_intra_estudo` | `analises_revisao/saidas/r5/` | within-study analysis |
| 8 | `python -m analises_revisao.r2_profundidade_esparsidade` | `analises_revisao/saidas/r2/` | sequencing depth, rarefaction, rate estimators |
| 9 | `python -m analises_revisao.r4_sitio_a` | `analises_revisao/saidas/r4/` | A-site assignment |
| 10 | `python -m analises_revisao.r3_serina` | `analises_revisao/saidas/r3/` | codon-specific signatures |
| 11 | `python -m analises_revisao.r6_remapeamento_adaptadores` | `analises_revisao/saidas/r6/` | comparison of NC_000913.2 and NC_000913.3 (seven sequence events, three overlapping CDS) |
| 12 | `python reprodutibilidade/verificar_numeros_artigo.py` | table of article value against recomputed value | must end with 0 divergences |

Notes:

- Steps 2 and 3 are deterministic. Steps 4 and 5 use seeds derived from 20260911 (`modelo_latente/config.py`).
- Steps 6 to 11 read the outputs of steps 2 to 5 and do not rewrite them. Each of them recomputes the corresponding values of the article first and aborts if they do not match (anchor check). Their shared definitions and seed rule are in `analises_revisao/comum.py`.
- Step 8 uses `analises_revisao/dados/profundidade_ena.csv`, the read count of each sequencing run in the European Nucleotide Archive, queried once and frozen together with the query URL. The script only accesses the network with `--atualizar-ena`.
- Steps 8 and 9 build a track cache in `analises_revisao/_cache/` on the first run, which adds a few minutes.
- Step 11 takes about 18 minutes and needs the two FASTQ files listed in the top-level README.
- Approximate run times: dataset selection and baseline with robustness, about 40 minutes in total; latent stage, about 5 minutes; steps 6, 7 and 10, seconds to about half a minute; step 8, about 100 seconds with cache; step 9, about 380 seconds with cache.
