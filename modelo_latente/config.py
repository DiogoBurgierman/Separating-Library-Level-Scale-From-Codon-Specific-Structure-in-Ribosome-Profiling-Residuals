"""Configuration for the latent condition-state stage.

This stage consumes the residual matrix R_ic produced by `modelo_baseline` and
fits the latent condition representation Z_c, defined as

    R_ic = l_i^T Z_c + xi_ic .

Nothing here re-derives k_obs or k_seq: the sequence baseline is frozen input.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

BASELINE_DIR = ROOT / "modelo_baseline" / "saidas"
FREEZE_DIR = ROOT / "modelo_baseline" / "saidas_robustez_freeze_mg"

# Two legitimate sequence baselines exist as comparators for Z_c. Both are run;
# "pooled" is primary because it is the baseline fit reported in the article.
COMPARATORS = {
    "pooled": {
        "residuals": BASELINE_DIR / "residuals_codon_condition.csv",
        "conditions": BASELINE_DIR / "conditions.csv",
        "n_obs": BASELINE_DIR / "n_obs_codon_condition.csv",
        "fit_summary": BASELINE_DIR / "fit_summary.json",
        "description": "k_seq fitted on all 14 libraries (pooled baseline)",
    },
    "freeze": {
        "residuals": FREEZE_DIR / "residuals_using_freeze_kseq.csv",
        "conditions": FREEZE_DIR / "conditions.csv",
        "n_obs": None,
        "fit_summary": FREEZE_DIR / "fit_summary.json",
        "description": "k_seq fitted only on the two untreated freeze+Mg libraries",
    },
}

PRIMARY_COMPARATOR = "pooled"

OUTPUT_ROOT = ROOT / "modelo_latente" / "saidas"

# 14 conditions minus one degree of freedom for the codon-level offset m_i leaves
# at most 13 identifiable components.
MAX_COMPONENTS = 13

# Latent dimensionalities scanned for every estimator.
D_GRID = tuple(range(1, 9))

RANDOM_SEED = 20260911

N_PERMUTATIONS = 2000
N_BOOTSTRAP = 2000
N_SPLIT_HALF = 500

# Variational Bayesian PCA stopping rules.
VB_MAX_ITER = 5000
VB_TOL = 1e-9
# A component is counted as active when its ARD precision stays below this
# multiple of the smallest precision in the model.
VB_ARD_ACTIVE_RATIO = 100.0

# Probabilistic PCA by EM (maximum-likelihood reference for the VB fit).
EM_MAX_ITER = 5000
EM_TOL = 1e-11

# Exploratory VAE. With 14 conditions this is deliberately tiny,
# and it is scanned over a short d grid only: the branch is descriptive, so a wide
# sweep would suggest an inferential weight it cannot carry.
VAE_D_GRID = (1, 2, 3)
VAE_HIDDEN = 16
VAE_EPOCHS = 2000
VAE_LR = 1e-3
VAE_WEIGHT_DECAY = 1e-3
VAE_RESTARTS = 3
