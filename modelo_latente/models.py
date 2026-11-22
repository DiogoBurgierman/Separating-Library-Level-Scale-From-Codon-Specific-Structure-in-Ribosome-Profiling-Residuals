"""Estimators for the latent condition vector Z_c.

All estimators share the decomposition

    R_ic = m_i + l_i^T Z_c + xi_ic ,

where m_i is a codon-level offset and l_i^T Z_c is the latent term g(Z_c).

Why m_i is separated and never credited to Z_c
----------------------------------------------
k_seq is fitted by OLS weighted by position-level n, while the codon x condition
table averages cells unweighted, so R has a non-zero per-codon mean. That offset
is a function of the codon index i alone, i.e. a sequence-level quantity, not a
condition-level one. Centring is therefore a correction to k_seq, and crediting
it to Z_c would inflate the latent term with variance no latent variable
explained. Every reported gain is measured against the m_i-only model.

Orientation convention: R is (codons x conditions) = (D x N); X = R.T is
(samples x features) with conditions as samples.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class LatentFit:
    """A fitted latent model, in the article's notation."""

    name: str
    d: int
    offset: np.ndarray  # m_i, shape (D,)
    loadings: np.ndarray  # l_i, shape (D, d)
    scores: np.ndarray  # Z_c, shape (N, d)
    extra: dict = field(default_factory=dict)

    def reconstruct(self) -> np.ndarray:
        """g(Z_c) + m_i as an (N x D) array."""
        return self.offset + self.scores @ self.loadings.T

    def latent_term(self) -> np.ndarray:
        """g(Z_c) alone as an (N x D) array."""
        return self.scores @ self.loadings.T


def centre(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (X centred over conditions, codon offset m_i)."""
    x = matrix.T
    offset = x.mean(axis=0)
    return x - offset, offset


# ---------------------------------------------------------------------------
# Linear dimensionality reduction (PCA)
# ---------------------------------------------------------------------------


def fit_pca(matrix: np.ndarray, d: int) -> LatentFit:
    xc, offset = centre(matrix)
    u, s, vt = np.linalg.svd(xc, full_matrices=False)
    loadings = vt[:d].T
    scores = u[:, :d] * s[:d]
    return LatentFit(
        name="pca",
        d=d,
        offset=offset,
        loadings=loadings,
        scores=scores,
        extra={
            "singular_values": s.tolist(),
            "explained_variance_ratio": (s**2 / np.sum(s**2)).tolist(),
        },
    )


def singular_spectrum(matrix: np.ndarray) -> np.ndarray:
    xc, _ = centre(matrix)
    return np.linalg.svd(xc, compute_uv=False)


# ---------------------------------------------------------------------------
# Probabilistic latent model
# ---------------------------------------------------------------------------


def fit_ppca_em(
    matrix: np.ndarray,
    d: int,
    max_iter: int,
    tol: float,
    seed: int,
) -> LatentFit:
    """Probabilistic PCA (Tipping & Bishop) fitted by EM.

    Generative model: x_c = W z_c + m + eps, z_c ~ N(0, I), eps ~ N(0, sigma^2 I).
    This is the maximum-likelihood reference against which the variational
    Bayesian fit below is compared.
    """
    xc, offset = centre(matrix)
    n, dim = xc.shape
    rng = np.random.default_rng(seed)
    w = rng.normal(scale=0.1, size=(dim, d))
    sigma2 = float(np.var(xc)) or 1.0

    prev = None
    iters = 0
    for iters in range(1, max_iter + 1):
        m_mat = w.T @ w + sigma2 * np.eye(d)
        m_inv = np.linalg.inv(m_mat)
        ez = xc @ w @ m_inv.T  # (n, d)
        ezz = n * sigma2 * m_inv + ez.T @ ez  # (d, d)

        w_new = (xc.T @ ez) @ np.linalg.inv(ezz)
        trace_term = float(np.trace(ezz @ (w_new.T @ w_new)))
        cross = float(np.sum((xc @ w_new) * ez))
        sigma2_new = (float(np.sum(xc**2)) - 2.0 * cross + trace_term) / (n * dim)
        sigma2_new = max(sigma2_new, 1e-12)

        delta = abs(sigma2_new - sigma2) + float(np.max(np.abs(w_new - w)))
        w, sigma2 = w_new, sigma2_new
        if prev is not None and delta < tol:
            break
        prev = delta

    cov = w @ w.T + sigma2 * np.eye(dim)
    sign, logdet = np.linalg.slogdet(cov)
    sample_cov = (xc.T @ xc) / n
    loglik = -0.5 * n * (
        dim * np.log(2.0 * np.pi)
        + logdet
        + float(np.trace(np.linalg.solve(cov, sample_cov)))
    )
    n_params = dim * d - d * (d - 1) / 2 + 1 + dim
    bic = -2.0 * loglik + n_params * np.log(n)

    m_mat = w.T @ w + sigma2 * np.eye(d)
    scores = xc @ w @ np.linalg.inv(m_mat).T

    return LatentFit(
        name="ppca_em",
        d=d,
        offset=offset,
        loadings=w,
        scores=scores,
        extra={
            "sigma2": sigma2,
            "loglik": loglik,
            "bic": bic,
            "n_free_params": n_params,
            "em_iterations": iters,
            "logdet_sign": float(sign),
        },
    )


def fit_vbpca_ard(
    matrix: np.ndarray,
    d: int,
    max_iter: int,
    tol: float,
    seed: int,
) -> LatentFit:
    """Variational Bayesian PCA with an ARD prior over latent directions.

    q(W), q(Z) are Gaussian; the ARD precisions alpha_k and the noise precision
    tau are updated by type-II maximum likelihood. Components whose alpha_k grows
    without bound are switched off, which is how this estimator reports the
    dimensionality it actually supports rather than the d it was given.
    """
    xc, offset = centre(matrix)
    n, dim = xc.shape
    rng = np.random.default_rng(seed)

    mw = rng.normal(scale=0.1, size=(dim, d))
    sw = np.eye(d)
    tau = 1.0 / (float(np.var(xc)) or 1.0)
    alpha = np.ones(d)

    noise_var = 1.0 / tau
    iters = 0
    for iters in range(1, max_iter + 1):
        # q(Z)
        eww = mw.T @ mw + dim * sw
        sz = np.linalg.inv(np.eye(d) + tau * eww)
        mz = tau * (xc @ mw) @ sz.T  # (n, d)

        # q(W)
        ezz = n * sz + mz.T @ mz
        sw = np.linalg.inv(np.diag(alpha) + tau * ezz)
        mw = tau * (xc.T @ mz) @ sw.T  # (dim, d)

        # ARD precisions
        alpha = dim / (np.sum(mw**2, axis=0) + dim * np.diag(sw))
        alpha = np.clip(alpha, 1e-12, 1e12)

        # noise precision
        eww = mw.T @ mw + dim * sw
        ezz = n * sz + mz.T @ mz
        recon = float(np.sum(xc**2)) - 2.0 * float(np.sum((xc @ mw) * mz))
        recon += float(np.trace(eww @ ezz))
        noise_var_new = max(recon / (n * dim), 1e-12)
        tau = 1.0 / noise_var_new

        if abs(noise_var_new - noise_var) < tol:
            noise_var = noise_var_new
            break
        noise_var = noise_var_new

    column_power = np.sum(mw**2, axis=0)
    order = np.argsort(column_power)[::-1]
    mw = mw[:, order]
    mz = mz[:, order]
    alpha = alpha[order]
    column_power = column_power[order]

    return LatentFit(
        name="vbpca_ard",
        d=d,
        offset=offset,
        loadings=mw,
        scores=mz,
        extra={
            "alpha": alpha.tolist(),
            "alpha_min": float(alpha.min()),
            "column_power": column_power.tolist(),
            "noise_variance": noise_var,
            "vb_iterations": iters,
        },
    )


def count_active_ard(alpha: np.ndarray, ratio: float) -> int:
    alpha = np.asarray(alpha, dtype=float)
    return int(np.sum(alpha < ratio * alpha.min()))


# ---------------------------------------------------------------------------
# Non-linear, explicitly exploratory
# ---------------------------------------------------------------------------


def fit_vae(
    matrix: np.ndarray,
    d: int,
    hidden: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    restarts: int,
    seed: int,
) -> LatentFit | None:
    """Variational autoencoder on the same residual matrix.

    This branch is exploratory. With N = 14 conditions the encoder sees
    fourteen training vectors, so the fit is reported as a descriptive comparison
    against PCA at equal d and is not used for any inferential claim.
    """
    try:
        import torch
        from torch import nn
    except ImportError:  # pragma: no cover - torch is optional
        return None

    xc, offset = centre(matrix)
    n, dim = xc.shape
    scale = float(np.std(xc)) or 1.0
    x = torch.tensor(xc / scale, dtype=torch.float32)

    best = None
    for restart in range(restarts):
        torch.manual_seed(seed + restart)

        encoder = nn.Sequential(nn.Linear(dim, hidden), nn.Tanh())
        to_mu = nn.Linear(hidden, d)
        to_logvar = nn.Linear(hidden, d)
        decoder = nn.Sequential(nn.Linear(d, hidden), nn.Tanh(), nn.Linear(hidden, dim))
        params = (
            list(encoder.parameters())
            + list(to_mu.parameters())
            + list(to_logvar.parameters())
            + list(decoder.parameters())
        )
        opt = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

        for _ in range(epochs):
            opt.zero_grad()
            h = encoder(x)
            mu = to_mu(h)
            logvar = torch.clamp(to_logvar(h), min=-10.0, max=10.0)
            std = torch.exp(0.5 * logvar)
            z = mu + std * torch.randn_like(std)
            recon = decoder(z)
            recon_loss = ((recon - x) ** 2).sum()
            kl = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp())
            loss = recon_loss + kl
            loss.backward()
            opt.step()

        with torch.no_grad():
            h = encoder(x)
            mu = to_mu(h)
            recon = decoder(mu)
            final_recon = float(((recon - x) ** 2).sum())
            final_kl = float(-0.5 * torch.sum(1 + torch.clamp(to_logvar(h), -10, 10)
                                              - mu**2
                                              - torch.clamp(to_logvar(h), -10, 10).exp()))
            elbo = -(final_recon + final_kl)
            scores = mu.numpy()
            fitted = recon.numpy() * scale

        if best is None or elbo > best["elbo"]:
            best = {
                "elbo": elbo,
                "recon": final_recon,
                "kl": final_kl,
                "scores": scores,
                "fitted": fitted,
                "restart": restart,
            }

    return LatentFit(
        name="vae",
        d=d,
        offset=offset,
        loadings=np.zeros((dim, d)),
        scores=best["scores"],
        extra={
            "elbo": best["elbo"],
            "reconstruction_sse_scaled": best["recon"],
            "kl": best["kl"],
            "best_restart": best["restart"],
            "input_scale": scale,
            # The decoder is non-linear, so the fitted values cannot be written as
            # l_i^T Z_c; they are carried explicitly for the variance accounting.
            "fitted_centred": best["fitted"],
        },
    )
