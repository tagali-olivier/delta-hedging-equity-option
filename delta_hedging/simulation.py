"""
Génération de trajectoires de prix synthétiques (mouvement brownien géométrique).

    S_t = S_{t-1} · exp[(r - σ²/2)·Δt + σ·√Δt·Z_t],   Z_t ~ N(0, 1)

On simule sous la probabilité risque-neutre (dérive = r), comme dans l'énoncé.
Pour une couverture delta, la dérive a d'ailleurs peu d'importance : c'est la
volatilité réalisée qui fait le PnL.
"""

import numpy as np


def simulate_stock(
    s0: float,
    r: float,
    sigma: float,
    dt: float,
    n_days: int,
    n_paths: int = 1,
    seed: int | None = None,
) -> np.ndarray:
    """
    Simule `n_paths` trajectoires de `n_days` jours.

    Retourne un tableau de forme (n_paths, n_days + 1) : la colonne 0 contient
    S0, la colonne j le prix à la clôture du jour j.
    """
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n_paths, n_days))

    log_returns = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
    # Somme cumulée des log-rendements, en ajoutant 0 au jour 0 (S = S0)
    cumulative = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(log_returns, axis=1)], axis=1)
    return s0 * np.exp(cumulative)
