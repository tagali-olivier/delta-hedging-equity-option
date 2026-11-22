"""
Prix et sensibilités (« Grecs ») d'un call européen dans le modèle de Black-Scholes.

Toutes les fonctions acceptent des scalaires OU des tableaux numpy : on peut
donc pricer en un seul appel l'option sur des centaines de trajectoires à la fois.

Formules (sans dividende) :
    d1 = [ln(S/K) + (r + σ²/2)·T] / (σ·√T)
    d2 = d1 - σ·√T
    C  = S·N(d1) - K·e^(-rT)·N(d2)
    Δ  = N(d1)
    Γ  = n(d1) / (S·σ·√T)
    ν  = S·√T·n(d1)                       (pour +1,00 de volatilité, soit +100 %)
    Θ  = -S·n(d1)·σ / (2√T) - r·K·e^(-rT)·N(d2)     (par an)
où N est la fonction de répartition de la loi normale et n sa densité.
"""

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

# En dessous de cette maturité résiduelle (≈ 1 seconde), on considère l'option échue.
_T_EPS = 1e-10


@dataclass
class Greeks:
    """Résultat de `bs_greeks` : prix + sensibilités, pour UNE option (1 action)."""

    price: np.ndarray
    delta: np.ndarray
    gamma: np.ndarray
    vega: np.ndarray
    theta: np.ndarray


def bs_greeks(S, K, r, T, sigma, s_floor: float = 0.01) -> Greeks:
    """
    Calcule le prix et les Grecs d'un call européen.

    Cas limites gérés :
      * S -> 0 : le prix de l'action ne descend pas sous le plancher `s_floor`
        (évite log(0) = -inf) ;
      * T -> 0 : à l'échéance, l'option vaut sa valeur intrinsèque max(S-K, 0),
        son delta vaut 1 si S > K et 0 sinon, gamma, vega et theta sont nuls.
    """
    S = np.maximum(np.asarray(S, dtype=float), s_floor)
    T = np.asarray(T, dtype=float)
    S, T = np.broadcast_arrays(S, T)

    expired = T <= _T_EPS
    # On remplace T par une petite valeur positive là où l'option est échue,
    # uniquement pour éviter les divisions par zéro. Ces valeurs sont ensuite
    # écrasées par la valeur intrinsèque.
    T_safe = np.where(expired, 1.0, T)
    sqrt_T = np.sqrt(T_safe)

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T_safe) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    discount = np.exp(-r * T_safe)
    pdf_d1 = norm.pdf(d1)

    price = S * norm.cdf(d1) - K * discount * norm.cdf(d2)
    delta = norm.cdf(d1)
    gamma = pdf_d1 / (S * sigma * sqrt_T)
    vega = S * sqrt_T * pdf_d1
    theta = -S * pdf_d1 * sigma / (2 * sqrt_T) - r * K * discount * norm.cdf(d2)

    # Option échue : valeur intrinsèque
    price = np.where(expired, np.maximum(S - K, 0.0), price)
    delta = np.where(expired, (S > K).astype(float), delta)
    gamma = np.where(expired, 0.0, gamma)
    vega = np.where(expired, 0.0, vega)
    theta = np.where(expired, 0.0, theta)

    return Greeks(price=price, delta=delta, gamma=gamma, vega=vega, theta=theta)


def bs_put_price(S, K, r, T, sigma) -> np.ndarray:
    """Prix d'un put européen, via la parité call-put (utilisé dans les tests)."""
    call = bs_greeks(S, K, r, T, sigma).price
    return call - np.asarray(S) + K * np.exp(-r * np.asarray(T))
