"""
Indicateurs de performance de la couverture.

Deux questions guident ces mesures :
  1. Combien a-t-on gagné ou perdu ?             -> PnL total et sa décomposition
  2. La couverture a-t-elle réduit le risque ?   -> efficacité de la couverture
"""

import numpy as np
import pandas as pd

from .config import TRADING_DAYS_PER_YEAR
from .hedging import HedgeResult


def realized_volatility(spot: np.ndarray) -> np.ndarray:
    """Volatilité réalisée annualisée de chaque trajectoire (écart-type des log-rendements)."""
    log_returns = np.diff(np.log(spot), axis=1)
    return log_returns.std(axis=1, ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)


def path_summary(res: HedgeResult) -> pd.DataFrame:
    """Une ligne par trajectoire : PnL total, décomposition, risque."""
    daily = res.pnl[:, 1:]             # on ignore le jour 0 (pas de mouvement de marché)
    daily_unhedged = res.pnl_unhedged[:, 1:]

    df = pd.DataFrame(
        {
            "S_final": res.spot[:, -1],
            "vol_realisee": realized_volatility(res.spot),
            "pnl_total": res.pnl.sum(axis=1),
            "pnl_sans_couverture": res.pnl_unhedged.sum(axis=1),
            "pnl_gamma": res.pnl_gamma.sum(axis=1),
            "pnl_theta": res.pnl_theta.sum(axis=1),
            "pnl_delta": res.pnl_delta.sum(axis=1),
            "pnl_interets": res.pnl_interest.sum(axis=1),
            "frais": res.pnl_costs.sum(axis=1),
            "pnl_inexplique": res.pnl_unexplained.sum(axis=1),
            "ecart_type_pnl_jour": daily.std(axis=1, ddof=1),
            "ecart_type_pnl_jour_sans_couv": daily_unhedged.std(axis=1, ddof=1),
            "nb_rebalancements": res.rebalanced.sum(axis=1),
        }
    )
    # Part de la variance quotidienne éliminée par la couverture, chemin par chemin
    df["efficacite_couverture"] = 1.0 - df["ecart_type_pnl_jour"] ** 2 / df["ecart_type_pnl_jour_sans_couv"] ** 2
    df.index.name = "trajectoire"
    return df


def hedge_effectiveness(res: HedgeResult) -> float:
    """
    Efficacité globale = 1 - Var(PnL quotidien couvert) / Var(PnL quotidien non couvert),
    calculée sur tous les jours de toutes les trajectoires.
    1 = risque de prix totalement éliminé, 0 = la couverture ne sert à rien.
    """
    hedged = res.pnl[:, 1:].ravel()
    unhedged = res.pnl_unhedged[:, 1:].ravel()
    return 1.0 - hedged.var() / unhedged.var()


def overall_summary(res: HedgeResult) -> dict:
    """Résumé agrégé sur toutes les trajectoires."""
    totals = res.pnl.sum(axis=1)
    totals_unhedged = res.pnl_unhedged.sum(axis=1)
    return {
        "nb_trajectoires": res.n_paths,
        "pnl_total_moyen": float(totals.mean()),
        "pnl_total_ecart_type": float(totals.std(ddof=1)) if res.n_paths > 1 else float("nan"),
        "pnl_sans_couv_moyen": float(totals_unhedged.mean()),
        "pnl_sans_couv_ecart_type": float(totals_unhedged.std(ddof=1)) if res.n_paths > 1 else float("nan"),
        "pnl_gamma_moyen": float(res.pnl_gamma.sum(axis=1).mean()),
        "pnl_theta_moyen": float(res.pnl_theta.sum(axis=1).mean()),
        "frais_moyens": float(res.pnl_costs.sum(axis=1).mean()),
        "efficacite_couverture": float(hedge_effectiveness(res)),
    }
