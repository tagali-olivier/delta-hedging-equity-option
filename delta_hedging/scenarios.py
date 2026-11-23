"""
Fonctions « clé en main » pour lancer une simulation complète et comparer des scénarios.
"""

from dataclasses import replace

import pandas as pd

from .analytics import overall_summary
from .config import HedgeParams, MarketParams, OptionSpec, SimulationParams
from .hedging import HedgeResult, run_delta_hedge
from .simulation import simulate_stock


def run_scenario(
    market: MarketParams, 
    option: OptionSpec,
    sim: SimulationParams,
    hedge: HedgeParams,
) -> HedgeResult:
    """Simule les prix (avec la vol réalisée) puis rejoue la couverture (avec la vol implicite)."""
    paths = simulate_stock(
        s0=market.s0,
        r=market.r,
        sigma=market.sigma_realized,
        dt=sim.dt,
        n_days=sim.n_days,
        n_paths=sim.n_paths,
        seed=sim.seed,
    )
    return run_delta_hedge(paths, market, option, hedge, sim.dt)


def compare_volatility_levels(
    sigmas, market: MarketParams, option: OptionSpec, sim: SimulationParams, hedge: HedgeParams
) -> pd.DataFrame:
    """
    Test de robustesse : σ = 10 %, 20 %, 40 %.
    Ici vol implicite = vol réalisée : le marché se comporte comme le modèle le prévoit.
    """
    rows = {}
    for sigma in sigmas:
        m = replace(market, sigma_implied=sigma, sigma_realized=sigma)
        rows[f"σ = {sigma:.0%}"] = overall_summary(run_scenario(m, option, sim, hedge))
    return pd.DataFrame(rows).T


def compare_realized_vs_implied(
    realized_sigmas, market: MarketParams, option: OptionSpec, sim: SimulationParams, hedge: HedgeParams
) -> pd.DataFrame:
    """
    Le cas réaliste : on a vendu l'option à 20 % de vol implicite,
    mais le marché bouge à 10 %, 20 % ou 40 %. C'est là que le coût gamma apparaît.
    """
    rows = {}
    for sigma in realized_sigmas:
        m = replace(market, sigma_realized=sigma)
        rows[f"réalisée {sigma:.0%}"] = overall_summary(run_scenario(m, option, sim, hedge))
    return pd.DataFrame(rows).T


def compare_rebalancing(
    frequencies: dict, costs, market: MarketParams, option: OptionSpec, sim: SimulationParams
) -> pd.DataFrame:
    """
    Fréquence de rebalancement × frais de transaction.
    `frequencies` associe un libellé à un nombre de jours, ex. {"quotidien": 1, "hebdo": 5}.
    """
    rows = []
    for label, every in frequencies.items():
        for cost in costs:
            h = HedgeParams(rebalance_every=every, transaction_cost=cost)
            summary = overall_summary(run_scenario(market, option, sim, h))
            rows.append({"rebalancement": label, "frais": f"{cost:.1%}", **summary})
    return pd.DataFrame(rows).set_index(["rebalancement", "frais"])
