"""
Point d'entrée du projet : lance toutes les analyses et écrit les résultats.

    python main.py                 # analyses complètes (≈ 1 seconde)
    python main.py --mc-paths 5000 --seed 7

Sorties :
    results/  -> tableaux CSV (journal quotidien, résumés, scénarios)
    figures/  -> graphiques PNG utilisés dans le README
"""

import argparse
import time
from dataclasses import replace
from pathlib import Path

import pandas as pd

from delta_hedging import plots
from delta_hedging.analytics import overall_summary, path_summary
from delta_hedging.config import HedgeParams, MarketParams, OptionSpec, SimulationParams
from delta_hedging.scenarios import (
    compare_realized_vs_implied,
    compare_rebalancing,
    compare_volatility_levels,
    run_scenario,
)

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)
pd.set_option("display.float_format", lambda v: f"{v:,.2f}")


def pct(df: pd.DataFrame) -> pd.DataFrame:
    """Affiche l'efficacité de couverture en pourcentage (0.9991 -> 99.91 %)."""
    out = df.copy()
    if "efficacite_couverture" in out:
        out["efficacite_couverture"] = out["efficacite_couverture"].map(lambda v: f"{v:.2%}")
    return out


def title(text: str) -> None:
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}")


def main(mc_paths: int, seed: int) -> None:
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)

    market = MarketParams()
    option = OptionSpec()
    hedge = HedgeParams()
    base_sim = SimulationParams(n_paths=5, seed=seed)
    mc_sim = replace(base_sim, n_paths=mc_paths)

    # ------------------------------------------------------------------
    # 1. Cas de base : 5 trajectoires, rebalancement quotidien, sans frais
    # ------------------------------------------------------------------
    title("1. CAS DE BASE — 5 trajectoires, couverture quotidienne, sans frais")
    start = time.perf_counter()
    base = run_scenario(market, option, base_sim, hedge)
    elapsed = time.perf_counter() - start
    print(f"Simulation + couverture de 5 × 21 jours : {elapsed * 1000:.1f} ms")
    print(f"Prime encaissée au jour 0 : {base.option_price[0, 0]:.4f} € par option, "
          f"soit {base.option_price[0, 0] * option.notional_shares:,.0f} € au total")
    print(f"Delta initial : {base.delta[0, 0]:.4f} -> achat de {base.hedge_shares[0, 0]:,.0f} actions")

    for i in range(base.n_paths):
        base.to_dataframe(i).to_csv(RESULTS / f"journal_trajectoire_{i}.csv", float_format="%.6f")

    journal = base.to_dataframe(0)
    print("\nJournal de la trajectoire 0 (extrait) :")
    print(journal[["S", "prix_call", "delta", "gamma", "vega", "actions_detenues",
                   "pnl", "pnl_cumule", "pnl_sans_couverture"]].round(4))

    summary = path_summary(base)
    summary.to_csv(RESULTS / "resume_trajectoires.csv", float_format="%.4f")
    print("\nRésumé par trajectoire :")
    print(pct(summary)[["S_final", "vol_realisee", "pnl_total", "pnl_sans_couverture",
                   "pnl_gamma", "pnl_theta", "efficacite_couverture"]])

    plots.plot_paths(base, FIGURES / "01_trajectoires.png")
    plots.plot_path_dashboard(base, 0, FIGURES / "02_tableau_de_bord_trajectoire_0.png")
    plots.plot_attribution(summary, FIGURES / "03_attribution_pnl.png")

    # ------------------------------------------------------------------
    # 2. Monte-Carlo : beaucoup de trajectoires pour juger l'efficacité
    # ------------------------------------------------------------------
    title(f"2. MONTE-CARLO — {mc_paths} trajectoires")
    mc = run_scenario(market, option, mc_sim, hedge)
    mc_summary = overall_summary(mc)
    for k, v in mc_summary.items():
        print(f"  {k:<28} {v:>14,.4f}")
    pd.Series(mc_summary).to_csv(RESULTS / "resume_monte_carlo.csv", header=["valeur"])

    plots.plot_distribution(mc, FIGURES / "04_distribution_pnl.png")
    small = run_scenario(market, option, replace(base_sim, n_paths=300), hedge)
    plots.plot_pnl_vs_move(small, FIGURES / "05_pnl_vs_mouvement.png")

    # ------------------------------------------------------------------
    # 3. Robustesse : σ = 10 %, 20 %, 40 % (vol implicite = vol réalisée)
    # ------------------------------------------------------------------
    title("3. ROBUSTESSE — niveau de volatilité (implicite = réalisée)")
    vol_table = compare_volatility_levels([0.10, 0.20, 0.40], market, option, mc_sim, hedge)
    vol_table.to_csv(RESULTS / "scenarios_volatilite.csv", float_format="%.4f")
    print(pct(vol_table)[["pnl_total_moyen", "pnl_total_ecart_type", "pnl_sans_couv_ecart_type",
                     "pnl_gamma_moyen", "pnl_theta_moyen", "efficacite_couverture"]])

    # ------------------------------------------------------------------
    # 4. Le vrai risque : vol réalisée ≠ vol implicite
    # ------------------------------------------------------------------
    title("4. VOL RÉALISÉE ≠ VOL IMPLICITE (option vendue à 20 %)")
    mismatch_table = compare_realized_vs_implied([0.10, 0.20, 0.40], market, option, mc_sim, hedge)
    mismatch_table.to_csv(RESULTS / "scenarios_vol_realisee.csv", float_format="%.4f")
    print(pct(mismatch_table)[["pnl_total_moyen", "pnl_total_ecart_type", "pnl_gamma_moyen",
                          "pnl_theta_moyen", "efficacite_couverture"]])
    mismatch_results = {
        f"Vol réalisée {s:.0%}": run_scenario(replace(market, sigma_realized=s), option, mc_sim, hedge)
        for s in [0.10, 0.20, 0.40]
    }
    plots.plot_vol_mismatch(mismatch_results, FIGURES / "06_vol_realisee_vs_implicite.png")

    # ------------------------------------------------------------------
    # 5. Frais de transaction et fréquence de rebalancement
    # ------------------------------------------------------------------
    title("5. FRAIS — rebalancement quotidien vs hebdomadaire, avec ou sans frais (0,1 %)")
    rebal_table = compare_rebalancing({"quotidien": 1, "hebdomadaire": 5}, [0.0, 0.001],
                                      market, option, mc_sim)
    rebal_table.to_csv(RESULTS / "scenarios_rebalancement.csv", float_format="%.4f")
    print(pct(rebal_table)[["pnl_total_moyen", "pnl_total_ecart_type", "frais_moyens", "efficacite_couverture"]])
    plots.plot_rebalancing(rebal_table, FIGURES / "07_rebalancement_frais.png")

    print(f"\nTerminé. Tableaux dans {RESULTS.relative_to(ROOT)}/, graphiques dans {FIGURES.relative_to(ROOT)}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulation d'une couverture delta d'un call sur action.")
    parser.add_argument("--mc-paths", type=int, default=2000, help="trajectoires Monte-Carlo (défaut : 2000)")
    parser.add_argument("--seed", type=int, default=42, help="graine aléatoire (défaut : 42)")
    args = parser.parse_args()
    main(args.mc_paths, args.seed)
