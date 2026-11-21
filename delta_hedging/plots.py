"""
Graphiques du projet (matplotlib). Chaque fonction renvoie une figure et,
si `path` est fourni, l'enregistre en PNG.

Code couleur constant dans tout le projet :
    bleu   = portefeuille COUVERT
    orange = option seule, SANS couverture
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # pas d'affichage à l'écran : on écrit directement des fichiers
import matplotlib.pyplot as plt
import numpy as np

from .hedging import HedgeResult

# Palette catégorielle (ordre fixe), validée pour les daltoniens
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
GREY, INK, MUTED = "#8a8a85", "#2b2b29", "#6b6b66"
PATH_COLORS = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA]

plt.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#c9c8c2",
        "axes.labelcolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlecolor": INK,
        "axes.grid": True,
        "grid.color": "#e6e5e0",
        "grid.linewidth": 0.8,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "legend.frameon": False,
        "lines.linewidth": 2,
        "font.size": 10,
    }
)


def _save(fig, path):
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path)
        plt.close(fig)
    return fig


def _euros(ax, axis="y"):
    fmt = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f} €".replace(",", " "))
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def plot_paths(res: HedgeResult, path=None):
    """Les trajectoires simulées du sous-jacent."""
    fig, ax = plt.subplots(figsize=(8, 4.2))
    days = np.arange(res.spot.shape[1])
    for i in range(min(res.n_paths, len(PATH_COLORS))):
        ax.plot(days, res.spot[i], color=PATH_COLORS[i], label=f"Trajectoire {i}")
    ax.axhline(100, color=GREY, lw=1, ls="--")
    ax.text(days[-1], 100, "  K = 100", va="center", color=MUTED, fontsize=9)
    ax.set(title="Trajectoires simulées de l'action (GBM, σ = 20 %)", xlabel="Jour de bourse", ylabel="Prix (€)")
    ax.legend(ncol=5, loc="upper left", fontsize=8.5)
    return _save(fig, path)


def plot_path_dashboard(res: HedgeResult, i: int = 0, path=None):
    """Vue d'ensemble d'une trajectoire : prix, couverture, PnL et Grecs."""
    df = res.to_dataframe(i)
    days = df.index
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))

    # (a) Prix de l'action
    ax = axes[0, 0]
    ax.plot(days, df["S"], color=BLUE)
    ax.set(title="(a) Prix de l'action", xlabel="Jour", ylabel="S (€)")

    # (b) Nombre d'actions détenues = couverture
    ax = axes[0, 1]
    ax.step(days, df["actions_detenues"], where="post", color=AQUA)
    ax.set(title="(b) Couverture : actions détenues (= Δ × 10 000)", xlabel="Jour", ylabel="Actions")

    # (c) PnL cumulé couvert vs non couvert
    ax = axes[1, 0]
    ax.plot(days, df["pnl_cumule"], color=BLUE, label="Couvert")
    ax.plot(days, df["pnl_sans_couverture"].cumsum(), color=ORANGE, label="Sans couverture")
    ax.axhline(0, color=GREY, lw=1)
    ax.set(title="(c) PnL cumulé", xlabel="Jour")
    _euros(ax)
    ax.legend()

    # (d) Grecs de la position (en valeurs relatives au jour 0 pour tout voir sur un axe)
    ax = axes[1, 1]
    for col, color, label in [("delta", BLUE, "Delta"), ("gamma", ORANGE, "Gamma"), ("vega", AQUA, "Vega")]:
        ax.plot(days, df[col] / df[col].iloc[0], color=color, label=label)
    ax.axhline(1, color=GREY, lw=1, ls="--")
    ax.set(title="(d) Grecs d'un call, base 1 au jour 0", xlabel="Jour", ylabel="Valeur / valeur initiale")
    ax.legend()

    fig.suptitle(f"Couverture delta quotidienne — trajectoire {i}", fontsize=14, fontweight="bold", color=INK)
    fig.tight_layout()
    return _save(fig, path)


def plot_pnl_vs_move(res: HedgeResult, path=None):
    """
    PnL quotidien en fonction du mouvement de l'action.
    Sans couverture : une droite (on perd quand S monte).
    Couvert : une parabole retournée = la signature du gamma négatif.
    """
    dS = np.diff(res.spot, axis=1).ravel()
    hedged = res.pnl[:, 1:].ravel()
    unhedged = res.pnl_unhedged[:, 1:].ravel()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    ax.scatter(dS, unhedged, s=10, color=ORANGE, alpha=0.5, label="Sans couverture", edgecolors="none")
    ax.scatter(dS, hedged, s=10, color=BLUE, alpha=0.6, label="Couvert", edgecolors="none")
    ax.set(title="PnL quotidien vs variation de l'action", xlabel="Variation journalière de S (€)")
    _euros(ax)
    ax.legend()

    ax = axes[1]
    ax.scatter(dS, hedged, s=10, color=BLUE, alpha=0.6, edgecolors="none", label="PnL couvert observé")
    grid = np.linspace(dS.min(), dS.max(), 200)
    # Approximation théorique au jour 0 : PnL ≈ Θ·dt + intérêts + ½·Γ·dS² (position vendeuse)
    gamma0 = res.net_gamma[:, 0].mean()
    carry0 = (res.pnl_theta[:, 1] + res.pnl_interest[:, 1]).mean()
    ax.plot(grid, carry0 + 0.5 * gamma0 * grid**2, color=INK, lw=1.5, label="Θ·dt + intérêts + ½·Γ·dS²")
    ax.axhline(0, color=GREY, lw=1)
    ax.set(title="Zoom sur le portefeuille couvert : l'effet gamma", xlabel="Variation journalière de S (€)")
    _euros(ax)
    ax.legend()
    fig.tight_layout()
    return _save(fig, path)


def plot_attribution(summary_df, path=None):
    """Décomposition du PnL total de chaque trajectoire (barres groupées)."""
    cols = [("pnl_theta", AQUA, "Theta"), ("pnl_gamma", ORANGE, "Gamma"),
            ("pnl_interets", YELLOW, "Intérêts"), ("pnl_inexplique", MAGENTA, "Inexpliqué")]
    n = len(summary_df)
    x = np.arange(n)
    width = 0.17
    fig, ax = plt.subplots(figsize=(10, 4.6))
    for k, (col, color, label) in enumerate(cols):
        ax.bar(x + (k - 1.5) * width, summary_df[col], width * 0.9, color=color, label=label)
    ax.scatter(x, summary_df["pnl_total"], color=INK, zorder=5, s=40, marker="D", label="PnL total")
    for xi, v in zip(x, summary_df["pnl_total"]):
        ax.annotate(f"{v:,.0f} €".replace(",", " "), (xi, v), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8.5, color=INK)
    ax.axhline(0, color=GREY, lw=1)
    ax.set_xticks(x, [f"Traj. {i}\nvol réal. {v:.0%}" for i, v in zip(summary_df.index, summary_df["vol_realisee"])])
    ax.set(title="D'où vient le PnL ? Attribution par les Grecs (1 mois)")
    _euros(ax)
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    return _save(fig, path)


def plot_distribution(res: HedgeResult, path=None):
    """Distribution du PnL total sur un mois, couvert vs non couvert (Monte-Carlo)."""
    hedged = res.pnl.sum(axis=1)
    unhedged = res.pnl_unhedged.sum(axis=1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))

    ax = axes[0]
    bins = np.linspace(np.percentile(unhedged, 0.5), np.percentile(unhedged, 99.5), 60)
    ax.hist(unhedged, bins=bins, color=ORANGE, alpha=0.75, label=f"Sans couverture (σ = {unhedged.std():,.0f} €)".replace(",", " "))
    ax.hist(hedged, bins=bins, color=BLUE, alpha=0.9, label=f"Couvert (σ = {hedged.std():,.0f} €)".replace(",", " "))
    ax.set(title=f"PnL sur 1 mois — {res.n_paths} trajectoires", ylabel="Nombre de trajectoires")
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
    _euros(ax, "x")
    ax.legend()

    ax = axes[1]
    ax.hist(hedged, bins=50, color=BLUE, alpha=0.9)
    ax.axvline(hedged.mean(), color=INK, lw=1.5, ls="--")
    ax.text(hedged.mean(), ax.get_ylim()[1] * 0.95, f"  moyenne {hedged.mean():,.0f} €".replace(",", " "),
            color=INK, fontsize=9, va="top")
    ax.set(title="Zoom : PnL du portefeuille couvert", ylabel="Nombre de trajectoires")
    _euros(ax, "x")
    fig.tight_layout()
    return _save(fig, path)


def plot_vol_mismatch(results: dict, path=None):
    """PnL total couvert quand la vol réalisée diffère de la vol implicite (20 %)."""
    fig, ax = plt.subplots(figsize=(9, 4.3))
    colors = [AQUA, BLUE, ORANGE]
    all_totals = np.concatenate([r.pnl.sum(axis=1) for r in results.values()])
    bins = np.linspace(all_totals.min(), all_totals.max(), 70)
    for (label, res), color in zip(results.items(), colors):
        totals = res.pnl.sum(axis=1)
        ax.hist(totals, bins=bins, color=color, alpha=0.75,
                label=f"{label} : moyenne {totals.mean():,.0f} €".replace(",", " "))
    ax.axvline(0, color=GREY, lw=1)
    ax.set(title="Option vendue à 20 % de vol implicite : PnL selon la vol réalisée",
           ylabel="Nombre de trajectoires")
    _euros(ax, "x")
    ax.legend()
    return _save(fig, path)


def plot_rebalancing(table, path=None):
    """Compromis rebalancement quotidien vs hebdomadaire : coût moyen vs risque."""
    fig, ax = plt.subplots(figsize=(8, 4.6))
    markers = {"0.0%": "o", "0.1%": "s"}
    colors = {"quotidien": BLUE, "hebdomadaire": ORANGE}
    for (freq, cost), row in table.iterrows():
        ax.scatter(row["pnl_total_ecart_type"], row["pnl_total_moyen"], s=90,
                   color=colors.get(freq, GREY), marker=markers.get(cost, "o"),
                   edgecolors="white", linewidths=2, zorder=3)
        ax.annotate(f"{freq}, frais {cost}", (row["pnl_total_ecart_type"], row["pnl_total_moyen"]),
                    textcoords="offset points", xytext=(8, -14), fontsize=9, color=INK)
    ax.margins(x=0.2, y=0.12)
    ax.axhline(0, color=GREY, lw=1)
    ax.set(title="Rebalancer plus souvent : moins de risque, plus de frais",
           xlabel="Risque : écart-type du PnL mensuel (€)", ylabel="PnL mensuel moyen (€)")
    _euros(ax)
    _euros(ax, "x")
    return _save(fig, path)
