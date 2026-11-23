"""
Cœur du projet : la boucle de couverture delta, jour après jour.

Situation : un client achète un call ATM 1 an à la banque. La banque est donc 
COURTE de l'option. Si l'action monte, l'option qu'elle a vendue prend de la
valeur et la banque perd de l'argent. Pour neutraliser ce risque, le trader
ACHÈTE Δ actions par option vendue, et ajuste cette quantité chaque jour.

Le portefeuille du trader contient trois lignes :
    1. l'option          : position × N × C_t        (N = 10 000 actions notionnelles)
    2. les actions       : h_t actions, avec h_t = -position × N × Δ_t
    3. un compte cash    : la prime encaissée, moins les achats d'actions,
                           rémunéré au taux r

Valeur du portefeuille : V_t = position·N·C_t + h_t·S_t + B_t
PnL du jour           : PnL_t = V_t - V_{t-1}
                             = variation de l'option
                             + gain/perte sur les actions  (h_{t-1} · (S_t - S_{t-1}))
                             + intérêts sur le cash        (B_{t-1} · (e^{rΔt} - 1))
                             - frais de transaction

Toutes les quantités sont vues du côté de la banque vendeuse.

Toute la boucle est vectorisée sur les trajectoires : on traite 5 ou 5 000
chemins en même temps, avec le même code.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .black_scholes import bs_greeks
from .config import HedgeParams, MarketParams, OptionSpec


@dataclass
class HedgeResult:
    """Toutes les séries produites par la simulation, de forme (n_paths, n_days + 1)."""

    spot: np.ndarray            # S_t
    time_to_maturity: np.ndarray
    option_price: np.ndarray    # C_t, prix d'UN call
    delta: np.ndarray           # Δ_t d'un call
    gamma: np.ndarray
    vega: np.ndarray
    theta: np.ndarray
    hedge_shares: np.ndarray    # h_t, nombre d'actions détenues
    cash: np.ndarray            # B_t
    portfolio_value: np.ndarray # V_t
    # PnL quotidien et sa décomposition « comptable »
    pnl: np.ndarray
    pnl_option: np.ndarray
    pnl_stock: np.ndarray
    pnl_interest: np.ndarray
    pnl_costs: np.ndarray
    # Décomposition « par les Grecs » (explication du PnL par le trader)
    pnl_delta: np.ndarray
    pnl_gamma: np.ndarray
    pnl_theta: np.ndarray
    pnl_unexplained: np.ndarray
    # Référence : ce qu'aurait donné la même option SANS couverture
    pnl_unhedged: np.ndarray
    # Mesures de risque du portefeuille (option + actions)
    net_delta: np.ndarray       # en nombre d'actions
    net_gamma: np.ndarray       # en actions par 1 € de mouvement de S
    net_vega: np.ndarray        # en € pour +1 point de volatilité
    rebalanced: np.ndarray      # True les jours où l'on a ajusté la couverture

    @property
    def n_paths(self) -> int:
        return self.spot.shape[0]

    def to_dataframe(self, path: int = 0) -> pd.DataFrame:
        """Tableau jour par jour pour une trajectoire donnée."""
        df = pd.DataFrame(
            {
                "jour": np.arange(self.spot.shape[1]),
                "S": self.spot[path],
                "T_restant": self.time_to_maturity[path],
                "prix_call": self.option_price[path],
                "delta": self.delta[path],
                "gamma": self.gamma[path],
                "vega": self.vega[path],
                "actions_detenues": self.hedge_shares[path],
                "cash": self.cash[path],
                "valeur_portefeuille": self.portfolio_value[path],
                "pnl": self.pnl[path],
                "pnl_cumule": np.cumsum(self.pnl[path]),
                "pnl_option": self.pnl_option[path],
                "pnl_actions": self.pnl_stock[path],
                "pnl_interets": self.pnl_interest[path],
                "pnl_frais": self.pnl_costs[path],
                "pnl_delta": self.pnl_delta[path],
                "pnl_gamma": self.pnl_gamma[path],
                "pnl_theta": self.pnl_theta[path],
                "pnl_inexplique": self.pnl_unexplained[path],
                "pnl_sans_couverture": self.pnl_unhedged[path],
                "delta_net": self.net_delta[path],
                "gamma_net": self.net_gamma[path],
                "vega_net": self.net_vega[path],
                "rebalance": self.rebalanced[path],
            }
        )
        return df.set_index("jour")


def run_delta_hedge(
    spot_paths: np.ndarray,
    market: MarketParams,
    option: OptionSpec,
    hedge: HedgeParams,
    dt: float,
) -> HedgeResult:
    """
    Rejoue la stratégie de couverture sur des trajectoires de prix déjà simulées.

    `spot_paths` est un tableau (n_paths, n_days + 1). Le pricing et le delta
    utilisent la volatilité IMPLICITE `market.sigma_implied`, comme un vrai
    trader qui ne connaît pas à l'avance la volatilité qui va se réaliser.
    """
    spot_paths = np.atleast_2d(np.asarray(spot_paths, dtype=float))
    spot_paths = np.maximum(spot_paths, hedge.s_floor)
    n_paths, n_steps = spot_paths.shape
    n_days = n_steps - 1

    q = option.position              # -1 : la banque a vendu l'option
    N = option.notional_shares       # 10 000 actions notionnelles
    growth = np.exp(market.r * dt)   # capitalisation du cash sur une journée

    # ------------------------------------------------------------------
    # 1) Prix et Grecs de l'option, pour tous les jours et tous les chemins
    # ------------------------------------------------------------------
    tau = np.maximum(option.maturity - np.arange(n_steps) * dt, 0.0)
    tau = np.broadcast_to(tau, spot_paths.shape)
    g = bs_greeks(spot_paths, option.strike, market.r, tau, market.sigma_implied, hedge.s_floor)

    # ------------------------------------------------------------------
    # 2) Boucle de couverture
    # ------------------------------------------------------------------
    shares = np.zeros_like(spot_paths)   # h_t
    cash = np.zeros_like(spot_paths)     # B_t
    costs = np.zeros_like(spot_paths)    # frais payés le jour t
    rebalanced = np.zeros(spot_paths.shape, dtype=bool)

    for t in range(n_steps):
        S_t = spot_paths[:, t]
        target = -q * N * g.delta[:, t]  # quantité d'actions qui annule le delta

        # On ajuste la couverture le jour 0, puis tous les `rebalance_every` jours.
        # Le dernier jour, on se contente d'évaluer le portefeuille (pas de trade).
        is_rebalance_day = (t % hedge.rebalance_every == 0) and (t < n_days)

        if t == 0:
            shares[:, 0] = target
            costs[:, 0] = hedge.transaction_cost * np.abs(target) * S_t
            # La banque encaisse la prime de l'option et paie les actions achetées
            cash[:, 0] = -q * N * g.price[:, 0] - target * S_t - costs[:, 0]
            rebalanced[:, 0] = True
            continue

        previous = shares[:, t - 1]
        shares[:, t] = target if is_rebalance_day else previous
        traded = shares[:, t] - previous
        costs[:, t] = hedge.transaction_cost * np.abs(traded) * S_t
        cash[:, t] = cash[:, t - 1] * growth - traded * S_t - costs[:, t]
        rebalanced[:, t] = is_rebalance_day

    option_value = q * N * g.price
    portfolio = option_value + shares * spot_paths + cash

    # ------------------------------------------------------------------
    # 3) Décomposition du PnL quotidien
    # ------------------------------------------------------------------
    dS = np.diff(spot_paths, axis=1)
    pnl_option = _prepend_zero(q * N * np.diff(g.price, axis=1))
    pnl_stock = _prepend_zero(shares[:, :-1] * dS)
    pnl_interest = _prepend_zero(cash[:, :-1] * (growth - 1.0))
    pnl_costs = -costs
    pnl = pnl_option + pnl_stock + pnl_interest + pnl_costs

    # Explication par les Grecs (développement de Taylor à l'ordre 2) :
    #   delta : exposition résiduelle × dS        (≈ 0 si on rebalance chaque jour)
    #   gamma : ½ · Γ · dS²                        (négatif pour un vendeur d'option)
    #   theta : Θ · dt                             (positif pour un vendeur d'option)
    # La vega ne contribue pas ici car la volatilité implicite reste constante.
    net_delta_prev = q * N * g.delta[:, :-1] + shares[:, :-1]
    pnl_delta = _prepend_zero(net_delta_prev * dS)
    pnl_gamma = _prepend_zero(q * N * 0.5 * g.gamma[:, :-1] * dS**2)
    pnl_theta = _prepend_zero(q * N * g.theta[:, :-1] * dt)
    pnl_unexplained = pnl - (pnl_delta + pnl_gamma + pnl_theta + pnl_interest + pnl_costs)

    # Référence sans couverture : on garde juste l'option et la prime placée au taux r
    cash_unhedged = -q * N * g.price[:, :1] * growth ** np.arange(n_steps)
    pnl_unhedged = _prepend_zero(q * N * np.diff(g.price, axis=1) + cash_unhedged[:, :-1] * (growth - 1.0))

    return HedgeResult(
        spot=spot_paths,
        time_to_maturity=np.array(tau),
        option_price=g.price,
        delta=g.delta,
        gamma=g.gamma,
        vega=g.vega,
        theta=g.theta,
        hedge_shares=shares,
        cash=cash,
        portfolio_value=portfolio,
        pnl=pnl,
        pnl_option=pnl_option,
        pnl_stock=pnl_stock,
        pnl_interest=pnl_interest,
        pnl_costs=pnl_costs,
        pnl_delta=pnl_delta,
        pnl_gamma=pnl_gamma,
        pnl_theta=pnl_theta,
        pnl_unexplained=pnl_unexplained,
        pnl_unhedged=pnl_unhedged,
        net_delta=q * N * g.delta + shares,
        net_gamma=q * N * g.gamma,
        net_vega=q * N * g.vega / 100.0,   # par point de volatilité (1 %)
        rebalanced=rebalanced,
    )


def _prepend_zero(x: np.ndarray) -> np.ndarray:
    """Ajoute une colonne de zéros au début (pas de PnL de marché le jour 0)."""
    return np.concatenate([np.zeros((x.shape[0], 1)), x], axis=1)
