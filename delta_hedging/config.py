"""
Paramètres du projet, regroupés au même endroit.

On utilise des dataclasses "gelées" (frozen) : une fois créés, les paramètres
ne peuvent plus être modifiés par erreur au milieu d'une simulation.
Pour tester un autre scénario, on crée une copie avec `dataclasses.replace`.
"""

from dataclasses import dataclass

TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class MarketParams:
    """Données de marché."""

    s0: float = 100.0              # prix initial de l'action
    r: float = 0.02                # taux sans risque, composé en continu
    sigma_implied: float = 0.20    # volatilité utilisée pour PRICER et couvrir
    sigma_realized: float = 0.20   # volatilité qui génère VRAIMENT les prix
    q: float = 0.0                 # rendement du dividende (aucun ici)


@dataclass(frozen=True)
class OptionSpec:
    """Caractéristiques de l'option vendue au client."""

    strike: float = 100.0          # K = 100 -> option à la monnaie (ATM)
    maturity: float = 1.0          # 1 an
    n_contracts: int = 100         # 100 contrats ...
    contract_size: int = 100       # ... de 100 actions = 10 000 actions notionnelles
    # Position de la banque sur l'option : -1 = vendeuse (le client a acheté),
    # +1 = acheteuse. Par défaut, la banque est courte (le client a acheté).
    position: int = -1

    @property
    def notional_shares(self) -> int:
        return self.n_contracts * self.contract_size


@dataclass(frozen=True)
class SimulationParams:
    """Horizon et discrétisation de la simulation."""

    n_days: int = 21               # 1 mois de bourse
    n_paths: int = 5               # nombre de trajectoires simulées
    seed: int = 42                 # graine aléatoire, pour des résultats reproductibles

    @property
    def dt(self) -> float:
        return 1.0 / TRADING_DAYS_PER_YEAR


@dataclass(frozen=True)
class HedgeParams:
    """Règles de gestion de la couverture."""

    rebalance_every: int = 1       # 1 = chaque jour, 5 = chaque semaine
    transaction_cost: float = 0.0  # en fraction du montant échangé (0.001 = 0,1 %)
    s_floor: float = 0.01          # plancher sur le prix (cas limite S -> 0)
