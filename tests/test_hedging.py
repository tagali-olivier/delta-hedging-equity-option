"""Tests de la boucle de couverture : cohérence comptable et comportement attendu."""

import time
from dataclasses import replace

import numpy as np
import pytest

from delta_hedging.analytics import hedge_effectiveness
from delta_hedging.config import HedgeParams, MarketParams, OptionSpec, SimulationParams 
from delta_hedging.scenarios import run_scenario

MARKET = MarketParams()
OPTION = OptionSpec()
SIM = SimulationParams(n_paths=200, seed=1)


@pytest.fixture(scope="module")
def daily():
    return run_scenario(MARKET, OPTION, SIM, HedgeParams())


def test_delta_neutre_au_depart(daily):
    np.testing.assert_allclose(daily.net_delta[:, 0], 0.0, atol=1e-9)
    # La banque est courte de l'option : elle achète des actions
    assert (daily.hedge_shares[:, 0] > 0).all()


def test_decomposition_du_pnl(daily):
    components = daily.pnl_option + daily.pnl_stock + daily.pnl_interest + daily.pnl_costs
    np.testing.assert_allclose(daily.pnl, components, atol=1e-8)


def test_pnl_cumule_egal_valeur_finale(daily):
    np.testing.assert_allclose(daily.pnl.sum(axis=1), daily.portfolio_value[:, -1], rtol=1e-10, atol=1e-6)


def test_attribution_grecs_presque_complete(daily):
    # Le résidu (termes d'ordre 3 et plus) doit rester petit devant gamma et theta
    residual = np.abs(daily.pnl_unexplained[:, 1:]).mean()
    gamma = np.abs(daily.pnl_gamma[:, 1:]).mean()
    assert residual < 0.05 * gamma


def test_la_couverture_reduit_le_risque(daily):
    assert hedge_effectiveness(daily) > 0.99
    assert daily.pnl.sum(axis=1).std() < 0.1 * daily.pnl_unhedged.sum(axis=1).std()


def test_pnl_moyen_proche_de_zero_si_vols_egales():
    res = run_scenario(MARKET, OPTION, replace(SIM, n_paths=3000), HedgeParams())
    totals = res.pnl.sum(axis=1)
    standard_error = totals.std() / np.sqrt(len(totals))
    # Couverture « parfaite » en moyenne : PnL ≈ 0 (petit biais de discrétisation)
    assert abs(totals.mean()) < 4 * standard_error + 50


def test_vol_realisee_superieure_fait_perdre_le_vendeur():
    high = run_scenario(replace(MARKET, sigma_realized=0.40), OPTION, SIM, HedgeParams())
    low = run_scenario(replace(MARKET, sigma_realized=0.10), OPTION, SIM, HedgeParams())
    assert high.pnl.sum(axis=1).mean() < 0 < low.pnl.sum(axis=1).mean()


def test_frais_de_transaction():
    no_cost = run_scenario(MARKET, OPTION, SIM, HedgeParams())
    with_cost = run_scenario(MARKET, OPTION, SIM, HedgeParams(transaction_cost=0.001))
    costs = with_cost.pnl_costs.sum(axis=1)
    assert (costs < 0).all()
    # Les frais réduisent le PnL de leur montant, plus un petit effet d'intérêts sur le cash
    np.testing.assert_allclose(with_cost.pnl.sum(axis=1) - no_cost.pnl.sum(axis=1), costs, rtol=5e-3)


def test_rebalancement_hebdomadaire():
    weekly = run_scenario(MARKET, OPTION, SIM, HedgeParams(rebalance_every=5))
    assert weekly.rebalanced[0].sum() == 5          # jours 0, 5, 10, 15, 20
    daily = run_scenario(MARKET, OPTION, SIM, HedgeParams())
    assert weekly.pnl.sum(axis=1).std() > daily.pnl.sum(axis=1).std()


def test_performance():
    start = time.perf_counter()
    run_scenario(MARKET, OPTION, SimulationParams(n_paths=5), HedgeParams())
    assert time.perf_counter() - start < 1.0
