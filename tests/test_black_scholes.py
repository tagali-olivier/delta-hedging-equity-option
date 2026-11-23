"""Tests du pricer Black-Scholes : valeurs connues, Grecs, cas limites."""

import numpy as np
import pytest

from delta_hedging.black_scholes import bs_greeks, bs_put_price 

S, K, R, T, SIGMA = 100.0, 100.0, 0.02, 1.0, 0.20


def test_prix_de_reference():
    # Valeur de référence d'un call ATM 1 an, r = 2 %, σ = 20 %
    assert bs_greeks(S, K, R, T, SIGMA).price == pytest.approx(8.9160, abs=1e-4)


def test_parite_call_put():
    call = bs_greeks(S, K, R, T, SIGMA).price
    put = bs_put_price(S, K, R, T, SIGMA)
    assert call - put == pytest.approx(S - K * np.exp(-R * T), abs=1e-10)


@pytest.mark.parametrize("spot", [80.0, 100.0, 120.0])
def test_grecs_contre_differences_finies(spot):
    g = bs_greeks(spot, K, R, T, SIGMA)
    h = 1e-3
    price = lambda s=spot, t=T, v=SIGMA: bs_greeks(s, K, R, t, v).price

    delta_fd = (price(s=spot + h) - price(s=spot - h)) / (2 * h)
    gamma_fd = (price(s=spot + h) - 2 * price() + price(s=spot - h)) / h**2
    vega_fd = (price(v=SIGMA + h) - price(v=SIGMA - h)) / (2 * h)
    theta_fd = -(price(t=T + h) - price(t=T - h)) / (2 * h)  # le temps qui passe réduit T

    assert g.delta == pytest.approx(delta_fd, rel=1e-5)
    assert g.gamma == pytest.approx(gamma_fd, rel=1e-3)
    assert g.vega == pytest.approx(vega_fd, rel=1e-5)
    assert g.theta == pytest.approx(theta_fd, rel=1e-5)


def test_echeance_valeur_intrinseque():
    g = bs_greeks(np.array([90.0, 110.0]), K, R, 0.0, SIGMA)
    np.testing.assert_allclose(g.price, [0.0, 10.0])
    np.testing.assert_allclose(g.delta, [0.0, 1.0])
    np.testing.assert_allclose(g.gamma, [0.0, 0.0])
    np.testing.assert_allclose(g.vega, [0.0, 0.0])


def test_prix_action_proche_de_zero():
    g = bs_greeks(0.0, K, R, T, SIGMA)  # plancher à 0,01 : pas de NaN ni d'infini
    assert np.isfinite([g.price, g.delta, g.gamma, g.vega, g.theta]).all()
    assert g.price == pytest.approx(0.0, abs=1e-12)
    assert g.delta == pytest.approx(0.0, abs=1e-12)


def test_vectorisation():
    spots = np.linspace(50, 150, 11)
    vector = bs_greeks(spots, K, R, T, SIGMA).price
    scalar = [float(bs_greeks(s, K, R, T, SIGMA).price) for s in spots]
    np.testing.assert_allclose(vector, scalar)
