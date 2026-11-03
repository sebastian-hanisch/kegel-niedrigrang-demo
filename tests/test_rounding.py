"""Rundung und exakter Schnitt (kbm_rounding): Erwartungswert von Hand, Garantie, Monte Carlo, Aufzählung."""

import math

import numpy as np
import pytest

import kbm_graphs as G
import kbm_rounding as R
import kbm_sdp as S


def test_the_expected_cut_by_hand():
    """Fünfeck: jede Kante wird mit Wahrscheinlichkeit θ/π = (4π/5)/π = 4/5 geschnitten: 5 · 4/5 = 4; K₃: Winkel 120°, Wahrscheinlichkeit 2/3: 3 · 2/3 = 2; K₄: 6 · arccos(−1/3)/π = 3,649;
    bipartit mit X = s sᵀ: jede Kante hat X = −1, Wahrscheinlichkeit 1: der Schnitt ist die Zahl der Kanten."""
    c5 = G.cycle(5)
    ang = 4.0 * math.pi * np.arange(5) / 5.0
    V = np.stack([np.cos(ang), np.sin(ang)], axis=1)
    assert R.expected_rounding(c5, V @ V.T) == pytest.approx(4.0)
    assert R.expected_rounding(G.complete(3), (3 * np.eye(3) - np.ones((3, 3))) / 2) == pytest.approx(2.0)
    assert R.expected_rounding(G.complete(4), (4 * np.eye(4) - np.ones((4, 4))) / 3) == pytest.approx(6 * math.acos(-1 / 3) / math.pi) and 3.649 < R.expected_rounding(G.complete(4), (4 * np.eye(4) - np.ones((4, 4))) / 3) < 3.6491
    s = np.array([1.0, 1.0, -1.0, -1.0, -1.0])
    assert R.expected_rounding(G.bipartite(2, 3), np.outer(s, s)) == pytest.approx(6.0)


def test_the_guarantee_holds_for_random_graphs_and_all_families():
    """E[Schnitt] ≥ α_GW · SDP-Wert (Goemans, Williamson): 80 Zufallsgraphen (mit und ohne Gewichte) und die Familien."""
    rng = np.random.default_rng(5)
    graphs = [G.cycle(5), G.cycle(7), G.complete(5), G.petersen(), G.wheel(9), G.grid(3, 3), G.tree(10, 1)]
    for _ in range(80):
        graphs.append(G.gnp(int(rng.integers(4, 15)), float(rng.choice([0.2, 0.5, 0.8])), int(rng.integers(0, 10 ** 6)), "zufall" if rng.random() < 0.3 else "eins"))
    worst = 1.0
    for W in graphs:
        if G.total_weight(W) == 0.0:
            continue
        r = S.solve(W)
        worst = min(worst, R.expected_rounding(W, r.X) / r.value)
        assert R.expected_rounding(W, r.X) >= R.ALPHA_GW * r.value - 1e-6
    assert 0.8785 <= worst < 0.93                                                  # Petersen liegt knapp über der Garantie


def test_alpha_gw_is_the_minimum_of_the_ratio():
    assert R.alpha_gw() == pytest.approx(0.8785672, abs=1e-6) and R.ALPHA_GW == pytest.approx(R.alpha_gw(), abs=1e-6)
    th = np.linspace(1e-3, math.pi, 1000)
    assert (2 / math.pi * th / (1 - np.cos(th)) >= R.ALPHA_GW - 1e-6).all()


def test_monte_carlo_agrees_with_the_formula():
    W = G.gnp(12, 0.5, 3)
    r = S.solve(W)
    values, best = R.round_values(W, S.vectors(r.X), 20000, np.random.default_rng(0))
    stderr = values.std() / math.sqrt(len(values))
    assert abs(values.mean() - R.expected_rounding(W, r.X)) < 4 * stderr and G.cut_value(W, best) == values.max()


def test_round_values_match_the_cut_value_of_each_sign_vector():
    W = G.gnp(9, 0.6, 2)
    r = S.solve(W)
    V = S.vectors(r.X)
    values, best = R.round_values(W, V, 50, np.random.default_rng(3))
    R_ = np.random.default_rng(3).normal(size=(V.shape[1], 50))
    for k in range(50):
        s = np.where(V @ R_[:, k] >= 0, 1.0, -1.0)
        assert values[k] == pytest.approx(G.cut_value(W, s))
    assert values.max() <= r.value + 1e-9 and best.shape == (9,) and set(np.unique(best)) <= {-1.0, 1.0}


def test_best_of_k_grows_with_k():
    W = G.gnp(12, 0.5, 3)
    r = S.solve(W)
    values, _ = R.round_values(W, S.vectors(r.X), 2000, np.random.default_rng(1))
    curve = R.best_of_k_curve(values, (1, 2, 5, 10, 20, 50, 100))
    assert curve[0] == pytest.approx(values.mean(), abs=1e-9) and all(b >= a - 1e-9 for a, b in zip(curve, curve[1:])) and curve[-1] <= r.value + 1e-9


def test_the_exact_enumeration_equals_the_simple_brute_force():
    for seed in range(8):
        W = G.gnp(9, 0.5, seed, "zufall" if seed % 2 else "eins")
        value, s = R.exact_maxcut(W)
        assert value == pytest.approx(G.brute_force_maxcut(W), abs=1e-9) and G.cut_value(W, s) == pytest.approx(value)
    assert R.exact_maxcut(G.petersen())[0] == 12.0 and R.exact_maxcut(G.cycle(7))[0] == 6.0 and R.exact_maxcut(G.grid(4, 4))[0] == 24.0 and R.exact_maxcut(G.complete(6))[0] == 9.0


def test_the_enumeration_is_limited_and_rejects_bad_input():
    with pytest.raises(ValueError):
        R.exact_maxcut(np.zeros((R.EXACT_MAX + 1, R.EXACT_MAX + 1)))
    with pytest.raises(ValueError):
        R.exact_maxcut(np.zeros((1, 1)))
    assert R.EXACT_MAX == 20
