"""Die SDP-Relaxation (kbm_sdp): Handinstanzen, Eigenschaften, Dual, Randfälle."""

import math

import numpy as np
import pytest

import kegel_kern as K
import kbm_graphs as G
import kbm_rounding as R
import kbm_sdp as S


def test_the_pentagon_by_hand():
    """Fünf Einheitsvektoren (cos(4πk/5), sin(4πk/5)): X_{k,k+1} = cos(4π/5) = −(1 + √5)/4; zulässig (PSD, Diagonale 1), Wert (5/2)(1 + (1 + √5)/4) = 4,5225; das SDP findet nichts Besseres."""
    ang = 4.0 * math.pi * np.arange(5) / 5.0
    V = np.stack([np.cos(ang), np.sin(ang)], axis=1)
    X = V @ V.T
    W = G.cycle(5)
    assert np.allclose(np.diag(X), 1.0) and np.linalg.eigvalsh(X).min() > -1e-12 and X[0, 1] == pytest.approx(-(1 + math.sqrt(5)) / 4)
    value = float(np.sum(np.triu(W, 1) * (1 - X)) / 2)
    assert value == pytest.approx(2.5 * (1 + (1 + math.sqrt(5)) / 4)) and value == pytest.approx(4.522542485937368)
    assert S.solve(W).value == pytest.approx(value, abs=1e-7)


@pytest.mark.parametrize("name,W,expected,rank", [("K3", G.complete(3), 9 / 4, 2), ("K4", G.complete(4), 4.0, 3), ("C4", G.cycle(4), 4.0, 1), ("K2,3", G.bipartite(2, 3), 6.0, 1)])
def test_hand_values_and_ranks(name, W, expected, rank):
    """K₃: X = (3I − J)/2: 3 · (1 + 1/2)/2 = 9/4; K₄: X = (4I − J)/3: 6 · (1 + 1/3)/2 = 4; bipartit: X = s sᵀ mit Wert = Zahl der Kanten, Rang 1."""
    r = S.solve(W)
    assert r.value == pytest.approx(expected, abs=1e-6) and r.rank == rank, name
    if name == "K3":
        assert np.allclose(r.X, (3 * np.eye(3) - np.ones((3, 3))) / 2, atol=1e-5)
    if name == "K4":
        assert np.allclose(r.X, (4 * np.eye(4) - np.ones((4, 4))) / 3, atol=1e-5)


def test_the_solution_is_feasible_and_bounds_the_exact_cut_from_above():
    rng = np.random.default_rng(0)
    for _ in range(12):
        n = int(rng.integers(3, 10))
        W = G.gnp(n, 0.6, int(rng.integers(0, 10 ** 6)), "zufall" if rng.random() < 0.5 else "eins")
        if G.total_weight(W) == 0.0:
            continue
        r = S.solve(W)
        assert np.allclose(np.diag(r.X), 1.0, atol=1e-7) and np.linalg.eigvalsh(r.X).min() > -1e-7 and np.allclose(r.X, r.X.T)
        assert G.brute_force_maxcut(W) <= r.value + 1e-6 and r.value <= G.total_weight(W) + 1e-9


def test_the_dual_is_a_certificate():
    """Z = Diag(d) + W/2 ist PSD mit Außerdiagonale w_ij/2; (Σ w + tr Z)/2 ist gleich dem SDP-Wert (starke Dualität) und über jedem Schnitt."""
    for W in (G.cycle(5), G.petersen(), G.gnp(12, 0.5, 3), G.gnp(10, 0.5, 2, "zufall")):
        r = S.solve(W)
        assert r.z_min_eig > -1e-8 and r.z_offdiag < 1e-9 and abs(r.gap) < 1e-6
        if len(W) <= 12:
            assert r.dual_bound >= G.brute_force_maxcut(W) - 1e-6
        assert np.allclose(r.Z - np.diag(np.diag(r.Z)), (W - np.diag(np.diag(W))) / 2.0, atol=1e-8)


def test_the_kernel_barrier_finds_the_same_value():
    for W in (G.cycle(5), G.gnp(8, 0.5, 1)):
        prob, pairs = S.build(W)
        res = K.solve_barrier(prob, tol=1e-9)
        assert (G.total_weight(W) - res.value) / 2 == pytest.approx(S.solve(W).value, abs=1e-6)


def test_the_program_has_the_expected_shape():
    prob, pairs = S.build(G.cycle(5))
    assert len(pairs) == 10 and prob.A.shape == (15, 10) and prob.cones == (("s", 5),) and prob.b.sum() == 5.0 and (prob.A.min() == -K.SQ2) and prob.c.sum() == 5.0
    assert np.allclose(S.vectors(S.solve(G.cycle(5)).X) @ S.vectors(S.solve(G.cycle(5)).X).T, S.solve(G.cycle(5)).X, atol=1e-7)


def test_edge_cases():
    r = S.solve(np.zeros((4, 4)))                                                  # keine Kanten: der Schnitt ist 0, X = I
    assert r.value == 0.0 and np.allclose(r.X, np.eye(4)) and r.dual_bound == 0.0 and r.rank == 4
    one = S.solve(np.array([[0.0, 1.0], [1.0, 0.0]]))                              # eine Kante: SDP = 1 = exakt
    assert one.value == pytest.approx(1.0, abs=1e-7) and one.rank == 1
    heavy = G.cycle(5) * 1e-4
    assert S.solve(heavy).value == pytest.approx(4.522542485937368e-4, rel=1e-4)    # Skala der Gewichte
    zero_edge = G.gnp(8, 0.5, 4).copy()
    zero_edge[0, 1] = zero_edge[1, 0] = 0.0
    assert np.isfinite(S.solve(zero_edge).value)
    for bad in (np.array([[1.0, 0.0], [0.0, 1.0]]), np.array([[0.0, 1.0], [2.0, 0.0]]), np.array([[0.0, -1.0], [-1.0, 0.0]])):
        with pytest.raises(ValueError):
            S.solve(bad)
    with pytest.raises(ValueError):
        S.solve(np.zeros((1, 1)))
    assert R.ALPHA_GW == pytest.approx(0.878567, abs=1e-6)
