"""ADMM in Matrixform (Gegenprobe), Speicher und das Zertifikat von Rakoto Endor und Waldspurger (kbm_admm, kbm_rank)."""

import math

import numpy as np
import pytest

import kbm_admm as A
import kbm_bm as B
import kbm_graphs as G
import kbm_rank as K
import kbm_sdp as S


def test_admm_finds_the_sdp_value_of_the_pentagon_and_of_random_graphs():
    r = A.admm(G.cycle(5), tol=1e-9)
    assert r.converged and r.value == pytest.approx(5.0 / 2.0 * (1.0 - math.cos(4.0 * math.pi / 5.0)), abs=1e-5)
    assert np.allclose(np.diag(r.X), 1.0, atol=1e-6) and np.linalg.eigvalsh(r.X)[0] > -1e-6
    for n, seed in [(8, 0), (12, 1), (16, 2)]:
        W = G.gnp(n, 0.4, seed)
        assert A.admm(W, tol=1e-8).value == pytest.approx(S.solve(W).value, abs=1e-4)


def test_admm_rejects_a_single_node_and_handles_no_edges():
    with pytest.raises(ValueError):
        A.admm(np.zeros((1, 1)))
    r = A.admm(np.zeros((4, 4)))
    assert r.value == 0.0


def test_burer_monteiro_and_admm_agree():
    W = G.gnp(20, 0.3, 3)
    bm = B.mixing(W, B.rank_for_theory(20) + 1, np.random.default_rng(0), tol=1e-12)
    assert bm.value == pytest.approx(A.admm(W, tol=1e-9).value, abs=1e-4)


def test_memory_is_eight_bytes_per_number():
    assert K.bytes_matrix(100) == 80_000 and K.bytes_factor(100, 15) == 12_000
    assert K.bytes_matrix(10_000) == 800_000_000 and K.bytes_factor(10_000, 142) == 11_360_000
    assert K.bytes_matrix(100_000) == 80_000_000_000 and K.bytes_factor(100_000, 448) == 358_400_000
    assert np.zeros((100, 100)).nbytes == K.bytes_matrix(100) and np.zeros((100, 15)).nbytes == K.bytes_factor(100, 15)


def test_human_bytes():
    assert K.human_bytes(800_000_000) == "800 MB" and K.human_bytes(11_360_000) == "11,4 MB" and K.human_bytes(80_000_000_000) == "80 GB" and K.human_bytes(358_400_000) == "358 MB" and K.human_bytes(512) == "512 B"


def test_the_certificate_of_the_complete_bipartite_graph():
    """K_{a,a}: L = D + W hat die Eigenwerte 0, a (mehrfach), 2a, also κ = 2 und p ≥ 3; K_{a,b}: Eigenwerte 0, min(a,b), max(a,b), a + b."""
    for a in (3, 5, 8):
        c = K.rew_certificate(G.bipartite(a, a))
        assert c["lam2"] == pytest.approx(a) and c["lamn"] == pytest.approx(2 * a) and c["kappa"] == pytest.approx(2.0) and c["p_min"] == 3
    c = K.rew_certificate(G.bipartite(3, 7))
    assert c["lam2"] == pytest.approx(3.0) and c["lamn"] == pytest.approx(10.0) and c["kappa"] == pytest.approx(10.0 / 3.0) and c["p_min"] == 4


def test_the_certificate_matrix_is_the_signless_laplacian_with_kernel_x():
    W = G.random_bipartite(8, 8, 0.6, 1)
    c = K.rew_certificate(W)
    x = c["x"]
    L = np.diag(W.sum(axis=1)) + W
    assert np.allclose(L @ x, 0.0, atol=1e-12) and np.linalg.eigvalsh(L)[0] > -1e-9
    assert c["kappa"] == pytest.approx(np.linalg.eigvalsh(L)[-1] / np.linalg.eigvalsh(L)[1])
    assert B.objective(W, x[:, None]) == pytest.approx(G.total_weight(W))            # x xᵀ schneidet alle Kanten: Wert = Σ w


def test_the_certificate_does_not_apply_outside_its_assumptions():
    assert K.rew_certificate(G.cycle(5)) is None and K.rew_certificate(G.complete(4)) is None and K.rew_certificate(np.zeros((4, 4))) is None
    two = np.zeros((6, 6))
    two[0, 1] = two[1, 0] = two[2, 3] = two[3, 2] = two[4, 5] = two[5, 4] = 1.0
    assert K.rew_certificate(two) is None                                              # nicht zusammenhängend: λ₂ = 0
    assert K.rew_certificate(G.cycle(6)) is not None                                    # gerader Kreis: bipartit


def test_the_sufficient_rank_works_on_bipartite_graphs():
    """p > κ ⇒ jeder Lauf endet im Optimum (hier: Wert = Σ w): 10 Zufallsstarts je Graph, p = ⌊κ⌋ + 1."""
    rng = np.random.default_rng(0)
    for a, b, pe, seed in [(5, 5, 1.0, 0), (6, 6, 0.6, 1), (8, 8, 0.5, 2)]:
        W = G.bipartite(a, b) if pe == 1.0 else G.random_bipartite(a, b, pe, seed)
        c = K.rew_certificate(W)
        if c is None:
            continue
        for _ in range(10):
            assert G.total_weight(W) - B.mixing(W, c["p_min"], rng, tol=1e-12).value < 1e-4
