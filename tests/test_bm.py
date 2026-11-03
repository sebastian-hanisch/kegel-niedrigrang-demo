"""Burer–Monteiro (kbm_bm): Handwerte, Fixpunktgleichung, Monotonie, Zertifikat, Referenzwert gegen den SDP-Löser aus Stück 6."""

import math

import numpy as np
import pytest

import kbm_bm as B
import kbm_graphs as G
import kbm_sdp as S


def _pentagon():
    ang = 4.0 * math.pi * np.arange(5) / 5.0
    return np.stack([np.cos(ang), np.sin(ang)], axis=1)


def _polygon(n):
    ang = 2.0 * math.pi * np.arange(n) / n
    return np.stack([np.cos(ang), np.sin(ang)], axis=1)


def test_the_objective_at_the_hand_examples():
    """Fünfeck im Winkel 144°: (5/2)(1 − cos(4π/5)) = 4,5225; K_n mit regelmäßigem n-Eck (Σ v = 0): n²/4; p = 1: der Schnittwert."""
    assert B.objective(G.cycle(5), _pentagon()) == pytest.approx(5.0 / 2.0 * (1.0 - math.cos(4.0 * math.pi / 5.0)), abs=1e-12)
    for n in (4, 5, 6, 8):
        assert B.objective(G.complete(n), _polygon(n)) == pytest.approx(n * n / 4.0, abs=1e-12)
    s = np.array([[1.0], [-1.0], [1.0], [-1.0], [1.0]])
    assert B.objective(G.cycle(5), s) == pytest.approx(4.0)
    assert B.objective(G.complete(5), np.array([[1.0], [1.0], [1.0], [-1.0], [-1.0]])) == pytest.approx(6.0)


def test_the_update_is_the_exact_minimizer_of_one_row():
    """Bei festen übrigen Zeilen maximiert v_i = −g_i/‖g_i‖ den Wert über alle Einheitsvektoren der Ebene (Gitter von 20 000 Winkeln)."""
    rng = np.random.default_rng(4)
    W = G.gnp(8, 0.6, 2)
    V = B.random_rows(8, 2, rng)
    for i in range(8):
        g = W[i] @ V
        if np.linalg.norm(g) < 1e-9:
            continue
        cand = -g / np.linalg.norm(g)
        grid = np.stack([np.cos(np.linspace(0, 2 * math.pi, 20000)), np.sin(np.linspace(0, 2 * math.pi, 20000))], axis=1)
        best = float(np.max(-grid @ g))
        assert float(-cand @ g) == pytest.approx(best, abs=1e-6) and float(-cand @ g) >= best - 1e-9


def test_rows_stay_unit_vectors_and_the_value_never_decreases():
    rng = np.random.default_rng(1)
    for seed in range(5):
        W = G.gnp(14, 0.4, seed)
        r = B.mixing(W, 3, rng, tol=1e-10)
        assert np.allclose(np.linalg.norm(r.V, axis=1), 1.0, atol=1e-12)
        assert np.all(np.diff(r.history) >= -1e-10) and r.history[-1] == pytest.approx(B.objective(W, r.V)) and r.value == r.history[-1]
        assert r.sweeps == len(r.history) - 1 and r.V.shape == (14, 3) and r.p == 3 and r.n == 14


def test_rank_one_is_the_flip_search_and_gives_integer_cuts():
    rng = np.random.default_rng(2)
    for seed in range(6):
        W = G.gnp(12, 0.5, seed)
        r = B.mixing(W, 1, rng, tol=1e-12)
        assert np.all(np.abs(r.V) == pytest.approx(1.0)) and abs(r.value - round(r.value)) < 1e-9
        s = np.where(r.V[:, 0] >= 0, 1.0, -1.0)
        assert r.value == pytest.approx(G.cut_value(W, s))
        for i in range(12):                                                           # lokales Optimum der Flip-Nachbarschaft
            t = s.copy()
            t[i] = -t[i]
            assert G.cut_value(W, t) <= r.value + 1e-9


def test_the_value_never_exceeds_the_sdp_value_and_a_large_enough_rank_reaches_it():
    rng = np.random.default_rng(3)
    for n, pe, seed in [(8, 0.5, 0), (12, 0.4, 1), (16, 0.3, 2), (20, 0.3, 3)]:
        W = G.gnp(n, pe, seed)
        sdp = S.solve(W).value
        for p in (1, 2, 3, 5):
            assert B.mixing(W, p, rng, tol=1e-10).value <= sdp + 1e-6
        r = B.mixing(W, B.rank_for_theory(n) + 1, rng, tol=1e-12)
        assert r.value == pytest.approx(sdp, abs=1e-4), (n, seed)


def test_the_dual_matrix_at_the_square_of_k4_is_psd_and_gives_the_bound():
    """K₄ mit den Ecken des Quadrats: g_i = −v_i, ‖g_i‖ = 1, Z = I/2 + (J − I)/2 = J/2 (Eigenwerte 0, 0, 0, 2); Schranke (6 + 2)/2 = 4 = Wert."""
    V = _polygon(4)
    W = G.complete(4)
    assert np.allclose(W @ V, -V, atol=1e-12)
    assert np.allclose(B.dual_matrix(W, V), np.ones((4, 4)) / 2.0, atol=1e-12)
    c = B.certificate(W, V)
    assert abs(c["lam_min"]) < 1e-12 and c["bound"] == pytest.approx(4.0) and B.objective(W, V) == pytest.approx(4.0)
    assert B.is_certified(c, 1e-8)
    assert np.allclose(B.dual_matrix(W, V) @ V, 0.0, atol=1e-12)                    # Z V = 0 am Fixpunkt


def test_the_certificate_refuses_a_trap():
    """K₅ mit Rang 1 und s = (+, +, +, −, −): Schnitt 6 < 6,25 (SDP). Z(V) hat einen deutlich negativen Eigenwert."""
    W = G.complete(5)
    V = np.array([[1.0], [1.0], [1.0], [-1.0], [-1.0]])
    c = B.certificate(W, V)
    assert c["lam_min"] == pytest.approx(-0.5) and not B.is_certified(c, 1e-8) and B.objective(W, V) == pytest.approx(6.0)


def test_a_certified_run_is_at_the_sdp_optimum_for_every_rank():
    """Zertifikat ja ⇒ Wert gleich SDP-Wert (schwache Dualität: die Schranke aus Z(V) ist gleich dem Wert), über alle Ränge und Zufallsstarts einer Stichprobe."""
    rng = np.random.default_rng(5)
    checked = 0
    for n, seed in [(10, 0), (14, 1), (18, 2)]:
        W = G.gnp(n, 0.4, seed)
        sdp = S.solve(W).value
        for p in (1, 2, 3, 4, 6):
            for _ in range(4):
                r = B.mixing(W, p, rng, tol=1e-10)
                c = B.certificate(W, r.V)
                assert c["bound"] >= r.value - 1e-9                                   # (Σ w + tr Z)/2 liegt stets über dem Wert des Laufs (Cauchy–Schwarz); ein Beweis für den SDP-Wert ist sie nur bei Z ⪰ 0
                if B.is_certified(c, 1e-10):
                    checked += 1
                    assert r.value == pytest.approx(sdp, abs=1e-3)
    assert checked > 10


def test_the_reference_value_equals_the_sdp_value_and_is_certified():
    for n, pe, seed in [(8, 0.5, 1), (12, 0.4, 2), (16, 0.3, 3), (20, 0.3, 3)]:
        W = G.gnp(n, pe, seed)
        ref = B.reference(W, np.random.default_rng(seed))
        assert ref["certified"] and ref["value"] == pytest.approx(S.solve(W).value, abs=1e-4) and ref["p"] == B.rank_for_theory(n) + 1 and ref["V"].shape == (n, ref["p"])


def test_the_theory_rank_is_the_smallest_p_with_p_p1_over_2_above_n():
    for n in (2, 5, 10, 20, 30, 40, 60, 80, 160, 320):
        p = B.rank_for_theory(n)
        assert p * (p + 1) // 2 > n and (p - 1) * p // 2 <= n
    assert [B.rank_for_theory(n) for n in (10, 20, 30, 40, 60, 80)] == [5, 6, 8, 9, 11, 13]
    assert B.sqrt_rank(20) == 7 and B.sqrt_rank(320) == 26 and B.sqrt_rank(2) == 2


def test_the_global_gap_and_the_threshold():
    assert B.global_gap(10.0) == 5e-3 and B.global_gap(8833.0) == pytest.approx(2e-5 * 8833.0)
    assert B.cert_threshold(1e-8) == pytest.approx(1e-3) and B.cert_threshold(1e-6) == pytest.approx(1e-2) and B.cert_threshold(1e-12) == pytest.approx(1e-5)


def test_a_smaller_tolerance_needs_at_least_as_many_sweeps():
    W = G.gnp(20, 0.3, 3)
    sweeps = []
    for tol in (1e-6, 1e-8, 1e-10, 1e-12):
        sweeps.append(B.mixing(W, 6, np.random.default_rng(0), tol=tol).sweeps)
    assert sweeps == sorted(sweeps) and sweeps[-1] > sweeps[0]


def test_the_same_seed_gives_the_same_run_and_a_start_matrix_is_used():
    W = G.gnp(10, 0.5, 0)
    a = B.mixing(W, 3, np.random.default_rng(7), tol=1e-9)
    b = B.mixing(W, 3, np.random.default_rng(7), tol=1e-9)
    assert np.array_equal(a.V, b.V) and a.sweeps == b.sweeps
    c = B.mixing(W, 3, np.random.default_rng(0), tol=1e-9, V0=a.V)
    assert c.sweeps <= 3 and c.value == pytest.approx(a.value, abs=1e-8)
    with pytest.raises(ValueError):
        B.mixing(W, 3, np.random.default_rng(0), V0=np.ones((10, 2)))


def test_bad_inputs_are_refused():
    with pytest.raises(ValueError):
        B.mixing(np.zeros((1, 1)), 2, np.random.default_rng(0))
    with pytest.raises(ValueError):
        B.mixing(G.cycle(5), 0, np.random.default_rng(0))
