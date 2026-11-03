"""Randfälle, an denen die Faktorisierung erfahrungsgemäß bricht (DEMO-PLAYBOOK §12)."""

import pickle

import numpy as np
import pytest

import kbm_bm as B
import kbm_constants as C
import kbm_evaluation as ev
import kbm_graphs as G
import kbm_sdp as S


def test_two_nodes_and_one_edge():
    W = G.gnp(2, 1.0, 0)
    for p in (1, 2, 3):
        r = B.mixing(W, p, np.random.default_rng(0), tol=1e-12)
        assert r.value == pytest.approx(1.0, abs=1e-9)
    assert B.certificate(W, B.mixing(W, 2, np.random.default_rng(0), tol=1e-12).V)["lam_min"] > -1e-9


def test_a_graph_without_edges():
    W = np.zeros((6, 6))
    r = B.mixing(W, 3, np.random.default_rng(0))
    c = B.certificate(W, r.V)
    assert r.value == 0.0 and r.sweeps == 1 and r.converged and c["lam_min"] == 0.0 and c["bound"] == 0.0 and B.is_certified(c, 1e-8)
    seed = next(s for s in range(200) if G.total_weight(G.gnp(10, 0.1, s)) == 0.0)
    m = ev.main_view(ev.Settings(n=10, p=0.1, seed=seed))
    assert m["value"] == 0.0 and m["ref"]["value"] == 0.0 and m["global"] and m["certified"] and m["edges"] == 0
    assert ev.verdict_main(ev.Settings(n=10, p=0.1, seed=seed), m)[0] == "success"


def test_the_scan_survives_graphs_without_edges():
    seed = next(s for s in range(200) if all(G.total_weight(G.gnp(10, 0.1, s + g)) == 0.0 for g in range(C.SCAN_GRAPHS)) or s > 150)
    sc = ev.scan_view(ev.Settings(n=10, p=0.1, seed=seed))
    assert set(sc["ranks"]) == {1, 2, 3, 4, 5, 6} and all(0.0 <= v <= 1.0 for v in sc["rate"].values())
    t = ev.certificate_table(sc)
    assert t["total"] == len(sc["runs"])
    assert ev.verdict_scan(sc, ev.Settings(n=10, p=0.1, seed=seed))[0] == "info" and ev.verdict_certificate(t, ev.Settings())[0] in ("success", "warning")


def test_isolated_nodes_do_not_change_the_value_and_their_rows_stay_put():
    A = G.gnp(12, 0.5, 1)
    W = np.zeros((16, 16))
    W[:12, :12] = A
    rng = np.random.default_rng(0)
    big = B.mixing(W, 5, np.random.default_rng(3), tol=1e-12)
    ref = B.reference(A, rng)
    assert big.value == pytest.approx(ref["value"], abs=1e-3)
    start = B.random_rows(16, 5, np.random.default_rng(3))
    assert np.allclose(big.V[12:], start[12:])                                              # keine Nachbarn: die Zeile wird nicht angefasst


def test_a_disconnected_graph_is_the_sum_of_its_parts():
    W = np.zeros((8, 8))
    W[:5, :5] = G.cycle(5)
    W[5:, 5:] = G.complete(3)
    ref = B.reference(W, np.random.default_rng(0))
    assert ref["certified"] and ref["value"] == pytest.approx(S.solve(G.cycle(5)).value + S.solve(G.complete(3)).value, abs=1e-4)


def test_scaling_the_weights_scales_the_value_and_keeps_the_rows():
    W = G.gnp(12, 0.5, 2)
    a = B.mixing(W, 3, np.random.default_rng(1), tol=1e-10)
    b = B.mixing(10.0 * W, 3, np.random.default_rng(1), tol=1e-10)
    assert np.allclose(a.V, b.V, atol=1e-9) and b.value == pytest.approx(10.0 * a.value, rel=1e-9) and a.sweeps == b.sweeps


def test_relabelling_the_nodes_keeps_the_reference_value():
    W = G.gnp(14, 0.4, 5)
    perm = np.random.default_rng(0).permutation(14)
    P = W[np.ix_(perm, perm)]
    assert B.reference(W, np.random.default_rng(0))["value"] == pytest.approx(B.reference(P, np.random.default_rng(1))["value"], abs=1e-4)


def test_a_rank_above_n_is_capped_in_the_views():
    m = ev.main_view(ev.Settings(n=10, rank=12))
    assert m["rank"] == 10 and m["V"].shape == (10, 10) and m["global"]


def test_weights_with_a_zero_entry_and_a_complete_graph():
    W = G.complete(6)
    W[0, 1] = W[1, 0] = 0.0
    r = B.reference(W, np.random.default_rng(0))
    assert r["certified"] and r["value"] == pytest.approx(S.solve(W).value, abs=1e-4)
    k = B.mixing(G.complete(7), 2, np.random.default_rng(0), tol=1e-12)
    assert k.value == pytest.approx(49.0 / 4.0, abs=1e-6)                                  # K_n mit p = 2: n²/4 (Ecken des regelmäßigen Vielecks)


def test_the_extreme_settings_run():
    for n in (C.NS[0], C.NS[-1]):
        for rank in (C.RANK_CHOICES[0], C.RANK_CHOICES[-1]):
            m = ev.main_view(ev.Settings(n=n, rank=rank, p=C.P_CHOICES[0] if n == C.NS[-1] else C.P_CHOICES[-1]))
            assert m["value"] <= m["ref"]["value"] + 1e-3 and np.isfinite(m["lam_min"])


def test_same_settings_give_the_same_results():
    a, b = ev.main_view(ev.Settings(n=30, seed=7)), ev.main_view(ev.Settings(n=30, seed=7))
    assert a["value"] == b["value"] and a["sweeps"] == b["sweeps"] and np.array_equal(a["V"], b["V"])
    assert not np.array_equal(a["W"], ev.main_view(ev.Settings(n=30, seed=8))["W"])


def test_cached_results_can_be_pickled():
    """st.cache_data speichert die Ergebnisse als Pickle: keine lokalen Funktionen oder Closures in den Ergebnissen."""
    s = ev.Settings(n=20)
    m = ev.main_view(s)
    for obj in (m, ev.small_view("fuenfeck", 2, 3), ev.scan_view(s), ev.rounding_view(m, 3), ev.memory_rows(), ev.rew_rows()):
        assert pickle.loads(pickle.dumps(obj)) is not None
