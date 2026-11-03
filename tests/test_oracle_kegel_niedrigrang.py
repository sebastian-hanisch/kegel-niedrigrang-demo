"""Orakel (DEMO-PLAYBOOK §12, ORACLE-PLAYBOOK §5): die Faktorisierung gegen einen anderen Rechenweg.

- cvxpy formuliert das SDP **selbst** (`cp.Variable((n, n), PSD=True)`, `cp.diag(X) == 1`, Zielfunktion über die Kanten) und löst es mit Clarabel und SCS: gleicher Wert wie der zertifizierte Referenzwert aus Burer–Monteiro (C₅, Petersen, Zufallsgraphen)
- die Dualschranke aus Z(V) bei Zertifikat „ja“ gleich dem cvxpy-Optimum
- ADMM in Matrixform gleicher Wert wie Clarabel

Fehlt cvxpy im Test-venv, überspringen sich diese Tests (pytest.importorskip), sie raten nie."""

import numpy as np
import pytest

import kbm_admm as A
import kbm_bm as B
import kbm_graphs as G


def _cp():
    cp = pytest.importorskip("cvxpy")
    if "CLARABEL" not in cp.installed_solvers():
        pytest.skip("Clarabel fehlt")
    return cp


def _cvxpy_value(cp, W, solver=None):
    n = len(W)
    X = cp.Variable((n, n), PSD=True)
    obj = sum(W[i, j] * (1 - X[i, j]) / 2 for i in range(n) for j in range(i + 1, n) if W[i, j] != 0)
    prob = cp.Problem(cp.Maximize(obj), [cp.diag(X) == 1])
    kw = dict(solver=cp.CLARABEL, tol_gap_abs=1e-10, tol_gap_rel=1e-10, tol_feas=1e-10) if solver is None else dict(solver=solver, eps=1e-9, max_iters=200000)
    prob.solve(**kw)
    return float(prob.value)


@pytest.mark.filterwarnings("ignore:Solution may be inaccurate")
@pytest.mark.parametrize("name,W", [("C5", G.cycle(5)), ("Petersen", G.petersen()), ("K4", G.complete(4)), ("G(10, ½)", G.gnp(10, 0.5, 2)), ("G(16, ½)", G.gnp(16, 0.5, 3)), ("G(20, 0,3)", G.gnp(20, 0.3, 3))])
def test_cvxpy_finds_the_same_value_as_the_certified_reference(name, W):
    cp = _cp()
    ref = B.reference(W, np.random.default_rng(0))
    assert ref["certified"] and ref["value"] == pytest.approx(_cvxpy_value(cp, W), abs=1e-4), name


@pytest.mark.filterwarnings("ignore:Solution may be inaccurate")
def test_scs_the_admm_of_cvxpy_finds_the_same_value():
    cp = _cp()
    if "SCS" not in cp.installed_solvers():
        pytest.skip("SCS fehlt")
    W = G.gnp(12, 0.5, 1)
    assert B.reference(W, np.random.default_rng(0))["value"] == pytest.approx(_cvxpy_value(cp, W, solver=cp.SCS), abs=1e-4)


@pytest.mark.filterwarnings("ignore:Solution may be inaccurate")
def test_the_bound_from_z_of_v_equals_the_cvxpy_optimum_when_certified():
    cp = _cp()
    for W in (G.cycle(5), G.gnp(10, 0.5, 2), G.gnp(14, 0.4, 4)):
        r = B.mixing(W, B.rank_for_theory(len(W)) + 1, np.random.default_rng(0), tol=1e-12)
        c = B.certificate(W, r.V)
        assert B.is_certified(c, 1e-12) and c["bound"] == pytest.approx(_cvxpy_value(cp, W), abs=1e-4)


@pytest.mark.filterwarnings("ignore:Solution may be inaccurate")
def test_a_trap_is_below_the_cvxpy_optimum():
    cp = _cp()
    W = G.petersen()
    opt = _cvxpy_value(cp, W)
    assert max(B.mixing(W, 3, np.random.default_rng(s), tol=1e-10).value for s in range(10)) < opt - 0.03


@pytest.mark.filterwarnings("ignore:Solution may be inaccurate")
def test_matrix_admm_equals_clarabel():
    cp = _cp()
    W = G.gnp(12, 0.5, 5)
    assert A.admm(W, tol=1e-9).value == pytest.approx(_cvxpy_value(cp, W), abs=1e-4)
